# -*- coding: utf-8 -*-
# Copyright 1999-2020 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import random
import time

from ...actors import new_client, FunctionActor

logger = logging.getLogger(__name__)


class K8SPodsIPWatcher(object):
    """
    Pods watcher class, compatible with SchedulerDiscoverer
    """
    dynamic = True

    def __init__(self, k8s_config=None, k8s_namespace=None, label_selector=None):
        from kubernetes import config, client
        from gevent.threadpool import ThreadPool

        if k8s_config is not None:
            self._k8s_config = k8s_config
        elif os.environ.get('KUBE_API_ADDRESS'):
            self._k8s_config = client.Configuration()
            self._k8s_config.host = os.environ['KUBE_API_ADDRESS']
        else:
            self._k8s_config = config.load_incluster_config()

        self._k8s_namespace = k8s_namespace or os.environ.get('MARS_K8S_POD_NAMESPACE') or 'default'
        self._label_selector = label_selector
        self._client = client.CoreV1Api(client.ApiClient(self._k8s_config))
        self._pool = ThreadPool(1)

        self._pod_to_ep = None

    def __reduce__(self):
        return type(self), (self._k8s_config, self._k8s_namespace, self._label_selector)

    def _extract_pod_name_ep(self, pod_data):
        svc_port = pod_data['spec']['containers'][0]['ports'][0]['container_port']
        return pod_data['metadata']['name'], '%s:%s' % (pod_data['status']['pod_ip'], svc_port)

    @staticmethod
    def _extract_pod_ready(obj_data):
        # if conditions not supported, always return True
        if 'status' not in obj_data or 'conditions' not in obj_data['status']:
            return True
        return any(cond['type'] == 'Ready' and cond['status'] == 'True'
                   for cond in obj_data['status']['conditions'])

    def _get_pod_to_ep(self):
        query = self._pool.spawn(self._client.list_namespaced_pod,
                                 namespace=self._k8s_namespace,
                                 label_selector=self._label_selector).result().to_dict()
        result = dict()
        for el in query['items']:
            name, pod_ep = self._extract_pod_name_ep(el)
            if pod_ep is not None and not self._extract_pod_ready(el):
                pod_ep = None
            result[name] = pod_ep
        return result

    def get(self, update=False):
        if self._pod_to_ep is None or update:
            self._pod_to_ep = self._get_pod_to_ep()
        return sorted(a for a in self._pod_to_ep.values() if a is not None)

    def is_all_ready(self):
        self.get(True)
        return all(a is not None for a in self._pod_to_ep.values())

    def watch(self):
        from urllib3.exceptions import ReadTimeoutError
        from kubernetes import watch

        cur_pods = set(self.get(True))
        w = watch.Watch()

        while True:
            # when some schedulers are not ready, we refresh faster
            linger = 10 if self.is_all_ready() else 1
            streamer = w.stream(self._client.list_namespaced_pod,
                                namespace=self._k8s_namespace,
                                label_selector=self._label_selector,
                                timeout_seconds=linger)
            while True:
                try:
                    event = self._pool.spawn(next, streamer, StopIteration).result()
                    if event is StopIteration:
                        raise StopIteration
                except (ReadTimeoutError, StopIteration):
                    new_pods = set(self.get(True))
                    if new_pods != cur_pods:
                        cur_pods = new_pods
                        yield self.get(False)
                    break
                except:  # noqa: E722
                    logger.exception('Unexpected error when watching on kubernetes')
                    break

                obj_dict = event['object'].to_dict()
                pod_name, endpoint = self._extract_pod_name_ep(obj_dict)
                self._pod_to_ep[pod_name] = endpoint \
                    if endpoint and self._extract_pod_ready(obj_dict) else None
                yield self.get(False)


class ReadinessActor(FunctionActor):
    """
    Dummy actor indicating service start
    """
    @classmethod
    def default_uid(cls):
        return 'k:0:%s' % cls.__name__


class K8SServiceMixin:
    @staticmethod
    def write_pid_file():
        with open('/tmp/mars-service.pid', 'w') as pid_file:
            pid_file.write(str(os.getpid()))

    def wait_all_schedulers_ready(self):
        """
        Wait till all containers are ready, both in kubernetes and in ClusterInfoActor
        """
        from ...scheduler.utils import SchedulerClusterInfoActor

        # check if all schedulers are ready using Kubernetes API
        sleep_fun = (getattr(self, 'pool', None) or time).sleep
        while not self.scheduler_discoverer.is_all_ready():
            sleep_fun(1)
        kube_schedulers = self.scheduler_discoverer.get()

        logger.debug('Schedulers all ready in kubernetes, waiting ClusterInfoActor to be ready')
        # check if all schedulers are registered in ClusterInfoActor
        actor_client = new_client()
        while True:
            cluster_info = actor_client.actor_ref(
                SchedulerClusterInfoActor.default_uid(), address=random.choice(kube_schedulers))
            cluster_info_schedulers = cluster_info.get_schedulers()
            if set(cluster_info_schedulers) == set(kube_schedulers):
                from ...cluster_info import INITIAL_SCHEDULER_FILE
                with open(INITIAL_SCHEDULER_FILE, 'w') as scheduler_file:
                    scheduler_file.write(','.join(cluster_info_schedulers))

                logger.debug('Scheduler detection finished. Result: %r', kube_schedulers)
                break
            sleep_fun(1)  # pragma: no cover

    def create_scheduler_discoverer(self):
        self.scheduler_discoverer = K8SPodsIPWatcher(label_selector='name=marsscheduler')
