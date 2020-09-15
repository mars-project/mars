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

    def __init__(self, k8s_config=None, k8s_namespace=None):
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
        self._full_label_selector = None
        self._client = client.CoreV1Api(client.ApiClient(self._k8s_config))
        self._pool = ThreadPool(1)

        self._service_pod_to_ep = dict()

    def __reduce__(self):
        return type(self), (self._k8s_config, self._k8s_namespace)

    def _get_label_selector(self, service_type):
        if self._full_label_selector is not None:
            return self._full_label_selector

        selectors = [f'mars/service-type={service_type}']
        if 'MARS_K8S_GROUP_LABELS' in os.environ:
            group_labels = os.environ['MARS_K8S_GROUP_LABELS'].split(',')
            cur_pod_info = self._pool.spawn(self._client.read_namespaced_pod,
                                            os.environ['MARS_K8S_POD_NAME'],
                                            namespace=self._k8s_namespace).result().to_dict()
            for label in group_labels:
                label_val = cur_pod_info['metadata']['labels'][label]
                selectors.append(f'{label}={label_val}')
        self._full_label_selector = ','.join(selectors)
        logger.debug('Using pod selector %s', self._full_label_selector)
        return self._full_label_selector

    def _extract_pod_name_ep(self, pod_data):
        pod_ip = pod_data["status"]["pod_ip"]
        svc_port = pod_data['spec']['containers'][0]['ports'][0]['container_port']
        return pod_data['metadata']['name'], f'{pod_ip}:{svc_port}'

    @staticmethod
    def _extract_pod_ready(obj_data):
        if obj_data['status']['phase'] != 'Running':
            return False
        # if conditions not supported, always return True
        if 'status' not in obj_data or 'conditions' not in obj_data['status']:
            return True
        return any(cond['type'] == 'Ready' and cond['status'] == 'True'
                   for cond in obj_data['status']['conditions'])

    def _get_pod_to_ep(self, service_type):
        query = self._pool.spawn(
            self._client.list_namespaced_pod,
            namespace=self._k8s_namespace,
            label_selector=self._get_label_selector(service_type)
        ).result().to_dict()

        result = dict()
        for el in query['items']:
            name, pod_ep = self._extract_pod_name_ep(el)
            if pod_ep is not None and not self._extract_pod_ready(el):
                pod_ep = None
            result[name] = pod_ep
        return result

    def _get_endpoints_by_service_type(self, service_type, update=False):
        if not self._service_pod_to_ep.get(service_type) or update:
            self._service_pod_to_ep[service_type] = self._get_pod_to_ep(service_type)
        return sorted(a for a in self._service_pod_to_ep[service_type].values() if a is not None)

    def get_schedulers(self, update=False):
        from .config import MarsSchedulersConfig
        return self._get_endpoints_by_service_type(MarsSchedulersConfig.rc_name, update=update)

    def is_all_schedulers_ready(self):
        from .config import MarsSchedulersConfig
        self.get_schedulers(True)
        pod_to_ep = self._service_pod_to_ep[MarsSchedulersConfig.rc_name]
        if not pod_to_ep:
            return False
        return all(a is not None for a in pod_to_ep.values())

    def _watch_service(self, service_type, linger=10):
        from urllib3.exceptions import ReadTimeoutError
        from kubernetes import watch

        cur_pods = set(self._get_endpoints_by_service_type(service_type, update=True))
        w = watch.Watch()

        pod_to_ep = self._service_pod_to_ep[service_type]
        while True:
            # when some pods are not ready, we refresh faster
            linger_seconds = linger() if callable(linger) else linger
            streamer = w.stream(
                self._client.list_namespaced_pod,
                namespace=self._k8s_namespace,
                label_selector=self._get_label_selector(service_type),
                timeout_seconds=linger_seconds
            )
            while True:
                try:
                    event = self._pool.spawn(next, streamer, StopIteration).result()
                    if event is StopIteration:
                        raise StopIteration
                except (ReadTimeoutError, StopIteration):
                    new_pods = set(self._get_endpoints_by_service_type(service_type, update=True))
                    if new_pods != cur_pods:
                        cur_pods = new_pods
                        yield self._get_endpoints_by_service_type(service_type, update=False)
                    break
                except:  # noqa: E722  # pragma: no cover  # pylint: disable=bare-except
                    logger.exception('Unexpected error when watching on kubernetes')
                    break

                obj_dict = event['object'].to_dict()
                pod_name, endpoint = self._extract_pod_name_ep(obj_dict)
                pod_to_ep[pod_name] = endpoint \
                    if endpoint and self._extract_pod_ready(obj_dict) else None
                yield self._get_endpoints_by_service_type(service_type, update=False)

    def watch_schedulers(self):
        from .config import MarsSchedulersConfig
        return self._watch_service(MarsSchedulersConfig.rc_name)

    def watch_workers(self):
        from .config import MarsWorkersConfig
        return self._watch_service(MarsWorkersConfig.rc_name)

    def rescale_workers(self, new_scale):
        from .config import MarsWorkersConfig
        self._client.patch_namespaced_replication_controller_scale(
            MarsWorkersConfig.rc_name, self._k8s_namespace, {"spec": {"replicas": new_scale}}
        )


class ReadinessActor(FunctionActor):
    """
    Dummy actor indicating service start
    """
    @classmethod
    def default_uid(cls):
        return f'k:0:{cls.__name__}'


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
        while not self.scheduler_discoverer.is_all_schedulers_ready():
            sleep_fun(1)
        kube_schedulers = self.scheduler_discoverer.get_schedulers()

        logger.debug('Schedulers all ready in kubernetes, waiting ClusterInfoActor to be ready')
        # check if all schedulers are registered in ClusterInfoActor
        actor_client = new_client()
        while True:
            try:
                cluster_info = actor_client.actor_ref(
                    SchedulerClusterInfoActor.default_uid(), address=random.choice(kube_schedulers))
                cluster_info_schedulers = cluster_info.get_schedulers()
            except ConnectionError:  # pragma: no cover
                time.sleep(0.1)
                continue

            if set(cluster_info_schedulers) == set(kube_schedulers):
                from ...cluster_info import INITIAL_SCHEDULER_FILE
                with open(INITIAL_SCHEDULER_FILE, 'w') as scheduler_file:
                    scheduler_file.write(','.join(cluster_info_schedulers))

                logger.debug('Scheduler detection finished. Result: %r', kube_schedulers)
                break
            sleep_fun(1)  # pragma: no cover

    def create_scheduler_discoverer(self):
        self.scheduler_discoverer = K8SPodsIPWatcher()
