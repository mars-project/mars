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

import asyncio
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

from ...actors import new_client, FunctionActor

logger = logging.getLogger(__name__)


class K8SPodsIPWatcher(object):
    """
    Pods watcher class, compatible with SchedulerDiscoverer
    """
    dynamic = True

    def __init__(self, k8s_config=None, k8s_namespace=None, label_selector=None):
        from kubernetes import config, client

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
        self._pool = ThreadPoolExecutor(1)

        self._pod_to_ep = None

    def __reduce__(self):
        return type(self), (self._k8s_config, self._k8s_namespace, self._label_selector)

    @staticmethod
    def _extract_pod_name_ep(obj_data):
        svc_port = obj_data['spec']['containers'][0]['ports'][0]['container_port']
        return obj_data['metadata']['name'], '%s:%s' % (obj_data['status']['pod_ip'], svc_port)

    @staticmethod
    def _extract_pod_ready(obj_data):
        # if conditions not supported, always return True
        if 'status' not in obj_data or 'conditions' not in obj_data['status']:
            return True
        return any(cond['type'] == 'Ready' and cond['status'] == 'True'
                   for cond in obj_data['status']['conditions'])

    async def _get_pod_to_ep(self):
        query = (await asyncio.get_event_loop().run_in_executor(
            self._pool, lambda: self._client.list_namespaced_pod(
                namespace=self._k8s_namespace, label_selector=self._label_selector))).to_dict()
        result = dict()
        for el in query['items']:
            name, pod_ep = self._extract_pod_name_ep(el)
            if not self._extract_pod_ready(el):
                pod_ep = None
            result[name] = pod_ep
        return result

    async def get(self, update=False):
        if self._pod_to_ep is None or update:
            self._pod_to_ep = await self._get_pod_to_ep()
        return sorted(a for a in self._pod_to_ep.values() if a is not None)

    async def is_all_ready(self):
        await self.get(True)
        return all(a is not None for a in self._pod_to_ep.values())

    def watch(self):
        from urllib3.exceptions import ReadTimeoutError
        from kubernetes import watch

        cur_pods = set()
        this = self
        streamer = None
        w = watch.Watch()

        class _AsyncIterator:
            async def __aiter__(self):
                nonlocal cur_pods
                cur_pods = set(await this.get(True))

            async def __anext__(self):
                nonlocal cur_pods, streamer
                while True:
                    if streamer is None:
                        linger = 10 if await this.is_all_ready() else 1
                        streamer = w.stream(this._client.list_namespaced_pod,
                                            namespace=this._k8s_namespace,
                                            label_selector=this._label_selector,
                                            timeout_seconds=linger)
                    try:
                        event = await asyncio.get_event_loop().run_in_executor(
                            this._pool, next, streamer, StopIteration)
                        if event is StopIteration:
                            raise StopIteration
                    except (ReadTimeoutError, StopIteration):
                        new_pods = set(await this.get(True))
                        if new_pods != cur_pods:
                            cur_pods = new_pods
                            return await this.get(False)
                        streamer = None
                        continue
                    except:  # noqa: E722
                        logger.exception('Unexpected error when watching on kubernetes')
                        streamer = None
                        continue

                    obj_dict = event['object'].to_dict()
                    pod_name, endpoint = this._extract_pod_name_ep(obj_dict)
                    this._pod_to_ep[pod_name] = endpoint \
                        if endpoint and this._extract_pod_ready(obj_dict) else None
                    return await this.get(False)

        return _AsyncIterator()


class ReadinessActor(FunctionActor):
    """
    Dummy actor indicating service start
    """
    @classmethod
    def default_uid(cls):
        return 'k:0:%s' % cls.__name__


class K8SServiceMixin(object):
    @staticmethod
    def write_pid_file():
        with open('/tmp/mars-service.pid', 'w') as pid_file:
            pid_file.write(str(os.getpid()))

    async def wait_all_schedulers_ready(self):
        """
        Wait till all containers are ready, both in kubernetes and in ClusterInfoActor
        """
        from ...scheduler.utils import SchedulerClusterInfoActor

        # check if all schedulers are ready using Kubernetes API
        while not await self.scheduler_discoverer.is_all_ready():
            await asyncio.sleep(0.5)
        kube_schedulers = await self.scheduler_discoverer.get()

        logger.debug('Schedulers all ready in kubernetes, waiting ClusterInfoActor to be ready')
        # check if all schedulers are registered in ClusterInfoActor
        actor_client = new_client()
        while True:
            cluster_info = actor_client.actor_ref(
                SchedulerClusterInfoActor.default_uid(), address=random.choice(kube_schedulers))
            cluster_info_schedulers = await cluster_info.get_schedulers()
            if set(cluster_info_schedulers) == set(kube_schedulers):
                break
            await asyncio.sleep(0.5)  # pragma: no cover

    def create_scheduler_discoverer(self):
        self.scheduler_discoverer = K8SPodsIPWatcher(label_selector='name=marsscheduler')
