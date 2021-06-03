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
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Callable, List, TypeVar

from ...services.cluster import ClusterAPI
from ...services.cluster.backends import register_cluster_backend, \
    AbstractClusterBackend
from .config import MarsReplicationConfig, MarsSupervisorsConfig

logger = logging.getLogger(__name__)
RetType = TypeVar('RetType')


@register_cluster_backend
class K8SClusterBackend(AbstractClusterBackend):
    name = "k8s"

    def __init__(self, k8s_config=None, k8s_namespace=None):
        from kubernetes import client

        self._k8s_config = k8s_config

        verify_ssl = bool(int(os.environ.get('KUBE_VERIFY_SSL', '1').strip('"')))
        if not verify_ssl:
            c = client.Configuration()
            c.verify_ssl = False
            client.Configuration.set_default(c)

        self._k8s_namespace = k8s_namespace or os.environ.get('MARS_K8S_POD_NAMESPACE') or 'default'
        self._full_label_selector = None
        self._client = client.CoreV1Api(client.ApiClient(self._k8s_config))
        self._executor = ThreadPoolExecutor(2)

        self._service_pod_to_ep = dict()

    @classmethod
    async def create(cls, lookup_address: str) -> "AbstractClusterBackend":
        from kubernetes import config, client

        if lookup_address is None:
            k8s_namespace = None
            k8s_config = config.load_incluster_config()
        else:
            address_parts = lookup_address.rsplit('?', 1)
            k8s_namespace = None if len(address_parts) == 1 else address_parts[1]

            k8s_config = client.Configuration()
            if '://' in address_parts[0]:
                k8s_config.host = address_parts[0]
            else:
                config.load_kube_config(address_parts[0], client_configuration=k8s_config)
        return cls(k8s_config, k8s_namespace)

    def __reduce__(self):
        return type(self), (self._k8s_config, self._k8s_namespace)

    async def _spawn_in_pool(self, func: Callable[..., RetType], *args, **kwargs) -> RetType:
        def func_in_thread():
            return func(*args, **kwargs)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func_in_thread)

    async def _get_label_selector(self, service_type):
        if self._full_label_selector is not None:
            return self._full_label_selector

        selectors = [f'mars/service-type={service_type}']
        if 'MARS_K8S_GROUP_LABELS' in os.environ:
            group_labels = os.environ['MARS_K8S_GROUP_LABELS'].split(',')
            cur_pod_info = (await self._spawn_in_pool(self._client.read_namespaced_pod,
                                                      os.environ['MARS_K8S_POD_NAME'],
                                                      namespace=self._k8s_namespace)).to_dict()
            for label in group_labels:
                label_val = cur_pod_info['metadata']['labels'][label]
                selectors.append(f'{label}={label_val}')
        self._full_label_selector = ','.join(selectors)
        logger.debug('Using pod selector %s', self._full_label_selector)
        return self._full_label_selector

    def _extract_pod_name_ep(self, pod_data):
        pod_ip = pod_data["status"].get("podIP") or pod_data["status"].get("pod_ip")
        ports_def = pod_data['spec']['containers'][0]['ports'][0]
        svc_port = ports_def.get('containerPort') or ports_def.get('container_port')
        return pod_data['metadata']['name'], f'{pod_ip}:{svc_port}'

    @staticmethod
    def _extract_pod_ready(obj_data):
        return obj_data['status']['phase'] == 'Running'

    async def _get_pod_to_ep(self, service_type: str, filter_ready: bool = False):
        query = (await self._spawn_in_pool(
            self._client.list_namespaced_pod,
            namespace=self._k8s_namespace,
            label_selector=await self._get_label_selector(service_type),
            resource_version='0'
        )).to_dict()

        result = dict()
        for el in query['items']:
            name, pod_ep = self._extract_pod_name_ep(el)
            if filter_ready and pod_ep is not None and not self._extract_pod_ready(el):
                pod_ep = None
            result[name] = pod_ep
        return result

    async def _get_endpoints_by_service_type(self, service_type: str, update: bool = False,
                                             filter_ready: bool = True):
        if not self._service_pod_to_ep.get(service_type) or update:
            self._service_pod_to_ep[service_type] = \
                await self._get_pod_to_ep(service_type, filter_ready=filter_ready)
        return sorted(a for a in self._service_pod_to_ep[service_type].values() if a is not None)

    async def _watch_service(self, service_type, linger=10):
        from urllib3.exceptions import ReadTimeoutError
        from kubernetes import watch

        cur_pods = set(await self._get_endpoints_by_service_type(service_type, update=True))
        w = watch.Watch()

        pod_to_ep = self._service_pod_to_ep[service_type]
        while True:
            # when some pods are not ready, we refresh faster
            linger_seconds = linger() if callable(linger) else linger
            streamer = w.stream(
                self._client.list_namespaced_pod,
                namespace=self._k8s_namespace,
                label_selector=await self._get_label_selector(service_type),
                timeout_seconds=linger_seconds,
                resource_version='0'
            )
            while True:
                try:
                    event = await self._spawn_in_pool(next, streamer, StopIteration)
                    if event is StopIteration:
                        # todo change this into a continuous watch
                        #  when watching in a master node is implemented
                        return
                except (ReadTimeoutError, StopIteration):
                    new_pods = set(await self._get_endpoints_by_service_type(service_type, update=True))
                    if new_pods != cur_pods:
                        cur_pods = new_pods
                        yield await self._get_endpoints_by_service_type(service_type, update=False)
                    break
                except:  # noqa: E722  # pragma: no cover  # pylint: disable=bare-except
                    logger.exception('Unexpected error when watching on kubernetes')
                    break

                obj_dict = event['object'].to_dict()
                pod_name, endpoint = self._extract_pod_name_ep(obj_dict)
                pod_to_ep[pod_name] = endpoint \
                    if endpoint and self._extract_pod_ready(obj_dict) else None
                yield await self._get_endpoints_by_service_type(service_type, update=False)

    def watch_supervisors(self) -> AsyncGenerator[List[str], None]:
        return self._watch_service(MarsSupervisorsConfig.rc_name)

    async def get_supervisors(self) -> List[str]:
        return await self._get_endpoints_by_service_type(MarsSupervisorsConfig.rc_name, update=False)

    async def get_expected_supervisors(self) -> List[str]:
        expected_supervisors = await self._get_endpoints_by_service_type(
            MarsSupervisorsConfig.rc_name, filter_ready=False)
        return expected_supervisors


class K8SServiceMixin:
    @staticmethod
    def write_pid_file():
        with open('/tmp/mars-service.pid', 'w') as pid_file:
            pid_file.write(str(os.getpid()))

    async def wait_all_supervisors_ready(self):
        """
        Wait till all containers are ready, both in kubernetes and in ClusterInfoActor
        """
        cluster_api = None
        while True:
            try:
                cluster_api = await ClusterAPI.create(self.args.endpoint)
                break
            except:  # noqa: E722  # pylint: disable=bare-except  # pragma: no cover
                await asyncio.sleep(0.1)
                continue

        assert cluster_api is not None
        await cluster_api.wait_all_supervisors_ready()

    async def start_readiness_server(self):
        readiness_port = os.environ.get('MARS_K8S_READINESS_PORT',
                                        MarsReplicationConfig.default_readiness_port)
        self._readiness_server = await asyncio.start_server(
            lambda r, w: None, port=readiness_port)

    async def stop_readiness_server(self):
        self._readiness_server.close()
        await self._readiness_server.wait_closed()
