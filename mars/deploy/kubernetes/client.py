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
import random
import time
import uuid
from urllib.parse import urlparse

from ...session import new_session
from .config import NamespaceConfig, RoleConfig, RoleBindingConfig, ServiceConfig, \
    MarsSchedulersConfig, MarsWorkersConfig, MarsWebsConfig

try:
    from kubernetes.client.rest import ApiException as K8SApiException
except ImportError:  # pragma: no cover
    K8SApiException = None

logger = logging.getLogger(__name__)


class KubernetesClusterClient(object):
    def __init__(self, kube_api_client, namespace, endpoint):
        self._kube_api_client = kube_api_client
        self._namespace = namespace
        self._endpoint = endpoint
        self._session = new_session(endpoint).as_default()

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def namespace(self):
        return self._namespace

    @property
    def session(self):
        return self._session

    def stop(self, wait=False, timeout=0):
        from kubernetes.client import CoreV1Api
        api = CoreV1Api(self._kube_api_client)
        api.delete_namespace(self._namespace)
        if wait:
            start_time = time.time()
            while True:
                try:
                    api.read_namespace(self._namespace)
                except K8SApiException as ex:
                    if ex.status != 404:  # pragma: no cover
                        raise
                    break
                else:
                    time.sleep(1)
                    if time.time() - start_time > timeout:  # pragma: no cover
                        raise TimeoutError


def _get_free_namespace(kube_api):
    while True:
        namespace = 'mars-ns-%s' % uuid.uuid4().hex
        try:
            kube_api.read_namespace(namespace)
        except K8SApiException as ex:
            if ex.status != 404:  # pragma: no cover
                raise
            return namespace


def _create_node_port_service(kube_api, namespace):
    """
    :type kube_api: kubernetes.client.CoreV1Api
    """
    from .config import DEFAULT_SERVICE_PORT
    service_config = ServiceConfig(
        'marsservice', service_type='NodePort', port=DEFAULT_SERVICE_PORT,
        selector=dict(name=MarsWebsConfig.rc_name),
    )
    kube_api.create_namespaced_service(namespace, service_config.build())

    time.sleep(1)

    svc_data = kube_api.read_namespaced_service('marsservice', namespace).to_dict()
    port = svc_data['spec']['ports'][0]['node_port']

    web_pods = kube_api.list_namespaced_pod(
        namespace, label_selector='name=' + MarsWebsConfig.rc_name).to_dict()
    host_ip = random.choice(web_pods['items'])['status']['host_ip']

    # docker desktop use a VM to hold docker processes, hence
    # we need to use API address instead
    desktop_nodes = kube_api.list_node(field_selector='metadata.name=docker-desktop').to_dict()
    if desktop_nodes['items']:  # pragma: no cover
        addresses = set(
            addr_info['address']
            for addr_info in desktop_nodes['items'][0].get('status', {}).get('addresses', ())
        )
        if host_ip in addresses:
            host_ip = urlparse(kube_api.api_client.configuration.host).netloc.split(':', 1)[0]
    return 'http://%s:%s' % (host_ip, port)


def _get_ready_pod_count(kube_api, label_selector, namespace):
    query = kube_api.list_namespaced_pod(
        namespace=namespace, label_selector=label_selector).to_dict()
    cnt = 0
    for el in query['items']:
        if el['status']['phase'] in ('Error', 'Failed'):
            raise SystemError(el['status']['message'])
        if 'status' not in el or 'conditions' not in el['status']:
            cnt += 1
        if any(cond['type'] == 'Ready' and cond['status'] == 'True'
               for cond in el['status'].get('conditions') or ()):
            cnt += 1
    return cnt


def new_cluster(kube_api_client=None, image=None, scheduler_num=1, scheduler_cpu=None,
                scheduler_mem=None, worker_num=1, worker_cpu=None, worker_mem=None,
                worker_spill_paths=None, worker_cache_mem=None, min_worker_num=None,
                web_num=1, web_cpu=None, web_mem=None, service_type=None,
                timeout=None, log_when_fail=False, **kwargs):
    """
    :param kube_api_client: Kubernetes API client, can be created with ``new_client_from_config``
    :param image: Docker image to use, ``marsproject/mars:<mars version>`` by default
    :param scheduler_num: Number of schedulers in the cluster, 1 by default
    :param scheduler_cpu: Number of CPUs for every scheduler
    :param scheduler_mem: Memory size for every scheduler
    :param worker_num: Number of workers in the cluster, 1 by default
    :param worker_cpu: Number of CPUs for every worker
    :param worker_mem: Memory size for every worker
    :param worker_spill_paths: Spill paths for worker pods on hosts
    :param worker_cache_mem: Size or ratio of cache memory for every worker
    :param min_worker_num: Minimal ready workers
    :param web_num: Number of web services in the cluster, 1 by default
    :param web_cpu: Number of CPUs for every web service
    :param web_mem: Memory size for every web service
    :param service_type: Type of Kubernetes Service, currently only ``NodePort`` supported
    :param timeout: Timeout when creating clusters
    """
    from kubernetes import client as kube_client

    core_api = kube_client.CoreV1Api(kube_api_client)
    rbac_api = kube_client.RbacAuthorizationV1Api(kube_api_client)

    def _override_envs(src, updates):
        ret = src.copy()
        ret.update(updates)
        return ret

    service_type = service_type or 'NodePort'
    extra_volumes = kwargs.pop('extra_volumes', ())
    pre_stop_command = kwargs.pop('pre_stop_command', None)
    scheduler_extra_modules = kwargs.pop('scheduler_extra_modules', None)
    worker_extra_modules = kwargs.pop('worker_extra_modules', None)
    web_extra_modules = kwargs.pop('web_extra_modules', None)

    extra_envs = kwargs.pop('extra_volumes', dict())
    scheduler_extra_env = _override_envs(extra_envs, kwargs.pop('scheduler_extra_env', dict()))
    worker_extra_env = _override_envs(extra_envs, kwargs.pop('worker_extra_env', dict()))
    web_extra_env = _override_envs(extra_envs, kwargs.pop('web_extra_env', dict()))

    namespace = None
    try:
        # create namespace
        namespace = _get_free_namespace(core_api)
        core_api.create_namespace(NamespaceConfig(namespace).build())

        # create role and binding
        role_config = RoleConfig('mars-pod-reader', namespace, '', 'pods', 'get,watch,list')
        rbac_api.create_namespaced_role(namespace, role_config.build())
        role_binding_config = RoleBindingConfig(
            'mars-pod-reader-binding', namespace, 'mars-pod-reader', 'default')
        rbac_api.create_namespaced_role_binding(namespace, role_binding_config.build())

        # create replication controller of schedulers
        schedulers_config = MarsSchedulersConfig(
            scheduler_num, image=image, cpu=scheduler_cpu, memory=scheduler_mem,
            modules=scheduler_extra_modules, volumes=extra_volumes,
            pre_stop_command=pre_stop_command,
        )
        schedulers_config.add_simple_envs(scheduler_extra_env)
        core_api.create_namespaced_replication_controller(
            namespace, schedulers_config.build())

        # create replication controller of workers
        workers_config = MarsWorkersConfig(
            worker_num, image=image, cpu=worker_cpu, memory=worker_mem,
            spill_volumes=worker_spill_paths, modules=worker_extra_modules,
            volumes=extra_volumes, worker_cache_mem=worker_cache_mem,
            pre_stop_command=pre_stop_command,
        )
        workers_config.add_simple_envs(worker_extra_env)
        core_api.create_namespaced_replication_controller(
            namespace, workers_config.build())

        # create replication controller of webs
        webs_config = MarsWebsConfig(
            web_num, image=image, cpu=web_cpu, memory=web_mem, modules=web_extra_modules,
            volumes=extra_volumes, pre_stop_command=pre_stop_command,
        )
        webs_config.add_simple_envs(web_extra_env)
        core_api.create_namespaced_replication_controller(
            namespace, webs_config.build())

        # create service
        if service_type != 'NodePort':  # pragma: no cover
            raise NotImplementedError('Service type %s not supported' % service_type)

        # wait until schedulers and expected num of workers are ready
        min_worker_num = int(min_worker_num or worker_num)
        readies = [0, 0, 0]
        limits = [scheduler_num, min_worker_num, web_num]
        selectors = [
            'name=' + MarsSchedulersConfig.rc_name,
            'name=' + MarsWorkersConfig.rc_name,
            'name=' + MarsWebsConfig.rc_name,
        ]
        start_time = time.time()
        while True:
            all_satisfy = True
            for idx, selector in enumerate(selectors):
                if readies[idx] < limits[idx]:
                    all_satisfy = False
                    readies[idx] = _get_ready_pod_count(core_api, selector, namespace)
                    break
            if all_satisfy:
                break
            if timeout and timeout + start_time < time.time():
                raise TimeoutError('Wait kubernetes cluster start timeout')
            time.sleep(1)

        web_ep = _create_node_port_service(core_api, namespace)
        return KubernetesClusterClient(kube_api_client, namespace, web_ep)
    except:  # noqa: E722
        if log_when_fail:  # pargma: no cover
            pod_items = core_api.list_namespaced_pod(namespace).to_dict()
            for item in pod_items['items']:
                log_resp = core_api.read_namespaced_pod_log(
                    name=item['metadata']['name'], namespace=namespace)
                logger.error('Error when creating cluster:\n%s', log_resp)

        if namespace is not None:
            try:
                core_api.delete_namespace(namespace)
            except K8SApiException:  # pragma: no cover
                pass
        raise
