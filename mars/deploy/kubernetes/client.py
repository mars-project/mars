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

import functools
import logging
import random
import time
import uuid
from urllib.parse import urlparse

from ...session import new_session
from ..utils import wait_services_ready
from .config import NamespaceConfig, RoleConfig, RoleBindingConfig, ServiceConfig, \
    MarsSchedulersConfig, MarsWorkersConfig, MarsWebsConfig

try:
    from kubernetes.client.rest import ApiException as K8SApiException
except ImportError:  # pragma: no cover
    K8SApiException = None

logger = logging.getLogger(__name__)


class KubernetesClusterClient:
    def __init__(self, cluster):
        self._cluster = cluster
        self._endpoint = None
        self._session = None

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def namespace(self):
        return self._cluster.namespace

    @property
    def session(self):
        return self._session

    def start(self):
        self._endpoint = self._cluster.start()
        self._session = new_session(self._endpoint).as_default()

    def stop(self, wait=False, timeout=0):
        self._cluster.stop(wait=wait, timeout=timeout)

    def rescale_workers(self, new_scale, min_workers=None, wait=True, timeout=None):
        self._session._sess.rescale_workers(
            new_scale, min_workers=min_workers, wait=wait, timeout=timeout)


class KubernetesCluster:
    _scheduler_config_cls = MarsSchedulersConfig
    _worker_config_cls = MarsWorkersConfig
    _web_config_cls = MarsWebsConfig
    _default_service_port = 7103

    def __init__(self, kube_api_client=None, image=None, namespace=None,
                 scheduler_num=1, scheduler_cpu=None, scheduler_mem=None,
                 scheduler_mem_limit_ratio=None,
                 worker_num=1, worker_cpu=None, worker_mem=None,
                 worker_spill_paths=None, worker_cache_mem=None, min_worker_num=None,
                 worker_min_cache_mem=None, worker_mem_limit_ratio=None,
                 web_num=1, web_cpu=None, web_mem=None, web_mem_limit_ratio=None,
                 service_type=None, timeout=None, **kwargs):
        from kubernetes import client as kube_client

        self._api_client = kube_api_client
        self._core_api = kube_client.CoreV1Api(kube_api_client)
        self._rbac_api = kube_client.RbacAuthorizationV1Api(kube_api_client)

        self._namespace = namespace
        self._image = image
        self._timeout = timeout
        self._service_type = service_type or 'NodePort'
        self._extra_volumes = kwargs.pop('extra_volumes', ())
        self._pre_stop_command = kwargs.pop('pre_stop_command', None)
        self._log_when_fail = kwargs.pop('log_when_fail', False)

        extra_modules = kwargs.pop('extra_modules', None) or []
        extra_modules = extra_modules.split(',') if isinstance(extra_modules, str) \
            else extra_modules
        extra_envs = kwargs.pop('extra_env', None) or dict()
        extra_labels = kwargs.pop('extra_labels', None) or dict()
        service_port = kwargs.pop('service_port', None) or self._default_service_port

        def _override_modules(updates):
            modules = set(extra_modules)
            updates = updates.split(',') if isinstance(updates, str) \
                else updates
            modules.update(updates)
            return sorted(modules)

        def _override_dict(d, updates):
            updates = updates or dict()
            ret = d.copy()
            ret.update(updates)
            return ret

        _override_envs = functools.partial(_override_dict, extra_envs)
        _override_labels = functools.partial(_override_dict, extra_labels)

        self._scheduler_num = scheduler_num
        self._scheduler_cpu = scheduler_cpu
        self._scheduler_mem = scheduler_mem
        self._scheduler_mem_limit_ratio = scheduler_mem_limit_ratio
        self._scheduler_extra_modules = _override_modules(kwargs.pop('scheduler_extra_modules', []))
        self._scheduler_extra_env = _override_envs(kwargs.pop('scheduler_extra_env', None))
        self._scheduler_extra_labels = _override_labels(kwargs.pop('scheduler_extra_labels', None))
        self._scheduler_service_port = kwargs.pop('scheduler_service_port', None) or service_port

        self._worker_num = worker_num
        self._worker_cpu = worker_cpu
        self._worker_mem = worker_mem
        self._worker_mem_limit_ratio = worker_mem_limit_ratio
        self._worker_spill_paths = worker_spill_paths
        self._worker_cache_mem = worker_cache_mem
        self._worker_min_cache_men = worker_min_cache_mem
        self._min_worker_num = min_worker_num
        self._worker_extra_modules = _override_modules(kwargs.pop('worker_extra_modules', []))
        self._worker_extra_env = _override_envs(kwargs.pop('worker_extra_env', None))
        self._worker_extra_labels = _override_labels(kwargs.pop('worker_extra_labels', None))
        self._worker_service_port = kwargs.pop('worker_service_port', None) or service_port

        self._web_num = web_num
        self._web_cpu = web_cpu
        self._web_mem = web_mem
        self._web_mem_limit_ratio = web_mem_limit_ratio
        self._web_extra_modules = _override_modules(kwargs.pop('web_extra_modules', []))
        self._web_extra_env = _override_envs(kwargs.pop('web_extra_env', None))
        self._web_extra_labels = _override_labels(kwargs.pop('web_extra_labels', None))
        self._web_service_port = kwargs.pop('web_service_port', None) or service_port

    @property
    def namespace(self):
        return self._namespace

    def _get_free_namespace(self):
        while True:
            namespace = 'mars-ns-' + str(uuid.uuid4().hex)
            try:
                self._core_api.read_namespace(namespace)
            except K8SApiException as ex:
                if ex.status != 404:  # pragma: no cover
                    raise
                return namespace

    def _create_kube_service(self):
        if self._service_type != 'NodePort':  # pragma: no cover
            raise NotImplementedError(f'Service type {self._service_type} not supported')

        service_config = ServiceConfig(
            'marsservice', service_type='NodePort', port=self._web_service_port,
            selector={'mars/service-type': MarsWebsConfig.rc_name},
        )
        self._core_api.create_namespaced_service(self._namespace, service_config.build())

        time.sleep(1)

        svc_data = self._core_api.read_namespaced_service('marsservice', self._namespace).to_dict()
        port = svc_data['spec']['ports'][0]['node_port']

        web_pods = self._core_api.list_namespaced_pod(
            self._namespace, label_selector='mars/service-type=' + MarsWebsConfig.rc_name).to_dict()
        host_ip = random.choice(web_pods['items'])['status']['host_ip']

        # docker desktop use a VM to hold docker processes, hence
        # we need to use API address instead
        desktop_nodes = self._core_api.list_node(field_selector='metadata.name=docker-desktop').to_dict()
        if desktop_nodes['items']:  # pragma: no cover
            addresses = set(
                addr_info['address']
                for addr_info in desktop_nodes['items'][0].get('status', {}).get('addresses', ())
            )
            if host_ip in addresses:
                host_ip = urlparse(self._core_api.api_client.configuration.host).netloc.split(':', 1)[0]
        return f'http://{host_ip}:{port}'

    def _get_ready_pod_count(self, label_selector):
        query = self._core_api.list_namespaced_pod(
            namespace=self._namespace, label_selector=label_selector).to_dict()
        cnt = 0
        for el in query['items']:
            if el['status']['phase'] in ('Error', 'Failed'):
                logger.warning('Error in starting pod, message: %s', el['status']['message'])
                continue
            if 'status' not in el or 'conditions' not in el['status']:
                cnt += 1
            elif any(cond['type'] == 'Ready' and cond['status'] == 'True'
                     for cond in el['status'].get('conditions') or ()):
                cnt += 1
        return cnt

    def _create_namespace(self):
        if self._namespace is None:
            namespace = self._namespace = self._get_free_namespace()
        else:
            namespace = self._namespace

        self._core_api.create_namespace(NamespaceConfig(namespace).build())

    def _create_roles_and_bindings(self):
        # create role and binding
        role_config = RoleConfig('mars-pod-operator', self._namespace, api_groups='',
                                 resources='pods,replicationcontrollers/scale',
                                 verbs='get,watch,list,patch')
        self._rbac_api.create_namespaced_role(self._namespace, role_config.build())
        role_binding_config = RoleBindingConfig(
            'mars-pod-operator-binding', self._namespace, 'mars-pod-operator', 'default')
        self._rbac_api.create_namespaced_role_binding(self._namespace, role_binding_config.build())

    def _create_schedulers(self):
        schedulers_config = self._scheduler_config_cls(
            self._scheduler_num, image=self._image, cpu=self._scheduler_cpu,
            memory=self._scheduler_mem, memory_limit_ratio=self._scheduler_mem_limit_ratio,
            modules=self._scheduler_extra_modules, volumes=self._extra_volumes,
            service_port=self._scheduler_service_port,
            pre_stop_command=self._pre_stop_command,
        )
        schedulers_config.add_simple_envs(self._scheduler_extra_env)
        schedulers_config.add_labels(self._scheduler_extra_labels)
        self._core_api.create_namespaced_replication_controller(
            self._namespace, schedulers_config.build())

    def _create_workers(self):
        workers_config = self._worker_config_cls(
            self._worker_num, image=self._image, cpu=self._worker_cpu,
            memory=self._worker_mem, memory_limit_ratio=self._worker_mem_limit_ratio,
            spill_volumes=self._worker_spill_paths,
            modules=self._worker_extra_modules, volumes=self._extra_volumes,
            worker_cache_mem=self._worker_cache_mem,
            min_cache_mem=self._worker_min_cache_men,
            service_port=self._worker_service_port,
            pre_stop_command=self._pre_stop_command,
        )
        workers_config.add_simple_envs(self._worker_extra_env)
        workers_config.add_labels(self._worker_extra_labels)
        self._core_api.create_namespaced_replication_controller(
            self._namespace, workers_config.build())

    def _create_webs(self):
        webs_config = self._web_config_cls(
            self._web_num, image=self._image, cpu=self._web_cpu, memory=self._web_mem,
            memory_limit_ratio=self._web_mem_limit_ratio, modules=self._web_extra_modules,
            volumes=self._extra_volumes, service_port=self._web_service_port,
            pre_stop_command=self._pre_stop_command,
        )
        webs_config.add_simple_envs(self._web_extra_env)
        webs_config.add_labels(self._web_extra_labels)
        self._core_api.create_namespaced_replication_controller(
            self._namespace, webs_config.build())

    def _create_services(self):
        self._create_schedulers()
        self._create_workers()
        self._create_webs()

    def _wait_services_ready(self):
        min_worker_num = int(self._min_worker_num or self._worker_num)
        limits = [self._scheduler_num, min_worker_num, self._web_num]
        selectors = [
            'mars/service-type=' + MarsSchedulersConfig.rc_name,
            'mars/service-type=' + MarsWorkersConfig.rc_name,
            'mars/service-type=' + MarsWebsConfig.rc_name,
            ]
        logger.debug('Start waiting pods to be ready')
        wait_services_ready(selectors, limits,
                            lambda sel: self._get_ready_pod_count(sel),
                            timeout=self._timeout)
        logger.info('All service pods ready.')

    def _load_cluster_logs(self):
        log_dict = dict()
        pod_items = self._core_api.list_namespaced_pod(self._namespace).to_dict()
        for item in pod_items['items']:
            log_dict[item['metadata']['name']] = self._core_api.read_namespaced_pod_log(
                name=item['metadata']['name'], namespace=self._namespace)
        return log_dict

    def start(self):
        try:
            self._create_namespace()
            self._create_roles_and_bindings()

            self._create_services()

            self._wait_services_ready()
            return self._create_kube_service()
        except:  # noqa: E722
            if self._log_when_fail:  # pargma: no cover
                logger.error('Error when creating cluster')
                for name, log in self._load_cluster_logs().items():
                    logger.error('Error logs for %s:\n%s', name, log)
            self.stop()
            raise

    def stop(self, wait=False, timeout=0):
        from kubernetes.client import CoreV1Api
        api = CoreV1Api(self._api_client)
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
                    if timeout and time.time() - start_time > timeout:  # pragma: no cover
                        raise TimeoutError


def new_cluster(kube_api_client=None, image=None, scheduler_num=1, scheduler_cpu=None,
                scheduler_mem=None, worker_num=1, worker_cpu=None, worker_mem=None,
                worker_spill_paths=None, worker_cache_mem=None, min_worker_num=None,
                web_num=1, web_cpu=None, web_mem=None, service_type=None,
                timeout=None, **kwargs):
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
    cluster_cls = kwargs.pop('cluster_cls', KubernetesCluster)
    cluster = cluster_cls(
        kube_api_client, image=image, scheduler_num=scheduler_num, scheduler_cpu=scheduler_cpu,
        scheduler_mem=scheduler_mem, worker_num=worker_num, worker_cpu=worker_cpu,
        worker_mem=worker_mem, worker_spill_paths=worker_spill_paths, worker_cache_mem=worker_cache_mem,
        min_worker_num=min_worker_num, web_num=web_num, web_cpu=web_cpu, web_mem=web_mem,
        service_type=service_type, timeout=timeout, **kwargs)
    client = KubernetesClusterClient(cluster)
    client.start()
    return client
