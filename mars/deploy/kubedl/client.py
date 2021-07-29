# Copyright 1999-2021 Alibaba Group Holding Ltd.
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
import time
import warnings

import requests

from ...session import new_session
from .config import MarsSchedulerSpecConfig, MarsWorkerSpecConfig, MarsWebSpecConfig, \
    MarsJobConfig

try:
    from kubernetes.client.rest import ApiException as K8SApiException
except ImportError:  # pragma: no cover
    K8SApiException = None

KUBEDL_API_VERSION = 'kubedl.io/v1alpha1'
KUBEDL_MARS_PLURAL = 'marsjobs'


logger = logging.getLogger(__name__)


class KubeDLClusterClient:
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
        self._session = new_session(self._endpoint, verify_ssl=self._cluster.verify_ssl)

    def stop(self, wait=False, timeout=0):
        self._cluster.stop(wait=wait, timeout=timeout)


class KubeDLCluster:
    def __init__(self, kube_api_client=None, image=None, job_name=None, namespace=None,
                 scheduler_num=1, scheduler_cpu=None, scheduler_mem=None,
                 worker_num=1, worker_cpu=None, worker_mem=None, worker_spill_paths=None,
                 worker_cache_mem=None, min_worker_num=None,
                 web_num=1, web_cpu=None, web_mem=None,
                 slb_endpoint=None, verify_ssl=True, timeout=None, **kwargs):
        from kubernetes import client as kube_client
        self._kube_api_client = kube_api_client
        self._custom_api = kube_client.CustomObjectsApi(kube_api_client)

        self._slb_endpoint = slb_endpoint.rstrip("/")
        self._verify_ssl = verify_ssl

        self._job_name = job_name
        self._mars_endpoint = None
        self._namespace = namespace or 'default'
        self._image = image
        self._timeout = timeout
        self._extra_volumes = kwargs.pop('extra_volumes', ())
        self._pre_stop_command = kwargs.pop('pre_stop_command', None)
        self._log_when_fail = kwargs.pop('log_when_fail', False)
        self._node_selectors = kwargs.pop('node_selectors', None)

        extra_modules = kwargs.pop('extra_modules', None) or []
        extra_modules = extra_modules.split(',') if isinstance(extra_modules, str) \
            else extra_modules
        extra_envs = kwargs.pop('extra_env', None) or dict()

        if not verify_ssl:
            extra_envs['KUBE_VERIFY_SSL'] = '0'

        def _override_modules(updates):
            modules = set(extra_modules)
            updates = updates.split(',') if isinstance(updates, str) \
                else updates
            modules.update(updates)
            return sorted(modules)

        def _override_envs(updates):
            ret = extra_envs.copy()
            ret.update(updates)
            return ret

        self._scheduler_num = scheduler_num
        self._scheduler_cpu = scheduler_cpu
        self._scheduler_mem = scheduler_mem
        self._scheduler_extra_modules = _override_modules(kwargs.pop('scheduler_extra_modules', []))
        self._scheduler_extra_env = _override_envs(kwargs.pop('scheduler_extra_env', None) or dict())

        self._worker_num = worker_num
        self._worker_cpu = worker_cpu
        self._worker_mem = worker_mem
        self._worker_spill_paths = worker_spill_paths
        self._worker_cache_mem = worker_cache_mem
        self._min_worker_num = min_worker_num or worker_num
        self._worker_extra_modules = _override_modules(kwargs.pop('worker_extra_modules', []))
        self._worker_extra_env = _override_envs(kwargs.pop('worker_extra_env', None) or dict())

        self._web_num = web_num
        self._web_cpu = web_cpu
        self._web_mem = web_mem
        self._web_extra_modules = _override_modules(kwargs.pop('web_extra_modules', []))
        self._web_extra_env = _override_envs(kwargs.pop('web_extra_env', None) or dict())

    @property
    def verify_ssl(self):
        return self._verify_ssl

    def _check_if_exist(self):
        if self._job_name is None:
            return False
        try:
            api, version = KUBEDL_API_VERSION.rsplit('/', 1)
            service_obj = self._custom_api.get_namespaced_custom_object_status(
                api, version, self._namespace, KUBEDL_MARS_PLURAL, self._job_name)
            if len(service_obj.get('status', dict()).get('conditions', [])) > 0:
                status = service_obj['status']['conditions'][-1]['type']
                if status == 'Running' or status == 'Created':
                    logger.warning(f'Reusing cluster: {self._job_name}')
                    return True
                else:
                    return False
            else:
                return False
        except K8SApiException:
            return False

    def _create_service(self):
        scheduler_cfg = MarsSchedulerSpecConfig(
            self._image, self._scheduler_num, cpu=self._scheduler_cpu, memory=self._scheduler_mem,
            node_selectors=self._node_selectors, modules=self._scheduler_extra_modules,
        )
        scheduler_cfg.add_simple_envs(self._scheduler_extra_env)

        worker_cfg = MarsWorkerSpecConfig(
            self._image, self._worker_num, cpu=self._worker_cpu, memory=self._worker_mem,
            cache_mem=self._worker_cache_mem, spill_dirs=self._worker_spill_paths,
            node_selectors=self._node_selectors, modules=self._worker_extra_modules
        )
        worker_cfg.add_simple_envs(self._worker_extra_env)

        web_cfg = MarsWebSpecConfig(
            self._image, self._web_num, cpu=self._web_cpu, memory=self._web_mem,
            node_selectors=self._node_selectors, modules=self._web_extra_modules
        )
        web_cfg.add_simple_envs(self._web_extra_env)

        job_cfg = MarsJobConfig(
            job_name=self._job_name, scheduler_config=scheduler_cfg, worker_config=worker_cfg,
            web_config=web_cfg, web_host=self._slb_endpoint
        )

        api, version = KUBEDL_API_VERSION.rsplit('/', 1)

        cfg_json = job_cfg.build()
        cfg_json['apiVersion'] = KUBEDL_API_VERSION

        response = self._custom_api.create_namespaced_custom_object(
            api, version, self._namespace, KUBEDL_MARS_PLURAL, cfg_json)
        self._job_name = response['metadata']['name']

    def _wait_service_ready(self):
        self._mars_endpoint = f'{self._slb_endpoint}/mars/{self._namespace}/{self._job_name}-webservice-0'
        logger.warning(f'Kubedl job name: {self._job_name}')
        check_start_time = time.time()
        worker_count_url = self._mars_endpoint + '/api/worker?action=count'
        while True:
            try:
                if self._timeout and time.time() - check_start_time > self._timeout:
                    raise TimeoutError('Check Mars service start timeout')

                if not self._verify_ssl:
                    try:
                        import urllib3
                        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    except ImportError:  # pragma: no cover
                        pass

                api, version = KUBEDL_API_VERSION.rsplit('/', 1)
                service_obj = self._custom_api.get_namespaced_custom_object_status(
                    api, version, self._namespace, KUBEDL_MARS_PLURAL, self._job_name)
                if len(service_obj.get('status', dict()).get('conditions', [])) > 0:
                    if service_obj['status']['conditions'][-1]['type'] == 'Failed':
                        raise SystemError(service_obj['status']['conditions'][-1]['message'])

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
                    resp = requests.get(worker_count_url, timeout=1, verify=self._verify_ssl)

                if int(resp.text) >= self._min_worker_num:
                    logger.warning(f'Web endpoint started at {self._mars_endpoint}')
                    break
            except (requests.Timeout, ValueError) as ex:
                if not isinstance(ex, requests.Timeout):
                    time.sleep(0.1)
                pass

    def start(self):
        try:
            if not self._check_if_exist():
                self._create_service()
            self._wait_service_ready()
            return self._mars_endpoint
        except:  # noqa: E722
            self.stop()
            raise

    def stop(self, wait=False, timeout=0):
        from kubernetes import client as kube_client

        custom_api = kube_client.CustomObjectsApi(self._kube_api_client)
        api, version = KUBEDL_API_VERSION.rsplit('/', 1)
        custom_api.delete_namespaced_custom_object(
            api, version, self._namespace, KUBEDL_MARS_PLURAL, self._job_name)

        if wait:
            start_time = time.time()
            while True:
                try:
                    custom_api.get_namespaced_custom_object(
                        api, version, self._namespace, KUBEDL_MARS_PLURAL, self._job_name)
                except K8SApiException as ex:
                    if ex.status != 404:  # pragma: no cover
                        raise
                    break
                else:
                    time.sleep(1)
                    if timeout and time.time() - start_time > timeout:  # pragma: no cover
                        raise TimeoutError('Check Mars service stop timeout')


def new_cluster(kube_api_client=None, image=None, scheduler_num=1, scheduler_cpu=2,
                scheduler_mem=4 * 1024 ** 3, worker_num=1, worker_cpu=8, worker_mem=32 * 1024 ** 3,
                worker_spill_paths=None, worker_cache_mem='45%', min_worker_num=None,
                web_num=1, web_cpu=1, web_mem=4 * 1024 ** 3, slb_endpoint=None, verify_ssl=True,
                job_name=None, timeout=None, **kwargs):
    worker_spill_paths = worker_spill_paths or ['/tmp/spill-dir']
    cluster = KubeDLCluster(kube_api_client, image=image, scheduler_num=scheduler_num,
                            scheduler_cpu=scheduler_cpu, scheduler_mem=scheduler_mem,
                            worker_num=worker_num, worker_cpu=worker_cpu, worker_mem=worker_mem,
                            worker_spill_paths=worker_spill_paths, worker_cache_mem=worker_cache_mem,
                            min_worker_num=min_worker_num, web_num=web_num, web_cpu=web_cpu,
                            web_mem=web_mem, slb_endpoint=slb_endpoint, verify_ssl=verify_ssl,
                            job_name=job_name, timeout=timeout, **kwargs)
    client = KubeDLClusterClient(cluster)
    client.start()
    return client
