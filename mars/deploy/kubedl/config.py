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

from urllib.parse import urlparse

from ...utils import parse_readable_size, calc_size_by_str
from ..kubernetes.config import ContainerEnvConfig

DEFAULT_SERVICE_ACCOUNT_NAME = 'kubedl-sa'


def _remove_nones(cfg):
    return dict((k, v) for k, v in cfg.items() if v is not None)


class ResourceConfig:
    """
    Configuration builder for Kubernetes computation resources
    """

    def __init__(self, cpu, memory):
        self._cpu = cpu
        self._memory, ratio = parse_readable_size(memory) if memory is not None else (None, False)
        assert not ratio

    def build(self):
        return {
            'cpu': str(self._cpu),
            'memory': str(int(self._memory)),
        }


class ReplicaSpecConfig:
    """
    Base configuration builder for Kubernetes replication controllers
    """
    container_name = 'mars'

    def __init__(self, name, image, replicas, resource_request=None, resource_limit=None,
                 node_selectors=None):
        self._name = name
        self._image = image
        self._replicas = replicas
        self._envs = dict()
        self._node_selectors = node_selectors

        self.add_default_envs()

        self._resource_request = resource_request
        self._resource_limit = resource_limit

    def add_env(self, name, value=None, field_path=None):
        self._envs[name] = ContainerEnvConfig(name, value=value, field_path=field_path)

    def add_simple_envs(self, envs):
        for k, v in envs.items() or ():
            self.add_env(k, v)

    def add_default_envs(self):
        pass  # pragma: no cover

    def build_container_command(self):
        raise NotImplementedError

    def build_container(self):
        resources_dict = {
            'requests': self._resource_request.build() if self._resource_request else None,
            'limits': self._resource_limit.build() if self._resource_limit else None,
        }
        return _remove_nones({
            'imagePullPolicy': 'Always',
            'command': self.build_container_command(),
            'env': [env.build() for env in self._envs.values()] or None,
            'image': self._image,
            'name': self.container_name,
            'resources': dict((k, v) for k, v in resources_dict.items() if v) or None,
        })

    def build_template_spec(self):
        return _remove_nones({
            'serviceAccountName': DEFAULT_SERVICE_ACCOUNT_NAME,
            'nodeSelector': self._node_selectors,
            'containers': [self.build_container()],
        })

    def build(self):
        return {
            'replicas': int(self._replicas),
            'restartPolicy': 'Never',
            'template': {
                'metadata': {
                    'labels': {'mars/service-type': self._name},
                },
                'spec': self.build_template_spec()
            }
        }


class MarsReplicaSpecConfig(ReplicaSpecConfig):
    service_name = None
    service_label = None

    def __init__(self, image, replicas, cpu=None, memory=None, limit_resources_ratio=1.2,
                 memory_limit_ratio=2, modules=None, node_selectors=None):
        self._cpu = cpu
        self._memory, ratio = parse_readable_size(memory) if memory is not None else (None, False)
        assert not ratio

        if isinstance(modules, str):
            self._modules = modules.split(',')
        else:
            self._modules = modules

        res_request = ResourceConfig(cpu, memory) if cpu or memory else None
        memory_limit_ratio = memory_limit_ratio if memory_limit_ratio is not None \
            else limit_resources_ratio
        res_limit = ResourceConfig(cpu * limit_resources_ratio,
                                   memory * memory_limit_ratio) if cpu or memory else None
        super().__init__(self.service_label, image, replicas, resource_request=res_request,
                         resource_limit=res_limit,
                         node_selectors=node_selectors)

    def build_container_command(self):
        cmd = [
            '/srv/entrypoint.sh', f'mars.deploy.kubernetes.{self.service_name}',
        ]
        return cmd

    def add_default_envs(self):
        if self._cpu:
            self.add_env('MARS_CPU_TOTAL', str(self._cpu))

        if self._memory:
            self.add_env('MARS_MEMORY_TOTAL', str(int(self._memory)))

        if self._modules:
            self.add_env('MARS_LOAD_MODULES', ','.join(self._modules))


class MarsSchedulerSpecConfig(MarsReplicaSpecConfig):
    service_name = 'scheduler'
    service_label = 'marsscheduler'


class MarsWorkerSpecConfig(MarsReplicaSpecConfig):
    service_name = 'worker'
    service_label = 'marsworker'

    def __init__(self, *args, **kwargs):
        cache_mem = kwargs.pop('cache_mem', None)
        self._spill_dirs = kwargs.pop('spill_dirs', None) or ()
        # set limits as 2*requests for worker replica defaulted.
        kwargs['limit_resources_ratio'] = kwargs.get('limit_resources_ratio', 1.2)
        super().__init__(*args, **kwargs)
        self._cache_mem = calc_size_by_str(cache_mem, self._memory)
        self.add_env('MARS_CACHE_MEM_SIZE', self._cache_mem)

    @property
    def spill_dirs(self):
        return self._spill_dirs

    @property
    def cache_mem(self):
        return self._cache_mem

    def add_default_envs(self):
        super().add_default_envs()
        if self._spill_dirs:
            self.add_env('MARS_SPILL_DIRS', ':'.join(self._spill_dirs))


class MarsWebSpecConfig(MarsReplicaSpecConfig):
    service_name = 'web'
    service_label = 'marsweb'


class MarsJobConfig:
    def __init__(self, job_name, scheduler_config, worker_config, web_config,
                 web_host=None):
        self._job_name = job_name
        self._scheduler_config = scheduler_config
        self._worker_config = worker_config
        self._web_config = web_config
        self._web_host = web_host

    def build(self):
        if self._job_name is None:
            metadata = {'generateName': 'mars-job-'}
        else:
            metadata = {'name': self._job_name}

        web_host = self._web_host
        if web_host is not None and '://' in web_host:
            web_host = urlparse(web_host).netloc

        return {
            'kind': 'MarsJob',
            'metadata': metadata,
            'spec': _remove_nones({
                'workerMemoryTuningPolicy': _remove_nones({
                    'spillDirs': self._worker_config.spill_dirs,
                    'workerCacheSize': self._worker_config.cache_mem,
                }),
                'cleanPodPolicy': 'None',
                'webHost': web_host,
                'marsReplicaSpecs': {
                    'Worker': self._worker_config.build(),
                    'Scheduler': self._scheduler_config.build(),
                    'WebService': self._web_config.build(),
                },
            }),
        }
