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

from urllib.parse import urlparse

from ...utils import parse_readable_size
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

    def __init__(self, name, image, replicas, resource_request=None, resource_limit=None):
        self._name = name
        self._image = image
        self._replicas = replicas
        self._envs = dict()

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
        return {
            'serviceAccountName': DEFAULT_SERVICE_ACCOUNT_NAME,
            'tolerations': [{'operator': 'Exists'}],
            'containers': [self.build_container()],
        }

    def build(self):
        return {
            'replicas': int(self._replicas),
            'restartPolicy': 'Never',
            'template': {
                'metadata': {
                    'labels': {'name': self._name},
                },
                'spec': self.build_template_spec()
            }
        }


class MarsReplicaSpecConfig(ReplicaSpecConfig):
    service_name = None
    service_label = None

    def __init__(self, image, replicas, cpu=None, memory=None, limit_resources=False,
                 modules=None):
        self._cpu = cpu
        self._memory, ratio = parse_readable_size(memory) if memory is not None else (None, False)
        assert not ratio

        if isinstance(modules, str):
            self._modules = modules.split(',')
        else:
            self._modules = modules

        res = ResourceConfig(cpu, memory) if cpu or memory else None
        super().__init__(self.service_label, image, replicas, resource_request=res,
                         resource_limit=res if limit_resources else None)

    def build_container_command(self):
        return ['/bin/sh', '-c', f'python -m mars.deploy.kubernetes.{self.service_name}']

    def add_default_envs(self):
        if self._modules:
            self.add_env('MARS_LOAD_MODULES', ','.join(self._modules))


class MarsSchedulerSpecConfig(MarsReplicaSpecConfig):
    service_name = 'scheduler'
    service_label = 'marsscheduler'


class MarsWorkerSpecConfig(MarsReplicaSpecConfig):
    service_name = 'worker'
    service_label = 'marsworker'

    def __init__(self, *args, **kwargs):
        self._cache_mem = kwargs.pop('cache_mem', None)
        self._spill_dirs = kwargs.pop('spill_dirs', None) or ()
        kwargs['limit_resources'] = kwargs.get('limit_resources', True)
        super().__init__(*args, **kwargs)

    def add_default_envs(self):
        super().add_default_envs()
        self.add_env('MARS_CACHE_MEM_SIZE', self._cache_mem)
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
                'cleanPodPolicy': 'None',
                'webHost': web_host,
                'marsReplicaSpecs': {
                    'Worker': self._worker_config.build(),
                    'Scheduler': self._scheduler_config.build(),
                    'WebService': self._web_config.build(),
                },
            }),
        }
