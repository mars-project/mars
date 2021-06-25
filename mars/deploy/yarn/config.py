# -*- coding: utf-8 -*-
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

import os
import textwrap
from urllib.parse import urlparse

from ...utils import parse_readable_size


def _remove_nones(cfg):
    return dict((k, v) for k, v in cfg.items() if v is not None)


def _get_local_app_module(mod_name):
    return __name__.rsplit('.', 1)[0] + '.' + mod_name.rsplit('.', 1)[-1]


class SecurityConfig:
    def __init__(self, cert_file=None, key_file=None):
        self._cert_file = cert_file
        self._key_file = key_file

    def build(self):
        return dict(cert_file=self._cert_file, key_file=self._key_file)


class AppFileConfig:
    def __init__(self, source, file_type=None, visibility=None,
                 size=None, timestamp=None):
        self._source = source
        self._file_type = file_type
        self._visibility = visibility
        self._size = size
        self._timestamp = timestamp

    def build(self):
        if all(v is None for v in (self._file_type, self._visibility, self._size, self._timestamp)):
            return self._source
        else:
            return _remove_nones(dict(
                source=self._source, type=self._file_type, visibility=self._visibility,
                size=self._size, timestamp=self._timestamp
            ))


class AppContainerConfig:
    def __init__(self, cpu=None, memory=None, env=None, files=None, script=None):
        self._cpu = cpu

        if memory is not None:
            real_mem, is_percent = parse_readable_size(memory)
            assert not is_percent
            self._memory = real_mem
        else:
            self._memory = None

        self._env = env
        self._script = script
        self._files = files

        self.add_default_envs()

    def build_script(self):
        return self._script

    def add_default_envs(self):
        pass

    def add_env(self, k, v):
        if self._env is None:
            self._env = dict()
        self._env[k] = v

    def build(self):
        return _remove_nones(dict(
            resources=dict(
                vcores=self._cpu,
                memory=f'{self._memory // 1024 ** 2} MiB' if self._memory else None,
            ),
            env=self._env,
            script=self.build_script(),
            files=dict((k, v.build()) for k, v in self._files.items()) if self._files else None,
        ))


class AppMasterConfig(AppContainerConfig):
    def __init__(self, security=None, **kwargs):
        super().__init__(**kwargs)
        self._security = security

    def build(self):
        d = super().build()
        if self._security is not None:
            d['security'] = self._security.build()
        return d


class AppServiceConfig(AppContainerConfig):
    def __init__(self, instances=1, depends=None, allow_failures=False,
                 max_restarts=0, **kwargs):
        super().__init__(**kwargs)
        if isinstance(depends, str):
            depends = [depends]

        self._allow_failures = allow_failures
        self._depends = depends or []
        self._max_restarts = max_restarts
        self._instances = instances

    def build(self):
        d = super().build()
        d.update(dict(
            instances=self._instances,
            depends=self._depends,
            allow_failures=self._allow_failures,
            max_restarts=self._max_restarts,
        ))
        return d


class MarsServiceConfig(AppServiceConfig):
    service_name = None

    def __init__(self, environment, modules=None, cmd_tmpl=None, cpu=None, memory=None,
                 log_config=None, extra_args=None, **kwargs):
        files = kwargs.pop('files', dict())
        kwargs['files'] = files

        parsed = urlparse(environment)
        self._env_scheme = parsed.scheme

        if parsed.scheme:
            import mars
            self._source_path = os.path.dirname(os.path.dirname(os.path.abspath(mars.__file__)))

            self._env_path = environment[len(parsed.scheme) + 3:]
            self._path_environ = os.environ['PATH']
        else:
            self._source_path = None
            self._env_path = environment
            self._path_environ = None

        self._cmd_tmpl = cmd_tmpl or '"{executable}"'
        if not self._env_scheme:
            files['mars_env'] = AppFileConfig(environment)

        self._log_config = log_config
        if log_config:
            files['logging.conf'] = AppFileConfig(log_config)

        self._modules = modules.split(',') if isinstance(modules, str) else modules

        self._extra_args = extra_args or ''

        cpu = cpu or 1
        memory = memory or '1 GiB'
        super().__init__(cpu=cpu, memory=memory, **kwargs)

    def add_default_envs(self):
        if self._cpu:
            self.add_env('MKL_NUM_THREADS', str(self._cpu))
            self.add_env('MARS_CPU_TOTAL', str(self._cpu))
            self.add_env('MARS_USE_PROCESS_STAT', '1')

        if self._memory:
            self.add_env('MARS_MEMORY_TOTAL', str(int(self._memory)))

        if self._modules:
            self.add_env('MARS_LOAD_MODULES', ','.join(self._modules))

        if self._path_environ:
            self.add_env('MARS_YARN_PATH', self._path_environ)

        if self._source_path:
            self.add_env('MARS_SOURCE_PATH', self._source_path)

    def build_script(self):
        bash_lines = [textwrap.dedent("""
        #!/bin/bash
        if [[ "$YARN_CONTAINER_RUNTIME_TYPE" == "docker" ]]; then
          export MARS_USE_CGROUP_STAT=1
        else
          export MARS_USE_PROCESS_STAT=1
        fi
        if [[ -n $MARS_SOURCE_PATH ]]; then export PYTHONPATH=$PYTHONPATH:$MARS_SOURCE_PATH; fi
        if [[ -n $MARS_YARN_PATH ]]; then export PATH=$MARS_YARN_PATH:$PATH; fi
        """).strip()]

        if not self._env_scheme:
            bash_lines.append('source mars_env/bin/activate')
            python_executable = 'mars_env/bin/python'
        elif self._env_scheme == 'conda':
            bash_lines.append(f'conda activate "{self._env_path}"')
            python_executable = 'python'
        elif self._env_scheme == 'venv':
            bash_lines.append(f'source "{self._env_path}/bin/activate"')
            python_executable = self._env_path + '/bin/python'
        else:  # pragma: no cover
            python_executable = self._env_path

        cmd = self._cmd_tmpl.format(executable=python_executable)
        bash_lines.append(f'{cmd} -m {_get_local_app_module(self.service_name)} {self._extra_args} > /tmp/{self.service_name}.stdout.log 2> /tmp/{self.service_name}.stderr.log')
        return '\n'.join(bash_lines) + '\n'


class MarsSupervisorConfig(MarsServiceConfig):
    service_name = 'mars.supervisor'
    web_service_name = 'mars.web'


class MarsWorkerConfig(MarsServiceConfig):
    service_name = 'mars.worker'

    def __init__(self, environment, worker_cache_mem=None, spill_dirs=None, **kwargs):
        kwargs['depends'] = MarsSupervisorConfig.service_name
        super().__init__(environment, **kwargs)

        if worker_cache_mem:
            self.add_env('MARS_CACHE_MEM_SIZE', worker_cache_mem)

        if spill_dirs:
            self.add_env('MARS_SPILL_DIRS',
                         spill_dirs if isinstance(spill_dirs, str) else ':'.join(spill_dirs))


class MarsApplicationConfig:
    def __init__(self, name=None, queue=None, file_systems=None, master=None,
                 supervisor_config=None, worker_config=None):
        self._name = name
        self._queue = queue or 'default'
        self._file_systems = file_systems or []
        self._master = master or AppMasterConfig(cpu=1, memory='512 MiB')
        self._supervisor_config = supervisor_config
        self._worker_config = worker_config

    def build(self):
        services = _remove_nones({
            MarsSupervisorConfig.service_name: self._supervisor_config.build() if self._supervisor_config else None,
            MarsWorkerConfig.service_name: self._worker_config.build() if self._worker_config else None,
        })
        return dict(
            name=self._name,
            queue=self._queue,
            file_systems=self._file_systems,
            master=self._master.build() if self._master else None,
            services=services,
        )
