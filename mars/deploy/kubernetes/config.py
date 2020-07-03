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

from ... import __version__ as mars_version
from ...utils import parse_readable_size

DEFAULT_IMAGE = 'marsproject/mars:v' + mars_version


def _remove_nones(cfg):
    return dict((k, v) for k, v in cfg.items() if v is not None)


class RoleConfig(object):
    """
    Configuration builder for Kubernetes RBAC roles
    """
    def __init__(self, name, namespace, api_groups, resources, verbs):
        self._name = name
        self._namespace = namespace
        self._api_groups = api_groups.split(',')
        self._resources = resources.split(',')
        self._verbs = verbs.split(',')

    def build(self):
        return {
            'kind': 'Role',
            'metadata': {'name': self._name, 'namespace': self._namespace},
            'rules': [{
                'apiGroups': self._api_groups,
                'resources': self._resources,
                'verbs': self._verbs,
            }]
        }


class RoleBindingConfig(object):
    """
    Configuration builder for Kubernetes RBAC role bindings
    """
    def __init__(self, name, namespace, role_name, service_account_name):
        self._name = name
        self._namespace = namespace
        self._role_name = role_name
        self._service_account_name = service_account_name

    def build(self):
        return {
            'kind': 'RoleBinding',
            'metadata': {'name': self._name, 'namespace': self._namespace},
            'roleRef': {
                'apiGroup': 'rbac.authorization.k8s.io',
                'kind': 'Role',
                'name': self._role_name,
            },
            'subjects': [{
                'kind': 'ServiceAccount',
                'name': self._service_account_name,
                'namespace': self._namespace,
            }]
        }


class NamespaceConfig(object):
    """
    Configuration builder for Kubernetes namespaces
    """
    def __init__(self, name):
        self._name = name

    def build(self):
        return {
            'kind': 'Namespace',
            'metadata': {
                'name': self._name,
                'labels': {
                    'name': self._name,
                }
            }
        }


class ServiceConfig(object):
    """
    Configuration builder for Kubernetes services
    """
    def __init__(self, name, service_type, selector, port, target_port=None,
                 protocol=None):
        self._name = name
        self._type = service_type
        self._protocol = protocol or 'TCP'
        self._selector = selector
        self._port = port
        self._target_port = target_port

    def build(self):
        return {
            'kind': 'Service',
            'metadata': {
                'name': self._name,
            },
            'spec': _remove_nones({
                'type': self._type,
                'selector': self._selector,
                'ports': [
                    _remove_nones({
                        'protocol': self._protocol,
                        'port': self._port,
                        'targetPort': self._target_port,
                    }),
                ]
            }),
        }


class ResourceConfig(object):
    """
    Configuration builder for Kubernetes computation resources
    """
    def __init__(self, cpu, memory):
        self._cpu = cpu
        self._memory, ratio = parse_readable_size(memory) if memory is not None else (None, False)
        assert not ratio

    def build(self):
        return {
            'cpu': '%dm' % int(self._cpu * 1000),
            'memory': '%d' % int(self._memory),
        }


class PortConfig(object):
    """
    Configuration builder for Kubernetes ports definition for containers
    """
    def __init__(self, container_port):
        self._container_port = int(container_port)

    def build(self):
        return {
            'containerPort': self._container_port,
        }


class VolumeConfig(object):
    """
    Base configuration builder for Kubernetes volumes
    """
    def __init__(self, name, mount_path):
        self.name = name
        self.mount_path = mount_path

    def build(self):
        raise NotImplementedError

    def build_mount(self):
        return {
            'name': self.name,
            'mountPath': self.mount_path,
        }


class HostPathVolumeConfig(VolumeConfig):
    """
    Configuration builder for Kubernetes host volumes
    """
    def __init__(self, name, mount_path, host_path, volume_type=None):
        super().__init__(name, mount_path)
        self._host_path = host_path
        self._volume_type = volume_type or 'DirectoryOrCreate'

    def build(self):
        return {
            'name': self.name,
            'hostPath': {'path': self._host_path, 'type': self._volume_type},
        }


class EmptyDirVolumeConfig(VolumeConfig):
    """
    Configuration builder for Kubernetes empty-dir volumes
    """
    def __init__(self, name, mount_path, use_memory=False):
        super().__init__(name, mount_path)
        self._medium = 'Memory' if use_memory else None

    def build(self):
        result = {
            'name': self.name,
            'emptyDir': {}
        }
        if self._medium:
            result['emptyDir']['medium'] = self._medium
        return result


class ContainerEnvConfig(object):
    """
    Configuration builder for Kubernetes container environments
    """
    def __init__(self, name, value=None, field_path=None):
        self._name = name
        self._value = value
        self._field_path = field_path

    def build(self):
        result = dict(name=self._name)
        if self._value is not None:
            result['value'] = str(self._value)
        elif self._field_path is not None:  # pragma: no branch
            result['valueFrom'] = {'fieldRef': {'fieldPath': self._field_path}}
        return result


class ProbeConfig(object):
    """
    Base configuration builder for Kubernetes liveness and readiness probes
    """
    def __init__(self, initial_delay=5, period=1, timeout=None,
                 success_thresh=None, failure_thresh=None):
        self._initial_delay = initial_delay
        self._period = period
        self._timeout = timeout
        self._success_thresh = success_thresh
        self._failure_thresh = failure_thresh

    def build(self):
        return _remove_nones({
            'initialDelaySeconds': self._initial_delay,
            'periodSeconds': self._period,
            'timeoutSeconds': self._timeout,
            'successThreshold': self._success_thresh,
            'failureThreshold': self._failure_thresh,
        })


class ExecProbeConfig(ProbeConfig):
    """
    Configuration builder for Kubernetes probes by executing commands
    """
    def __init__(self, command, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._command = command

    def build(self):
        result = {
            'exec': {'command': self._command}
        }
        result.update(super().build())
        return result


class TcpProbeConfig(ProbeConfig):
    """
    Configuration builder for Kubernetes probes by checking TCP ports
    """
    def __init__(self, port, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._port = port

    def build(self):
        result = {
            'tcpSocket': {'port': self._port}
        }
        result.update(super().build())
        return result


class ReplicationControllerConfig(object):
    """
    Base configuration builder for Kubernetes replication controllers
    """
    def __init__(self, name, image, replicas, resource_request=None, resource_limit=None,
                 liveness_probe=None, readiness_probe=None, pre_stop_command=None):
        self._name = name
        self._image = image
        self._replicas = replicas
        self._ports = []
        self._volumes = []
        self._envs = dict()

        self.add_default_envs()

        self._resource_request = resource_request
        self._resource_limit = resource_limit

        self._liveness_probe = liveness_probe
        self._readiness_probe = readiness_probe

        self._pre_stop_command = pre_stop_command

    def add_env(self, name, value=None, field_path=None):
        self._envs[name] = ContainerEnvConfig(name, value=value, field_path=field_path)

    def add_simple_envs(self, envs):
        for k, v in envs.items() or ():
            self.add_env(k, v)

    def add_port(self, container_port):
        self._ports.append(PortConfig(container_port))

    def add_default_envs(self):
        pass  # pragma: no cover

    def add_volume(self, vol):
        self._volumes.append(vol)

    def build_container_command(self):
        raise NotImplementedError

    def build_container(self):
        resources_dict = {
            'requests': self._resource_request.build() if self._resource_request else None,
            'limits': self._resource_limit.build() if self._resource_limit else None,
        }
        lifecycle_dict = _remove_nones({
            'preStop': {
                'exec': {'command': self._pre_stop_command},
            } if self._pre_stop_command else None,
        })
        return _remove_nones({
            'command': self.build_container_command(),
            'env': [env.build() for env in self._envs.values()] or None,
            'image': self._image,
            'name': self._name,
            'resources': dict((k, v) for k, v in resources_dict.items() if v) or None,
            'ports': [p.build() for p in self._ports] or None,
            'volumeMounts': [vol.build_mount() for vol in self._volumes] or None,
            'livenessProbe': self._liveness_probe.build() if self._liveness_probe else None,
            'readinessProbe': self._readiness_probe.build() if self._readiness_probe else None,
            'lifecycle': lifecycle_dict or None,
        })

    def build_template_spec(self):
        result = {
            'containers': [self.build_container()],
            'volumes': [vol.build() for vol in self._volumes]
        }
        return dict((k, v) for k, v in result.items() if v)

    def build(self):
        return {
            'kind': 'ReplicationController',
            'metadata': {
                'name': self._name,
            },
            'spec': {
                'replicas': int(self._replicas),
                'selector': {'name': self._name},
                'template': {
                    'metadata': {
                        'labels': {'name': self._name},
                    },
                    'spec': self.build_template_spec()
                }
            },
        }


class MarsReplicationControllerConfig(ReplicationControllerConfig):
    """
    Base configuration builder for replication controllers for Mars
    """
    rc_name = None

    def __init__(self, replicas, cpu=None, memory=None, limit_resources=False,
                 image=None, modules=None, volumes=None, service_port=None, **kwargs):
        self._cpu = cpu
        self._memory, ratio = parse_readable_size(memory) if memory is not None else (None, False)
        assert not ratio

        if isinstance(modules, str):
            self._modules = modules.split(',')
        else:
            self._modules = modules

        res = ResourceConfig(cpu, memory) if cpu or memory else None

        self._service_port = service_port

        super().__init__(
            self.rc_name, image or DEFAULT_IMAGE, replicas,
            resource_request=res, resource_limit=res if limit_resources else None,
            readiness_probe=self.config_readiness_probe(), **kwargs
        )
        if service_port:
            self.add_port(service_port)

        for vol in volumes or ():
            self.add_volume(vol)

    def add_default_envs(self):
        self.add_env('MARS_K8S_POD_NAME', field_path='metadata.name')
        self.add_env('MARS_K8S_POD_NAMESPACE', field_path='metadata.namespace')
        self.add_env('MARS_K8S_POD_IP', field_path='status.podIP')

        if self._service_port:
            self.add_env('MARS_K8S_SERVICE_PORT', str(self._service_port))

        if self._cpu:
            self.add_env('MKL_NUM_THREADS', str(self._cpu))
            self.add_env('MARS_CPU_TOTAL', str(self._cpu))
            if getattr(self, 'stat_type', 'cgroup') == 'cgroup':
                self.add_env('MARS_USE_CGROUP_STAT', '1')

        if self._memory:
            self.add_env('MARS_MEMORY_TOTAL', str(int(self._memory)))

        if self._modules:
            self.add_env('MARS_LOAD_MODULES', ','.join(self._modules))

    def config_readiness_probe(self):
        raise NotImplementedError

    @staticmethod
    def get_local_app_module(mod_name):
        return __name__.rsplit('.', 1)[0] + '.' + mod_name


class MarsSchedulersConfig(MarsReplicationControllerConfig):
    """
    Configuration builder for Mars scheduler service
    """
    rc_name = 'marsscheduler'

    def config_readiness_probe(self):
        readiness_cmd = [
            '/srv/entrypoint.sh', self.get_local_app_module('probe'),
        ]
        return ExecProbeConfig(readiness_cmd, timeout=5, failure_thresh=3)

    def build_container_command(self):
        cmd = [
            '/srv/entrypoint.sh', self.get_local_app_module('scheduler'),
        ]
        if self._service_port:
            cmd += ['-p', str(self._service_port)]
        return cmd


class MarsWorkersConfig(MarsReplicationControllerConfig):
    """
    Configuration builder for Mars worker service
    """
    rc_name = 'marsworker'

    def __init__(self, *args, **kwargs):
        spill_volumes = kwargs.pop('spill_volumes', None) or ()
        mount_shm = kwargs.pop('mount_shm', True)
        self._limit_resources = kwargs['limit_resources'] = kwargs.get('limit_resources', True)
        worker_cache_mem = kwargs.pop('worker_cache_mem', None)

        super().__init__(*args, **kwargs)

        self._spill_volumes = []
        for idx, vol in enumerate(spill_volumes):
            if isinstance(vol, str):
                path = '/mnt/hostpath%d' % idx
                self.add_volume(HostPathVolumeConfig('host-path-vol-%d' % idx, path, vol))
                self._spill_volumes.append(path)
            else:
                self.add_volume(vol)
                self._spill_volumes.append(vol.mount_path)
        if self._spill_volumes:
            self.add_env('MARS_SPILL_DIRS', ':'.join(self._spill_volumes))

        if mount_shm:
            self.add_volume(EmptyDirVolumeConfig('shm-volume', '/dev/shm', True))
        if worker_cache_mem:
            self.add_env('MARS_CACHE_MEM_SIZE', worker_cache_mem)

    def config_readiness_probe(self):
        readiness_cmd = [
            '/srv/entrypoint.sh', self.get_local_app_module('probe'),
        ]
        return ExecProbeConfig(readiness_cmd, timeout=60, failure_thresh=3)

    def build_container_command(self):
        cmd = [
            '/srv/entrypoint.sh', self.get_local_app_module('worker'),
        ]
        if self._service_port:
            cmd += ['-p', str(self._service_port)]
        return cmd


class MarsWebsConfig(MarsReplicationControllerConfig):
    """
    Configuration builder for Mars web service
    """
    rc_name = 'marsweb'

    def config_readiness_probe(self):
        if self._service_port:
            return TcpProbeConfig(self._service_port, timeout=60, failure_thresh=3)

    def build_container_command(self):
        cmd = [
            '/srv/entrypoint.sh', self.get_local_app_module('web'),
        ]
        if self._service_port:
            cmd += ['-p', str(self._service_port)]
        return cmd
