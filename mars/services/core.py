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

import abc
import asyncio
import enum
import importlib
import inspect
import warnings
from typing import Dict, Iterable, List, Union

_ModulesType = Union[List, str, None]


class NodeRole(enum.Enum):
    SUPERVISOR = 0
    WORKER = 1


class AbstractService(abc.ABC):
    _instances = dict()

    def __init__(self, config: Dict, address: str):
        self._config = config
        self._address = address

    @classmethod
    def get_instance(cls, address: str, config: Dict = None):
        type_addr = (cls, address)
        if type_addr not in cls._instances:
            inst = cls._instances[type_addr] = cls(config, address)
        else:
            inst = cls._instances[type_addr]
        return inst

    @classmethod
    def clear(cls):
        cls._instances = dict()

    @abc.abstractmethod
    async def start(self):
        raise NotImplementedError

    @abc.abstractmethod
    async def stop(self):
        raise NotImplementedError

    async def create_session(self, session_id: str):
        pass

    async def destroy_session(self, session_id: str):
        pass


class EmptyService(AbstractService):
    async def start(self):
        pass

    async def stop(self):
        pass


def _find_service_entries(node_role: NodeRole,
                          services: List,
                          modules: List):
    svc_entries_list = []

    web_handlers = {}
    for svc_names in services:
        if isinstance(svc_names, str):
            svc_names = [svc_names]
        svc_entries = []
        for svc_name in svc_names:
            svc_mod = None
            for mod_name in modules:
                try:
                    full_mod_name = f'{mod_name}.{svc_name}.{node_role.name.lower()}'
                    svc_mod = importlib.import_module(full_mod_name)

                    abstract_derivatives = []
                    valid_derivatives = []
                    for attr_name in dir(svc_mod):
                        obj = getattr(svc_mod, attr_name)
                        if obj is not AbstractService \
                                and isinstance(obj, type) \
                                and issubclass(obj, AbstractService):
                            if inspect.isabstract(obj):
                                abstract_derivatives.append(obj)
                            else:
                                valid_derivatives.append(obj)

                    svc_entries.extend(valid_derivatives)
                    if not valid_derivatives and abstract_derivatives:
                        warnings.warn(f'Module {full_mod_name} does not have non-abstract '
                                      f'service classes, but abstract classes '
                                      f'{abstract_derivatives} found.', RuntimeWarning)

                    try:
                        web_mod = importlib.import_module(
                            mod_name + '.' + svc_name + '.api.web')
                        web_handlers.update(getattr(web_mod, 'web_handlers', {}))
                    except ImportError:
                        pass
                except ImportError:
                    pass
            if svc_mod is None:
                raise ImportError(f'Cannot discover {node_role} for service {svc_name}')
        svc_entries_list.append(svc_entries)

    return svc_entries_list, web_handlers


def _normalize_modules(modules: _ModulesType):
    if modules is None:
        modules = []
    elif isinstance(modules, str):
        modules = [modules]
    else:
        modules = list(modules)
    modules = ['mars.services'] + modules
    return modules


def _iter_service_instances(node_role: NodeRole,
                            config: Dict,
                            address: str = None,
                            reverse: bool = False) -> Iterable[List[AbstractService]]:
    modules = _normalize_modules(config.get('modules'))
    service_names = config['services']
    if reverse:
        service_names = service_names[::-1]

    svc_entries_list, _ = _find_service_entries(
        node_role, service_names, modules)
    for entries in svc_entries_list:
        yield [svc_entry.get_instance(address, config) for svc_entry in entries]


async def start_services(node_role: NodeRole, config: Dict,
                         address: str = None,
                         mark_ready: bool = True):
    modules = _normalize_modules(config.get('modules'))

    # discover services
    service_names = config['services']

    svc_entries_list, web_handlers = _find_service_entries(
        node_role, service_names, modules)

    if 'web' in service_names:
        try:
            web_config = config['web']
        except KeyError:
            web_config = config['web'] = dict()

        web_config['web_handlers'] = web_handlers

    for entries in svc_entries_list:
        instances = [svc_entry.get_instance(address, config) for svc_entry in entries]
        await asyncio.gather(*[inst.start() for inst in instances])

    if mark_ready and 'cluster' in service_names:
        from .cluster import ClusterAPI
        cluster_api = await ClusterAPI.create(address)
        await cluster_api.mark_node_ready()


async def stop_services(node_role: NodeRole,
                        config: Dict,
                        address: str = None):
    for instances in _iter_service_instances(node_role, config, address, reverse=True):
        await asyncio.gather(*[inst.stop() for inst in instances])

    AbstractService.clear()


async def create_service_session(node_role: NodeRole,
                                 config: Dict,
                                 session_id: str = None,
                                 address: str = None):
    for instances in _iter_service_instances(node_role, config, address):
        await asyncio.gather(*[inst.create_session(session_id)
                               for inst in instances])


async def destroy_service_session(node_role: NodeRole,
                                  config: Dict,
                                  session_id: str = None,
                                  address: str = None):
    for instances in _iter_service_instances(node_role, config, address, reverse=True):
        await asyncio.gather(*[inst.destroy_session(session_id)
                               for inst in instances])
