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
import enum
import importlib
from typing import Dict, List, Union


class NodeRole(enum.Enum):
    SUPERVISOR = 0
    WORKER = 1


async def start_services(node_role: NodeRole, config: Dict,
                         modules: Union[List, str, None] = None,
                         address: str = None):
    if modules is None:
        modules = []
    elif isinstance(modules, str):
        modules = [modules]
    modules.append('mars.services')

    # discover services
    svc_entries_list = []
    service_names = []

    web_handlers = {}
    bokeh_apps = {}
    for svc_names in config['services']:
        if isinstance(svc_names, str):
            svc_names = [svc_names]
        service_names.extend(svc_names)

        svc_entries = []
        for svc_name in svc_names:
            svc_mod = None
            for mod_name in modules:
                try:
                    svc_mod = importlib.import_module(
                        mod_name + '.' + svc_name + '.' + node_role.name.lower())
                    svc_entries.append(getattr(svc_mod, 'start'))

                    try:
                        web_mod = importlib.import_module(
                            mod_name + '.' + svc_name + '.api.web')
                        web_handlers.update(getattr(web_mod, 'web_handlers', {}))
                        bokeh_apps.update(getattr(web_mod, 'bokeh_apps', {}))
                    except ImportError:
                        pass
                except ImportError:
                    pass
            if svc_mod is None:
                raise ImportError(f'Cannot discover {node_role} for service {svc_name}')
        svc_entries_list.append(svc_entries)

    if 'web' in service_names:
        try:
            web_config = config['web']
        except KeyError:
            web_config = config['web'] = dict()

        web_config['web_handlers'] = web_handlers
        web_config['bokeh_apps'] = bokeh_apps

    for entries in svc_entries_list:
        await asyncio.gather(*[entry(config, address=address) for entry in entries])
