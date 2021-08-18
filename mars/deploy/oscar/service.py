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
from typing import List, Dict, Union

from ...services import start_services, stop_services, NodeRole
from ..utils import load_service_config_file


def load_config(filename=None):
    # use default config
    if not filename:  # pragma: no cover
        d = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(d, 'config.yml')
    return load_service_config_file(filename)


async def start_supervisor(address: str,
                           lookup_address: str = None,
                           modules: Union[List, str, None] = None,
                           config: Dict = None,
                           web: Union[str, bool] = 'auto'):
    if not config or isinstance(config, str):
        config = load_config(config)
    lookup_address = lookup_address or address
    backend = config['cluster'].get('backend', 'fixed')
    if backend == 'fixed' and config['cluster'].get('lookup_address') is None:
        config['cluster']['lookup_address'] = lookup_address
    if web:
        # try to append web to services
        config['services'].append('web')
    if modules:
        config['modules'] = modules
    try:
        await start_services(NodeRole.SUPERVISOR, config, address=address)
    except ImportError:
        if web == 'auto':
            config['services'] = [service for service in config['services']
                                  if service != 'web']
            await start_services(NodeRole.SUPERVISOR, config, address=address)
            return False
        else:  # pragma: no cover
            raise
    else:
        return bool(web)


async def stop_supervisor(address: str,
                          config: Dict = None):
    if not config or isinstance(config, str):
        config = load_config(config)
    await stop_services(NodeRole.SUPERVISOR, address=address, config=config)


async def start_worker(address: str,
                       lookup_address: str,
                       band_to_slots: Dict[str, int],
                       modules: Union[List, str, None] = None,
                       config: Dict = None,
                       mark_ready: bool = True):
    if not config or isinstance(config, str):
        config = load_config(config)
    backend = config['cluster'].get('backend', 'fixed')
    if backend == 'fixed' and config['cluster'].get('lookup_address') is None:
        config['cluster']['lookup_address'] = lookup_address
    if config['cluster'].get('resource') is None:
        config['cluster']['resource'] = band_to_slots
    if any(band_name.startswith('gpu-') for band_name in band_to_slots):  # pragma: no cover
        if 'cuda' not in config['storage']['backends']:
            config['storage']['backends'].append('cuda')
    if modules:
        config['modules'] = modules
    await start_services(NodeRole.WORKER, config, address=address,
                         mark_ready=mark_ready)


async def stop_worker(address: str,
                      config: Dict = None):
    if not config or isinstance(config, str):
        config = load_config(config)
    await stop_services(NodeRole.WORKER, address=address, config=config)
