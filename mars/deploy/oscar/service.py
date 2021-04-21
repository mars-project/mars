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

import os
from typing import List, Dict, Union

import yaml

from ...services import start_services, stop_services, NodeRole


def _load_config(filename=None):
    # use default config
    if not filename:  # pragma: no cover
        d = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(d, 'config.yml')
    with open(filename) as f:
        return yaml.safe_load(f)


async def start_supervisor(address: str,
                           modules: Union[List, str, None] = None,
                           config: Dict = None):
    if not config or isinstance(config, str):
        config = _load_config(config)
    await start_services(NodeRole.SUPERVISOR, config,
                         modules=modules, address=address)


async def stop_supervisor(address: str,
                          config: Dict = None):
    if not config or isinstance(config, str):
        config = _load_config(config)
    await stop_services(NodeRole.SUPERVISOR, address, config)


async def start_worker(address: str,
                       lookup_address: str,
                       band_to_slots: Dict[str, int],
                       modules: Union[List, str, None] = None,
                       config: Dict = None):
    if not config or isinstance(config, str):
        config = _load_config(config)
    if config['cluster'].get('lookup_address') is None:
        config['cluster']['lookup_address'] = lookup_address
    if config['cluster'].get('resource') is None:
        config['cluster']['resource'] = band_to_slots
    await start_services(NodeRole.WORKER, config,
                         modules=modules, address=address)


async def stop_worker(address: str,
                      config: Dict = None):
    if not config or isinstance(config, str):
        config = _load_config(config)
    await stop_services(NodeRole.WORKER, address, config)
