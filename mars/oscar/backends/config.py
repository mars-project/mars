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

from typing import Union, List, Dict


class ActorPoolConfig:
    __slots__ = '_conf',

    def __init__(self, conf: Dict = None):
        if conf is None:
            conf = dict()
        self._conf = conf
        if 'pools' not in self._conf:
            self._conf['pools'] = dict()
        if 'mapping' not in self._conf:
            self._conf['mapping'] = dict()

    @property
    def n_pool(self):
        return len(self._conf['pools'])

    def add_pool_conf(self,
                      process_index: int,
                      label: str,
                      internal_address: str,
                      external_address: Union[str, List[str]],
                      env: Dict = None):
        pools: Dict = self._conf['pools']
        if not isinstance(external_address, list):
            external_address = [external_address]
        pools[process_index] = {
            'label': label,
            'internal_address': internal_address,
            'external_address': external_address,
            'env': env
        }
        for addr in external_address:
            mapping: Dict = self._conf['mapping']
            mapping[addr] = internal_address

    def get_pool_config(self, process_index: int):
        return self._conf['pools'][process_index]

    def get_external_address(self, process_index: int) -> str:
        return self._conf['pools'][process_index]['external_address'][0]

    def get_process_indexes(self):
        return list(self._conf['pools'])

    def get_process_index(self, external_address):
        for process_index, conf in self._conf['pools'].items():
            if external_address in conf['external_address']:
                return process_index
        raise ValueError(f'Cannot get process_index '
                         f'for {external_address}')  # pragma: no cover

    def get_external_addresses(self, label=None) -> List[str]:
        result = []
        for c in self._conf['pools'].values():
            if label is not None:
                if label == c['label']:
                    result.append(c['external_address'][0])
            else:
                result.append(c['external_address'][0])
        return result

    @property
    def external_to_internal_address_map(self) -> Dict[str, str]:
        return self._conf['mapping']

    def as_dict(self):
        return self._conf
