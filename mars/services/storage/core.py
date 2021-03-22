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

import logging
from collections import defaultdict
from typing import Dict, List, Union, Tuple

from ... import oscar as mo
from ...storage import StorageLevel, get_storage_backend

logger = logging.getLogger(__name__)


class StorageQuota:
    def __init__(self, total_size: Union[int, None]):
        self._total_size = total_size
        self._used_size = 0

    @property
    def total_size(self):
        return self._total_size

    @property
    def used_size(self):
        return self._used_size

    def request(self, size: int) -> bool:
        if self._total_size is None:
            self._used_size += size
            return True
        elif self._used_size + size >= self._total_size:
            return False
        else:
            self._used_size += size
            return True

    def release(self, size: int):
        self._used_size -= size


class DataMetaManagerActor(mo.Actor):
    def __init__(self):
        # mapping key is (session_id, data_key)
        # mapping value is list of (object_id, level, size)
        self._mapping = defaultdict(list)

    def put(self,
            session_id: str,
            data_key: str,
            object_id,
            level: StorageLevel,
            size: int):
        self._mapping[(session_id, data_key)].append(
            (object_id, level, size))

    def get(self,
            session_id: str,
            data_key: str) -> List:
        return self._mapping.get((session_id, data_key))

    def get_lowest(self,
                   session_id: str,
                   data_key: str):
        # if the data is stored in multiply levels,
        # return the lowest level info
        infos = sorted(self._mapping.get((session_id, data_key)),
                       key=lambda x: x[1])
        return infos[0]

    def delete(self,
               session_id: str,
               data_key: str,
               level: StorageLevel):
        if (session_id, data_key) in self._mapping:
            infos = self._mapping[(session_id, data_key)]
            rest = [info for info in infos if info[1] != level]
            if len(rest) == 0:
                del self._mapping[(session_id, data_key)]
            else:  # pragma: no cover
                self._mapping[(session_id, data_key)] = rest


class StorageManagerActor(mo.Actor):
    def __init__(self,
                 storage_configs: Dict,
                 ):
        self._storage_configs = storage_configs
        # params to init and teardown
        self._init_params = dict()
        self._teardown_params = dict()
        # pinned_keys
        self._pinned_keys = []

    async def __post_create__(self):
        # stores data key to storage object id
        self._data_meta_ref = await mo.actor_ref(DataMetaManagerActor.default_uid(),
                                                 address=self.address)
        # setup storage backend
        clients = dict()
        quotas = dict()
        for backend, setup_params in self._storage_configs.items():
            client = await self._setup_storage(backend, setup_params)
            for level in StorageLevel.__members__.values():
                if client.level & level:
                    clients[level] = client
                    quotas[level] = StorageQuota(client.size)

        self._clients = clients
        self._quotas = quotas

    async def __pre_destroy__(self):
        for backend, teardown_params in self._teardown_params.items():
            backend_cls = get_storage_backend(backend)
            await backend_cls.teardown(**teardown_params)

    async def _setup_storage(self,
                             storage_backend: str,
                             storage_config: Dict):
        backend = get_storage_backend(storage_backend)
        init_params, teardown_params = await backend.setup(**storage_config)
        client = backend(**init_params)
        self._init_params[storage_backend] = init_params
        self._teardown_params[storage_backend] = teardown_params
        return client

    def get_client_params(self):
        return self._init_params

    def allocate_size(self,
                      size: int,
                      level: StorageLevel) -> Tuple[bool, int]:
        if self._quotas[level].request(size):
            return True, 0
        else:  # pragma: no cover
            quota = self._quotas[level]
            exceeded_size = (quota.used_size + size) - quota.total_size
            return False, exceeded_size

    def release_size(self,
                     size: int,
                     level: StorageLevel
                     ):
        self._quotas[level].release(size)

    def pin(self, object_id):
        self._pinned_keys.append(object_id)

    def unpin(self, object_id):
        self._pinned_keys.remove(object_id)
