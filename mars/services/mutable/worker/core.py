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

from typing import Tuple
from collections import OrderedDict
import numpy as np


class Chunk:
    def __init__(self,
                idx: int,
                shape: Tuple,
                worker_adress,
                storage_api,
                value=None) -> None:
        self._idx = idx
        self._shape = shape
        self._worker_address = worker_adress
        self._storage_api = storage_api
        self._value = value

    async def initstorage(self):
        await self._storage_api.put('data'+str(self._idx), np.full(self._shape, self._value))

    async def write(self, index, value, version_time):
        try:
            index_data: OrderedDict = await self._storage_api.get('data'+str(self._idx)+str(index))
            await self._storage_api.delete('data'+str(self._idx)+str(index))
        except Exception:
            index_data = OrderedDict()
        index_data[version_time] = value
        await self._storage_api.put('data'+str(self._idx)+str(index), index_data)

    async def read(self, index, version_time):
        try:
            index_data: OrderedDict = await self._storage_api.get('data'+str(self._idx)+str(index))
        except Exception:
            index_data = OrderedDict()
        result = self._value
        for k, v in index_data.items():
            if k <= version_time:
                result = v
            else:
                break
        return result
