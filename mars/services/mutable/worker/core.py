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
                chunk_key,
                worker_adress,
                storage_api,
                value=None) -> None:
        self._idx = idx
        self._shape = shape
        self._chunk_key = chunk_key
        self._worker_address = worker_adress
        self._storage_api = storage_api
        self._value = value

        self._ops = OrderedDict()

    async def write(self, index, value, version_time):
        try:
            index_data: OrderedDict = self._ops[index]
        except Exception:
            index_data = OrderedDict()
        index_data[version_time] = value
        self._ops[index] = index_data

    async def read(self, index, version_time):
        try:
            index_data: OrderedDict = self._ops[index]
        except Exception:
            index_data = OrderedDict()
        last_version = 0
        for k in index_data.keys():
            if k <= version_time:
                last_version = k if k > last_version else last_version
        result = index_data[last_version] if last_version != 0 else self._value
        return result

    async def seal(self, version_time):
        _tensor = np.full(self._shape, self._value)
        for k, v in self._ops.items():
            last_version = 0

            for version_t, _ in v.items():
                if version_t <= version_time:
                    last_version = version_t if version_t > last_version else last_version
            result = self._value if last_version == 0 else v[last_version]
            _tensor[k] = result
        return _tensor
