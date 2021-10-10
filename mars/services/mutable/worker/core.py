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

import bisect
from collections import defaultdict
import sys
from typing import Union

import numpy as np

from ....typing import ChunkType


class Chunk:
    def __init__(self,
                 chunk: ChunkType,
                 manager_address: str,
                 worker_address: str,
                 default_value: Union[int, float] = 0) -> None:
        self._chunk = chunk
        self._manager_address = manager_address
        self._worker_address = worker_address
        self._default_value = default_value

        self._records = defaultdict(list)

    @property
    def chunk(self):
        return self._chunk

    async def write(self, records):
        for flat_index, value, ts in records:
            self._records[flat_index].append((ts, value))

    async def read(self, records, chunk_value_shape, timestamp):
        result = np.full(shape=chunk_value_shape, fill_value=self._default_value)
        for flat_index, value_index in records:
            if flat_index not in self._records:
                continue
            # Find the newest one.
            #
            # FIXME Python doesn't have things like SortedDict or SortedList,
            # we trigger a `sorted` here to ensure the correct semantic and try
            # to be as efficient as possible.
            self._records[flat_index].sort()
            # bitsect will compare on first element in the tuple.
            index = bisect.bisect_left(self._records[flat_index], (timestamp, sys.float_info.min))
            if index >= len(self._records[flat_index]):
                index = -1
            result[value_index] = self._records[flat_index][index][1]  # take the value
        return result

    async def seal(self, timestamp):
        result = np.full(self._chunk.shape, self._default_value)
        for flat_index, values in self._records.items():
            if flat_index not in self._records:
                continue
            # compute value
            values.sort()
            index = bisect.bisect_left(values, (timestamp, sys.float_info.min))
            if index >= len(values):
                index = -1
            # compute value index
            value_index = np.unravel_index(flat_index, self._chunk.shape)
            result[value_index] = values[index][1]  # take the value
        return result
