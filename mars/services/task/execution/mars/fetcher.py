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

import operator

from collections import namedtuple, defaultdict
from typing import Dict, List
from ..api import Fetcher, register_fetcher_cls


_GetWithIndex = namedtuple("GetWithIndex", ["get", "index"])


@register_fetcher_cls
class MarsFetcher(Fetcher):
    name = "mars"
    required_meta_keys = ("bands",)

    def __init__(self, get_storage_api, **kwargs):
        self._get_storage_api = get_storage_api
        self._storage_api_to_gets = defaultdict(list)
        self._counter = 0

    async def append(self, chunk_key: str, chunk_meta: Dict, conditions: List = None):
        band = None
        if chunk_meta:
            bands = chunk_meta.get("bands")
            if bands:
                band = bands[0]
        storage_api = await self._get_storage_api(band)
        get = _GetWithIndex(
            storage_api.get.delay(chunk_key, conditions=conditions), self._counter
        )
        self._storage_api_to_gets[storage_api].append(get)
        self._counter += 1

    async def get(self):
        results = [None] * self._counter
        for storage_api in self._storage_api_to_gets:
            gets = self._storage_api_to_gets[storage_api]
            fetched_data = await storage_api.get.batch(
                *map(operator.itemgetter(0), gets)
            )
            for get, data in zip(gets, fetched_data):
                results[get.index] = data
        return results
