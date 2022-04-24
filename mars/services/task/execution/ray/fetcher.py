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

import asyncio
from collections import namedtuple
from typing import Dict, List
from .....utils import lazy_import
from ..api import Fetcher, register_fetcher_cls

ray = lazy_import("ray")
_FetchInfo = namedtuple("FetchInfo", ["key", "object_ref", "conditions"])


@register_fetcher_cls
class RayFetcher(Fetcher):
    name = "ray"
    required_meta_keys = ("object_refs",)

    def __init__(self, **kwargs):
        self._fetch_info_list = []
        self._no_conditions = True

    async def append(self, chunk_key: str, chunk_meta: Dict, conditions: List = None):
        if conditions is not None:
            self._no_conditions = False
        self._fetch_info_list.append(
            _FetchInfo(chunk_key, chunk_meta["object_refs"][0], conditions)
        )

    async def get(self):
        if self._no_conditions:
            return await asyncio.gather(
                *(info.object_ref for info in self._fetch_info_list)
            )
        refs = [None] * len(self._fetch_info_list)
        for index, fetch_info in enumerate(self._fetch_info_list):
            if fetch_info.conditions is None:
                refs[index] = fetch_info.object_ref
            else:
                refs[index] = query_object_with_condition.remote(
                    fetch_info.object_ref, fetch_info.conditions
                )
        return await asyncio.gather(*refs)


def query_object_with_condition(o, conditions):
    try:
        return o.iloc[conditions]
    except AttributeError:
        return o[conditions]


if ray:
    query_object_with_condition = ray.remote(query_object_with_condition)
