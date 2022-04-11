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

from ..api import Fetcher, register_fetcher_cls


@register_fetcher_cls
class RayObjectFetcher(Fetcher):
    __slots__ = ("_object_ref",)
    name = "ray"

    def __init__(self, meta, **kwargs):
        object_refs = meta["object_refs"]
        assert len(object_refs) == 1
        self._object_ref = object_refs[0]

    async def get(self, conditions=None):
        data = await self._object_ref
        if conditions is None:
            return data
        try:
            return data.iloc[tuple(conditions)]
        except AttributeError:
            return data[tuple(conditions)]
