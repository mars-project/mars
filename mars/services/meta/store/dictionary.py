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

from dataclasses import asdict
from typing import Dict, List, Tuple

from ....utils import implements
from ..core import _CommonMeta, _ChunkMeta
from .base import AbstractMetaStore, register_meta_store


@register_meta_store
class DictMetaStore(AbstractMetaStore):
    name = 'dict'

    def __init__(self, session_id: str, **kw):
        super().__init__(session_id)
        self._store: Dict[str, _CommonMeta] = dict()
        if kw:  # pragma: no cover
            raise TypeError(f'Keyword arguments {kw!r} cannot be recognized.')

    @classmethod
    @implements(AbstractMetaStore.create)
    async def create(cls, config) -> Dict:
        # Nothing needs to do for dict-based meta store.
        # no extra kwargs.
        return dict()

    @implements(AbstractMetaStore.set_meta)
    async def set_meta(self,
                       object_id: str,
                       meta: _CommonMeta):
        self._store[object_id] = meta

    @implements(AbstractMetaStore.get_meta)
    async def get_meta(self,
                       object_id: str,
                       fields: List[str] = None) -> Dict:
        meta = asdict(self._store[object_id])
        if fields:
            return {k: meta[k] for k in fields}
        return meta

    @implements(AbstractMetaStore.del_meta)
    async def del_meta(self,
                       object_id: str):
        del self._store[object_id]

    @implements(AbstractMetaStore.add_chunk_bands)
    async def add_chunk_bands(self,
                              object_id: str,
                              bands: List[Tuple[str, str]]):
        meta = self._store[object_id]
        assert isinstance(meta, _ChunkMeta)
        meta.bands = list(set(meta.bands) | set(bands))
