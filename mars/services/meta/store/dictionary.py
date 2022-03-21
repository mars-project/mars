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

import functools
from collections import defaultdict
from dataclasses import fields as dataclass_fields
from typing import Dict, List, Set

from .... import oscar as mo
from ....utils import implements
from ....typing import BandType
from ..core import _CommonMeta, _ChunkMeta
from .base import AbstractMetaStore, register_meta_store


@functools.lru_cache(100)
def _get_meta_fields(meta_cls):
    return [f.name for f in dataclass_fields(meta_cls)]


@register_meta_store
class DictMetaStore(AbstractMetaStore):
    name = "dict"

    def __init__(self, session_id: str, **kw):
        super().__init__(session_id)
        self._store: Dict[str, _CommonMeta] = dict()
        self._band_chunks: Dict[BandType, Set[str]] = defaultdict(set)
        if kw:  # pragma: no cover
            raise TypeError(f"Keyword arguments {kw!r} cannot be recognized.")

    @classmethod
    @implements(AbstractMetaStore.create)
    async def create(cls, config) -> Dict:
        # Nothing needs to do for dict-based meta store.
        # no extra kwargs.
        return dict()

    def _set_meta(self, object_id: str, meta: _CommonMeta):
        if isinstance(meta, _ChunkMeta):
            for band in meta.bands:
                self._band_chunks[band].add(object_id)
        prev_meta = self._store.get(object_id)
        if prev_meta:
            meta = meta.merge_from(prev_meta)
        self._store[object_id] = meta

    @implements(AbstractMetaStore.set_meta)
    @mo.extensible
    async def set_meta(self, object_id: str, meta: _CommonMeta):
        self._set_meta(object_id, meta)

    @set_meta.batch
    async def batch_set_meta(self, args_list, kwargs_list):
        for args, kwargs in zip(args_list, kwargs_list):
            self._set_meta(*args, **kwargs)

    def _get_meta(
        self, object_id: str, fields: List[str] = None, error: str = "raise"
    ) -> Dict:
        if error not in ("raise", "ignore"):  # pragma: no cover
            raise ValueError("error must be raise or ignore")
        try:
            meta = self._store[object_id]
            if fields is None:
                fields = _get_meta_fields(type(meta))
            return {k: getattr(meta, k) for k in fields}
        except KeyError:
            if error == "raise":
                raise
            else:
                return

    @implements(AbstractMetaStore.get_meta)
    @mo.extensible
    async def get_meta(
        self, object_id: str, fields: List[str] = None, error: str = "raise"
    ) -> Dict:
        return self._get_meta(object_id, fields=fields, error=error)

    @get_meta.batch
    async def batch_get_meta(self, args_list, kwargs_list):
        metas = []
        for args, kwargs in zip(args_list, kwargs_list):
            metas.append(self._get_meta(*args, **kwargs))
        return metas

    def _del_meta(self, object_id: str):
        meta = self._store[object_id]
        if isinstance(meta, _ChunkMeta):
            for band in meta.bands:
                chunks = self._band_chunks[band]
                chunks.remove(object_id)
                if len(chunks) == 0:
                    del self._band_chunks[band]
        del self._store[object_id]

    @implements(AbstractMetaStore.del_meta)
    @mo.extensible
    async def del_meta(self, object_id: str):
        self._del_meta(object_id)

    @del_meta.batch
    async def batch_del_meta(self, args_list, kwargs_list):
        for args, kwargs in zip(args_list, kwargs_list):
            self._del_meta(*args, **kwargs)

    def _add_chunk_bands(self, object_id: str, bands: List[BandType]):
        meta = self._store[object_id]
        assert isinstance(meta, _ChunkMeta)
        meta.bands = list(set(meta.bands) | set(bands))
        for band in bands:
            self._band_chunks[band].add(object_id)

    @implements(AbstractMetaStore.add_chunk_bands)
    @mo.extensible
    async def add_chunk_bands(self, object_id: str, bands: List[BandType]):
        self._add_chunk_bands(object_id, bands)

    @add_chunk_bands.batch
    async def batch_add_chunk_bands(self, args_list, kwargs_list):
        for args, kwargs in zip(args_list, kwargs_list):
            self._add_chunk_bands(*args, **kwargs)

    def _remove_chunk_bands(self, object_id: str, bands: List[BandType]):
        meta = self._store[object_id]
        assert isinstance(meta, _ChunkMeta)
        meta.bands = list(set(meta.bands) - set(bands))
        for band in bands:
            self._band_chunks[band].remove(object_id)

    @implements(AbstractMetaStore.remove_chunk_bands)
    @mo.extensible
    async def remove_chunk_bands(self, object_id: str, bands: List[BandType]):
        self._remove_chunk_bands(object_id, bands)

    @remove_chunk_bands.batch
    async def batch_remove_chunk_bands(self, args_list, kwargs_list):
        for args, kwargs in zip(args_list, kwargs_list):
            self._remove_chunk_bands(*args, **kwargs)

    async def get_band_chunks(self, band: BandType) -> List[str]:
        return list(self._band_chunks[band])
