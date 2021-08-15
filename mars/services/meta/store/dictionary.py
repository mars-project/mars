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

from dataclasses import asdict
from typing import Dict, List

from .... import oscar as mo
from ....utils import implements
from ....typing import BandType
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

    def _set_meta(self,
                  object_id: str,
                  meta: _CommonMeta):
        self._store[object_id] = meta

    @implements(AbstractMetaStore.set_meta)
    @mo.extensible
    async def set_meta(self,
                       object_id: str,
                       meta: _CommonMeta):
        self._set_meta(object_id, meta)

    @set_meta.batch
    async def batch_set_meta(self, args_list, kwargs_list):
        for args, kwargs in zip(args_list, kwargs_list):
            self._set_meta(*args, **kwargs)

    def _get_meta(self,
                  object_id: str,
                  fields: List[str] = None,
                  error: str = 'raise') -> Dict:
        if error not in ('raise', 'ignore'):  # pragma: no cover
            raise ValueError('error must be raise or ignore')
        try:
            meta = asdict(self._store[object_id])
            if fields:
                return {k: meta[k] for k in fields}
            return meta
        except KeyError:
            if error == 'raise':
                raise
            else:
                return

    @implements(AbstractMetaStore.get_meta)
    @mo.extensible
    async def get_meta(self,
                       object_id: str,
                       fields: List[str] = None,
                       error: str = 'raise') -> Dict:
        return self._get_meta(object_id, fields=fields, error=error)

    @get_meta.batch
    async def batch_get_meta(self, args_list, kwargs_list):
        metas = []
        for args, kwargs in zip(args_list, kwargs_list):
            metas.append(self._get_meta(*args, **kwargs))
        return metas

    def _del_meta(self, object_id: str):
        del self._store[object_id]

    @implements(AbstractMetaStore.del_meta)
    @mo.extensible
    async def del_meta(self,
                       object_id: str):
        self._del_meta(object_id)

    @del_meta.batch
    async def batch_del_meta(self, args_list, kwargs_list):
        for args, kwargs in zip(args_list, kwargs_list):
            self._del_meta(*args, **kwargs)

    def _add_chunk_bands(self,
                         object_id: str,
                         bands: List[BandType]):
        meta = self._store[object_id]
        assert isinstance(meta, _ChunkMeta)
        meta.bands = list(set(meta.bands) | set(bands))

    @implements(AbstractMetaStore.add_chunk_bands)
    @mo.extensible
    async def add_chunk_bands(self,
                              object_id: str,
                              bands: List[BandType]):
        self._add_chunk_bands(object_id, bands)

    @add_chunk_bands.batch
    async def batch_add_chunk_bands(self, args_list, kwargs_list):
        for args, kwargs in zip(args_list, kwargs_list):
            self._add_chunk_bands(*args, **kwargs)
