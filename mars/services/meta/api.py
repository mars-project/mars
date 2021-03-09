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


from typing import Dict, List, Type

from ... import oscar as mo
from ...tensor.core import TENSOR_TYPE, TENSOR_CHUNK_TYPE, \
    Tensor, TensorChunk
from ...dataframe.core import DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE, \
    DataFrame, DataFrameChunk, \
    SERIES_TYPE, SERIES_CHUNK_TYPE, Series, SeriesChunk, \
    INDEX_TYPE, INDEX_CHUNK_TYPE, Index, IndexChunk
from ...core import OBJECT_TYPE, OBJECT_CHUNK_TYPE, Object, ObjectChunk
from .core import MetaStoreActor
from .store import AbstractMetaStore


class MetaAPI:
    def __init__(self,
                 session_id: str,
                 meta_store: AbstractMetaStore):
        self._session_id = session_id
        self._meta_store = meta_store

    @staticmethod
    async def create(config: Dict) -> "MetaAPI":
        """
        Create Meta API according to config.

        Parameters
        ----------
        config

        Returns
        -------
        meta_api
            Meta api.
        """
        config = config.copy()
        session_id = config.pop('session_id')
        supervisor_address = config.pop('supervisor_address')
        if config:
            raise TypeError(f'Config {config!r} cannot be recognized.')
        meta_store_ref = mo.create_actor_ref(
            supervisor_address, MetaStoreActor.gen_uid(session_id))
        return MetaAPI(session_id, meta_store_ref)

    async def set_tileable_meta(self,
                                tileable,
                                physical_size: int = None,
                                **extra):
        if isinstance(tileable, TENSOR_TYPE):
            tensor: Tensor = tileable
            return await self._meta_store.set_tensor_meta(
                tensor.key, shape=tensor.shape,
                dtype=tensor.dtype, order=tensor.order,
                nsplits=tensor.nsplits,
                physical_size=physical_size, extra=extra)
        elif isinstance(tileable, DATAFRAME_TYPE):
            df: DataFrame = tileable
            return await self._meta_store.set_dataframe_meta(
                df.key, shape=df.shape,
                dtypes_value=df.dtypes_value,
                index_value=df.index_value,
                nsplits=df.nsplits,
                physical_size=physical_size, extra=extra)
        elif isinstance(tileable, SERIES_TYPE):
            series: Series = tileable
            return await self._meta_store.set_series_meta(
                series.key, shape=series.shape,
                dtype=series.dtype,
                index_value=series.index_value,
                name=series.name, nsplits=series.nsplits,
                physical_size=physical_size, extra=extra)
        elif isinstance(tileable, INDEX_TYPE):
            index: Index = tileable
            return await self._meta_store.set_index_meta(
                index.key, shape=index.shape,
                dtype=index.dtype,
                index_value=index.index_value,
                name=index.name, nsplits=index.nsplits,
                physical_size=physical_size, extra=extra)
        elif isinstance(tileable, OBJECT_TYPE):
            obj: Object = tileable
            return await self._meta_store.set_object_meta(
                obj.key, nsplits=obj.nsplits,
                physical_size=physical_size, extra=extra)
        else:  # pragma: no cover
            raise TypeError('unknown tileable type to set meta')

    async def get_tileable_meta(self,
                                object_id: str,
                                tileable_type: Type,
                                fields: List[str] = None):
        if issubclass(tileable_type, TENSOR_TYPE):
            return await self._meta_store.get_tensor_meta(object_id, fields=fields)
        elif issubclass(tileable_type, DATAFRAME_TYPE):
            return await self._meta_store.get_dataframe_meta(object_id, fields=fields)
        elif issubclass(tileable_type, SERIES_TYPE):
            return await self._meta_store.get_series_meta(object_id, fields=fields)
        elif issubclass(tileable_type, INDEX_TYPE):
            return await self._meta_store.get_index_meta(object_id, fields=fields)
        elif issubclass(tileable_type, OBJECT_TYPE):
            return await self._meta_store.get_object_meta(object_id, fields=fields)
        else:  # pragma: no cover
            raise TypeError('unknown tileable type to get meta')

    async def del_tileable_meta(self,
                                object_id: str,
                                tileable_type: Type):
        if issubclass(tileable_type, TENSOR_TYPE):
            return await self._meta_store.del_tensor_meta(object_id)
        elif issubclass(tileable_type, DATAFRAME_TYPE):
            return await self._meta_store.del_dataframe_meta(object_id)
        elif issubclass(tileable_type, SERIES_TYPE):
            return await self._meta_store.del_series_meta(object_id)
        elif issubclass(tileable_type, INDEX_TYPE):
            return await self._meta_store.del_index_meta(object_id)
        elif issubclass(tileable_type, OBJECT_TYPE):
            return await self._meta_store.del_object_meta(object_id)
        else:  # pragma: no cover
            raise TypeError('unknown tileable type to del meta')

    async def set_chunk_meta(self,
                             chunk,
                             physical_size: int = None,
                             workers: List[str] = None,
                             **extra):
        if isinstance(chunk, TENSOR_CHUNK_TYPE):
            tensor: TensorChunk = chunk
            return await self._meta_store.set_tensor_chunk_meta(
                tensor.key, shape=tensor.shape,
                dtype=tensor.dtype, order=tensor.order,
                index=tensor.index, physical_size=physical_size,
                workers=workers, extra=extra)
        elif isinstance(chunk, DATAFRAME_CHUNK_TYPE):
            df: DataFrameChunk = chunk
            return await self._meta_store.set_dataframe_chunk_meta(
                df.key, shape=df.shape,
                dtypes_value=df.dtypes_value,
                index_value=df.index_value,
                index=df.index, physical_size=physical_size,
                workers=workers, extra=extra)
        elif isinstance(chunk, SERIES_CHUNK_TYPE):
            series: SeriesChunk = chunk
            return await self._meta_store.set_series_chunk_meta(
                series.key, shape=series.shape,
                dtype=series.dtype,
                index_value=series.index_value,
                name=series.name, index=series.index,
                physical_size=physical_size,
                workers=workers, extra=extra)
        elif isinstance(chunk, INDEX_CHUNK_TYPE):
            index: IndexChunk = chunk
            return await self._meta_store.set_index_chunk_meta(
                index.key, shape=index.shape,
                dtype=index.dtype,
                index_value=index.index_value,
                name=index.name,
                index=index.index, physical_size=physical_size,
                workers=workers, extra=extra)
        elif isinstance(chunk, OBJECT_CHUNK_TYPE):
            obj: ObjectChunk = chunk
            return await self._meta_store.set_object_chunk_meta(
                obj.key, index=obj.index,
                physical_size=physical_size,
                workers=workers, extra=extra)
        else:  # pragma: no cover
            raise TypeError('unknown tileable type to set meta')

    async def get_chunk_meta(self,
                             object_id: str,
                             chunk_type: Type,
                             fields: List[str] = None):
        if issubclass(chunk_type, TENSOR_CHUNK_TYPE):
            return await self._meta_store.get_tensor_chunk_meta(
                object_id, fields=fields)
        elif issubclass(chunk_type, DATAFRAME_CHUNK_TYPE):
            return await self._meta_store.get_dataframe_chunk_meta(
                object_id, fields=fields)
        elif issubclass(chunk_type, SERIES_CHUNK_TYPE):
            return await self._meta_store.get_series_chunk_meta(
                object_id, fields=fields)
        elif issubclass(chunk_type, INDEX_CHUNK_TYPE):
            return await self._meta_store.get_index_chunk_meta(
                object_id, fields=fields)
        elif issubclass(chunk_type, OBJECT_CHUNK_TYPE):
            return await self._meta_store.get_object_chunk_meta(
                object_id, fields=fields)
        else:  # pragma: no cover
            raise TypeError(f'unknown chunk type to get meta {chunk_type}')

    async def del_chunk_meta(self,
                             object_id: str,
                             chunk_type: Type):
        if issubclass(chunk_type, TENSOR_CHUNK_TYPE):
            return await self._meta_store.del_tensor_chunk_meta(object_id)
        elif issubclass(chunk_type, DATAFRAME_CHUNK_TYPE):
            return await self._meta_store.del_dataframe_chunk_meta(object_id)
        elif issubclass(chunk_type, SERIES_CHUNK_TYPE):
            return await self._meta_store.del_series_chunk_meta(object_id)
        elif issubclass(chunk_type, INDEX_CHUNK_TYPE):
            return await self._meta_store.del_index_chunk_meta(object_id)
        elif issubclass(chunk_type, OBJECT_CHUNK_TYPE):
            return await self._meta_store.del_object_chunk_meta(object_id)
        else:  # pragma: no cover
            raise TypeError('unknown chunk type to del meta')
