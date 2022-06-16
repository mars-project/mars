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


from typing import Dict, List, Any

from .... import oscar as mo
from ....core import ChunkType
from ....core.operand import Fuse
from ....lib.aio import alru_cache
from ....typing import BandType
from ....utils import get_chunk_params
from ..core import get_meta_type
from ..store import AbstractMetaStore
from ..supervisor.core import MetaStoreManagerActor, MetaStoreActor
from ..worker.core import WorkerMetaStoreManagerActor
from .core import AbstractMetaAPI


class BaseMetaAPI(AbstractMetaAPI):
    def __init__(self, session_id: str, meta_store: mo.ActorRefType[AbstractMetaStore]):
        # make sure all meta types registered
        from .. import metas

        del metas

        self._session_id = session_id
        self._meta_store = meta_store

    @mo.extensible
    async def set_tileable_meta(
        self, tileable, memory_size: int = None, store_size: int = None, **extra
    ):
        from ....dataframe.core import (
            DATAFRAME_TYPE,
            DATAFRAME_GROUPBY_TYPE,
            SERIES_GROUPBY_TYPE,
        )

        params = tileable.params.copy()
        if isinstance(
            tileable, (DATAFRAME_TYPE, DATAFRAME_GROUPBY_TYPE, SERIES_GROUPBY_TYPE)
        ):
            # dataframe needs some special process for now
            del params["columns_value"]
            del params["dtypes"]
            params.pop("key_dtypes", None)
            params["dtypes_value"] = tileable.dtypes_value
        params["nsplits"] = tileable.nsplits
        params.update(extra)
        meta = get_meta_type(type(tileable))(
            object_id=tileable.key,
            **params,
            memory_size=memory_size,
            store_size=store_size
        )
        return await self._meta_store.set_meta(tileable.key, meta)

    @mo.extensible
    async def get_tileable_meta(
        self, object_id: str, fields: List[str] = None
    ) -> Dict[str, Any]:
        return await self._meta_store.get_meta(object_id, fields=fields)

    @mo.extensible
    async def del_tileable_meta(self, object_id: str):
        return await self._meta_store.del_meta(object_id)

    @classmethod
    def _extract_chunk_meta(
        cls,
        chunk: ChunkType,
        memory_size: int = None,
        store_size: int = None,
        bands: List[BandType] = None,
        fields: List[str] = None,
        exclude_fields: List[str] = None,
        **extra
    ):
        if isinstance(chunk.op, Fuse):
            # fuse op
            chunk = chunk.chunk
        params = get_chunk_params(chunk)
        chunk_key = extra.pop("chunk_key", chunk.key)
        object_ref = extra.pop("object_ref", None)
        params.update(extra)

        if object_ref:
            object_refs = (
                [object_ref] if not isinstance(object_ref, list) else object_ref
            )
        else:
            object_refs = []

        if fields is not None:
            fields = set(fields)
            params = {k: v for k, v in params.items() if k in fields}
        elif exclude_fields is not None:
            exclude_fields = set(exclude_fields)
            params = {k: v for k, v in params.items() if k not in exclude_fields}

        return get_meta_type(type(chunk))(
            object_id=chunk_key,
            **params,
            bands=bands,
            memory_size=memory_size,
            store_size=store_size,
            object_refs=object_refs
        )

    @mo.extensible
    async def set_chunk_meta(
        self,
        chunk: ChunkType,
        memory_size: int = None,
        store_size: int = None,
        bands: List[BandType] = None,
        fields: List[str] = None,
        exclude_fields: List[str] = None,
        **extra
    ):
        """
        Parameters
        ----------
        chunk: ChunkType
            chunk to set meta
        memory_size: int
            memory size for chunk data
        store_size: int
            serialized size for chunk data
        bands:
            chunk data bands
        fields: list
            fields to include in meta
        exclude_fields: list
            fields to exclude in meta
        extra

        Returns
        -------

        """
        meta = self._extract_chunk_meta(
            chunk,
            memory_size=memory_size,
            store_size=store_size,
            bands=bands,
            fields=fields,
            exclude_fields=exclude_fields,
            **extra
        )
        return await self._meta_store.set_meta(meta.object_id, meta)

    @set_chunk_meta.batch
    async def batch_set_chunk_meta(self, args_list, kwargs_list):
        set_chunk_metas = []
        for args, kwargs in zip(args_list, kwargs_list):
            meta = self._extract_chunk_meta(*args, **kwargs)
            set_chunk_metas.append(
                self._meta_store.set_meta.delay(meta.object_id, meta)
            )
        return await self._meta_store.set_meta.batch(*set_chunk_metas)

    @mo.extensible
    async def get_chunk_meta(
        self, object_id: str, fields: List[str] = None, error="raise"
    ):
        return await self._meta_store.get_meta(object_id, fields=fields, error=error)

    @get_chunk_meta.batch
    async def batch_get_chunk_meta(self, args_list, kwargs_list):
        get_chunk_metas = []
        for args, kwargs in zip(args_list, kwargs_list):
            get_chunk_metas.append(self._meta_store.get_meta.delay(*args, **kwargs))
        return await self._meta_store.get_meta.batch(*get_chunk_metas)

    @mo.extensible
    async def del_chunk_meta(self, object_id: str):
        """
        Parameters
        ----------
        object_id: str
            chunk id
        """
        return await self._meta_store.del_meta(object_id)

    @del_chunk_meta.batch
    async def batch_del_chunk_meta(self, args_list, kwargs_list):
        del_chunk_metas = []
        for args, kwargs in zip(args_list, kwargs_list):
            del_chunk_metas.append(self._meta_store.del_meta.delay(*args, **kwargs))
        return await self._meta_store.del_meta.batch(*del_chunk_metas)

    @mo.extensible
    async def add_chunk_bands(self, object_id: str, bands: List[BandType]):
        return await self._meta_store.add_chunk_bands(object_id, bands)

    @add_chunk_bands.batch
    async def batch_add_chunk_bands(self, args_list, kwargs_list):
        add_chunk_bands_tasks = []
        for args, kwargs in zip(args_list, kwargs_list):
            add_chunk_bands_tasks.append(
                self._meta_store.add_chunk_bands.delay(*args, **kwargs)
            )
        return await self._meta_store.add_chunk_bands.batch(*add_chunk_bands_tasks)

    @mo.extensible
    async def remove_chunk_bands(self, object_id: str, bands: List[BandType]):
        return await self._meta_store.remove_chunk_bands(object_id, bands)

    @remove_chunk_bands.batch
    async def batch_remove_chunk_bands(self, args_list, kwargs_list):
        remove_chunk_bands_tasks = []
        for args, kwargs in zip(args_list, kwargs_list):
            remove_chunk_bands_tasks.append(
                self._meta_store.remove_chunk_bands.delay(*args, **kwargs)
            )
        return await self._meta_store.remove_chunk_bands.batch(
            *remove_chunk_bands_tasks
        )

    @mo.extensible
    async def get_band_chunks(self, band: BandType) -> List[str]:
        return await self._meta_store.get_band_chunks(band)


class MetaAPI(BaseMetaAPI):
    @classmethod
    @alru_cache(maxsize=1024, cache_exceptions=False)
    async def create(cls, session_id: str, address: str) -> "MetaAPI":
        """
        Create Meta API.

        Parameters
        ----------
        session_id : str
            Session ID.
        address : str
            Supervisor address.

        Returns
        -------
        meta_api
            Meta api.
        """
        meta_store_ref = await mo.actor_ref(address, MetaStoreActor.gen_uid(session_id))

        return MetaAPI(session_id, meta_store_ref)


class MockMetaAPI(MetaAPI):
    @classmethod
    async def create(cls, session_id: str, address: str) -> "MetaAPI":
        # create an Actor for mock
        try:
            meta_store_manager_ref = await mo.create_actor(
                MetaStoreManagerActor,
                "dict",
                dict(),
                address=address,
                uid=MetaStoreManagerActor.default_uid(),
            )
        except mo.ActorAlreadyExist:
            # ignore if actor exists
            meta_store_manager_ref = await mo.actor_ref(
                MetaStoreManagerActor,
                address=address,
                uid=MetaStoreManagerActor.default_uid(),
            )
        try:
            await meta_store_manager_ref.new_session_meta_store(session_id)
        except mo.ActorAlreadyExist:
            pass
        return await super().create(session_id=session_id, address=address)


class WorkerMetaAPI(BaseMetaAPI):
    @classmethod
    @alru_cache(cache_exceptions=False)
    async def create(cls, session_id: str, address: str) -> "WorkerMetaAPI":
        """
        Create worker meta API.

        Parameters
        ----------
        session_id : str
            Session ID.
        address : str
            Worker address.

        Returns
        -------
        meta_api
            Worker meta api.
        """
        worker_meta_store_manager_ref = await mo.actor_ref(
            uid=WorkerMetaStoreManagerActor.default_uid(), address=address
        )
        worker_meta_store_ref = (
            await worker_meta_store_manager_ref.new_session_meta_store(session_id)
        )
        return WorkerMetaAPI(session_id, worker_meta_store_ref)


class MockWorkerMetaAPI(WorkerMetaAPI):
    @classmethod
    async def create(cls, session_id: str, address: str) -> "WorkerMetaAPI":
        # create an Actor for mock
        try:
            await mo.create_actor(
                WorkerMetaStoreManagerActor,
                "dict",
                dict(),
                address=address,
                uid=WorkerMetaStoreManagerActor.default_uid(),
            )
        except mo.ActorAlreadyExist:
            # ignore if actor exists
            await mo.actor_ref(
                WorkerMetaStoreManagerActor,
                address=address,
                uid=WorkerMetaStoreManagerActor.default_uid(),
            )
        return await super().create(session_id, address)
