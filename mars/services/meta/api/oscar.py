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


from typing import Dict, List, Any, Union

from .... import oscar as mo
from ....core import ChunkType
from ....core.operand import Fuse
from ....dataframe.core import DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE, \
    DATAFRAME_GROUPBY_TYPE, DATAFRAME_GROUPBY_CHUNK_TYPE, \
    SERIES_GROUPBY_TYPE, SERIES_GROUPBY_CHUNK_TYPE
from ....lib.aio import alru_cache
from ....utils import extensible
from ...core import BandType
from ..core import get_meta_type
from ..store import AbstractMetaStore
from ..supervisor.core import MetaStoreManagerActor, MetaStoreActor
from .core import AbstractMetaAPI


class MetaAPI(AbstractMetaAPI):
    def __init__(self,
                 session_id: str,
                 meta_store: Union[AbstractMetaStore, mo.ActorRef]):
        self._session_id = session_id
        self._meta_store = meta_store

    @classmethod
    @alru_cache(cache_exceptions=False)
    async def create(cls,
                     session_id: str,
                     address: str) -> "MetaAPI":
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
        meta_store_ref = await mo.actor_ref(
            address, MetaStoreActor.gen_uid(session_id))

        return MetaAPI(session_id, meta_store_ref)

    @classmethod
    async def create_session(cls,
                             session_id: str,
                             address: str) -> "MetaAPI":
        """
        Creating a new meta store for the session, and return meta API.

        Parameters
        ----------
        session_id : str
            Session ID.
        address : str
            Supervisor address.

        Returns
        -------
        meta_api
            Meta API.
        """
        # get MetaStoreManagerActor ref.
        meta_store_manager_ref = await mo.actor_ref(
            address, MetaStoreManagerActor.default_uid())
        meta_store_ref = \
            await meta_store_manager_ref.new_session_meta_store(session_id)
        return MetaAPI(session_id, meta_store_ref)

    @classmethod
    async def destroy_session(cls,
                              session_id: str,
                              address: str):
        """
        Destroy a session.

        Parameters
        ----------
        session_id : str
            Session ID.
        address : str
            Supervisor address.
        """
        meta_store_ref = await mo.actor_ref(
            address, MetaStoreActor.gen_uid(session_id))
        return await mo.destroy_actor(meta_store_ref)

    @extensible
    async def set_tileable_meta(self,
                                tileable,
                                memory_size: int = None,
                                store_size: int = None,
                                **extra):
        params = tileable.params.copy()
        if isinstance(tileable, (DATAFRAME_TYPE, DATAFRAME_GROUPBY_TYPE,
                                 SERIES_GROUPBY_TYPE)):
            # dataframe needs some special process for now
            del params['columns_value']
            del params['dtypes']
            params.pop('key_dtypes', None)
            params['dtypes_value'] = tileable.dtypes_value
        params['nsplits'] = tileable.nsplits
        params.update(extra)
        meta = get_meta_type(type(tileable))(object_id=tileable.key,
                                             **params,
                                             memory_size=memory_size,
                                             store_size=store_size)
        return await self._meta_store.set_meta(tileable.key, meta)

    @extensible
    async def get_tileable_meta(self,
                                object_id: str,
                                fields: List[str] = None) -> Dict[str, Any]:
        return await self._meta_store.get_meta(object_id, fields=fields)

    @extensible
    async def del_tileable_meta(self,
                                object_id: str):
        return await self._meta_store.del_meta(object_id)

    @classmethod
    def _extract_chunk_meta(cls,
                            chunk: ChunkType,
                            memory_size: int = None,
                            store_size: int = None,
                            bands: List[BandType] = None,
                            **extra):
        if isinstance(chunk.op, Fuse):
            # fuse op
            chunk = chunk.chunk
        params = chunk.params.copy()
        if isinstance(chunk, (DATAFRAME_CHUNK_TYPE, DATAFRAME_GROUPBY_CHUNK_TYPE,
                              SERIES_GROUPBY_CHUNK_TYPE)):
            # dataframe chunk needs some special process for now
            params.pop('columns_value', None)
            params.pop('dtypes', None)
            params.pop('key_dtypes', None)
        params.update(extra)
        return get_meta_type(type(chunk))(object_id=chunk.key,
                                          **params,
                                          bands=bands,
                                          memory_size=memory_size,
                                          store_size=store_size)

    @extensible
    async def set_chunk_meta(self,
                             chunk: ChunkType,
                             memory_size: int = None,
                             store_size: int = None,
                             bands: List[BandType] = None,
                             **extra):
        meta = self._extract_chunk_meta(
            chunk, memory_size=memory_size, store_size=store_size,
            bands=bands, **extra)
        return await self._meta_store.set_meta(chunk.key, meta)

    @set_chunk_meta.batch
    async def batch_set_chunk_meta(self, args_list, kwargs_list):
        set_chunk_metas = []
        for args, kwargs in zip(args_list, kwargs_list):
            chunk = args[0]
            meta = self._extract_chunk_meta(*args, **kwargs)
            set_chunk_metas.append(self._meta_store.set_meta.delay(chunk.key, meta))
        return await self._meta_store.set_meta.batch(*set_chunk_metas)

    @extensible
    async def get_chunk_meta(self,
                             object_id: str,
                             fields: List[str] = None,
                             error='raise'):
        return await self._meta_store.get_meta(
            object_id, fields=fields, error=error)

    @get_chunk_meta.batch
    async def batch_get_chunk_meta(self, args_list, kwargs_list):
        get_chunk_metas = []
        for args, kwargs in zip(args_list, kwargs_list):
            get_chunk_metas.append(self._meta_store.get_meta.delay(*args, **kwargs))
        return await self._meta_store.get_meta.batch(*get_chunk_metas)

    @extensible
    async def del_chunk_meta(self,
                             object_id: str):
        return await self._meta_store.del_meta(object_id)

    @del_chunk_meta.batch
    async def batch_del_chunk_meta(self, args_list, kwargs_list):
        del_chunk_metas = []
        for args, kwargs in zip(args_list, kwargs_list):
            del_chunk_metas.append(self._meta_store.del_meta.delay(*args, **kwargs))
        return await self._meta_store.del_meta.batch(*del_chunk_metas)

    @extensible
    async def add_chunk_bands(self,
                              object_id: str,
                              bands: List[BandType]):
        return await self._meta_store.add_chunk_bands(object_id, bands)


class MockMetaAPI(MetaAPI):
    @classmethod
    async def create(cls, session_id: str, address: str) -> "MetaAPI":
        # create an Actor for mock
        try:
            meta_store_manager_ref = await mo.create_actor(
                MetaStoreManagerActor, 'dict', dict(),
                address=address,
                uid=MetaStoreManagerActor.default_uid())
        except mo.ActorAlreadyExist:
            # ignore if actor exists
            meta_store_manager_ref = await mo.actor_ref(
                MetaStoreManagerActor, address=address,
                uid=MetaStoreManagerActor.default_uid())
        try:
            await meta_store_manager_ref.new_session_meta_store(session_id)
        except mo.ActorAlreadyExist:
            pass
        return await super().create(session_id=session_id, address=address)
