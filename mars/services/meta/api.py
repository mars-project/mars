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


from typing import Dict, List, Tuple, Any, Union

from ... import oscar as mo
from ...oscar.backends.mars import allocate_strategy
from ...dataframe.core import DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE
from .core import get_meta_type
from .supervisor.core import MetaStoreManagerActor, MetaStoreActor
from .store import AbstractMetaStore


class MetaAPI:
    def __init__(self,
                 session_id: str,
                 meta_store: Union[AbstractMetaStore, mo.ActorRef]):
        self._session_id = session_id
        self._meta_store = meta_store

    @classmethod
    async def create(cls,
                     session_id: str,
                     address: str) -> "MetaAPI":
        """
        Create Meta API according to config.

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
            await meta_store_manager_ref.new_session_meta_store(session_id, address)
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

    async def set_tileable_meta(self,
                                tileable,
                                memory_size: int = None,
                                store_size: int = None,
                                **extra):
        params = tileable.params.copy()
        if isinstance(tileable, DATAFRAME_TYPE):
            # dataframe needs some special process for now
            del params['columns_value']
            del params['dtypes']
            params['dtypes_value'] = tileable.dtypes_value
        params['nsplits'] = tileable.nsplits
        params.update(extra)
        meta = get_meta_type(type(tileable))(object_id=tileable.key,
                                             **params,
                                             memory_size=memory_size,
                                             store_size=store_size)
        return await self._meta_store.set_meta(tileable.key, meta)

    async def get_tileable_meta(self,
                                object_id: str,
                                fields: List[str] = None) -> Dict[str, Any]:
        return await self._meta_store.get_meta(object_id, fields=fields)

    async def del_tileable_meta(self,
                                object_id: str):
        return await self._meta_store.del_meta(object_id)

    async def set_chunk_meta(self,
                             chunk,
                             memory_size: int = None,
                             store_size: int = None,
                             bands: List[Tuple[str, str]] = None,
                             **extra):
        params = chunk.params.copy()
        if isinstance(chunk, DATAFRAME_CHUNK_TYPE):
            # dataframe chunk needs some special process for now
            del params['columns_value']
            del params['dtypes']
            params['dtypes_value'] = chunk.dtypes_value
        params.update(extra)
        meta = get_meta_type(type(chunk))(object_id=chunk.key,
                                          **params,
                                          bands=bands,
                                          memory_size=memory_size,
                                          store_size=store_size)
        return await self._meta_store.set_meta(chunk.key, meta)

    async def get_chunk_meta(self,
                             object_id: str,
                             fields: List[str] = None):
        return await self._meta_store.get_meta(object_id, fields=fields)

    async def del_chunk_meta(self,
                             object_id: str):
        return await self._meta_store.del_meta(object_id)

    async def add_chunk_bands(self,
                              object_id: str,
                              bands: List[Tuple[str, str]]):
        return await self._meta_store.add_chunk_bands(object_id, bands)


class MockMetaAPI(MetaAPI):
    @classmethod
    async def create(cls, session_id: str, address: str) -> "MetaAPI":
        # create an Actor for mock
        try:
            await mo.create_actor(MetaStoreActor, 'dict', session_id,
                                  address=address,
                                  uid=MetaStoreActor.gen_uid(session_id),
                                  allocate_strategy=allocate_strategy.ProcessIndex(1))
        except mo.ActorAlreadyExist:
            # ignore if actor exists
            pass
        return await super().create(session_id=session_id, address=address)
