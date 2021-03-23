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


from typing import Dict, List, Type, Any

from ... import oscar as mo
from ...oscar.backends.mars import allocate_strategy
from ...dataframe.core import DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE
from .core import MetaStoreActor
from .store import AbstractMetaStore


class MetaAPI:
    def __init__(self,
                 session_id: str,
                 meta_store: AbstractMetaStore):
        self._session_id = session_id
        self._meta_store = meta_store

    @classmethod
    async def create(cls, session_id: str, address: str) -> "MetaAPI":
        """
        Create Meta API according to config.

        Parameters
        ----------
        session_id
        address

        Returns
        -------
        meta_api
            Meta api.
        """
        meta_store_ref = await mo.actor_ref(mo.create_actor_ref(
            address, MetaStoreActor.gen_uid(session_id)))
        return MetaAPI(session_id, meta_store_ref)

    async def set_tileable_meta(self,
                                tileable,
                                physical_size: int = None,
                                **extra):
        params = tileable.params.copy()
        if isinstance(tileable, DATAFRAME_TYPE):
            # dataframe needs some special process for now
            del params['columns_value']
            del params['dtypes']
            params['dtypes_value'] = tileable.dtypes_value
        params['nsplits'] = tileable.nsplits
        params.update(extra)
        return await self._meta_store.set_meta(
            type(tileable), tileable.key, **params, physical_size=physical_size)

    async def get_tileable_meta(self,
                                object_id: str,
                                tileable_type: Type,
                                fields: List[str] = None) -> Dict[str, Any]:
        return await self._meta_store.get_meta(tileable_type, object_id, fields=fields)

    async def del_tileable_meta(self,
                                object_id: str,
                                tileable_type: Type):
        return await self._meta_store.del_meta(tileable_type, object_id)

    async def set_chunk_meta(self,
                             chunk,
                             physical_size: int = None,
                             workers: List[str] = None,
                             **extra):
        params = chunk.params.copy()
        if isinstance(chunk, DATAFRAME_CHUNK_TYPE):
            # dataframe chunk needs some special process for now
            del params['columns_value']
            del params['dtypes']
            params['dtypes_value'] = chunk.dtypes_value
        params.update(extra)
        return await self._meta_store.set_meta(
            type(chunk), chunk.key, **params,
            physical_size=physical_size, workers=workers)

    async def get_chunk_meta(self,
                             object_id: str,
                             chunk_type: Type,
                             fields: List[str] = None):
        return await self._meta_store.get_meta(chunk_type, object_id, fields=fields)

    async def del_chunk_meta(self,
                             object_id: str,
                             chunk_type: Type):
        return await self._meta_store.del_meta(chunk_type, object_id)


class MockMetaAPI(MetaAPI):
    @classmethod
    async def create(cls, session_id: str, address: str) -> "MetaAPI":
        # create an Actor for mock
        try:
            await mo.create_actor(MetaStoreActor, 'mock', session_id,
                                  address=address,
                                  uid=MetaStoreActor.gen_uid(session_id),
                                  allocate_strategy=allocate_strategy.ProcessIndex(1))
        except mo.ActorAlreadyExist:
            # ignore if actor exists
            pass
        return await super().create(session_id=session_id, address=address)
