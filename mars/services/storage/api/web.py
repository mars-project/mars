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

from typing import Any, List

from ....lib.aio import alru_cache
from ....storage import StorageLevel
from ....utils import serialize_serializable, deserialize_serializable
from ...web import web_api, MarsServiceWebAPIHandler, MarsWebAPIClientMixin
from ..core import DataInfo
from .core import AbstractStorageAPI


class StorageWebAPIHandler(MarsServiceWebAPIHandler):
    _root_pattern = '/api/session/(?P<session_id>[^/]+)/storage'

    @alru_cache
    async def _get_cluster_api(self):
        from ...cluster import ClusterAPI
        return await ClusterAPI.create(self._supervisor_addr)

    @alru_cache
    async def _get_oscar_meta_api(self, session_id: str):
        from ...meta import MetaAPI
        cluster_api = await self._get_cluster_api()
        [address] = await cluster_api.get_supervisors_by_keys([session_id])
        return await MetaAPI.create(session_id, address)

    async def _get_storage_api_by_object_id(self, session_id: str,
                                            object_id: str):
        from .oscar import StorageAPI
        meta_api = await self._get_oscar_meta_api(session_id)
        bands = (await meta_api.get_chunk_meta(object_id, ['bands'])).get('bands')
        if not bands:
            raise KeyError
        return await StorageAPI.create(session_id, bands[0][0])

    @web_api('(?P<data_key>[^/]+)', method='get')
    async def get(self, session_id: str, data_key: str):
        oscar_api = await self._get_storage_api_by_object_id(session_id, data_key)
        result = await oscar_api.get(data_key)
        self.write(serialize_serializable(result))

    @web_api('(?P<data_key>[^/]+)', method='post', arg_filter={'action': ''})
    async def get_by_post(self, session_id: str, data_key: str):
        conditions_raw = self.get_argument('conditions', None)
        if conditions_raw is not None:
            conditions = deserialize_serializable(conditions_raw)
        else:
            conditions = None

        oscar_api = await self._get_storage_api_by_object_id(session_id, data_key)
        result = await oscar_api.get(data_key, conditions)
        self.write(serialize_serializable(result))

    @web_api('(?P<data_key>[^/]+)', method='put')
    async def put(self, session_id: str, data_key: str):
        level = self.get_argument('level', None)
        if level is not None:
            level = getattr(StorageLevel, level.upper())
        else:
            level = StorageLevel.MEMORY

        oscar_api = await self._get_storage_api_by_object_id(session_id, data_key)
        await oscar_api.put(
            data_key, deserialize_serializable(self.get_argument('data')), level)


web_handlers = {
    StorageWebAPIHandler.get_root_pattern(): StorageWebAPIHandler
}


class WebStorageAPI(AbstractStorageAPI, MarsWebAPIClientMixin):
    def __init__(self, session_id: str, address: str):
        self._session_id = session_id
        self._address = address

    async def get(self, data_key: str, conditions: List = None) -> Any:
        path = f'{self._address}/api/session/{self._session_id}/storage/{data_key}'
        if conditions is None:
            res = await self._request_url(path)
        else:
            res = await self._request_url(
                path, method='post',
                headers={'Content-Type': 'multipart/form-data'},
                body=self._make_multipart_form({
                    'conditions': serialize_serializable(conditions),
                }))
        return deserialize_serializable(res.body)

    async def put(self, data_key: str,
                  obj: object,
                  level: StorageLevel = StorageLevel.MEMORY) -> DataInfo:
        path = f'{self._address}/api/session/{self._session_id}/storage/{data_key}' \
               f'?level={level.name.lower()}'
        res = await self._request_url(
            path, method='put',
            headers={'Content-Type': 'multipart/form-data'},
            body=self._make_multipart_form({
                'data': serialize_serializable(obj),
            }))
        return deserialize_serializable(res.body)
