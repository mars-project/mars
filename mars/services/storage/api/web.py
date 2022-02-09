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

from collections import defaultdict
from typing import Any, Callable, List

from .... import oscar as mo
from ....storage import StorageLevel
from ....utils import serialize_serializable, deserialize_serializable
from ...web import web_api, MarsServiceWebAPIHandler, MarsWebAPIClientMixin
from ..core import DataInfo
from .core import AbstractStorageAPI


class StorageWebAPIHandler(MarsServiceWebAPIHandler):
    _root_pattern = "/api/session/(?P<session_id>[^/]+)/storage"

    async def _get_oscar_meta_api(self, session_id: str):
        from ...meta import MetaAPI

        return await self._get_api_by_key(MetaAPI, session_id)

    async def _get_storage_api_by_object_id(self, session_id: str, object_id: str):
        from .oscar import StorageAPI

        meta_api = await self._get_oscar_meta_api(session_id)
        bands = (await meta_api.get_chunk_meta(object_id, ["bands"])).get("bands")
        if not bands:
            raise KeyError
        return await StorageAPI.create(session_id, bands[0][0], bands[0][1])

    @web_api("(?P<data_key>[^/]+)", method="get")
    async def get_data(self, session_id: str, data_key: str):
        oscar_api = await self._get_storage_api_by_object_id(session_id, data_key)
        result = await oscar_api.get(data_key)
        self.write(serialize_serializable(result))

    @web_api("batch/get", method="post")
    async def get_batch_data(self, session_id: str):
        body_args = deserialize_serializable(self.request.body)
        storage_api_to_gets = defaultdict(list)
        storage_api_to_idx = defaultdict(list)
        results = [None] * len(body_args)
        for i, (data_key, conditions, error) in enumerate(body_args):
            oscar_api = await self._get_storage_api_by_object_id(session_id, data_key)
            storage_api_to_idx[oscar_api].append(i)
            storage_api_to_gets[oscar_api].append(
                oscar_api.get.delay(data_key, conditions=conditions, error=error)
            )
        for api, fetches in storage_api_to_gets.items():
            data_list = await api.get.batch(*fetches)
            for idx, data in zip(storage_api_to_idx[api], data_list):
                results[idx] = data
        res_data = serialize_serializable(results)
        self.write(res_data)

    @web_api("(?P<data_key>[^/]+)", method="post")
    async def get_data_by_post(self, session_id: str, data_key: str):
        body_args = (
            deserialize_serializable(self.request.body) if self.request.body else None
        )
        conditions = body_args.get("conditions")

        oscar_api = await self._get_storage_api_by_object_id(session_id, data_key)
        result = await oscar_api.get(data_key, conditions)
        self.write(serialize_serializable(result))

    @web_api("(?P<data_key>[^/]+)", method="put")
    async def put_data(self, session_id: str, data_key: str):
        level = self.get_argument("level", None) or "MEMORY"
        level = getattr(StorageLevel, level.upper())

        oscar_api = await self._get_storage_api_by_object_id(session_id, data_key)
        res = await oscar_api.put(
            data_key, deserialize_serializable(self.request.body), level
        )
        self.write(serialize_serializable(res))

    @web_api("(?P<data_key>[^/]+)", method="get", arg_filter={"action": "get_infos"})
    async def get_infos(self, session_id: str, data_key: str):
        oscar_api = await self._get_storage_api_by_object_id(session_id, data_key)
        res = await oscar_api.get_infos(data_key)
        self.write(serialize_serializable(res))


web_handlers = {StorageWebAPIHandler.get_root_pattern(): StorageWebAPIHandler}


class WebStorageAPI(AbstractStorageAPI, MarsWebAPIClientMixin):
    def __init__(
        self,
        session_id: str,
        address: str,
        band_name: str,
        request_rewriter: Callable = None,
    ):
        self._session_id = session_id
        self._address = address.rstrip("/")
        self._band_name = band_name
        self.request_rewriter = request_rewriter

    @mo.extensible
    async def get(
        self, data_key: str, conditions: List = None, error: str = "raise"
    ) -> Any:
        path = f"{self._address}/api/session/{self._session_id}/storage/{data_key}"
        params = dict(error=error)
        if conditions is not None:
            params["conditions"] = conditions
        body = serialize_serializable(params)
        res = await self._request_url(
            path=path,
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
            data=body,
        )
        return deserialize_serializable(res.body)

    @get.batch
    async def get_batch(self, args_list, kwargs_list):
        get_chunks = []
        for args, kwargs in zip(args_list, kwargs_list):
            data_key, conditions, error = self.get.bind(*args, **kwargs)
            get_chunks.append([data_key, conditions, error])

        path = f"{self._address}/api/session/{self._session_id}/storage/batch/get"
        res = await self._request_url(
            path=path,
            method="POST",
            data=serialize_serializable(get_chunks),
        )
        return deserialize_serializable(res.body)

    @mo.extensible
    async def put(
        self, data_key: str, obj: object, level: StorageLevel = StorageLevel.MEMORY
    ) -> DataInfo:
        params = dict(level=level.name.lower())
        path = f"{self._address}/api/session/{self._session_id}/storage/{data_key}"
        res = await self._request_url(
            path=path,
            method="PUT",
            params=params,
            headers={"Content-Type": "application/octet-stream"},
            data=serialize_serializable(obj),
        )
        return deserialize_serializable(res.body)

    @mo.extensible
    async def get_infos(self, data_key: str) -> List[DataInfo]:
        path = f"{self._address}/api/session/{self._session_id}/storage/{data_key}"
        params = dict(action="get_infos")
        res = await self._request_url(
            path=path,
            method="GET",
            headers={"Content-Type": "application/octet-stream"},
            params=params,
        )
        return deserialize_serializable(res.body)
