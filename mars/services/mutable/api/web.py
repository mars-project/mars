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

from typing import Union, Callable

import numpy as np

from ....lib.aio import alru_cache
from ....utils import deserialize_serializable, serialize_serializable
from ...web import web_api, MarsServiceWebAPIHandler, MarsWebAPIClientMixin
from .core import AbstractMutableAPI


class MutableWebAPIHandler(MarsServiceWebAPIHandler):
    _root_pattern = "/api/session/(?P<session_id>[^/]+)/mutable"

    @alru_cache(cache_exceptions=False)
    async def _get_cluster_api(self):
        from ...cluster import ClusterAPI

        return await ClusterAPI.create(self._supervisor_addr)

    @alru_cache(cache_exceptions=False)
    async def _get_oscar_mutable_api(self, session_id: str):
        from .oscar import MutableAPI

        cluster_api = await self._get_cluster_api()
        [address] = await cluster_api.get_supervisors_by_keys([session_id])
        return await MutableAPI.create(session_id, address)

    @web_api("", method="post")
    async def create_mutable_tensor(self, session_id: str):
        body_args = (
            deserialize_serializable(self.request.body) if self.request.body else None
        )
        shape = body_args.get("shape")
        dtype = body_args.get("dtype")
        name = body_args.get("name")
        default_value = body_args.get("default_value")
        chunk_size = body_args.get("chunk_size")

        oscar_api = await self._get_oscar_mutable_api(session_id)
        res = await oscar_api.create_mutable_tensor(
            shape, dtype, name, default_value, chunk_size
        )
        self.write(serialize_serializable(res))

    @web_api("(?P<name>[^/]+)", method="get")
    async def get_mutable_tensor(self, session_id: str, name: str):
        oscar_api = await self._get_oscar_mutable_api(session_id)
        res = await oscar_api.get_mutable_tensor(name)
        self.write(serialize_serializable(res))

    @web_api("(?P<name>[^/]+)/seal", method="post")
    async def seal_mutable_tensor(self, session_id: str, name: str):  # pragma: no cover
        body_args = (
            deserialize_serializable(self.request.body) if self.request.body else None
        )
        timestamp = body_args.get("timestamp")

        oscar_api = await self._get_oscar_mutable_api(session_id)
        res = await oscar_api.seal_mutable_tensor(name, timestamp)
        self.write(serialize_serializable(res))

    @web_api("(?P<name>[^/]+)/read", method="post")
    async def read_mutable(self, session_id: str, name: str):  # pragma: no cover
        body_args = (
            deserialize_serializable(self.request.body) if self.request.body else None
        )
        index = body_args.get("index")
        timestamp = body_args.get("timestamp")

        oscar_api = await self._get_oscar_mutable_api(session_id)
        res = await oscar_api.read(name, index, timestamp)
        self.write(serialize_serializable(res))

    @web_api("(?P<name>[^/]+)/write", method="post")
    async def write_mutable(self, session_id: str, name: str):  # pragma: no cover
        body_args = (
            deserialize_serializable(self.request.body) if self.request.body else None
        )
        index = body_args.get("index")
        value = body_args.get("value")
        timestamp = body_args.get("timestamp")

        oscar_api = await self._get_oscar_mutable_api(session_id)
        res = await oscar_api.write(name, index, value, timestamp)
        self.write(serialize_serializable(res))


web_handlers = {
    MutableWebAPIHandler.get_root_pattern(): MutableWebAPIHandler,
}


class WebMutableAPI(AbstractMutableAPI, MarsWebAPIClientMixin):
    def __init__(
        self, session_id: str, address: str, request_rewriter: Callable = None
    ):
        self._session_id = session_id
        self._address = address.rstrip("/")
        self.request_rewriter = request_rewriter

    async def create_mutable_tensor(
        self,
        shape: tuple,
        dtype: Union[np.dtype, str],
        name: str = None,
        default_value: Union[int, float] = 0,
        chunk_size: Union[tuple, int] = None,
    ):
        path = f"{self._address}/api/session/{self._session_id}/mutable"
        params = dict(
            shape=shape,
            dtype=dtype,
            name=name,
            default_value=default_value,
            chunk_size=chunk_size,
        )
        body = serialize_serializable(params)
        res = await self._request_url(
            path=path,
            method="POST",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        return deserialize_serializable(res.body)

    async def get_mutable_tensor(self, name: str):
        path = f"{self._address}/api/session/{self._session_id}/mutable/{name}"
        res = await self._request_url(
            path=path,
            method="GET",
            headers={"Content-Type": "application/octet-stream"},
        )
        return deserialize_serializable(res.body)

    async def seal_mutable_tensor(self, name: str, timestamp=None):
        path = f"{self._address}/api/session/{self._session_id}/mutable/{name}/seal"
        params = dict(timestamp=timestamp)
        body = serialize_serializable(params)
        res = await self._request_url(
            path=path,
            method="POST",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            request_timeout=3600,
        )
        return deserialize_serializable(res.body)

    async def read(self, name: str, index, timestamp=None):
        path = f"{self._address}/api/session/{self._session_id}/mutable/{name}/read"
        params = dict(index=index, timestamp=timestamp)
        body = serialize_serializable(params)
        res = await self._request_url(
            path=path,
            method="POST",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            request_timeout=3600,
        )
        return deserialize_serializable(res.body)

    async def write(self, name: str, index, value, timestamp=None):
        path = f"{self._address}/api/session/{self._session_id}/mutable/{name}/write"
        params = dict(index=index, value=value, timestamp=timestamp)
        body = serialize_serializable(params)
        res = await self._request_url(
            path=path,
            method="POST",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            request_timeout=3600,
        )
        return deserialize_serializable(res.body)
