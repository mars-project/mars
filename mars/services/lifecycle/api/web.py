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

from typing import Callable, Dict, List

from ....utils import serialize_serializable, deserialize_serializable
from ...web import web_api, MarsServiceWebAPIHandler, MarsWebAPIClientMixin
from .core import AbstractLifecycleAPI


class LifecycleWebAPIHandler(MarsServiceWebAPIHandler):
    _root_pattern = "/api/session/(?P<session_id>[^/]+)/lifecycle"

    async def _get_oscar_lifecycle_api(self, session_id: str):
        from .oscar import LifecycleAPI

        return await self._get_api_by_key(LifecycleAPI, session_id)

    @web_api("", method="post", arg_filter={"action": "decref_tileables"})
    async def decref_tileables(self, session_id: str):
        tileable_keys = self.get_argument("tileable_keys").split(",")

        oscar_api = await self._get_oscar_lifecycle_api(session_id)
        await oscar_api.decref_tileables(tileable_keys)

    @web_api("", method="get", arg_filter={"action": "get_all_chunk_ref_counts"})
    async def get_all_chunk_ref_counts(self, session_id: str):
        oscar_api = await self._get_oscar_lifecycle_api(session_id)
        res = await oscar_api.get_all_chunk_ref_counts()
        self.write(serialize_serializable(res))


web_handlers = {LifecycleWebAPIHandler.get_root_pattern(): LifecycleWebAPIHandler}


class WebLifecycleAPI(AbstractLifecycleAPI, MarsWebAPIClientMixin):
    def __init__(
        self, session_id: str, address: str, request_rewriter: Callable = None
    ):
        self._session_id = session_id
        self._address = address.rstrip("/")
        self.request_rewriter = request_rewriter

    async def decref_tileables(self, tileable_keys: List[str]):
        path = f"{self._address}/api/session/{self._session_id}/lifecycle"
        params = dict(action="decref_tileables")
        await self._request_url(
            path=path,
            method="POST",
            params=params,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data="tileable_keys=" + ",".join(tileable_keys),
        )

    async def get_all_chunk_ref_counts(self) -> Dict[str, int]:
        params = dict(action="get_all_chunk_ref_counts")
        path = f"{self._address}/api/session/{self._session_id}/lifecycle"
        res = await self._request_url("GET", path, params=params)
        return deserialize_serializable(res.body)
