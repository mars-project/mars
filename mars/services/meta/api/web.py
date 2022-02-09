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

from typing import Callable, Dict, List, Optional

from .... import oscar as mo
from ....utils import serialize_serializable, deserialize_serializable
from ...web import web_api, MarsServiceWebAPIHandler, MarsWebAPIClientMixin
from .core import AbstractMetaAPI


class MetaWebAPIHandler(MarsServiceWebAPIHandler):
    _root_pattern = "/api/session/(?P<session_id>[^/]+)/meta"

    async def _get_oscar_meta_api(self, session_id: str):
        from .oscar import MetaAPI

        return await self._get_api_by_key(MetaAPI, session_id)

    @web_api("(?P<data_key>[^/]+)", method="get")
    async def get_chunk_meta(self, session_id: str, data_key: str):
        fields_str = self.get_argument("fields", None)
        error = self.get_argument("error", "raise")
        fields = fields_str.split(",") if fields_str else None

        oscar_api = await self._get_oscar_meta_api(session_id)
        result = await oscar_api.get_chunk_meta(data_key, fields=fields, error=error)
        self.write(serialize_serializable(result))

    @web_api("", method="post")
    async def get_chunks_meta(self, session_id: str):
        body_args = deserialize_serializable(self.request.body)
        oscar_api = await self._get_oscar_meta_api(session_id)
        get_metas = []
        for data_key, fields, error in body_args:
            get_metas.append(oscar_api.get_chunk_meta.delay(data_key, fields, error))
        results = await oscar_api.get_chunk_meta.batch(*get_metas)
        self.write(serialize_serializable(results))


web_handlers = {MetaWebAPIHandler.get_root_pattern(): MetaWebAPIHandler}


class WebMetaAPI(AbstractMetaAPI, MarsWebAPIClientMixin):
    def __init__(
        self, session_id: str, address: str, request_rewriter: Callable = None
    ):
        self._session_id = session_id
        self._address = address.rstrip("/")
        self.request_rewriter = request_rewriter

    @mo.extensible
    async def get_chunk_meta(
        self, object_id: str, fields: List[str] = None, error: str = "raise"
    ) -> Optional[Dict]:
        params = dict(error=error)
        req_addr = f"{self._address}/api/session/{self._session_id}/meta/{object_id}"
        if fields:
            params["fields"] = ",".join(fields)
        res = await self._request_url("GET", req_addr, params=params)
        return deserialize_serializable(res.body)

    @get_chunk_meta.batch
    async def get_chunks_meta(self, args_list, kwargs_list):
        get_chunk_metas = []
        for args, kwargs in zip(args_list, kwargs_list):
            object_id, fields, error = self.get_chunk_meta.bind(*args, **kwargs)
            get_chunk_metas.append([object_id, fields, error])

        req_addr = f"{self._address}/api/session/{self._session_id}/meta"
        res = await self._request_url(
            "POST", req_addr, data=serialize_serializable(get_chunk_metas)
        )
        return deserialize_serializable(res.body)
