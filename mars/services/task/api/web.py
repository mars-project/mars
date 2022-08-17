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

import asyncio
import base64
import json
from typing import Callable, List, Optional, Union

from ....core import TileableGraph, Tileable
from ....lib.tbcode import load_traceback_code, dump_traceback_code
from ....utils import serialize_serializable, deserialize_serializable
from ...web import web_api, MarsServiceWebAPIHandler, MarsWebAPIClientMixin
from ..core import TaskResult, TaskStatus
from .core import AbstractTaskAPI


def _json_serial_task_result(result: Optional[TaskResult]):
    if result is None:
        return {}
    res_json = {
        "task_id": result.task_id,
        "session_id": result.session_id,
        "stage_id": result.stage_id,
        "start_time": result.start_time,
        "end_time": result.end_time,
        "progress": result.progress,
        "status": result.status.value,
        "profiling": result.profiling,
    }
    if result.error is not None:
        res_json["error"] = base64.b64encode(
            serialize_serializable(result.error)
        ).decode()
        res_json["traceback"] = base64.b64encode(
            serialize_serializable(result.traceback)
        ).decode()
        res_json["traceback_code"] = dump_traceback_code(result.traceback)
    return res_json


def _json_deserial_task_result(d: dict) -> Optional[TaskResult]:
    if not d:
        return None
    if "error" in d:
        d["error"] = deserialize_serializable(base64.b64decode(d["error"]))
        d["traceback"] = deserialize_serializable(base64.b64decode(d["traceback"]))
        load_traceback_code(d.pop("traceback_code"))
    d["status"] = TaskStatus(d["status"])
    return TaskResult(**d)


class TaskWebAPIHandler(MarsServiceWebAPIHandler):
    _root_pattern = "/api/session/(?P<session_id>[^/]+)/task"

    async def _get_oscar_task_api(self, session_id: str):
        from .oscar import TaskAPI

        return await self._get_api_by_key(TaskAPI, session_id)

    @web_api("", method="post")
    async def submit_tileable_graph(self, session_id: str):
        body_args = (
            deserialize_serializable(self.request.body) if self.request.body else None
        )

        fuse_enabled = body_args.get("fuse")

        graph = body_args["graph"]
        extra_config = body_args.get("extra_config", None)
        if extra_config:
            extra_config = deserialize_serializable(extra_config)

        oscar_api = await self._get_oscar_task_api(session_id)
        task_id = await oscar_api.submit_tileable_graph(
            graph,
            fuse_enabled=fuse_enabled,
            extra_config=extra_config,
        )
        self.write(task_id)

    @web_api("", method="get", cache_blocking=True)
    async def get_task_results(self, session_id: str):
        progress = bool(int(self.get_argument("progress", "0")))
        oscar_api = await self._get_oscar_task_api(session_id)
        res = await oscar_api.get_task_results(progress=progress)
        self.write(json.dumps({"tasks": [_json_serial_task_result(r) for r in res]}))

    @web_api(
        "(?P<task_id>[^/]+)",
        method="get",
        arg_filter={"action": "fetch_tileables"},
        cache_blocking=True,
    )
    async def get_fetch_tileables(self, session_id: str, task_id: str):
        oscar_api = await self._get_oscar_task_api(session_id)
        res = await oscar_api.get_fetch_tileables(task_id)
        self.write(serialize_serializable(res))

    @web_api("(?P<task_id>[^/]+)", method="get", cache_blocking=True)
    async def get_task_result(self, session_id: str, task_id: str):
        oscar_api = await self._get_oscar_task_api(session_id)
        res = await oscar_api.get_task_result(task_id)
        self.write(json.dumps(_json_serial_task_result(res)))

    @web_api(
        "(?P<task_id>[^/]+)/tileable_graph",
        method="get",
        arg_filter={"action": "get_tileable_graph_as_json"},
        cache_blocking=True,
    )
    async def get_tileable_graph_as_json(self, session_id: str, task_id: str):
        oscar_api = await self._get_oscar_task_api(session_id)
        res = await oscar_api.get_tileable_graph_as_json(task_id)
        self.write(json.dumps(res))

    @web_api("(?P<task_id>[^/]+)/tileable_detail", method="get", cache_blocking=True)
    async def get_tileable_details(self, session_id: str, task_id: str):
        oscar_api = await self._get_oscar_task_api(session_id)
        res = await oscar_api.get_tileable_details(task_id)
        self.write(json.dumps(res))

    @web_api(
        "(?P<task_id>[^/]+)/(?P<tileable_id>[^/]+)/subtask",
        method="get",
        cache_blocking=True,
    )
    async def get_tileable_subtasks(
        self, session_id: str, task_id: str, tileable_id: str
    ):
        with_input_output = self.get_argument("with_input_output", "false") == "true"
        oscar_api = await self._get_oscar_task_api(session_id)
        res = await oscar_api.get_tileable_subtasks(
            task_id, tileable_id, with_input_output
        )
        self.write(json.dumps(res))

    @web_api(
        "(?P<task_id>[^/]+)",
        method="get",
        arg_filter={"action": "progress"},
        cache_blocking=True,
    )
    async def get_task_progress(self, session_id: str, task_id: str):
        oscar_api = await self._get_oscar_task_api(session_id)
        res = await oscar_api.get_task_progress(task_id)
        self.write(str(res))

    @web_api("", method="get", arg_filter={"action": "last_idle_time"})
    async def get_last_idle_time(self, session_id: str):
        oscar_api = await self._get_oscar_task_api(session_id)
        res = await oscar_api.get_last_idle_time()
        if res:
            self.write(str(res))

    @web_api("(?P<task_id>[^/]+)", method="get", arg_filter={"action": "wait"})
    async def wait_task(self, session_id: str, task_id: str):
        timeout = self.get_argument("timeout", None) or None
        timeout = float(timeout) if timeout is not None else None
        oscar_api = await self._get_oscar_task_api(session_id)
        if timeout:
            try:
                res = await asyncio.wait_for(
                    asyncio.shield(oscar_api.wait_task(task_id, timeout)),
                    timeout=timeout,
                )
                self.write(json.dumps(_json_serial_task_result(res)))
            except asyncio.TimeoutError:
                self.write(json.dumps({}))
        else:
            res = await oscar_api.wait_task(task_id, timeout)
            self.write(json.dumps(_json_serial_task_result(res)))

    @web_api("(?P<task_id>[^/]+)", method="delete")
    async def cancel_task(self, session_id: str, task_id: str):
        oscar_api = await self._get_oscar_task_api(session_id)
        await oscar_api.cancel_task(task_id)


web_handlers = {TaskWebAPIHandler.get_root_pattern(): TaskWebAPIHandler}


class WebTaskAPI(AbstractTaskAPI, MarsWebAPIClientMixin):
    def __init__(
        self, session_id: str, address: str, request_rewriter: Callable = None
    ):
        self._session_id = session_id
        self._address = address.rstrip("/")
        self.request_rewriter = request_rewriter

    async def get_task_results(self, progress: bool = False) -> List[TaskResult]:
        path = f"{self._address}/api/session/{self._session_id}/task"
        params = {"progress": int(progress)}
        res = await self._request_url("GET", path, params=params)
        return [
            _json_deserial_task_result(d)
            for d in json.loads(res.body.decode())["tasks"]
        ]

    async def submit_tileable_graph(
        self,
        graph: TileableGraph,
        fuse_enabled: bool = True,
        extra_config: dict = None,
    ) -> str:
        path = f"{self._address}/api/session/{self._session_id}/task"
        extra_config_ser = (
            serialize_serializable(extra_config) if extra_config else None
        )
        body = serialize_serializable(
            {
                "fuse": fuse_enabled,
                "graph": graph,
                "extra_config": extra_config_ser,
            }
        )
        res = await self._request_url(
            path=path,
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
            data=body,
        )
        return res.body.decode().strip()

    async def get_fetch_tileables(self, task_id: str) -> List[Tileable]:
        path = (
            f"{self._address}/api/session/{self._session_id}/task/{task_id}"
            f"?action=fetch_tileables"
        )
        res = await self._request_url("GET", path)
        return deserialize_serializable(res.body)

    async def get_task_result(self, task_id: str) -> TaskResult:
        path = f"{self._address}/api/session/{self._session_id}/task/{task_id}"
        res = await self._request_url("GET", path)
        return _json_deserial_task_result(json.loads(res.body.decode()))

    async def get_task_progress(self, task_id: str) -> float:
        path = f"{self._address}/api/session/{self._session_id}/task/{task_id}"
        params = dict(action="progress")
        res = await self._request_url("GET", path, params=params)
        return float(res.body.decode())

    async def get_last_idle_time(self) -> Union[float, None]:
        path = f"{self._address}/api/session/{self._session_id}/task"
        params = dict(action="last_idle_time")
        res = await self._request_url("GET", path, params=params)
        content = res.body.decode()
        return float(content) if content else None

    async def wait_task(self, task_id: str, timeout: float = None):
        path = f"{self._address}/api/session/{self._session_id}/task/{task_id}"
        # increase client timeout to handle network overhead during entire request
        client_timeout = timeout + 3 if timeout else 0
        params = {"action": "wait", "timeout": "" if timeout is None else str(timeout)}
        res = await self._request_url(
            "GET", path, params=params, request_timeout=client_timeout
        )
        return _json_deserial_task_result(json.loads(res.body.decode()))

    async def cancel_task(self, task_id: str):
        path = f"{self._address}/api/session/{self._session_id}/task/{task_id}"
        await self._request_url(path=path, method="DELETE")

    async def get_tileable_graph_as_json(self, task_id: str):
        path = f"{self._address}/api/session/{self._session_id}/task/{task_id}/tileable_graph"
        params = dict(action="get_tileable_graph_as_json")
        res = await self._request_url(path=path, params=params, method="GET")
        return json.loads(res.body.decode())

    async def get_tileable_details(self, task_id: str):
        path = f"{self._address}/api/session/{self._session_id}/task/{task_id}/tileable_detail"
        res = await self._request_url(path=path, method="GET")
        return json.loads(res.body.decode())

    async def get_tileable_subtasks(
        self, task_id: str, tileable_id: str, with_input_output: bool
    ):

        with_input_output = "true" if with_input_output else "false"
        path = f"{self._address}/api/session/{self._session_id}/task/{task_id}/{tileable_id}/subtask"
        params = {
            "action": "fetch_graph",
            "with_input_output": with_input_output,
        }
        res = await self._request_url(path=path, params=params, method="GET")
        return json.loads(res.body.decode())
