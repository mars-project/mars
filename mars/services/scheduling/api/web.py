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

import json
from typing import Callable, List, Optional

from ....lib.aio import alru_cache
from ...web import web_api, MarsServiceWebAPIHandler, MarsWebAPIClientMixin
from ..core import SubtaskScheduleSummary
from .core import AbstractSchedulingAPI


class SchedulingWebAPIHandler(MarsServiceWebAPIHandler):
    _root_pattern = "/api/session/(?P<session_id>[^/]+)/scheduling"

    @alru_cache(cache_exceptions=False)
    async def _get_cluster_api(self):
        from ...cluster import ClusterAPI

        return await ClusterAPI.create(self._supervisor_addr)

    @alru_cache(cache_exceptions=False)
    async def _get_oscar_scheduling_api(self, session_id: str):
        from ..api import SchedulingAPI

        cluster_api = await self._get_cluster_api()
        [address] = await cluster_api.get_supervisors_by_keys([session_id])
        return await SchedulingAPI.create(session_id, address)

    @web_api("subtasks", method="get")
    async def get_subtask_schedule_summaries(self, session_id: str):
        oscar_api = await self._get_oscar_scheduling_api(session_id)
        task_id = self.get_argument("task_id", None) or None

        result = await oscar_api.get_subtask_schedule_summaries(task_id)
        self.write(
            json.dumps(
                {
                    summary.subtask_id: {
                        "task_id": summary.task_id,
                        "subtask_id": summary.subtask_id,
                        "bands": [
                            {
                                "endpoint": band[0],
                                "band_name": band[1],
                            }
                            for band in summary.bands
                        ],
                        "num_reschedules": summary.num_reschedules,
                        "is_finished": summary.is_finished,
                        "is_cancelled": summary.is_cancelled,
                    }
                    for summary in result
                }
            )
        )


web_handlers = {SchedulingWebAPIHandler.get_root_pattern(): SchedulingWebAPIHandler}


class WebSchedulingAPI(AbstractSchedulingAPI, MarsWebAPIClientMixin):
    def __init__(
        self, session_id: str, address: str, request_rewriter: Callable = None
    ):
        self._session_id = session_id
        self._address = address.rstrip("/")
        self.request_rewriter = request_rewriter

    async def get_subtask_schedule_summaries(
        self, task_id: Optional[str] = None
    ) -> List[SubtaskScheduleSummary]:
        task_id = task_id or ""
        path = (
            f"{self._address}/api/session/{self._session_id}/scheduling/subtasks"
            f"?task_id={task_id}"
        )

        res = await self._request_url("GET", path)
        res_json = json.loads(res.body)

        return [
            SubtaskScheduleSummary(
                task_id=summary_json["task_id"],
                subtask_id=summary_json["subtask_id"],
                bands=[
                    (band_json["endpoint"], band_json["band_name"])
                    for band_json in summary_json["bands"]
                ],
                num_reschedules=summary_json["num_reschedules"],
                is_finished=summary_json["is_finished"],
                is_cancelled=summary_json["is_cancelled"],
            )
            for summary_json in res_json.values()
        ]
