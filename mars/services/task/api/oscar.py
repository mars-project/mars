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

from typing import List, Union

from .... import oscar as mo
from ....core import Tileable
from ....lib.aio import alru_cache
from ...subtask import SubtaskResult
from ..core import TileableGraph, TaskResult
from ..supervisor.manager import TaskManagerActor
from .core import AbstractTaskAPI


class TaskAPI(AbstractTaskAPI):
    def __init__(self,
                 session_id: str,
                 task_manager_ref: Union[TaskManagerActor, mo.ActorRef]):
        self._session_id = session_id
        self._task_manager_ref = task_manager_ref

    @classmethod
    @alru_cache(cache_exceptions=False)
    async def create(cls,
                     session_id: str,
                     address: str) -> "TaskAPI":
        """
        Create Task API.

        Parameters
        ----------
        session_id : str
            Session ID
        address : str
            Supervisor address.

        Returns
        -------
        task_api
            Task API.
        """
        task_manager_ref = await mo.actor_ref(
            address, TaskManagerActor.gen_uid(session_id))
        return TaskAPI(session_id, task_manager_ref)

    async def get_task_results(self, progress: bool = False) -> List[TaskResult]:
        return await self._task_manager_ref.get_task_results(progress)

    async def submit_tileable_graph(self,
                                    graph: TileableGraph,
                                    task_name: str = None,
                                    fuse_enabled: bool = True,
                                    extra_config: dict = None) -> str:
        try:
            return await self._task_manager_ref.submit_tileable_graph(
                graph, task_name, fuse_enabled=fuse_enabled,
                extra_config=extra_config)
        except mo.ActorNotExist:
            raise RuntimeError('Session closed already')

    async def get_tileable_graph_as_json(self, task_id: str):
        return await self._task_manager_ref.get_tileable_graph_dict_by_task_id(
            task_id
        )

    async def get_tileable_details(self, task_id: str):
        return await self._task_manager_ref.get_tileable_details(task_id)

    async def wait_task(self, task_id: str, timeout: float = None):
        return await self._task_manager_ref.wait_task(
            task_id, timeout=timeout)

    async def get_task_result(self, task_id: str) -> TaskResult:
        return await self._task_manager_ref.get_task_result(task_id)

    async def get_task_progress(self,
                                task_id: str) -> float:
        return await self._task_manager_ref.get_task_progress(task_id)

    async def cancel_task(self, task_id: str):
        return await self._task_manager_ref.cancel_task(task_id)

    async def get_fetch_tileables(self, task_id: str) -> List[Tileable]:
        return await self._task_manager_ref.get_task_result_tileables(task_id)

    async def set_subtask_result(self, subtask_result: SubtaskResult):
        return await self._task_manager_ref.set_subtask_result.tell(subtask_result)

    async def get_last_idle_time(self) -> Union[float, None]:
        return await self._task_manager_ref.get_last_idle_time()
