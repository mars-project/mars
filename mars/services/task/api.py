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

from abc import ABC, abstractmethod
from typing import List, Union

from ... import oscar as mo
from ...core import Tileable
from ...lib.aio import alru_cache
from ..session import OscarSessionAPI
from .core import TileableGraph, TaskResult
from .supervisor.task_manager import TaskManagerActor
from mars.services.web.core import ServiceWebAPIBase, _transfer_request_timeout


class TaskAPI(ABC):

    @abstractmethod
    async def submit_tileable_graph(self,
                                    graph: TileableGraph,
                                    task_name: str = None,
                                    fuse_enabled: bool = True,
                                    extra_config: dict = None) -> str:
        """
        Submit a tileable graph

        Parameters
        ----------
        graph : TileableGraph
            Tileable graph.
        task_name : str
            Task name
        fuse_enabled : bool
            Enable fuse optimization
        extra_config : dict
            Extra config.

        Returns
        -------
        task_id : str
            Task ID.
        """

    @abstractmethod
    async def get_fetch_tileables(self, task_id: str) -> List[Tileable]:
        """
        Get fetch tileable for a task.

        Parameters
        ----------
        task_id : str
            Task ID.

        Returns
        -------
        fetch_tileable_list
            Fetch tileable list.
        """

    @abstractmethod
    async def wait_task(self, task_id: str, timeout: float = None):
        """
        Wait for a task to finish.

        Parameters
        ----------
        task_id : str
            Task ID
        timeout: float
            Second to timeout
        """

    @abstractmethod
    async def get_task_progress(self,
                                task_id: str) -> float:
        """
        Get task progress.

        Parameters
        ----------
        task_id : str
            Task ID.

        Returns
        -------
        progress : float
            Get task progress.
        """


class OscarTaskAPI(TaskAPI):
    def __init__(self,
                 session_id: str,
                 task_manager_ref: Union[TaskManagerActor, mo.ActorRef]):
        self._session_id = session_id
        self._task_manager_ref = task_manager_ref

    @classmethod
    @alru_cache
    async def create(cls,
                     session_id: str,
                     address: str) -> "OscarTaskAPI":
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
        return OscarTaskAPI(session_id, task_manager_ref)

    @classmethod
    async def create_session(cls,
                             session_id: str,
                             address: str) -> "OscarTaskAPI":
        """
        Creating a new task API for the session.

        Parameters
        ----------
        session_id : str
            Session ID.
        address : str
            Supervisor address.

        Returns
        -------
        task_api
            Task API
        """
        session_api = await OscarSessionAPI.create(address)
        session_address = await session_api.get_session_address(session_id)
        allocate_strategy = mo.allocate_strategy.AddressSpecified(session_address)
        task_manager_ref = await mo.create_actor(
            TaskManagerActor, session_id, address=address,
            uid=TaskManagerActor.gen_uid(session_id),
            allocate_strategy=allocate_strategy)
        return OscarTaskAPI(session_id, task_manager_ref)

    @classmethod
    async def destroy_session(cls,
                              session_id: str,
                              address: str):
        """
        Destroy a session

        Parameters
        ----------
        session_id : str
            Session ID.
        address : str
            Supervisor address
        """
        task_manager_ref = await mo.actor_ref(
            address, TaskManagerActor.gen_uid(session_id))
        return await mo.destroy_actor(task_manager_ref)

    async def submit_tileable_graph(self,
                                    graph: TileableGraph,
                                    task_name: str = None,
                                    fuse_enabled: bool = True,
                                    extra_config: dict = None) -> str:
        return await self._task_manager_ref.submit_tileable_graph(
            graph, task_name, fuse_enabled=fuse_enabled,
            extra_config=extra_config)

    async def wait_task(self, task_id: str, timeout: float = None):
        return await self._task_manager_ref.wait_task(
            task_id, timeout=timeout)

    async def get_task_result(self, task_id: str) -> TaskResult:
        """
        Get task status.

        Parameters
        ----------
        task_id : str
            Task ID.

        Returns
        -------
        result : TaskResult
            Task result.
        """
        return await self._task_manager_ref.get_task_result(task_id)

    async def get_task_progress(self,
                                task_id: str) -> float:
        return await self._task_manager_ref.get_task_progress(task_id)

    async def cancel_task(self, task_id: str):
        """
        Cancel task.

        Parameters
        ----------
        task_id : str
            Task ID.
        """
        return await self._task_manager_ref.cancel_task(task_id)

    async def get_fetch_tileables(self, task_id: str) -> List[Tileable]:
        return await self._task_manager_ref.get_task_result_tileables(task_id)

    async def get_last_idle_time(self) -> Union[float, None]:
        """
        Get last idle time from task manager.

        Returns
        -------
        last_idle_time: float
            The last idle time if the task manager is idle else None.
        """
        return await self._task_manager_ref.get_last_idle_time()


class WebTaskAPI(ServiceWebAPIBase, TaskAPI):
    _service_name = 'task'

    @classmethod
    @alru_cache
    async def create(cls, web_address: str, session_id: str, address: str, **kwargs):
        return WebTaskAPI(web_address, 'create', session_id, address, **kwargs)

    async def submit_tileable_graph(self,
                                    graph: TileableGraph,
                                    task_name: str = None,
                                    fuse_enabled: bool = True,
                                    extra_config: dict = None) -> str:
        return await self._call_method(dict(request_timeout=_transfer_request_timeout),
                                       'submit_tileable_graph', graph, task_name, fuse_enabled, extra_config)

    async def get_fetch_tileables(self, task_id: str) -> List[Tileable]:
        return await self._call_method({}, 'get_fetch_tileables', task_id)

    async def wait_task(self, task_id: str, timeout: float = None):
        return await self._call_method({}, 'wait_task', task_id, timeout)

    async def get_task_progress(self, task_id: str) -> float:
        return await self._call_method({}, 'get_task_progress', task_id)
