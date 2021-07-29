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

from abc import ABC, abstractmethod
from typing import List, Union

from ....core import Tileable
from ..core import TileableGraph, TaskResult


class AbstractTaskAPI(ABC):
    @abstractmethod
    async def get_task_results(self, progress: bool = False) -> List[TaskResult]:
        """
        Get results of all tasks in the session

        Parameters
        ----------
        progress : bool
            If True, will return task progress

        Returns
        -------
        task_results: List[TaskResult]
            List of task results
        """

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
    async def cancel_task(self, task_id: str):
        """
        Cancel task.

        Parameters
        ----------
        task_id : str
            Task ID.
        """

    @abstractmethod
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
    async def get_last_idle_time(self) -> Union[float, None]:
        """
        Get last idle time from task manager.

        Returns
        -------
        last_idle_time: float
            The last idle time if the task manager is idle else None.
        """
