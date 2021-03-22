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
from typing import List, Dict, Any

from .core import TileableGraph, ChunkGraph, TaskStatus, SubTaskStatus


class AbstractTaskAPI(ABC):
    def __init__(self, session_id):
        self._session_id = session_id

    @abstractmethod
    async def submit_tileable_graph(self,
                                    graph: TileableGraph,
                                    append_to_task: str) -> str:
        """
        Submit a tileable graph

        Parameters
        ----------
        graph : TileableGraph
            Tileable graph.
        append_to_task : str
            Append to task ID.

        Returns
        -------
        task_id : str
            Task ID.
        """

    @abstractmethod
    async def get_task_status(self, task_id: str) -> TaskStatus:
        """
        Get task status.

        Parameters
        ----------
        task_id : str
            Task ID.

        Returns
        -------
        status : TaskStatus
            Task status.
        """

    @abstractmethod
    async def get_task_info(self,
                            task_id: str,
                            fields: List[str] = None) -> Dict[str, Any]:
        """
        Get task info.

        Parameters
        ----------
        task_id : str
            Task ID.
        fields : list
            Fields to filter, if not provided, get all fields.

        Returns
        -------
        info : dict
            Task information.
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
    async def cancel_task(self, task_id: str):
        """
        Cancel task.

        Parameters
        ----------
        task_id : str
            Task ID.
        """

    @abstractmethod
    async def get_subtask_count(self, task_id: str) -> int:
        """
        Get count of subtask

        Parameters
        ----------
        task_id : str
            Task ID.

        Returns
        -------
        count : int
            Count of subtasks.
        """

    @abstractmethod
    async def list_subtasks(self,
                            task_id: str,
                            offset: int = 0,
                            limit: int = None,
                            fields: List[str] = None) -> List[Dict]:
        """
        List all subtasks

        Parameters
        ----------
        task_id : str
            Task ID.
        offset : int
            Offset.
        limit : int
            Count limitation.
        fields : list
            Fields to filter, if not provided, all fields will be included.

        Returns
        -------
        infos: dict
            Subtasks information.
        """

    @abstractmethod
    async def submit_chunk_graph(self,
                                 graph: ChunkGraph,
                                 task_id: str) -> str:
        """
        Submit chunk graph to a Task.

        Parameters
        ----------
        graph : ChunkGraph
            Chunk Graph.
        task_id : str
            Task ID.

        Returns
        -------
        task_id : str
            Task ID.
        """

    @abstractmethod
    async def get_subtask_status(self,
                                 subtask_id: str) -> SubTaskStatus:
        """
        Get subtask status.

        Parameters
        ----------
        subtask_id : str
            Subtask ID.

        Returns
        -------
        status: SubTaskStatus
            Subtask status.
        """

    @abstractmethod
    async def get_subtask_info(self,
                               subtask_id: str,
                               fields: List[str] = None) -> Dict[str, Any]:
        """
        Get subtask info.

        Parameters
        ----------
        subtask_id : str
            Subtask ID.
        fields : list
            Fields to filter, if not provided, get all fields.

        Returns
        -------
        info : dict
            Subtask info.
        """

    @abstractmethod
    async def report_subtask_progress(self,
                                      subtask_id: str,
                                      op_key: str,
                                      progress: float):
        """
        Report subtask progress.

        Parameters
        ----------
        subtask_id : str
            Subtask ID.
        op_key : str
            Operand key.
        progress : float
            Progress.
        """

    @abstractmethod
    async def get_subtask_progress(self,
                                   subtask_id: str,
                                   op_key: str = None) -> float:
        """
        Get subtask progress.

        Parameters
        ----------
        subtask_id : str
            Subtask ID.
        op_key : str
            Operand key, if specified, only return operand progress.

        Returns
        -------
        progress : float
            Progress.
        """

    @abstractmethod
    async def cancel_subtask(self,
                             subtask_id: str):
        """
        Cancel subtask.

        Parameters
        ----------
        subtask_id : str
            Subtask ID.
        """
