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

from enum import Enum
from typing import List

from ...core import Tileable, Chunk
from ...graph import DAG


class EntityGraph(DAG):
    __slots__ = ()


class TileableGraph(EntityGraph):
    __slots__ = 'result_tileables',

    def __init__(self, result_tileables: List[Tileable]):
        super().__init__()
        self.result_tileables = result_tileables


class ChunkGraph(EntityGraph):
    __slots__ = 'result_chunks',

    def __init__(self, result_chunks: List[Chunk]):
        super().__init__()
        self.result_chunks = result_chunks


class TaskStatus(Enum):
    pending = 0
    running = 1
    terminated = 2


class SubTaskStatus(Enum):
    pending = 0
    running = 1
    terminated = 2


class Task:
    __slots__ = 'task_id', 'session_id', 'tileable_graph', \
                'task_name', 'status', 'rerun_time'

    task_id: str
    task_name: str
    session_id: str
    tileable_graph: TileableGraph
    status: TaskStatus
    rerun_time: int

    def __init__(self,
                 task_id: str,
                 session_id: str,
                 tileable_graph: TileableGraph,
                 task_name: str = None):
        self.task_id = task_id
        self.task_name = task_name
        self.session_id = session_id
        self.tileable_graph = tileable_graph

        self.status = TaskStatus.pending
        self.rerun_time = 0


class SubTask:
    __slots__ = 'subtask_id', 'session_id', 'chunk_graph', \
                'subtask_name', 'status', 'rerun_time'

    subtask_id: str
    subtask_name: str
    session_id: str
    chunk_graph: ChunkGraph
    status: TaskStatus
    rerun_time: int

    def __init__(self,
                 subtask_id: str,
                 session_id: str,
                 chunk_graph: ChunkGraph,
                 subtask_name: str):
        self.subtask_id = subtask_id
        self.subtask_name = subtask_name
        self.session_id = session_id
        self.chunk_graph = chunk_graph

        self.status = TaskStatus.pending
        self.rerun_time = 0
