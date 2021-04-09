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

import random
from enum import Enum
from string import ascii_letters, digits
from typing import Any, Tuple

from ...core import TileableGraph, ChunkGraph, DAG
from ...serialization.serializables import Serializable, StringField, \
    ReferenceField, Int32Field, Int64Field, BoolField, AnyField, TupleField


class TaskStatus(Enum):
    pending = 0
    running = 1
    terminated = 2


class SubTaskStatus(Enum):
    pending = 0
    running = 1
    succeeded = 2
    errored = 3
    cancelled = 4

    @property
    def is_done(self) -> bool:
        return self in (SubTaskStatus.succeeded,
                        SubTaskStatus.errored,
                        SubTaskStatus.cancelled)


class Task(Serializable):
    task_id: str = StringField('task_id')
    task_name: str = StringField('task_name')
    session_id: str = StringField('session_id')
    tileable_graph: TileableGraph = ReferenceField(
        'tileable_graph', TileableGraph)
    fuse_enabled: bool = BoolField('fuse_enabled')
    rerun_time: int = Int32Field('rerun_time')

    def __init__(self,
                 task_id: str = None,
                 session_id: str = None,
                 tileable_graph: TileableGraph = None,
                 task_name: str = None,
                 fuse_enabled: bool = True,
                 rerun_time: int = 0):
        super().__init__(task_id=task_id, task_name=task_name,
                         session_id=session_id,
                         tileable_graph=tileable_graph,
                         fuse_enabled=fuse_enabled,
                         rerun_time=rerun_time)


class TaskResult(Serializable):
    task_id: str = StringField('task_id')
    session_id: str = StringField('session_id')
    status: TaskStatus = ReferenceField('status', TaskStatus)
    error = AnyField('error')
    traceback = AnyField('traceback')

    def __init__(self,
                 task_id: str = None,
                 session_id: str = None,
                 status: TaskStatus = None,
                 error: Any = None,
                 traceback: Any = None):
        super().__init__(task_id=task_id,
                         session_id=session_id,
                         status=status,
                         error=error,
                         traceback=traceback)


class Subtask(Serializable):
    subtask_id: str = StringField('subtask_id')
    subtask_name: str = StringField('subtask_name')
    session_id: str = StringField('session_id')
    task_id: str = StringField('task_id')
    chunk_graph: ChunkGraph = ReferenceField('chunk_graph', ChunkGraph)
    expect_band: Tuple[str, str] = TupleField('expect_band')
    priority: int = Int32Field('priority')
    rerun_time: int = Int32Field('rerun_time')

    def __init__(self,
                 subtask_id: str = None,
                 session_id: str = None,
                 task_id: str = None,
                 chunk_graph: ChunkGraph = None,
                 subtask_name: str = None,
                 expect_band: Tuple[str, str] = None,
                 priority: int = None,
                 rerun_time: int = 0):
        super().__init__(subtask_id=subtask_id,
                         subtask_name=subtask_name,
                         session_id=session_id,
                         task_id=task_id,
                         chunk_graph=chunk_graph,
                         expect_band=expect_band,
                         priority=priority,
                         rerun_time=rerun_time)


class SubtaskResult(Serializable):
    subtask_id: str = StringField('subtask_id')
    session_id: str = StringField('session_id')
    task_id: str = StringField('task_id')
    status: SubTaskStatus = ReferenceField('status', SubTaskStatus)
    data_size: int = Int64Field('data_size', default=None)
    error = AnyField('error', default=None)
    traceback = AnyField('traceback', default=None)


class SubtaskGraph(DAG):
    """
    Subtask graph.
    """


def new_task_id():
    return ''.join(random.choice(ascii_letters + digits) for _ in range(24))
