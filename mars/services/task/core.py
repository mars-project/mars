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

import random
from enum import Enum
from string import ascii_letters, digits
from typing import Any, Optional

from ...core import TileableGraph
from ...serialization.serializables import Serializable, StringField, \
    ReferenceField, Int32Field, BoolField, AnyField, DictField, \
    Float64Field


class TaskStatus(Enum):
    pending = 0
    running = 1
    terminated = 2


class Task(Serializable):
    task_id: str = StringField('task_id')
    task_name: str = StringField('task_name')
    session_id: str = StringField('session_id')
    parent_task_id: str = StringField('parent_task_id')
    tileable_graph: TileableGraph = ReferenceField(
        'tileable_graph', TileableGraph)
    fuse_enabled: bool = BoolField('fuse_enabled')
    rerun_time: int = Int32Field('rerun_time')
    extra_config: dict = DictField('extra_config')

    def __init__(self,
                 task_id: str = None,
                 session_id: str = None,
                 tileable_graph: TileableGraph = None,
                 task_name: str = None,
                 parent_task_id: str = None,
                 fuse_enabled: bool = True,
                 rerun_time: int = 0,
                 extra_config: dict = None):
        super().__init__(task_id=task_id, task_name=task_name,
                         session_id=session_id,
                         parent_task_id=parent_task_id,
                         tileable_graph=tileable_graph,
                         fuse_enabled=fuse_enabled,
                         rerun_time=rerun_time,
                         extra_config=extra_config)


class TaskResult(Serializable):
    task_id: str = StringField('task_id')
    session_id: str = StringField('session_id')
    stage_id: str = StringField('stage_id')
    start_time: Optional[float] = Float64Field('start_time')
    end_time: Optional[float] = Float64Field('end_time')
    progress: Optional[float] = Float64Field('progress')
    status: TaskStatus = ReferenceField('status', TaskStatus)
    error = AnyField('error')
    traceback = AnyField('traceback')

    def __init__(self,
                 task_id: str = None,
                 session_id: str = None,
                 stage_id: str = None,
                 start_time: Optional[float] = None,
                 end_time: Optional[float] = None,
                 progress: Optional[float] = None,
                 status: TaskStatus = None,
                 error: Any = None,
                 traceback: Any = None):
        super().__init__(task_id=task_id,
                         session_id=session_id,
                         stage_id=stage_id,
                         start_time=start_time,
                         end_time=end_time,
                         progress=progress,
                         status=status,
                         error=error,
                         traceback=traceback)


def new_task_id():
    return ''.join(random.choice(ascii_letters + digits) for _ in range(24))
