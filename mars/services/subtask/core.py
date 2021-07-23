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

from enum import Enum
from typing import Iterable, List, Tuple

from ...core import ChunkGraph, DAG
from ...serialization.serializables import Serializable, StringField, \
    ReferenceField, Int32Field, Int64Field, Float64Field, \
    BoolField, AnyField, DictField, ListField, TupleField, FieldTypes
from ...typing import BandType


class SubtaskStatus(Enum):
    pending = 0
    running = 1
    succeeded = 2
    errored = 3
    cancelled = 4

    @property
    def is_done(self) -> bool:
        return self in (SubtaskStatus.succeeded,
                        SubtaskStatus.errored,
                        SubtaskStatus.cancelled)


class Subtask(Serializable):
    subtask_id: str = StringField('subtask_id')
    subtask_name: str = StringField('subtask_name')
    session_id: str = StringField('session_id')
    task_id: str = StringField('task_id')
    chunk_graph: ChunkGraph = ReferenceField('chunk_graph', ChunkGraph)
    expect_bands: List[BandType] = ListField('expect_bands', FieldTypes.tuple)
    virtual: bool = BoolField('virtual')
    priority: Tuple[int, int] = TupleField('priority', FieldTypes.int32)
    rerun_time: int = Int32Field('rerun_time')
    extra_config: dict = DictField('extra_config')

    def __init__(self,
                 subtask_id: str = None,
                 session_id: str = None,
                 task_id: str = None,
                 chunk_graph: ChunkGraph = None,
                 subtask_name: str = None,
                 expect_bands: List[BandType] = None,
                 priority: Tuple[int, int] = None,
                 virtual: bool = False,
                 rerun_time: int = 0,
                 extra_config: dict = None):
        super().__init__(subtask_id=subtask_id,
                         subtask_name=subtask_name,
                         session_id=session_id,
                         task_id=task_id,
                         chunk_graph=chunk_graph,
                         expect_bands=expect_bands,
                         priority=priority,
                         virtual=virtual,
                         rerun_time=rerun_time,
                         extra_config=extra_config)

    @property
    def expect_band(self):
        if self.expect_bands:
            return self.expect_bands[0]


class SubtaskResult(Serializable):
    subtask_id: str = StringField('subtask_id')
    session_id: str = StringField('session_id')
    task_id: str = StringField('task_id')
    status: SubtaskStatus = ReferenceField('status', SubtaskStatus)
    progress: float = Float64Field('progress', default=0.0)
    data_size: int = Int64Field('data_size', default=None)
    error = AnyField('error', default=None)
    traceback = AnyField('traceback', default=None)


class SubtaskGraph(DAG, Iterable[Subtask]):
    """
    Subtask graph.
    """
    @classmethod
    def _extract_operands(cls, node: Subtask):
        from ...core.operand import Fetch, FetchShuffle
        for node in node.chunk_graph:
            if isinstance(node.op, (Fetch, FetchShuffle)):
                continue
            yield node.op
