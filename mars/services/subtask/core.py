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
from typing import Iterable, List, Optional, Set, Tuple

from ...core import ChunkGraph, DAG, ChunkData
from ...resource import Resource
from ...serialization.serializables.field_type import TupleType
from ...serialization.serializables import (
    Serializable,
    StringField,
    ReferenceField,
    Int32Field,
    Int64Field,
    Float64Field,
    BoolField,
    AnyField,
    DictField,
    ListField,
    TupleField,
    FieldTypes,
)
from ...typing import BandType, ChunkType


class SubtaskStatus(Enum):
    pending = 0
    running = 1
    succeeded = 2
    errored = 3
    cancelled = 4

    @property
    def is_done(self) -> bool:
        return self in (
            SubtaskStatus.succeeded,
            SubtaskStatus.errored,
            SubtaskStatus.cancelled,
        )


class Subtask(Serializable):
    __slots__ = ("_repr", "_pure_depend_keys")

    subtask_id: str = StringField("subtask_id")
    subtask_name: str = StringField("subtask_name")
    session_id: str = StringField("session_id")
    task_id: str = StringField("task_id")
    chunk_graph: ChunkGraph = ReferenceField("chunk_graph", ChunkGraph)
    expect_bands: List[BandType] = ListField(
        "expect_bands", TupleType(FieldTypes.string, FieldTypes.string)
    )
    virtual: bool = BoolField("virtual")
    retryable: bool = BoolField("retryable")
    priority: Tuple[int, int] = TupleField("priority", FieldTypes.int32)
    rerun_time: int = Int32Field("rerun_time")
    extra_config: dict = DictField("extra_config")
    stage_id: str = StringField("stage_id")
    # chunks that need meta updated
    update_meta_chunks: List[ChunkType] = ListField(
        "update_meta_chunks", FieldTypes.reference(ChunkData)
    )
    # A unique and deterministic key for subtask compute logic. See logic_key in operator.py.
    logic_key: str = StringField("logic_key")
    # index for subtask with same compute logic.
    logic_index: int = Int32Field("logic_index")
    # parallelism for subtask with same compute logic.
    logic_parallelism: int = Int32Field("logic_parallelism")
    # subtask can only run in specified bands in `expect_bands`
    bands_specified: bool = BoolField("bands_specified")
    required_resource: Resource = AnyField("required_resource", Resource)
    # The count of result chunks that are the stage's results.
    stage_n_outputs: int = Int32Field("stage_n_outputs")

    def __init__(
        self,
        subtask_id: str = None,
        session_id: str = None,
        task_id: str = None,
        chunk_graph: ChunkGraph = None,
        subtask_name: str = None,
        expect_bands: List[BandType] = None,
        priority: Tuple[int, int] = None,
        virtual: bool = False,
        retryable: bool = True,
        rerun_time: int = 0,
        extra_config: dict = None,
        stage_id: str = None,
        update_meta_chunks: List[ChunkType] = None,
        logic_key: str = None,
        logic_index: int = None,
        logic_parallelism: int = None,
        bands_specified: bool = False,
        required_resource: Resource = None,
        stage_n_outputs: int = 0,
    ):
        super().__init__(
            subtask_id=subtask_id,
            subtask_name=subtask_name,
            session_id=session_id,
            task_id=task_id,
            chunk_graph=chunk_graph,
            expect_bands=expect_bands,
            priority=priority,
            virtual=virtual,
            retryable=retryable,
            rerun_time=rerun_time,
            extra_config=extra_config,
            stage_id=stage_id,
            update_meta_chunks=update_meta_chunks,
            logic_key=logic_key,
            logic_index=logic_index,
            logic_parallelism=logic_parallelism,
            bands_specified=bands_specified,
            required_resource=required_resource,
            stage_n_outputs=stage_n_outputs,
        )
        self._pure_depend_keys = None
        self._repr = None

    def __on_deserialize__(self):
        super(Subtask, self).__on_deserialize__()
        self._pure_depend_keys = None
        self._repr = None

    @property
    def expect_band(self):
        if self.expect_bands:
            return self.expect_bands[0]

    @property
    def pure_depend_keys(self) -> Set[str]:
        if self._pure_depend_keys is not None:
            return self._pure_depend_keys
        pure_dep_keys = set()
        for n in self.chunk_graph:
            pure_dep_keys.update(
                inp.key
                for inp, pure_dep in zip(n.inputs, n.op.pure_depends)
                if pure_dep
            )
        self._pure_depend_keys = pure_dep_keys
        return pure_dep_keys

    def __repr__(self):
        if self._repr is not None:
            return self._repr

        if self.chunk_graph:
            result_chunk_repr = " ".join(
                [
                    f"{type(chunk.op).__name__}({chunk.key})"
                    for chunk in self.chunk_graph.result_chunks
                ]
            )
        else:  # pragma: no cover
            result_chunk_repr = None
        self._repr = f"<Subtask id={self.subtask_id} results=[{result_chunk_repr}]>"
        return self._repr


class SubtaskResult(Serializable):
    subtask_id: str = StringField("subtask_id")
    session_id: str = StringField("session_id")
    task_id: str = StringField("task_id")
    stage_id: str = StringField("stage_id")
    status: SubtaskStatus = ReferenceField("status", SubtaskStatus)
    progress: float = Float64Field("progress", default=0.0)
    data_size: int = Int64Field("data_size", default=None)
    bands: List[BandType] = ListField("band", FieldTypes.tuple, default=None)
    error = AnyField("error", default=None)
    traceback = AnyField("traceback", default=None)
    # The following is the execution information of the subtask
    execution_start_time: float = Float64Field("execution_start_time")
    execution_end_time: float = Float64Field("execution_end_time")

    def update(self, result: Optional["SubtaskResult"]):
        if result and result.bands:
            bands = self.bands or []
            self.bands = sorted(set(bands + result.bands))
            self.execution_start_time = result.execution_start_time
            if hasattr(result, "execution_end_time"):
                self.execution_end_time = result.execution_end_time
        return self


class SubtaskGraph(DAG, Iterable[Subtask]):
    """
    Subtask graph.
    """

    def __init__(self):
        super().__init__()
        self._proxy_subtasks = []

    @classmethod
    def _extract_operands(cls, node: Subtask):
        from ...core.operand import Fetch, FetchShuffle

        for node in node.chunk_graph:
            if isinstance(node.op, (Fetch, FetchShuffle)):
                continue
            yield node.op

    def add_shuffle_proxy_subtask(self, proxy_subtask):
        self._proxy_subtasks.append(proxy_subtask)

    def num_shuffles(self) -> int:
        return len(self._proxy_subtasks)

    def get_shuffle_proxy_subtasks(self):
        return self._proxy_subtasks
