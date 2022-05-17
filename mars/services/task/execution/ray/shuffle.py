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

import numpy as np
from typing import List, Iterable

from mars.core.operand import ShuffleProxy, MapReduceOperand
from ....subtask import Subtask
from .....utils import lazy_import

ray = lazy_import("ray")


class ShuffleManager:
    def __init__(self, subtask_graph):
        self.subtask_graph = subtask_graph
        self._proxy_subtasks = []
        for subtask in subtask_graph:
            chunk = subtask.chunk_graph.results[0]
            if isinstance(chunk.op, ShuffleProxy):
                self._proxy_subtasks.append(subtask)
        self.num_shuffles = len(self._proxy_subtasks)
        self.mapper_output_refs = []
        self.mapper_indices = {}
        self.reducer_indices = {}
        for shuffle_index, proxy_subtask in enumerate(self._proxy_subtasks):
            mapper_subtasks = subtask_graph.predecessors(proxy_subtask)
            reducer_subtasks = subtask_graph.successors(proxy_subtask)
            num_mapper, num_reducer = len(mapper_subtasks), len(reducer_subtasks)
            mapper_output_arr = np.empty((num_mapper, num_reducer), dtype=object)
            self.mapper_output_refs.append(mapper_output_arr)
            self.mapper_indices.update(
                {
                    subtask: (shuffle_index, mapper_index)
                    for mapper_index, subtask in enumerate(mapper_subtasks)
                }
            )
            # reducers subtask should be sorted by reducer_index and MapReduceOperand.map should insert shuffle block
            # in reducers order, otherwise shuffle blocks will be sent to wrong reducers.
            sorted_reducer_subtasks = self._sort_reducers(reducer_subtasks)
            self.reducer_indices.update(
                {
                    subtask: (shuffle_index, reducer_index)
                    for reducer_index, subtask in enumerate(sorted_reducer_subtasks)
                }
            )

    @staticmethod
    def _sort_reducers(reducer_subtasks: Iterable[Subtask]):
        def sort_key(subtask: Subtask):
            return next(
                c.op.reducer_index
                for c in subtask.chunk_graph
                if isinstance(c.op, MapReduceOperand)
            )

        return sorted(reducer_subtasks, key=sort_key)

    def has_shuffle(self):
        return self.num_shuffles > 0

    def add_mapper_output_refs(
        self, subtask, output_object_refs: List["ray.ObjectRef"]
    ):
        shuffle_index, mapper_index = self.mapper_indices[subtask]
        self.mapper_output_refs[shuffle_index][mapper_index] = np.array(
            output_object_refs
        )

    def get_reducer_input_refs(self, subtask) -> List["ray.ObjectRef"]:
        shuffle_index, reducer_index = self.reducer_indices[subtask]
        return self.mapper_output_refs[shuffle_index][:, reducer_index]

    def get_num_reducers(self, subtask):
        mapper_index = self.mapper_indices.get(subtask)
        if mapper_index:
            shuffle_index = mapper_index[0]
        else:
            reducer_index = self.reducer_indices[subtask]
            shuffle_index = reducer_index[0]
        return self.mapper_output_refs[shuffle_index].shape[1]
