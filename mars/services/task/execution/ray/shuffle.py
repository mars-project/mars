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

from .....core.operand import MapReduceOperand, OperandStage
from .....utils import lazy_import
from ....subtask import Subtask, SubtaskGraph

ray = lazy_import("ray")


class ShuffleManager:
    """Manage shuffle execution for ray by resolve dependencies between mappers outputs and reducers inputs based on
    mapper and reducer index.
    """

    def __init__(self, subtask_graph: SubtaskGraph):
        self.subtask_graph = subtask_graph
        self._proxy_subtasks = subtask_graph.get_shuffle_proxy_subtasks()
        self.num_shuffles = subtask_graph.num_shuffles()
        self.mapper_output_refs = []
        self.mapper_indices = {}
        self.reducer_indices = {}
        for shuffle_index, proxy_subtask in enumerate(self._proxy_subtasks):
            # Note that the reducers can also be mappers such as `DuplicateOperand`.
            mapper_subtasks = subtask_graph.predecessors(proxy_subtask)
            reducer_subtasks = subtask_graph.successors(proxy_subtask)
            n_mappers = len(mapper_subtasks)
            n_reducers = proxy_subtask.chunk_graph.results[0].op.n_reducers
            mapper_output_arr = np.empty((n_mappers, n_reducers), dtype=object)
            self.mapper_output_refs.append(mapper_output_arr)
            self.mapper_indices.update(
                {
                    subtask: (shuffle_index, mapper_index)
                    for mapper_index, subtask in enumerate(mapper_subtasks)
                }
            )
            # reducers subtask should be sorted by reducer_index and MapReduceOperand.map should insert shuffle block
            # in reducers order, otherwise shuffle blocks will be sent to wrong reducers.
            sorted_filled_reducer_subtasks = self._get_sorted_filled_reducers(
                reducer_subtasks, n_reducers
            )
            self.reducer_indices.update(
                {
                    subtask: (shuffle_index, reducer_ordinal)
                    for reducer_ordinal, subtask in enumerate(
                        sorted_filled_reducer_subtasks
                    )
                }
            )

    @staticmethod
    def _get_sorted_filled_reducers(
        reducer_subtasks: Iterable[Subtask], n_reducers: int
    ):
        # For operands such as `PSRSAlign`, sometimes `reducer_subtasks` might be less than `n_reducers`.
        # fill missing reducers with `None`.
        filled_reducers = [None] * n_reducers
        for subtask in reducer_subtasks:
            reducer_ordinal = _get_reducer_operand(subtask.chunk_graph).reducer_ordinal
            filled_reducers[reducer_ordinal] = subtask
        return filled_reducers

    def has_shuffle(self):
        """
        Whether current subtask graph has shuffles to execute.
        """
        return self.num_shuffles > 0

    def add_mapper_output_refs(
        self, subtask, output_object_refs: List["ray.ObjectRef"]
    ):
        """
        Record mapper output ObjectRefs which will be used by reducers later.

        Parameters
        ----------
        subtask
        output_object_refs : List["ray.ObjectRef"]
            Mapper output ObjectRefs.
        """
        shuffle_index, mapper_index = self.mapper_indices[subtask]
        self.mapper_output_refs[shuffle_index][mapper_index] = np.array(
            output_object_refs
        )

    def get_reducer_input_refs(self, subtask) -> List["ray.ObjectRef"]:
        """
        Get the reducer inputs ObjectRefs output by mappers.

        Parameters
        ----------
        subtask : Subtask
            A reducer subtask.
        Returns
        -------
        input_refs : List["ray.ObjectRef"]
            The reducer inputs ObjectRefs output by mappers.
        """
        shuffle_index, reducer_ordinal = self.reducer_indices[subtask]
        return self.mapper_output_refs[shuffle_index][:, reducer_ordinal]

    def get_n_reducers(self, subtask):
        """
        Get the number of shuffle blocks that a mapper operand outputs,
        which is also the number of the reducers when tiling shuffle operands.
        Note that this might be greater than actual number of the reducers in the subtask graph,
        because some reducers may not be added to chunk graph.

        Parameters
        ----------
        subtask : Subtask
            A mapper or reducer subtask.
        Returns
        -------
        n_reducers : int
            The number of shuffle blocks that a mapper operand outputs.
        """
        index = self.mapper_indices.get(subtask) or self.reducer_indices.get(subtask)
        if index is None:
            raise Exception(f"The {subtask} should be a mapper or a reducer.")
        else:
            shuffle_index, _ = index
            return self.mapper_output_refs[shuffle_index].shape[1]

    def is_mapper(self, subtask):
        """
        Check whether a subtask is a mapper subtask. Note the even this a mapper subtask, it can be a reducer subtask
        at the same time such as `DuplicateOperand`, see
        https://user-images.githubusercontent.com/12445254/174305282-f7c682a9-0346-47fe-a34c-1e384e6a1775.svg
        """
        return subtask in self.mapper_indices


def _get_reducer_operand(subtask_chunk_graph):
    return next(
        c.op
        for c in subtask_chunk_graph
        if isinstance(c.op, MapReduceOperand) and c.op.stage == OperandStage.reduce
    )
