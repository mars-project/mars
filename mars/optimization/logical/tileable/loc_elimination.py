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

from typing import List

from ....core import OperandType, TileableType
from ....dataframe.indexing.getitem import DataFrameIndex
from ....dataframe.indexing.iloc import DataFrameIlocGetItem
from ....dataframe.indexing.loc import DataFrameLocGetItem
from ....dataframe.merge import DataFrameMerge
from ....utils import is_full_slice
from ..core import OptimizationRecord, OptimizationRecordType
from .core import (
    OptimizationRule,
    register_tileable_optimization_rule,
)


@register_tileable_optimization_rule([DataFrameMerge])
class LocElimination(OptimizationRule):
    def match(self, op: OperandType) -> bool:
        node = op.outputs[0]
        input_nodes = self._graph.predecessors(node)
        if any(self._can_be_eliminated(n) for n in input_nodes):
            return True
        else:
            return False

    def apply(self, op: OperandType):
        node = op.outputs[0]
        input_nodes = node.inputs
        eliminated_nodes = []
        for input_node in input_nodes:
            input_node = self._records.get_optimization_result(input_node, input_node)
            cur_node = input_node
            to_be_eliminated_nodes = []
            while self._can_be_eliminated(cur_node):
                to_be_eliminated_nodes.insert(0, cur_node)
                cur_node = self._graph.predecessors(cur_node)[0]
            eliminated_nodes.append(to_be_eliminated_nodes)
        self._eliminate_loc_nodes(node, eliminated_nodes)

    def _can_be_eliminated(self, node: TileableType):
        op = node.op
        if len(self._graph.successors(node)) == 1:
            if isinstance(op, DataFrameIndex) and op.mask is None:
                return True
            elif isinstance(
                op, (DataFrameLocGetItem, DataFrameIlocGetItem)
            ) and is_full_slice(op.indexes[0]):
                return True
            else:
                return False
        else:
            return False

    def _eliminate_loc_nodes(
        self, node: TileableType, loc_nodes_list: List[List[TileableType]]
    ):
        new_input_tileables = []
        index_functions_list = []
        for loc_nodes, input_node in zip(loc_nodes_list, node.inputs):
            if loc_nodes:
                new_input_tileables.append(self._graph.predecessors(loc_nodes[0])[0])
                index_functions = []
                for loc_node in loc_nodes:
                    index_functions.append(self._loc_op_to_func(loc_node.op))
                    self._records.append_record(
                        OptimizationRecord(
                            loc_node, None, OptimizationRecordType.delete
                        )
                    )
                    self._graph.remove_node(loc_node)
                index_functions_list.append(index_functions)
            else:
                new_input_tileables.append(
                    self._records.get_optimization_result(input_node, input_node)
                )
                index_functions_list.append([])

        new_op = node.op.copy()
        new_op.index_functions = index_functions_list
        params = node.params.copy()
        new_tileable = new_op.new_tileable(new_input_tileables, kws=[params]).data

        self._graph.add_node(new_tileable)
        for succ in self._graph.successors(node):
            self._graph.add_edge(new_tileable, succ)
        for input_tileable in new_input_tileables:
            self._graph.add_edge(input_tileable, new_tileable)
        self._graph.remove_node(node)
        self._records.append_record(
            OptimizationRecord(node, new_tileable, OptimizationRecordType.replace)
        )

    @classmethod
    def _loc_op_to_func(cls, op: OperandType):
        if isinstance(op, DataFrameIndex):
            col_names = op.col_names
            return lambda df: df[col_names]
        elif isinstance(op, DataFrameIlocGetItem):
            column_iloc_index = op.indexes[1]
            return lambda df: df.iloc[:, column_iloc_index]
        else:
            assert isinstance(op, DataFrameLocGetItem)
            column_loc_index = op.indexes[1]
            return lambda df: df.loc[:, column_loc_index]
