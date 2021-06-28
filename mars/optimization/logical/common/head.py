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

from ....core import OperandType, TileableType, CHUNK_TYPE
from ....dataframe.base.value_counts import DataFrameValueCounts
from ....dataframe.datasource.core import HeadOptimizedDataSource
from ....dataframe.sort.core import DataFrameSortOperand
from ....dataframe.utils import parse_index
from ....utils import implements
from ..core import OptimizationRule, OptimizationRecord, OptimizationRecordType


class HeadPushDown(OptimizationRule):
    @implements(OptimizationRule.match)
    def match(self, op: OperandType) -> bool:
        node = op.outputs[0]
        input_node = self._graph.predecessors(node)[0]
        successors = self._graph.successors(input_node)
        return self._all_successor_head_pushdown(successors)

    def _all_successor_head_pushdown(self,
                                     successors: List[TileableType]):
        for succ in successors:
            rule_types = self._optimizer_cls.get_rule_types(type(succ.op))
            if rule_types is None:
                return False
            for rule_type in rule_types:
                if not issubclass(rule_type, HeadPushDown):
                    return False
                rule = rule_type(self._graph, self._records,
                                 self._optimizer_cls)
                if not rule._can_push_down(succ.op):
                    return False
        return True

    def _can_push_down(self, op: OperandType) -> bool:
        input_nodes = self._graph.predecessors(op.outputs[0])
        accept_types = (HeadOptimizedDataSource,
                        DataFrameSortOperand,
                        DataFrameValueCounts)
        if len(input_nodes) == 1 and \
                op.can_be_optimized() and \
                isinstance(input_nodes[0].op, accept_types) and \
                input_nodes[0] not in self._graph.results:
            return True
        return False

    def apply(self, op: OperandType):
        node = op.outputs[0]
        input_node = self._graph.predecessors(node)[0]
        nrows = input_node.op.nrows or 0
        head = op.indexes[0].stop

        new_input_op = input_node.op.copy()
        new_input_op._key = input_node.op.key
        new_input_op._nrows = nrows = max(nrows, head)
        new_input_params = input_node.params.copy()
        new_input_params['shape'] = (nrows,) + input_node.shape[1:]
        pandas_index = node.index_value.to_pandas()[:nrows]
        new_input_params['index_value'] = parse_index(pandas_index, node)
        new_input_params.update(input_node.extra_params)
        new_entity = new_input_op.new_tileable \
            if not isinstance(node, CHUNK_TYPE) else new_input_op.new_chunk
        new_input_node = new_entity(
            input_node.inputs, kws=[new_input_params]).data

        if new_input_node.op.nrows == head and \
                self._graph.count_successors(input_node) == 1:
            new_input_node._key = node.key
            new_input_node._id = node.id
            # just remove the input data
            self._graph.add_node(new_input_node)
            for succ in self._graph.successors(node):
                self._graph.add_edge(new_input_node, succ)
            for pred in self._graph.predecessors(input_node):
                self._graph.add_edge(pred, new_input_node)
            self._graph.remove_node(input_node)
            self._graph.remove_node(node)

            # mark optimization record
            # the input node is removed
            self._records.append_record(
                OptimizationRecord(input_node, None,
                                   OptimizationRecordType.delete))
            self._records.append_record(
                OptimizationRecord(node, new_input_node,
                                   OptimizationRecordType.replace))
            new_node = new_input_node
        else:
            self._replace_node(input_node, new_input_node)
            new_op = op.copy()
            new_op._key = op.key
            params = node.params.copy()
            params.update(node.extra_params)
            new_entity = new_op.new_tileable \
                if not isinstance(node, CHUNK_TYPE) else new_op.new_chunk
            new_node = new_entity(
                [new_input_node], kws=[params]).data
            self._replace_node(node, new_node)

            # mark optimization record
            self._records.append_record(
                OptimizationRecord(input_node, new_input_node,
                                   OptimizationRecordType.replace))
            self._records.append_record(
                OptimizationRecord(node, new_node,
                                   OptimizationRecordType.replace))

        # check node if it's in result
        try:
            i = self._graph.results.index(node)
            self._graph.results[i] = new_node
        except ValueError:
            pass
