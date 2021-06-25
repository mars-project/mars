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


from abc import ABCMeta, abstractmethod
from typing import Any, List

from ....core import OperandType, TileableType, CHUNK_TYPE
from ....dataframe.datasource.core import ColumnPruneSupportedDataSourceMixin
from ....dataframe.utils import parse_index
from ....utils import implements
from ..core import OptimizationRule, OptimizationRecord, OptimizationRecordType


class PruneDataSource(OptimizationRule, metaclass=ABCMeta):
    def _all_successor_prune_pushdown(self,
                                      successors: List[TileableType]):
        for succ in successors:
            rule_types = self._optimizer_cls.get_rule_types(type(succ.op))
            if rule_types is None:
                return False
            for rule_type in rule_types:
                if not issubclass(rule_type, PruneDataSource):
                    return False
                rule = rule_type(self._graph, self._records,
                                 self._optimizer_cls)
                if not rule._need_prune(succ.op):
                    return False
        return True

    @implements(OptimizationRule.match)
    def match(self, op: OperandType) -> bool:
        node = op.outputs[0]
        input_node = self._graph.predecessors(node)[0]
        successors = self._graph.successors(input_node)
        return self._all_successor_prune_pushdown(successors)

    @abstractmethod
    def _need_prune(self, op: OperandType) -> bool:
        """
        Check if this operand can prune

        Returns
        -------
        need_prune : bool
        """

    @abstractmethod
    def _get_selected_columns(self, op: OperandType) -> List[Any]:
        """
        Get selected columns to prune data source.

        Parameters
        ----------
        op : OperandType
            Operand.

        Returns
        -------
        columns : list
            Columns selected.
        """

    def _merge_selected_columns(self,
                                selected_columns: List[Any],
                                op: OperandType):
        input_node = self._graph.predecessors(op.outputs[0])[0]
        original_node = self._records.get_original_chunk(input_node)
        if original_node is None:
            # not pruned before
            original_all_columns = input_node.dtypes.index.tolist()
            if set(selected_columns) != set(original_all_columns):
                # not prune all fields
                return [c for c in original_all_columns
                        if c in selected_columns]
            else:
                return []
        else:
            # pruned before
            original_all_columns = original_node.dtypes.index.tolist()
            original_pruned_columns = input_node.op.get_columns()
            pruned_columns_set = set(selected_columns) | \
                                 set(original_pruned_columns)
            # pruned before, cannot revert it,
            # so we just return pruned columns
            # even though no columns pruned
            return [c for c in original_all_columns
                    if c in pruned_columns_set]

    @implements(OptimizationRule.apply)
    def apply(self, op: OperandType):
        node = op.outputs[0]
        data_source_node = self._graph.predecessors(node)[0]

        if isinstance(node, CHUNK_TYPE) and \
                self._graph.count_successors(data_source_node) == 1:
            # merge into data source only for chunk
            data_source_params = node.params.copy()
            data_source_params.update(data_source_node.extra_params)
            data_source_op = data_source_node.op.copy()
            data_source_op._key = data_source_node.op.key
            data_source_op._output_types = op.output_types
            if node.ndim == 1:
                data_source_op.set_pruned_columns(node.name, keep_order=True)
            else:
                data_source_op.set_pruned_columns(node.dtypes.index.tolist(),
                                                  keep_order=True)
            new_entity = data_source_op.new_tileable \
                if not isinstance(node, CHUNK_TYPE) else data_source_op.new_chunk
            new_data_source_node = new_entity(
                data_source_node.inputs, kws=[data_source_params]).data
            new_data_source_node._key = node.key
            new_data_source_node._id = node.id
            # just remove the input data
            self._graph.add_node(new_data_source_node)
            for succ in self._graph.successors(node):
                self._graph.add_edge(new_data_source_node, succ)
            self._graph.remove_node(data_source_node)
            self._graph.remove_node(node)

            # mark optimization record
            # the input node is removed
            self._records.append_record(
                OptimizationRecord(data_source_node, None,
                                   OptimizationRecordType.delete))
            self._records.append_record(
                OptimizationRecord(node, new_data_source_node,
                                   OptimizationRecordType.replace))
            new_outputs = [new_data_source_node]
        else:
            selected_columns: List[Any] = self._get_selected_columns(op)
            original_node = self._records.get_original_chunk(data_source_node)
            if original_node is not None:
                # pruned before
                dtypes = original_node.dtypes
            else:
                dtypes = data_source_node.dtypes
            data_source_params = data_source_node.params.copy()
            data_source_params['shape'] = \
                (data_source_node.shape[0], len(selected_columns))
            data_source_params['dtypes'] = dtypes = dtypes[selected_columns]
            data_source_params['columns_value'] = \
                parse_index(dtypes.index, store_data=True)
            data_source_params.update(data_source_node.extra_params)
            data_source_node_op = data_source_node.op.copy()
            data_source_node_op._key = data_source_node.op.key
            data_source_node_op.set_pruned_columns(selected_columns, keep_order=True)
            new_data_source_node = data_source_node_op.new_tileable(
                data_source_node_op.inputs, kws=[data_source_params]).data

            self._replace_node(data_source_node, new_data_source_node)
            # mark optimization record
            self._records.append_record(
                OptimizationRecord(data_source_node, new_data_source_node,
                                   OptimizationRecordType.replace))

            new_op = op.copy()
            new_op._key = op.key
            kws = []
            for out in op.outputs:
                params = out.params.copy()
                params.update(out.extra_params)
                kws.append(params)
            new_entity = new_op.new_tileables \
                if not isinstance(node, CHUNK_TYPE) else new_op.new_chunks
            new_outputs = [t.data for t in
                           new_entity([new_data_source_node], kws=kws)]

            for out, new_out in zip(op.outputs, new_outputs):
                new_out._id = out.id
                new_out._key = out.key
                self._replace_node(out, new_out)
                # mark optimization record
                self._records.append_record(
                    OptimizationRecord(out, new_out,
                                       OptimizationRecordType.replace))

        for out, new_out in zip(op.outputs, new_outputs):
            # check out if it's in result
            try:
                i = self._graph.results.index(out)
                self._graph.results[i] = new_out
            except ValueError:
                pass


class GetitemPruneDataSource(PruneDataSource):
    def _need_prune(self, op: OperandType) -> bool:
        data_source_node = self._graph.predecessors(op.outputs[0])[0]
        input_can_be_pruned = \
            isinstance(data_source_node.op, ColumnPruneSupportedDataSourceMixin)
        if input_can_be_pruned and \
                data_source_node not in self._graph.results and \
                op.col_names is not None:
            selected_columns = self._get_selected_columns(op)
            if not isinstance(op.outputs[0], CHUNK_TYPE) and \
                    not selected_columns:
                # no columns selected, skip
                return False
            return True
        return False

    def _get_selected_columns(self, op: OperandType) -> List[str]:
        columns = op.col_names if isinstance(op.col_names, list) else [op.col_names]
        return self._merge_selected_columns(columns, op)
