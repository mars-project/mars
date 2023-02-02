# Copyright 2022 XProbe Inc.
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

from typing import List, Dict, Set, Any, Type, Union, Optional

import pandas as pd

from .input_column_selector import InputColumnSelector
from .self_column_selector import SelfColumnSelector
from ..core import register_optimization_rule
from ...core import (
    OptimizationRecord,
    OptimizationRecordType,
    OptimizationRule,
    OptimizationRecords,
    Optimizer,
)
from .....core import TileableData
from .....core.graph import EntityGraph
from .....dataframe.core import (
    parse_index,
    BaseSeriesData,
    BaseDataFrameData,
)
from .....dataframe.datasource.core import ColumnPruneSupportedDataSourceMixin
from .....dataframe.groupby.aggregation import DataFrameGroupByAgg
from .....dataframe.indexing.getitem import DataFrameIndex
from .....dataframe.merge import DataFrameMerge
from .....utils import implements

OPTIMIZABLE_OP_TYPES = (DataFrameMerge, DataFrameGroupByAgg)


@register_optimization_rule()
class ColumnPruningRule(OptimizationRule):
    def __init__(
        self,
        graph: EntityGraph,
        records: OptimizationRecords,
        optimizer_cls: Type["Optimizer"],
    ):
        super().__init__(graph, records, optimizer_cls)
        self._context: Dict[TileableData, Dict[TileableData, Set[Any]]] = {}

    def _get_successor_required_columns(self, data: TileableData) -> Set[Any]:
        """
        Get columns required by the successors of the given tileable data.
        """
        successors = self._get_successors(data)
        if successors:
            return set().union(
                *[self._context[successor][data] for successor in successors]
            )
        else:
            return self._get_all_columns(data)

    @staticmethod
    def _get_self_required_columns(data: TileableData) -> Set[Any]:
        return SelfColumnSelector.select(data)

    def _get_required_columns(self, data: TileableData) -> Optional[Set[Any]]:
        required_columns = set()
        successor_required_columns = self._get_successor_required_columns(data)
        if successor_required_columns is None:
            return None
        required_columns.update(successor_required_columns)
        self_required_columns = self._get_self_required_columns(data)
        required_columns.update(self_required_columns)
        return required_columns

    @staticmethod
    def _get_all_columns(data: TileableData) -> Union[Set[Any], None]:
        """
        Return all the columns of given tileable data. If the given tileable data is neither
        BaseDataFrameData nor BaseSeriesData, None will be returned, indicating that column pruning
        is not available for the given tileable data.
        """
        if isinstance(data, BaseDataFrameData) and data.dtypes is not None:
            return set(data.dtypes.index)
        elif isinstance(data, BaseSeriesData):
            return {data.name}
        else:
            return None

    def _get_successors(self, data: TileableData) -> List[TileableData]:
        """
        Get successors of the given tileable data.

        Column pruning is available only when every successor is available for column pruning
        (i.e. appears in the context).
        """
        successors = list(self._graph.successors(data))
        if all(successor in self._context for successor in successors):
            return successors
        else:
            return []

    def _build_context(self) -> None:
        """
        Select required columns for each tileable data in the graph.
        """
        for data in self._graph.topological_iter(reverse=True):
            if self._is_skipped_type(data):
                continue
            self._context[data] = InputColumnSelector.select(
                data, self._get_successor_required_columns(data)
            )

    def _prune_columns(self) -> List[TileableData]:
        pruned_nodes: List[TileableData] = []
        datasource_nodes: List[TileableData] = []

        node_list = list(self._graph.topological_iter())
        for data in node_list:
            if self._is_skipped_type(data):
                continue

            op = data.op

            successor_required_columns = self._get_successor_required_columns(data)
            if (
                isinstance(op, ColumnPruneSupportedDataSourceMixin)
                and successor_required_columns is not None
                and set(successor_required_columns) != self._get_all_columns(data)
            ):
                op.set_pruned_columns(list(successor_required_columns))
                self.effective = True
                pruned_nodes.append(data)
                datasource_nodes.append(data)
                continue

            if isinstance(op, OPTIMIZABLE_OP_TYPES):
                predecessors = list(self._graph.predecessors(data))
                for predecessor in predecessors:
                    if (
                        self._is_skipped_type(predecessor)
                        or predecessor in datasource_nodes
                        # if the group by key is a series, no need to do column pruning
                        or isinstance(predecessor, BaseSeriesData)
                    ):
                        continue

                    pruned_columns = list(self._context[data][predecessor])
                    if set(pruned_columns) == self._get_all_columns(predecessor):
                        continue

                    # new node init
                    new_node_op = DataFrameIndex(
                        col_names=pruned_columns,
                    )
                    new_params = predecessor.params.copy()
                    new_params["shape"] = (
                        new_params["shape"][0],
                        len(pruned_columns),
                    )
                    new_params["dtypes"] = new_params["dtypes"][pruned_columns]
                    new_params["columns_value"] = parse_index(
                        new_params["dtypes"].index, store_data=True
                    )
                    new_node = new_node_op.new_dataframe(
                        [predecessor], **new_params
                    ).data

                    # update context
                    del self._context[data][predecessor]
                    self._context[new_node] = {predecessor: set(pruned_columns)}
                    self._context[data][new_node] = set(pruned_columns)

                    # change edges and nodes
                    self._graph.remove_edge(predecessor, data)
                    self._graph.add_node(new_node)
                    self._graph.add_edge(predecessor, new_node)
                    self._graph.add_edge(new_node, data)

                    self._records.append_record(
                        OptimizationRecord(
                            predecessor, new_node, OptimizationRecordType.new
                        )
                    )
                    # update inputs
                    data.inputs[data.inputs.index(predecessor)] = new_node
                    self.effective = True
                    pruned_nodes.extend([predecessor])
        return pruned_nodes

    def _update_tileable_params(self, pruned_nodes: List[TileableData]) -> None:
        # change dtypes and columns_value
        queue = [n for n in pruned_nodes]
        affected_nodes = set()
        while len(queue) > 0:
            node = queue.pop(0)
            if isinstance(node.op, ColumnPruneSupportedDataSourceMixin):
                affected_nodes.add(node)
            for successor in self._graph.successors(node):
                if successor not in affected_nodes:
                    queue.append(successor)
                    if not self._is_skipped_type(successor):
                        affected_nodes.add(successor)

        for node in affected_nodes:
            required_columns = self._get_required_columns(node)
            if (
                isinstance(node, BaseDataFrameData)
                and required_columns is not None
                and set(required_columns) != set(node.dtypes.index)
            ):
                new_dtypes = pd.Series(
                    dict(
                        (col, dtype)
                        for col, dtype in node.dtypes.items()
                        if col in required_columns
                    )
                )
                new_columns_value = parse_index(new_dtypes.index, store_data=True)
                node._dtypes = new_dtypes
                node._columns_value = new_columns_value
                node._shape = (node.shape[0], len(new_dtypes))

    @implements(OptimizationRule.apply)
    def apply(self):
        self._build_context()
        pruned_nodes = self._prune_columns()
        self._update_tileable_params(pruned_nodes)

    @staticmethod
    def _is_skipped_type(data: TileableData) -> bool:
        """
        If column pruning should be applied to the given tileable data.
        """
        return not isinstance(data, (BaseSeriesData, BaseDataFrameData))
