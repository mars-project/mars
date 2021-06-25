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

from ....core import OperandType
from ....dataframe.datasource.core import ColumnPruneSupportedDataSourceMixin
from ....dataframe.groupby.aggregation import DataFrameGroupByAgg
from ....dataframe.indexing.getitem import DataFrameIndex
from ..common.column_pruning import PruneDataSource, GetitemPruneDataSource
from .core import register_tileable_optimization_rule


@register_tileable_optimization_rule([DataFrameGroupByAgg])
class GroupByPruneDataSource(PruneDataSource):
    def _need_prune(self, op: OperandType) -> bool:
        data_source_node = self._graph.predecessors(op.outputs[0])[0]
        input_can_be_pruned = \
            isinstance(data_source_node.op, ColumnPruneSupportedDataSourceMixin)
        if input_can_be_pruned and data_source_node not in self._graph.results:
            selected_columns = self._get_selected_columns(op)
            if not selected_columns:
                # no columns selected, skip
                return False
            return True
        return False

    def _get_selected_columns(self, op: OperandType) -> List[str]:
        by = op.groupby_params.get('by')
        by_cols = by if isinstance(by, (list, tuple)) else [by]

        # check all by columns
        for by_col in by_cols:
            if not isinstance(by_col, str):
                return []

        selected_columns = list(by_cols)

        # DataFrameGroupby
        selection = op.groupby_params.get('selection', list())
        selection = list(selection) \
            if isinstance(selection, (list, tuple)) else [selection]
        if isinstance(op.func, (str, list)):
            if not selection:
                # if func is str or list and no selection
                # cannot perform optimization
                return []
            else:
                selected_columns.extend(selection)
        else:
            # dict
            func_cols = list(op.func)
            selected_columns.extend(func_cols)

        return self._merge_selected_columns(selected_columns, op)


@register_tileable_optimization_rule([DataFrameIndex])
class TileableGetitemPruneDataSource(GetitemPruneDataSource):
    """
    Prune data source for getitem.
    """
