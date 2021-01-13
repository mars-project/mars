#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

from ...utils import copy_tileables
from ...dataframe.utils import parse_index
from ...dataframe.datasource.core import ColumnPruneSupportedDataSourceMixin
from ...dataframe.groupby.aggregation import DataFrameGroupByAgg
from ...dataframe.indexing.getitem import DataFrameIndex
from .core import TileableOptimizeRule, register


class _PruneDataSource(TileableOptimizeRule):
    def _get_selected_columns(self, op):  # pragma: no cover
        raise NotImplementedError

    def apply(self, node):
        input_node = node.inputs[0]
        selected_columns = self._get_selected_columns(node.op)
        if input_node in self._optimizer_context:
            new_input = self._optimizer_context[input_node]
            selected_columns = [
                c for c in list(input_node.dtypes.index)
                if c in selected_columns + new_input.op.get_columns()]
        else:
            new_input = copy_tileables([input_node])[0].data

        new_input._shape = (input_node.shape[0], len(selected_columns))
        new_input._dtypes = input_node.dtypes[selected_columns]
        new_input._columns_value = parse_index(new_input._dtypes.index, store_data=True)
        new_input.op.set_pruned_columns(selected_columns)
        new_node = copy_tileables([node], inputs=[new_input])[0].data

        self._optimizer_context[node] = new_node
        self._optimizer_context[input_node] = new_input
        return node


class GroupbyPruneDatasource(_PruneDataSource):
    """
    An experimental implementation for tileable optimization rule.
    This rule works only when groupby aggregation operation follows the read CSV files,
    we can prune the columns that not used by the following operations when read the files.
    """
    def match(self, node):
        if isinstance(node.inputs[0].op, ColumnPruneSupportedDataSourceMixin) and \
                node.inputs[0] not in self._optimizer_context.result_tileables:
            selected_columns = self._get_selected_columns(node.op)
            if not selected_columns:
                return False
            return True
        return False

    @classmethod
    def _get_selected_columns(cls, op):
        by = op.groupby_params.get('by')
        by_cols = by if isinstance(by, (list, tuple)) else [by]

        # check all by columns
        for by_col in by_cols:
            if not isinstance(by_col, str):
                return

        selected_columns = list(by_cols)

        # DataFrameGroupby
        selection = op.groupby_params.get('selection', list())
        selection = list(selection) \
            if isinstance(selection, (list, tuple)) else [selection]
        if isinstance(op.func, (str, list)):
            if not selection:
                # if func is str or list and no selection
                # cannot perform optimization
                return
            else:
                selected_columns.extend(selection)
        else:
            # dict
            func_cols = list(op.func)
            selected_columns.extend(func_cols)

        selected_columns = set(selected_columns)
        if len(selected_columns) == \
                len(op.inputs[0].op.get_columns() or op.inputs[0].dtypes):
            # If performs on all columns, no need to prune
            return

        return [c for c in op.inputs[0].dtypes.index if c in selected_columns]


class GetItemPruneDataSource(_PruneDataSource):
    def match(self, node):
        if isinstance(node.inputs[0].op, ColumnPruneSupportedDataSourceMixin) and \
                node.inputs[0] not in self._optimizer_context.result_tileables and \
                isinstance(node.op, DataFrameIndex) and node.op.col_names is not None:
            return True
        return False

    def _get_selected_columns(self, op):
        return op.col_names if isinstance(op.col_names, list) else [op.col_names]


register(DataFrameGroupByAgg, GroupbyPruneDatasource)
register(DataFrameIndex, GetItemPruneDataSource)
