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

from ...compat import six
from ...dataframe.groupby.aggregation import DataFrameGroupByAgg
from ...dataframe.datasource.read_csv import DataFrameReadCSV
from .core import TileableOptimizeRule, register


class GroupByPruningReadCSV(TileableOptimizeRule):
    """
    An experimental implementation for tileable optimization rule.
    This rule works only when groupby aggregation operation follows the read CSV files,
    we can prune the columns that not used by the following operations when read the files.
    """
    def _match(self, node):
        if isinstance(node.op, DataFrameGroupByAgg) and isinstance(node.inputs[0].op, DataFrameReadCSV):
            return True
        return False

    def _process(self, node):
        by_columns = node.op.by if isinstance(node.op.by, (list, tuple)) else [node.op.by]
        if isinstance(node.op.func, six.string_types):
            # Passing func name means perform on all columns.
            return node
        else:
            agg_columns = list(node.op.func)
        selected_columns = by_columns + agg_columns
        input_node = node.inputs[0]
        if len(selected_columns) >= len(input_node.dtypes):
            return node
        else:
            if input_node in self._optimizer_context:
                new_input = self._optimizer_context[input_node]
                selected_columns = list(set(selected_columns + new_input.op._usecols))
            else:
                new_input = input_node.op.copy().reset_key().new_tileables(
                    input_node.inputs, kws=[input_node.params.copy()],
                    _key=input_node.key, **input_node.extra_params)[0].data

            new_input._dtypes = input_node.dtypes[selected_columns]
            new_input._columns_value = new_input._dtypes.index
            new_input.op._usecols = selected_columns
            new_node = node.op.copy().reset_key().new_tileables(
                [new_input], kws=[node.params.copy()], _key=node.key, **node.extra_params)[0].data

            self._optimizer_context[node] = new_node
            self._optimizer_context[input_node] = new_input
            return new_node


register(DataFrameGroupByAgg, GroupByPruningReadCSV)

