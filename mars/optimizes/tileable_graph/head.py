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
from ...dataframe.base.value_counts import DataFrameValueCounts
from ...dataframe.datasource.core import HeadOptimizedDataSource
from ...dataframe.indexing.iloc import DataFrameIlocGetItem, SeriesIlocGetItem
from ...dataframe.sort.core import DataFrameSortOperand
from ...dataframe.utils import parse_index
from .core import TileableOptimizeRule, register


class HeadPushDown(TileableOptimizeRule):
    def match(self, node):
        op = node.op
        inputs = op.inputs
        accept_types = (HeadOptimizedDataSource, DataFrameSortOperand, DataFrameValueCounts)
        if len(inputs) == 1 and isinstance(op, (DataFrameIlocGetItem, SeriesIlocGetItem)) and \
                op.can_be_optimized() and isinstance(inputs[0].op, accept_types) and \
                inputs[0] not in self._optimizer_context.result_tileables:
            return True
        return False

    def apply(self, node):
        op = node.op
        inp = op.inputs[0]
        nrows = inp.op.nrows or 0
        head = op.indexes[0].stop

        if inp in self._optimizer_context:
            new_inp = self._optimizer_context[inp]
        else:
            new_inp = copy_tileables([inp])[0].data

        new_inp.op._nrows = max(nrows, head)
        new_inp._key = node.key
        new_inp._shape = (nrows,) + inp.shape[1:]
        pd_index = node.index_value.to_pandas()[:nrows]
        new_inp._index_value = parse_index(pd_index, node)

        self._optimizer_context[node] = new_inp
        return new_inp


register(DataFrameIlocGetItem, HeadPushDown)
register(SeriesIlocGetItem, HeadPushDown)
