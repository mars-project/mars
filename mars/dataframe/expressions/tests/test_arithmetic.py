# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import unittest
from weakref import ReferenceType

import numpy as np
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from mars.dataframe.expressions.utils import split_monotonic_index_min_max, \
    build_split_idx_to_origin_idx
from mars.dataframe.expressions.datasource.dataframe import from_pandas
from mars.dataframe.expressions.arithmetic import add, DataFrameAdd
from mars.dataframe.expressions.arithmetic.core import DataFrameIndexAlignMap
from mars.tests.core import TestBase


@unittest.skipIf(pd is None, 'pandas not installed')
class Test(TestBase):
    def testAddWithoutShuffle(self):
        # all the axes are monotonic
        # data1 with index split into [0...4], [5...9],
        # columns [3...7], [8...12]
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=np.arange(3, 13))
        df1 = from_pandas(data1, chunk_size=5)
        # data2 with index split into [6...11], [2, 5],
        # columns [4...9], [10, 13]
        data2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                             columns=np.arange(4, 14))
        df2 = from_pandas(data2, chunk_size=6)

        df3 = add(df1, df2)
        df3.tiles()

        data1_index_min_max = [(0, True, 4, True), (5, True, 9, True)]
        data1_columns_min_max = [[3, True, 7, True], [8, True, 12, True]]
        data2_index_min_max = [(2, True, 5, True), (6, True, 11, True)]
        data2_columns_min_max = [(4, True, 9, True), (10, True, 13, True)]

        left_index_splits, right_index_splits = split_monotonic_index_min_max(
            data1_index_min_max, True, data2_index_min_max, False)
        left_columns_splits, right_columns_splits = split_monotonic_index_min_max(
            data1_columns_min_max, True, data2_columns_min_max, True)

        left_index_idx_to_original_idx = build_split_idx_to_origin_idx(left_index_splits)
        right_index_idx_to_original_idx = build_split_idx_to_origin_idx(right_index_splits, False)
        left_columns_idx_to_original_idx = build_split_idx_to_origin_idx(left_columns_splits)
        right_columns_idx_to_original_idx = build_split_idx_to_origin_idx(right_columns_splits)

        self.assertEqual(df3.chunk_shape, (7, 7))
        for c in df3.chunks:
            self.assertIsInstance(c.op, DataFrameAdd)
            self.assertEqual(len(c.inputs), 2)
            idx = c.index
            # test the left side
            self.assertIsInstance(c.inputs[0].op, DataFrameIndexAlignMap)
            left_row_idx = left_index_idx_to_original_idx[idx[0]][0]
            left_col_idx = left_columns_idx_to_original_idx[idx[1]][0]
            self.assertIs(c.inputs[0].inputs[0], df1.cix[left_row_idx, left_col_idx].data)
            # test the right side
            self.assertIsInstance(c.inputs[0].op, DataFrameIndexAlignMap)
