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

import unittest

import numpy as np
import pandas as pd

from mars.operands import OperandStage
from mars.dataframe.initializer import DataFrame
from mars.dataframe.indexing.getitem import DataFrameIndex
from mars.dataframe.sort.sort_values import dataframe_sort_values, DataFrameSortValues
from mars.dataframe.sort.sort_index import sort_index, DataFrameSortIndex


class Test(unittest.TestCase):
    def testSortValues(self):
        raw = pd.DataFrame({'a': np.random.rand(10),
                            'b': np.random.randint(1000, size=10),
                            'c': np.random.rand(10),
                            'd': [np.random.bytes(10) for _ in range(10)],
                            'e': [pd.Timestamp('201{}'.format(i)) for i in range(10)],
                            'f': [pd.Timedelta('{} days'.format(i)) for i in range(10)]
                            },)
        df = DataFrame(raw)
        sorted_df = dataframe_sort_values(df, by='c')

        self.assertEqual(sorted_df.shape, raw.shape)
        self.assertIsInstance(sorted_df.op, DataFrameSortValues)

        tiled = sorted_df.tiles()

        self.assertEqual(len(tiled.chunks), 1)
        self.assertIsInstance(tiled.chunks[0].op, DataFrameSortValues)

        df = DataFrame(raw, chunk_size=6)
        sorted_df = dataframe_sort_values(df, by='c')

        self.assertEqual(sorted_df.shape, raw.shape)
        self.assertIsInstance(sorted_df.op, DataFrameSortValues)

        tiled = sorted_df.tiles()

        self.assertEqual(len(tiled.chunks), 2)
        self.assertEqual(tiled.chunks[0].op.stage, OperandStage.reduce)

        df = DataFrame(raw, chunk_size=3)
        sorted_df = dataframe_sort_values(df, by=['a', 'c'])

        self.assertEqual(sorted_df.shape, raw.shape)
        self.assertIsInstance(sorted_df.op, DataFrameSortValues)
        pd.testing.assert_index_equal(sorted_df.index_value.to_pandas(), pd.RangeIndex(10))

        tiled = sorted_df.tiles()

        self.assertEqual(len(tiled.chunks), 3)
        self.assertEqual(tiled.chunks[0].op.stage, OperandStage.reduce)
        pd.testing.assert_index_equal(tiled.chunks[0].index_value.to_pandas(), pd.RangeIndex(3))
        self.assertEqual(tiled.chunks[1].op.stage, OperandStage.reduce)
        pd.testing.assert_index_equal(tiled.chunks[1].index_value.to_pandas(), pd.RangeIndex(3, 6))
        self.assertEqual(tiled.chunks[2].op.stage, OperandStage.reduce)
        pd.testing.assert_index_equal(tiled.chunks[2].index_value.to_pandas(), pd.RangeIndex(6, 10))

    def testSortIndex(self):
        raw = pd.DataFrame(np.random.rand(10, 10), columns=np.random.rand(10), index=np.random.rand(10))
        df = DataFrame(raw)
        sorted_df = sort_index(df)

        self.assertEqual(sorted_df.shape, raw.shape)
        self.assertIsInstance(sorted_df.op, DataFrameSortIndex)

        tiled = sorted_df.tiles()

        self.assertEqual(len(tiled.chunks), 1)
        self.assertIsInstance(tiled.chunks[0].op, DataFrameSortIndex)

        df = DataFrame(raw, chunk_size=6)
        sorted_df = sort_index(df)

        self.assertEqual(sorted_df.shape, raw.shape)
        self.assertIsInstance(sorted_df.op, DataFrameSortIndex)

        tiled = sorted_df.tiles()

        self.assertEqual(len(tiled.chunks), 2)
        self.assertEqual(tiled.chunks[0].op.stage, OperandStage.reduce)

        df = DataFrame(raw, chunk_size=3)
        sorted_df = sort_index(df)

        self.assertEqual(sorted_df.shape, raw.shape)
        self.assertIsInstance(sorted_df.op, DataFrameSortIndex)

        tiled = sorted_df.tiles()

        self.assertEqual(len(tiled.chunks), 3)
        self.assertEqual(tiled.chunks[0].op.stage, OperandStage.reduce)
        self.assertEqual(tiled.chunks[1].op.stage, OperandStage.reduce)
        self.assertEqual(tiled.chunks[2].op.stage, OperandStage.reduce)

        # support on axis 1
        df = DataFrame(raw, chunk_size=4)
        sorted_df = sort_index(df, axis=1)

        self.assertEqual(sorted_df.shape, raw.shape)
        self.assertIsInstance(sorted_df.op, DataFrameSortIndex)

        tiled = sorted_df.tiles()

        self.assertTrue(all(isinstance(c.op, DataFrameIndex) for c in tiled.chunks))
