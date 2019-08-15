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

import numpy as np
import pandas as pd

from mars.executor import Executor
from mars.tests.core import TestBase
from mars.dataframe.core import DataFrame, Series
from mars.dataframe.datasource.dataframe import from_pandas
from mars.dataframe.indexing.iloc import DataFrameIlocGetItem, DataFrameIlocSetItem


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.executor = Executor()

    def testSetIndex(self):
        df1 = pd.DataFrame([[1,3,3], [4,2,6], [7, 8, 9]],
                            index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = from_pandas(df1, chunk_size=2)

        df3 = df2.set_index('y', drop=True)
        df3.tiles()
        self.assertEqual(df3.chunk_shape, (2, 2))
        pd.testing.assert_index_equal(df3.chunks[0].columns.to_pandas(), pd.Index(['x']))
        pd.testing.assert_index_equal(df3.chunks[1].columns.to_pandas(), pd.Index(['z']))

        df4 = df2.set_index('y', drop=False)
        df4.tiles()
        self.assertEqual(df4.chunk_shape, (2, 2))
        pd.testing.assert_index_equal(df4.chunks[0].columns.to_pandas(), pd.Index(['x', 'y']))
        pd.testing.assert_index_equal(df4.chunks[1].columns.to_pandas(), pd.Index(['z']))

    def testILocGetItem(self):
        df1 = pd.DataFrame([[1,3,3], [4,2,6], [7, 8, 9]],
                           index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = from_pandas(df1, chunk_size=2)

        # plain index
        df3 = df2.iloc[1]
        df3.tiles()
        self.assertIsInstance(df3, Series)
        self.assertIsInstance(df3.op, DataFrameIlocGetItem)
        self.assertEqual(df3.chunk_shape, (2,))
        self.assertEqual(df3.chunks[0].shape, (2,))
        self.assertEqual(df3.chunks[1].shape, (1,))
        self.assertEqual(df3.chunks[0].op.indexes, (1, slice(None, None, None)))
        self.assertEqual(df3.chunks[1].op.indexes, (1, slice(None, None, None)))
        self.assertEqual(df3.chunks[0].inputs[0].index, (0, 0))
        self.assertEqual(df3.chunks[0].inputs[0].shape, (2, 2))
        self.assertEqual(df3.chunks[1].inputs[0].index, (0, 1))
        self.assertEqual(df3.chunks[1].inputs[0].shape, (2, 1))

        # slice index
        df4 = df2.iloc[:, 2:4]
        df4.tiles()
        self.assertIsInstance(df4, DataFrame)
        self.assertIsInstance(df4.op, DataFrameIlocGetItem)
        self.assertEqual(df4.chunk_shape, (2, 1))
        self.assertEqual(df4.chunks[0].shape, (2, 1))
        self.assertEqual(df4.chunks[1].shape, (1, 1))
        self.assertEqual(df4.chunks[0].op.indexes, (slice(None, None, None), slice(None, None, None)))
        self.assertEqual(df4.chunks[1].op.indexes, (slice(None, None, None), slice(None, None, None)))
        self.assertEqual(df4.chunks[0].inputs[0].index, (0, 1))
        self.assertEqual(df4.chunks[0].inputs[0].shape, (2, 1))
        self.assertEqual(df4.chunks[1].inputs[0].index, (1, 1))
        self.assertEqual(df4.chunks[1].inputs[0].shape, (1, 1))

        # plain fancy index
        df5 = df2.iloc[[0], [0, 1, 2]]
        df5.tiles()
        self.assertIsInstance(df5, DataFrame)
        self.assertIsInstance(df5.op, DataFrameIlocGetItem)
        self.assertEqual(df5.shape, (1, 3))
        self.assertEqual(df5.chunk_shape, (1, 2))
        self.assertEqual(df5.chunks[0].shape, (1, 2))
        self.assertEqual(df5.chunks[1].shape, (1, 1))
        np.testing.assert_array_equal(df5.chunks[0].op.indexes[0], [0])
        np.testing.assert_array_equal(df5.chunks[0].op.indexes[1], [0, 1])
        np.testing.assert_array_equal(df5.chunks[1].op.indexes[0], [0])
        np.testing.assert_array_equal(df5.chunks[1].op.indexes[1], [0])
        self.assertEqual(df5.chunks[0].inputs[0].index, (0, 0))
        self.assertEqual(df5.chunks[0].inputs[0].shape, (2, 2))
        self.assertEqual(df5.chunks[1].inputs[0].index, (0, 1))
        self.assertEqual(df5.chunks[1].inputs[0].shape, (2, 1))

        # fancy index
        df6 = df2.iloc[[1, 2], [0, 1, 2]]
        df6.tiles()
        self.assertIsInstance(df6, DataFrame)
        self.assertIsInstance(df6.op, DataFrameIlocGetItem)
        self.assertEqual(df6.shape, (2, 3))
        self.assertEqual(df6.chunk_shape, (2, 2))
        self.assertEqual(df6.chunks[0].shape, (1, 2))
        self.assertEqual(df6.chunks[1].shape, (1, 1))
        self.assertEqual(df6.chunks[2].shape, (1, 2))
        self.assertEqual(df6.chunks[3].shape, (1, 1))
        np.testing.assert_array_equal(df6.chunks[0].op.indexes[0], [1])
        np.testing.assert_array_equal(df6.chunks[0].op.indexes[1], [0, 1])
        np.testing.assert_array_equal(df6.chunks[1].op.indexes[0], [1])
        np.testing.assert_array_equal(df6.chunks[1].op.indexes[1], [0])
        np.testing.assert_array_equal(df6.chunks[2].op.indexes[0], [0])
        np.testing.assert_array_equal(df6.chunks[2].op.indexes[1], [0, 1])
        np.testing.assert_array_equal(df6.chunks[3].op.indexes[0], [0])
        np.testing.assert_array_equal(df6.chunks[3].op.indexes[1], [0])
        self.assertEqual(df6.chunks[0].inputs[0].index, (0, 0))
        self.assertEqual(df6.chunks[0].inputs[0].shape, (2, 2))
        self.assertEqual(df6.chunks[1].inputs[0].index, (0, 1))
        self.assertEqual(df6.chunks[1].inputs[0].shape, (2, 1))
        self.assertEqual(df6.chunks[2].inputs[0].index, (1, 0))
        self.assertEqual(df6.chunks[2].inputs[0].shape, (1, 2))
        self.assertEqual(df6.chunks[3].inputs[0].index, (1, 1))
        self.assertEqual(df6.chunks[3].inputs[0].shape, (1, 1))

        # plain index
        df7 = df2.iloc[1, 2]
        df7.tiles()
        self.assertIsInstance(df7, Series)
        self.assertIsInstance(df7.op, DataFrameIlocGetItem)
        self.assertEqual(df7.chunk_shape, ())
        self.assertEqual(df7.chunks[0].shape, ())
        self.assertEqual(df7.chunks[0].op.indexes, (1, 0))
        self.assertEqual(df7.chunks[0].inputs[0].index, (0, 1))
        self.assertEqual(df7.chunks[0].inputs[0].shape, (2, 1))

    def testILocSetItem(self):
        df1 = pd.DataFrame([[1,3,3], [4,2,6], [7, 8, 9]],
                            index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = from_pandas(df1, chunk_size=2)
        df2.tiles()

        # plain index
        df3 = from_pandas(df1, chunk_size=2)
        df3.iloc[1] = 100
        df3.tiles()
        self.assertIsInstance(df3.op, DataFrameIlocSetItem)
        self.assertEqual(df3.chunk_shape, df2.chunk_shape)
        pd.testing.assert_index_equal(df2.index_value.to_pandas(), df3.index_value.to_pandas())
        pd.testing.assert_index_equal(df2.columns.to_pandas(), df3.columns.to_pandas())
        for c1, c2 in zip(df2.chunks, df3.chunks):
            self.assertEqual(c1.shape, c2.shape)
            pd.testing.assert_index_equal(c1.index_value.to_pandas(), c2.index_value.to_pandas())
            pd.testing.assert_index_equal(c1.columns.to_pandas(), c2.columns.to_pandas())
            if isinstance(c2.op, DataFrameIlocSetItem):
                self.assertEqual(c1.key, c2.inputs[0].key)
            else:
                self.assertEqual(c1.key, c2.key)
        self.assertEqual(df3.chunks[0].op.indexes, (1, slice(None, None, None)))
        self.assertEqual(df3.chunks[1].op.indexes, (1, slice(None, None, None)))

        # # slice index
        df4 = from_pandas(df1, chunk_size=2)
        df4.iloc[:, 2:4] = 1111
        df4.tiles()
        self.assertIsInstance(df4.op, DataFrameIlocSetItem)
        self.assertEqual(df4.chunk_shape, df2.chunk_shape)
        pd.testing.assert_index_equal(df2.index_value.to_pandas(), df4.index_value.to_pandas())
        pd.testing.assert_index_equal(df2.columns.to_pandas(), df4.columns.to_pandas())
        for c1, c2 in zip(df2.chunks, df4.chunks):
            self.assertEqual(c1.shape, c2.shape)
            pd.testing.assert_index_equal(c1.index_value.to_pandas(), c2.index_value.to_pandas())
            pd.testing.assert_index_equal(c1.columns.to_pandas(), c2.columns.to_pandas())
            if isinstance(c2.op, DataFrameIlocSetItem):
                self.assertEqual(c1.key, c2.inputs[0].key)
            else:
                self.assertEqual(c1.key, c2.key)
        self.assertEqual(df4.chunks[1].op.indexes, (slice(None, None, None), slice(None, None, None)))
        self.assertEqual(df4.chunks[3].op.indexes, (slice(None, None, None), slice(None, None, None)))

        # plain fancy index
        df5 = from_pandas(df1, chunk_size=2)
        df5.iloc[[0], [0, 1, 2]] = 2222
        df5.tiles()
        self.assertIsInstance(df5.op, DataFrameIlocSetItem)
        self.assertEqual(df5.chunk_shape, df2.chunk_shape)
        pd.testing.assert_index_equal(df2.index_value.to_pandas(), df5.index_value.to_pandas())
        pd.testing.assert_index_equal(df2.columns.to_pandas(), df5.columns.to_pandas())
        for c1, c2 in zip(df2.chunks, df5.chunks):
            self.assertEqual(c1.shape, c2.shape)
            pd.testing.assert_index_equal(c1.index_value.to_pandas(), c2.index_value.to_pandas())
            pd.testing.assert_index_equal(c1.columns.to_pandas(), c2.columns.to_pandas())
            if isinstance(c2.op, DataFrameIlocSetItem):
                self.assertEqual(c1.key, c2.inputs[0].key)
            else:
                self.assertEqual(c1.key, c2.key)
        np.testing.assert_array_equal(df5.chunks[0].op.indexes[0], [0])
        np.testing.assert_array_equal(df5.chunks[0].op.indexes[1], [0, 1])
        np.testing.assert_array_equal(df5.chunks[1].op.indexes[0], [0])
        np.testing.assert_array_equal(df5.chunks[1].op.indexes[1], [0])

        # fancy index
        df6 = from_pandas(df1, chunk_size=2)
        df6.iloc[[1, 2], [0, 1, 2]] = 3333
        df6.tiles()
        self.assertIsInstance(df6.op, DataFrameIlocSetItem)
        self.assertEqual(df6.chunk_shape, df2.chunk_shape)
        pd.testing.assert_index_equal(df2.index_value.to_pandas(), df6.index_value.to_pandas())
        pd.testing.assert_index_equal(df2.columns.to_pandas(), df6.columns.to_pandas())
        for c1, c2 in zip(df2.chunks, df6.chunks):
            self.assertEqual(c1.shape, c2.shape)
            pd.testing.assert_index_equal(c1.index_value.to_pandas(), c2.index_value.to_pandas())
            pd.testing.assert_index_equal(c1.columns.to_pandas(), c2.columns.to_pandas())
            if isinstance(c2.op, DataFrameIlocSetItem):
                self.assertEqual(c1.key, c2.inputs[0].key)
            else:
                self.assertEqual(c1.key, c2.key)
        np.testing.assert_array_equal(df6.chunks[0].op.indexes[0], [1])
        np.testing.assert_array_equal(df6.chunks[0].op.indexes[1], [0, 1])
        np.testing.assert_array_equal(df6.chunks[1].op.indexes[0], [1])
        np.testing.assert_array_equal(df6.chunks[1].op.indexes[1], [0])
        np.testing.assert_array_equal(df6.chunks[2].op.indexes[0], [0])
        np.testing.assert_array_equal(df6.chunks[2].op.indexes[1], [0, 1])
        np.testing.assert_array_equal(df6.chunks[3].op.indexes[0], [0])
        np.testing.assert_array_equal(df6.chunks[3].op.indexes[1], [0])

        # plain index
        df7 = from_pandas(df1, chunk_size=2)
        df7.iloc[1, 2] = 4444
        df7.tiles()
        self.assertIsInstance(df7.op, DataFrameIlocSetItem)
        self.assertEqual(df7.chunk_shape, df2.chunk_shape)
        pd.testing.assert_index_equal(df2.index_value.to_pandas(), df7.index_value.to_pandas())
        pd.testing.assert_index_equal(df2.columns.to_pandas(), df7.columns.to_pandas())
        for c1, c2 in zip(df2.chunks, df7.chunks):
            self.assertEqual(c1.shape, c2.shape)
            pd.testing.assert_index_equal(c1.index_value.to_pandas(), c2.index_value.to_pandas())
            pd.testing.assert_index_equal(c1.columns.to_pandas(), c2.columns.to_pandas())
            if isinstance(c2.op, DataFrameIlocSetItem):
                self.assertEqual(c1.key, c2.inputs[0].key)
            else:
                self.assertEqual(c1.key, c2.key)
        self.assertEqual(df7.chunks[1].op.indexes, (1, 0))
