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

import numpy as np
import pandas as pd

import mars.dataframe as md
from mars.tensor.core import CHUNK_TYPE as TENSOR_CHUNK_TYPE
from mars.tests.core import TestBase
from mars.dataframe.core import SERIES_CHUNK_TYPE, Series, DataFrame, DATAFRAME_CHUNK_TYPE
from mars.dataframe.indexing.iloc import DataFrameIlocGetItem, DataFrameIlocSetItem


class Test(TestBase):
    def testSetIndex(self):
        df1 = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]],
                           index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = md.DataFrame(df1, chunk_size=2)

        df3 = df2.set_index('y', drop=True)
        df3 = df3.tiles()
        self.assertEqual(df3.chunk_shape, (2, 2))
        pd.testing.assert_index_equal(df3.chunks[0].columns_value.to_pandas(), pd.Index(['x']))
        pd.testing.assert_index_equal(df3.chunks[1].columns_value.to_pandas(), pd.Index(['z']))

        df4 = df2.set_index('y', drop=False)
        df4 = df4.tiles()
        self.assertEqual(df4.chunk_shape, (2, 2))
        pd.testing.assert_index_equal(df4.chunks[0].columns_value.to_pandas(), pd.Index(['x', 'y']))
        pd.testing.assert_index_equal(df4.chunks[1].columns_value.to_pandas(), pd.Index(['z']))

    def testILocGetItem(self):
        df1 = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]],
                           index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = md.DataFrame(df1, chunk_size=2)

        # plain index
        df3 = df2.iloc[1]
        df3 = df3.tiles()
        self.assertIsInstance(df3, Series)
        self.assertIsInstance(df3.op, DataFrameIlocGetItem)
        self.assertEqual(df3.shape, (3,))
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
        df4 = df4.tiles()
        self.assertIsInstance(df4, DataFrame)
        self.assertIsInstance(df4.op, DataFrameIlocGetItem)
        self.assertEqual(df4.shape, (3, 1))
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
        df5 = df5.tiles()
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
        df6 = df6.tiles()
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
        df7 = df7.tiles()
        self.assertIsInstance(df7, Series)
        self.assertIsInstance(df7.op, DataFrameIlocGetItem)
        self.assertEqual(df7.shape, ())
        self.assertEqual(df7.chunk_shape, ())
        self.assertEqual(df7.chunks[0].dtype, df7.dtype)
        self.assertEqual(df7.chunks[0].shape, ())
        self.assertEqual(df7.chunks[0].op.indexes, (1, 0))
        self.assertEqual(df7.chunks[0].inputs[0].index, (0, 1))
        self.assertEqual(df7.chunks[0].inputs[0].shape, (2, 1))

        # test Series iloc getitem

        # slice
        series = md.Series(pd.Series(np.arange(10)), chunk_size=3).iloc[4:8]
        series = series.tiles()

        self.assertEqual(series.shape, (4,))

        self.assertEqual(len(series.chunks), 2)
        self.assertEqual(series.chunks[0].shape, (2,))
        self.assertEqual(series.chunks[0].index, (0,))
        self.assertEqual(series.chunks[0].op.indexes, slice(1, 3, 1))
        self.assertEqual(series.chunks[1].shape, (2,))
        self.assertEqual(series.chunks[1].op.indexes, slice(0, 2, 1))
        self.assertEqual(series.chunks[1].index, (1,))

        # fancy index
        series = md.Series(pd.Series(np.arange(10)), chunk_size=3).iloc[[2, 4, 8]]
        series = series.tiles()

        self.assertEqual(series.shape, (3,))

        self.assertEqual(len(series.chunks), 3)
        self.assertEqual(series.chunks[0].shape, (1,))
        self.assertEqual(series.chunks[0].index, (0,))
        self.assertEqual(series.chunks[0].op.indexes[0], [2])
        self.assertEqual(series.chunks[1].shape, (1,))
        self.assertEqual(series.chunks[1].op.indexes[0], [1])
        self.assertEqual(series.chunks[1].index, (1,))
        self.assertEqual(series.chunks[2].shape, (1,))
        self.assertEqual(series.chunks[2].op.indexes[0], [2])
        self.assertEqual(series.chunks[2].index, (2,))

    def testILocSetItem(self):
        df1 = pd.DataFrame([[1,3,3], [4,2,6], [7, 8, 9]],
                            index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = md.DataFrame(df1, chunk_size=2)
        df2 = df2.tiles()

        # plain index
        df3 = md.DataFrame(df1, chunk_size=2)
        df3.iloc[1] = 100
        df3 = df3.tiles()
        self.assertIsInstance(df3.op, DataFrameIlocSetItem)
        self.assertEqual(df3.chunk_shape, df2.chunk_shape)
        pd.testing.assert_index_equal(df2.index_value.to_pandas(), df3.index_value.to_pandas())
        pd.testing.assert_index_equal(df2.columns_value.to_pandas(), df3.columns_value.to_pandas())
        for c1, c2 in zip(df2.chunks, df3.chunks):
            self.assertEqual(c1.shape, c2.shape)
            pd.testing.assert_index_equal(c1.index_value.to_pandas(), c2.index_value.to_pandas())
            pd.testing.assert_index_equal(c1.columns_value.to_pandas(), c2.columns_value.to_pandas())
            if isinstance(c2.op, DataFrameIlocSetItem):
                self.assertEqual(c1.key, c2.inputs[0].key)
            else:
                self.assertEqual(c1.key, c2.key)
        self.assertEqual(df3.chunks[0].op.indexes, (1, slice(None, None, None)))
        self.assertEqual(df3.chunks[1].op.indexes, (1, slice(None, None, None)))

        # # slice index
        df4 = md.DataFrame(df1, chunk_size=2)
        df4.iloc[:, 2:4] = 1111
        df4 = df4.tiles()
        self.assertIsInstance(df4.op, DataFrameIlocSetItem)
        self.assertEqual(df4.chunk_shape, df2.chunk_shape)
        pd.testing.assert_index_equal(df2.index_value.to_pandas(), df4.index_value.to_pandas())
        pd.testing.assert_index_equal(df2.columns_value.to_pandas(), df4.columns_value.to_pandas())
        for c1, c2 in zip(df2.chunks, df4.chunks):
            self.assertEqual(c1.shape, c2.shape)
            pd.testing.assert_index_equal(c1.index_value.to_pandas(), c2.index_value.to_pandas())
            pd.testing.assert_index_equal(c1.columns_value.to_pandas(), c2.columns_value.to_pandas())
            if isinstance(c2.op, DataFrameIlocSetItem):
                self.assertEqual(c1.key, c2.inputs[0].key)
            else:
                self.assertEqual(c1.key, c2.key)
        self.assertEqual(df4.chunks[1].op.indexes, (slice(None, None, None), slice(None, None, None)))
        self.assertEqual(df4.chunks[3].op.indexes, (slice(None, None, None), slice(None, None, None)))

        # plain fancy index
        df5 = md.DataFrame(df1, chunk_size=2)
        df5.iloc[[0], [0, 1, 2]] = 2222
        df5 = df5.tiles()
        self.assertIsInstance(df5.op, DataFrameIlocSetItem)
        self.assertEqual(df5.chunk_shape, df2.chunk_shape)
        pd.testing.assert_index_equal(df2.index_value.to_pandas(), df5.index_value.to_pandas())
        pd.testing.assert_index_equal(df2.columns_value.to_pandas(), df5.columns_value.to_pandas())
        for c1, c2 in zip(df2.chunks, df5.chunks):
            self.assertEqual(c1.shape, c2.shape)
            pd.testing.assert_index_equal(c1.index_value.to_pandas(), c2.index_value.to_pandas())
            pd.testing.assert_index_equal(c1.columns_value.to_pandas(), c2.columns_value.to_pandas())
            if isinstance(c2.op, DataFrameIlocSetItem):
                self.assertEqual(c1.key, c2.inputs[0].key)
            else:
                self.assertEqual(c1.key, c2.key)
        np.testing.assert_array_equal(df5.chunks[0].op.indexes[0], [0])
        np.testing.assert_array_equal(df5.chunks[0].op.indexes[1], [0, 1])
        np.testing.assert_array_equal(df5.chunks[1].op.indexes[0], [0])
        np.testing.assert_array_equal(df5.chunks[1].op.indexes[1], [0])

        # fancy index
        df6 = md.DataFrame(df1, chunk_size=2)
        df6.iloc[[1, 2], [0, 1, 2]] = 3333
        df6 = df6.tiles()
        self.assertIsInstance(df6.op, DataFrameIlocSetItem)
        self.assertEqual(df6.chunk_shape, df2.chunk_shape)
        pd.testing.assert_index_equal(df2.index_value.to_pandas(), df6.index_value.to_pandas())
        pd.testing.assert_index_equal(df2.columns_value.to_pandas(), df6.columns_value.to_pandas())
        for c1, c2 in zip(df2.chunks, df6.chunks):
            self.assertEqual(c1.shape, c2.shape)
            pd.testing.assert_index_equal(c1.index_value.to_pandas(), c2.index_value.to_pandas())
            pd.testing.assert_index_equal(c1.columns_value.to_pandas(), c2.columns_value.to_pandas())
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
        df7 = md.DataFrame(df1, chunk_size=2)
        df7.iloc[1, 2] = 4444
        df7 = df7.tiles()
        self.assertIsInstance(df7.op, DataFrameIlocSetItem)
        self.assertEqual(df7.chunk_shape, df2.chunk_shape)
        pd.testing.assert_index_equal(df2.index_value.to_pandas(), df7.index_value.to_pandas())
        pd.testing.assert_index_equal(df2.columns_value.to_pandas(), df7.columns_value.to_pandas())
        for c1, c2 in zip(df2.chunks, df7.chunks):
            self.assertEqual(c1.shape, c2.shape)
            pd.testing.assert_index_equal(c1.index_value.to_pandas(), c2.index_value.to_pandas())
            pd.testing.assert_index_equal(c1.columns_value.to_pandas(), c2.columns_value.to_pandas())
            if isinstance(c2.op, DataFrameIlocSetItem):
                self.assertEqual(c1.key, c2.inputs[0].key)
            else:
                self.assertEqual(c1.key, c2.key)
        self.assertEqual(df7.chunks[1].op.indexes, (1, 0))

        # test Series

        # slice
        series = md.Series(pd.Series(np.arange(10)), chunk_size=3)
        series.iloc[:4] = 2
        series = series.tiles()

        self.assertEqual(series.shape, (10,))
        self.assertEqual(len(series.chunks), 4)

        self.assertEqual(series.chunks[0].op.indexes, slice(None, None, None))
        self.assertEqual(series.chunks[0].op.value, 2)
        self.assertEqual(series.chunks[1].op.indexes, slice(0, 1, 1))
        self.assertEqual(series.chunks[1].op.value, 2)

        # fancy index
        series = md.Series(pd.Series(np.arange(10)), chunk_size=3)
        series.iloc[[2, 4, 9]] = 3
        series = series.tiles()

        self.assertEqual(series.shape, (10,))

        self.assertEqual(len(series.chunks), 4)
        self.assertEqual(series.chunks[0].index, (0,))
        self.assertEqual(series.chunks[0].op.indexes, [2])
        self.assertEqual(series.chunks[0].op.value, 3)
        self.assertEqual(series.chunks[1].index, (1,))
        self.assertEqual(series.chunks[1].op.indexes, [1])
        self.assertEqual(series.chunks[1].op.value, 3)
        self.assertEqual(series.chunks[3].index, (3,))
        self.assertEqual(series.chunks[3].op.indexes, [0])
        self.assertEqual(series.chunks[3].op.value, 3)

    def testDataFrameGetitem(self):
        data = pd.DataFrame(np.random.rand(10, 5), columns=['c1', 'c2', 'c3', 'c4', 'c5'])
        df = md.DataFrame(data, chunk_size=2)

        series = df['c3']
        self.assertIsInstance(series, Series)
        self.assertEqual(series.shape, (10,))
        self.assertEqual(series.name, 'c3')
        self.assertEqual(series.dtype, data['c3'].dtype)
        self.assertEqual(series.index_value, df.index_value)

        series = series.tiles()
        self.assertEqual(series.nsplits, ((2, 2, 2, 2, 2),))
        self.assertEqual(len(series.chunks), 5)
        for i, c in enumerate(series.chunks):
            self.assertIsInstance(c, SERIES_CHUNK_TYPE)
            self.assertEqual(c.index, (i,))
            self.assertEqual(c.shape, (2,))

        df1 = df[['c1', 'c2', 'c3']]
        self.assertIsInstance(df1, DataFrame)
        self.assertEqual(df1.shape, (10, 3))
        self.assertEqual(df1.index_value, df.index_value)
        pd.testing.assert_index_equal(df1.columns_value.to_pandas(), data[['c1', 'c2', 'c3']].columns)
        pd.testing.assert_series_equal(df1.dtypes, data[['c1', 'c2', 'c3']].dtypes)

        df1 = df1.tiles()
        self.assertEqual(df1.nsplits, ((2, 2, 2, 2, 2), (2, 1)))
        self.assertEqual(len(df1.chunks), 10)
        for i, c in enumerate(df1.chunks[slice(0, 10, 2)]):
            self.assertIsInstance(c, DATAFRAME_CHUNK_TYPE)
            self.assertEqual(c.index, (i, 0))
            self.assertEqual(c.shape, (2, 2))
        for i, c in enumerate(df1.chunks[slice(1, 10, 2)]):
            self.assertIsInstance(c, DATAFRAME_CHUNK_TYPE)
            self.assertEqual(c.index, (i, 1))
            self.assertEqual(c.shape, (2, 1))

    def testDataFrameGetitemBool(self):
        data = pd.DataFrame(np.random.rand(10, 5), columns=['c1', 'c2', 'c3', 'c4', 'c5'])
        df = md.DataFrame(data, chunk_size=2)

        mask_data1 = data.c1 > 0.5
        mask_data2 = data.c1 < 0.5
        mask1 = md.Series(mask_data1, chunk_size=2)
        mask2 = md.Series(mask_data2, chunk_size=2)

        r1 = df[mask1]
        r2 = df[mask2]
        r3 = df[mask1]

        self.assertNotEqual(r1.index_value.key, df.index_value.key)
        self.assertNotEqual(r1.index_value.key, mask1.index_value.key)
        self.assertEqual(r1.columns_value.key, df.columns_value.key)
        self.assertIs(r1.columns_value, df.columns_value)

        self.assertNotEqual(r1.index_value.key, r2.index_value.key)
        self.assertEqual(r1.columns_value.key, r2.columns_value.key)
        self.assertIs(r1.columns_value, r2.columns_value)

        self.assertEqual(r1.index_value.key, r3.index_value.key)
        self.assertEqual(r1.columns_value.key, r3.columns_value.key)
        self.assertIs(r1.columns_value, r3.columns_value)

    def testSeriesGetitem(self):
        data = pd.Series(np.random.rand(10,), name='a')
        series = md.Series(data, chunk_size=3)

        result1 = series[2]
        self.assertEqual(result1.shape, ())

        result1 = result1.tiles()
        self.assertEqual(result1.nsplits, ())
        self.assertEqual(len(result1.chunks), 1)
        self.assertIsInstance(result1.chunks[0], TENSOR_CHUNK_TYPE)
        self.assertEqual(result1.chunks[0].shape, ())
        self.assertEqual(result1.chunks[0].dtype, data.dtype)

        result2 = series[[4, 5, 1, 2, 3]]
        self.assertEqual(result2.shape, (5,))

        result2 = result2.tiles()
        self.assertEqual(result2.nsplits, ((2, 2, 1),))
        self.assertEqual(len(result2.chunks), 3)
        self.assertEqual(result2.chunks[0].op.labels, [4, 5])
        self.assertEqual(result2.chunks[1].op.labels, [1, 2])
        self.assertEqual(result2.chunks[2].op.labels, [3])

        data = pd.Series(np.random.rand(10), index=['i' + str(i) for i in range(10)])
        series = md.Series(data, chunk_size=3)

        result1 = series['i2']
        self.assertEqual(result1.shape, ())

        result1 = result1.tiles()
        self.assertEqual(result1.nsplits, ())
        self.assertEqual(result1.chunks[0].dtype, data.dtype)
        self.assertTrue(result1.chunks[0].op.labels, ['i2'])

        result2 = series[['i2', 'i4']]
        self.assertEqual(result2.shape, (2,))

        result2 = result2.tiles()
        self.assertEqual(result2.nsplits, ((2,),))
        self.assertEqual(result2.chunks[0].dtype, data.dtype)
        self.assertTrue(result2.chunks[0].op.labels, [['i2', 'i4']])
