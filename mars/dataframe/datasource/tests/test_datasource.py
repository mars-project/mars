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

import os
import tempfile
import shutil
from collections import OrderedDict
from weakref import ReferenceType

import numpy as np
import pandas as pd

import mars.tensor as mt
from mars import opcodes as OperandDef
from mars.graph import DAG
from mars.tests.core import TestBase
from mars.dataframe.core import IndexValue, DataFrameChunk, DatetimeIndex, Int64Index, Float64Index
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.datasource.index import from_pandas as from_pandas_index, from_tileable
from mars.dataframe.datasource.from_tensor import dataframe_from_tensor, \
    series_from_tensor, dataframe_from_1d_tileables
from mars.dataframe.datasource.from_records import from_records
from mars.dataframe.datasource.read_csv import read_csv, DataFrameReadCSV
from mars.dataframe.datasource.read_sql import read_sql_table, read_sql_query, DataFrameReadSQL
from mars.dataframe.datasource.date_range import date_range


class Test(TestBase):
    def testChunkSerialize(self):
        data = pd.DataFrame(np.random.rand(10, 10), index=np.random.randint(-100, 100, size=(10,)),
                            columns=[np.random.bytes(10) for _ in range(10)])
        df = from_pandas_df(data).tiles()

        # pb
        chunk = df.chunks[0]
        serials = self._pb_serial(chunk)
        op, pb = serials[chunk.op, chunk.data]

        self.assertEqual(tuple(pb.index), chunk.index)
        self.assertEqual(pb.key, chunk.key)
        self.assertEqual(tuple(pb.shape), chunk.shape)
        self.assertEqual(int(op.type.split('.', 1)[1]), OperandDef.DATAFRAME_DATA_SOURCE)

        chunk2 = self._pb_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        pd.testing.assert_index_equal(chunk2.index_value.to_pandas(), chunk.index_value.to_pandas())
        pd.testing.assert_index_equal(chunk2.columns_value.to_pandas(), chunk.columns_value.to_pandas())

        # json
        chunk = df.chunks[0]
        serials = self._json_serial(chunk)

        chunk2 = self._json_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        pd.testing.assert_index_equal(chunk2.index_value.to_pandas(), chunk.index_value.to_pandas())
        pd.testing.assert_index_equal(chunk2.columns_value.to_pandas(), chunk.columns_value.to_pandas())

    def testDataFrameGraphSerialize(self):
        df = from_pandas_df(pd.DataFrame(np.random.rand(10, 10),
                                         columns=pd.timedelta_range(start='1 day', periods=10),
                                         index=pd.date_range('2020-1-1', periods=10)))
        graph = df.build_graph(tiled=False)

        pb = graph.to_pb()
        graph2 = DAG.from_pb(pb)
        self.assertEqual(len(graph), len(graph2))
        t = next(iter(graph))
        t2 = next(iter(graph2))
        self.assertTrue(t2.op.outputs[0], ReferenceType)  # make sure outputs are all weak reference
        self.assertBaseEqual(t.op, t2.op)
        self.assertEqual(t.shape, t2.shape)
        self.assertEqual(sorted(i.key for i in t.inputs), sorted(i.key for i in t2.inputs))
        pd.testing.assert_index_equal(t2.index_value.to_pandas(), t.index_value.to_pandas())
        pd.testing.assert_index_equal(t2.columns_value.to_pandas(), t.columns_value.to_pandas())

        jsn = graph.to_json()
        graph2 = DAG.from_json(jsn)
        self.assertEqual(len(graph), len(graph2))
        t = next(iter(graph))
        t2 = next(iter(graph2))
        self.assertTrue(t2.op.outputs[0], ReferenceType)  # make sure outputs are all weak reference
        self.assertBaseEqual(t.op, t2.op)
        self.assertEqual(t.shape, t2.shape)
        self.assertEqual(sorted(i.key for i in t.inputs), sorted(i.key for i in t2.inputs))
        pd.testing.assert_index_equal(t2.index_value.to_pandas(), t.index_value.to_pandas())
        pd.testing.assert_index_equal(t2.columns_value.to_pandas(), t.columns_value.to_pandas())

        # test graph with tiled DataFrame
        t2 = from_pandas_df(pd.DataFrame(np.random.rand(10, 10)), chunk_size=(5, 4)).tiles()
        graph = DAG()
        graph.add_node(t2)

        pb = graph.to_pb()
        graph2 = DAG.from_pb(pb)
        self.assertEqual(len(graph), len(graph2))
        chunks = next(iter(graph2)).chunks
        self.assertEqual(len(chunks), 6)
        self.assertIsInstance(chunks[0], DataFrameChunk)
        self.assertEqual(chunks[0].index, t2.chunks[0].index)
        self.assertBaseEqual(chunks[0].op, t2.chunks[0].op)
        pd.testing.assert_index_equal(chunks[0].index_value.to_pandas(), t2.chunks[0].index_value.to_pandas())
        pd.testing.assert_index_equal(chunks[0].columns_value.to_pandas(), t2.chunks[0].columns_value.to_pandas())

        jsn = graph.to_json()
        graph2 = DAG.from_json(jsn)
        self.assertEqual(len(graph), len(graph2))
        chunks = next(iter(graph2)).chunks
        self.assertEqual(len(chunks), 6)
        self.assertIsInstance(chunks[0], DataFrameChunk)
        self.assertEqual(chunks[0].index, t2.chunks[0].index)
        self.assertBaseEqual(chunks[0].op, t2.chunks[0].op)
        pd.testing.assert_index_equal(chunks[0].index_value.to_pandas(), t2.chunks[0].index_value.to_pandas())
        pd.testing.assert_index_equal(chunks[0].columns_value.to_pandas(), t2.chunks[0].columns_value.to_pandas())

    def testFromPandasDataFrame(self):
        data = pd.DataFrame(np.random.rand(10, 10), columns=['c' + str(i) for i in range(10)])
        df = from_pandas_df(data, chunk_size=4)

        pd.testing.assert_series_equal(df.op.dtypes, data.dtypes)
        self.assertIsInstance(df.index_value._index_value, IndexValue.RangeIndex)
        self.assertEqual(df.index_value._index_value._slice, slice(0, 10, 1))
        self.assertTrue(df.index_value.is_monotonic_increasing)
        self.assertFalse(df.index_value.is_monotonic_decreasing)
        self.assertTrue(df.index_value.is_unique)
        self.assertEqual(df.index_value.min_val, 0)
        self.assertEqual(df.index_value.max_val, 9)
        np.testing.assert_equal(df.columns_value._index_value._data, data.columns.values)

        df = df.tiles()

        self.assertEqual(len(df.chunks), 9)
        pd.testing.assert_frame_equal(df.chunks[0].op.data, df.op.data.iloc[:4, :4])
        self.assertEqual(df.chunks[0].index_value._index_value._slice, slice(0, 4, 1))
        self.assertTrue(df.chunks[0].index_value._index_value._is_monotonic_increasing)
        self.assertFalse(df.chunks[0].index_value._index_value._is_monotonic_decreasing)
        self.assertTrue(df.chunks[0].index_value._index_value._is_unique)
        pd.testing.assert_frame_equal(df.chunks[1].op.data, df.op.data.iloc[:4, 4:8])
        self.assertEqual(df.chunks[1].index_value._index_value._slice, slice(0, 4, 1))
        self.assertTrue(df.chunks[1].index_value._index_value._is_monotonic_increasing)
        self.assertFalse(df.chunks[1].index_value._index_value._is_monotonic_decreasing)
        self.assertTrue(df.chunks[1].index_value._index_value._is_unique)
        pd.testing.assert_frame_equal(df.chunks[2].op.data, df.op.data.iloc[:4, 8:])
        self.assertEqual(df.chunks[2].index_value._index_value._slice, slice(0, 4, 1))
        self.assertTrue(df.chunks[2].index_value._index_value._is_monotonic_increasing)
        self.assertFalse(df.chunks[2].index_value._index_value._is_monotonic_decreasing)
        self.assertTrue(df.chunks[2].index_value._index_value._is_unique)
        pd.testing.assert_frame_equal(df.chunks[3].op.data, df.op.data.iloc[4:8, :4])
        self.assertEqual(df.chunks[3].index_value._index_value._slice, slice(4, 8, 1))
        self.assertTrue(df.chunks[3].index_value._index_value._is_monotonic_increasing)
        self.assertFalse(df.chunks[3].index_value._index_value._is_monotonic_decreasing)
        self.assertTrue(df.chunks[3].index_value._index_value._is_unique)
        pd.testing.assert_frame_equal(df.chunks[4].op.data, df.op.data.iloc[4:8, 4:8])
        self.assertEqual(df.chunks[4].index_value._index_value._slice, slice(4, 8, 1))
        self.assertTrue(df.chunks[4].index_value._index_value._is_monotonic_increasing)
        self.assertFalse(df.chunks[4].index_value._index_value._is_monotonic_decreasing)
        self.assertTrue(df.chunks[4].index_value._index_value._is_unique)
        pd.testing.assert_frame_equal(df.chunks[5].op.data, df.op.data.iloc[4:8, 8:])
        self.assertEqual(df.chunks[5].index_value._index_value._slice, slice(4, 8, 1))
        self.assertTrue(df.chunks[5].index_value._index_value._is_monotonic_increasing)
        self.assertFalse(df.chunks[5].index_value._index_value._is_monotonic_decreasing)
        self.assertTrue(df.chunks[5].index_value._index_value._is_unique)
        pd.testing.assert_frame_equal(df.chunks[6].op.data, df.op.data.iloc[8:, :4])
        self.assertEqual(df.chunks[6].index_value._index_value._slice, slice(8, 10, 1))
        self.assertTrue(df.chunks[6].index_value._index_value._is_monotonic_increasing)
        self.assertFalse(df.chunks[6].index_value._index_value._is_monotonic_decreasing)
        self.assertTrue(df.chunks[6].index_value._index_value._is_unique)
        pd.testing.assert_frame_equal(df.chunks[7].op.data, df.op.data.iloc[8:, 4:8])
        self.assertEqual(df.chunks[7].index_value._index_value._slice, slice(8, 10, 1))
        self.assertTrue(df.chunks[7].index_value._index_value._is_monotonic_increasing)
        self.assertFalse(df.chunks[7].index_value._index_value._is_monotonic_decreasing)
        self.assertTrue(df.chunks[7].index_value._index_value._is_unique)
        pd.testing.assert_frame_equal(df.chunks[8].op.data, df.op.data.iloc[8:, 8:])
        self.assertEqual(df.chunks[8].index_value._index_value._slice, slice(8, 10, 1))
        self.assertTrue(df.chunks[8].index_value._index_value._is_monotonic_increasing)
        self.assertFalse(df.chunks[8].index_value._index_value._is_monotonic_decreasing)
        self.assertTrue(df.chunks[8].index_value._index_value._is_unique)

        data2 = data[::2]
        df2 = from_pandas_df(data2, chunk_size=4)

        pd.testing.assert_series_equal(df.op.dtypes, data2.dtypes)
        self.assertIsInstance(df2.index_value._index_value, IndexValue.RangeIndex)
        self.assertEqual(df2.index_value._index_value._slice, slice(0, 10, 2))

        df2 = df2.tiles()

        self.assertEqual(len(df2.chunks), 6)
        pd.testing.assert_frame_equal(df2.chunks[0].op.data, df2.op.data.iloc[:4, :4])
        self.assertEqual(df2.chunks[0].index_value._index_value._slice, slice(0, 8, 2))
        pd.testing.assert_frame_equal(df2.chunks[1].op.data, df2.op.data.iloc[:4, 4:8])
        self.assertEqual(df2.chunks[1].index_value._index_value._slice, slice(0, 8, 2))
        pd.testing.assert_frame_equal(df2.chunks[2].op.data, df2.op.data.iloc[:4, 8:])
        self.assertEqual(df2.chunks[2].index_value._index_value._slice, slice(0, 8, 2))
        pd.testing.assert_frame_equal(df2.chunks[3].op.data, df2.op.data.iloc[4:, :4])
        self.assertEqual(df2.chunks[3].index_value._index_value._slice, slice(8, 10, 2))
        pd.testing.assert_frame_equal(df2.chunks[4].op.data, df2.op.data.iloc[4:, 4:8])
        self.assertEqual(df2.chunks[3].index_value._index_value._slice, slice(8, 10, 2))
        pd.testing.assert_frame_equal(df2.chunks[5].op.data, df2.op.data.iloc[4:, 8:])
        self.assertEqual(df2.chunks[3].index_value._index_value._slice, slice(8, 10, 2))

    def testFromPandasSeries(self):
        data = pd.Series(np.random.rand(10), name='a')
        series = from_pandas_series(data, chunk_size=4)

        self.assertEqual(series.name, data.name)
        self.assertIsInstance(series.index_value._index_value, IndexValue.RangeIndex)
        self.assertEqual(series.index_value._index_value._slice, slice(0, 10, 1))
        self.assertTrue(series.index_value.is_monotonic_increasing)
        self.assertFalse(series.index_value.is_monotonic_decreasing)
        self.assertTrue(series.index_value.is_unique)
        self.assertEqual(series.index_value.min_val, 0)
        self.assertEqual(series.index_value.max_val, 9)

        series = series.tiles()

        self.assertEqual(len(series.chunks), 3)
        pd.testing.assert_series_equal(series.chunks[0].op.data, series.op.data.iloc[:4])
        self.assertEqual(series.chunks[0].index_value._index_value._slice, slice(0, 4, 1))
        self.assertTrue(series.chunks[0].index_value._index_value._is_monotonic_increasing)
        self.assertFalse(series.chunks[0].index_value._index_value._is_monotonic_decreasing)
        self.assertTrue(series.chunks[0].index_value._index_value._is_unique)
        pd.testing.assert_series_equal(series.chunks[1].op.data, series.op.data.iloc[4:8])
        self.assertEqual(series.chunks[1].index_value._index_value._slice, slice(4, 8, 1))
        self.assertTrue(series.chunks[1].index_value._index_value._is_monotonic_increasing)
        self.assertFalse(series.chunks[1].index_value._index_value._is_monotonic_decreasing)
        self.assertTrue(series.chunks[1].index_value._index_value._is_unique)
        pd.testing.assert_series_equal(series.chunks[2].op.data, series.op.data.iloc[8:])
        self.assertEqual(series.chunks[2].index_value._index_value._slice, slice(8, 10, 1))
        self.assertTrue(series.chunks[2].index_value._index_value._is_monotonic_increasing)
        self.assertFalse(series.chunks[2].index_value._index_value._is_monotonic_decreasing)
        self.assertTrue(series.chunks[2].index_value._index_value._is_unique)

    def testFromPandasIndex(self):
        data = pd.date_range('2020-1-1', periods=10, name='date')
        index = from_pandas_index(data, chunk_size=4)

        self.assertIsInstance(index, DatetimeIndex)
        self.assertEqual(index.name, data.name)
        self.assertEqual(index.dtype, data.dtype)
        self.assertIsInstance(index.index_value.value, IndexValue.DatetimeIndex)

        index = index.tiles()

        for i, c in enumerate(index.chunks):
            self.assertEqual(c.name, data.name)
            pd.testing.assert_index_equal(c.op.data, data[i * 4: (i + 1) * 4])
            self.assertEqual(c.dtype, data.dtype)
            self.assertIsInstance(c.index_value.value, IndexValue.DatetimeIndex)

    def testFromTileableIndex(self):
        t = mt.random.rand(10, 4)

        with self.assertRaises(ValueError):
            from_tileable(t)

        pd_df = pd.DataFrame(np.random.rand(10, 4),
                             index=np.arange(10, 0, -1).astype(np.int64))
        pd_df.index.name = 'ind'
        df = from_pandas_df(pd_df, chunk_size=6)

        for o in [df, df[0]]:
            index = o.index
            self.assertIsInstance(index, Int64Index)
            self.assertEqual(index.dtype, np.int64)
            self.assertEqual(index.name, pd_df.index.name)
            self.assertIsInstance(index.index_value.value, IndexValue.Int64Index)

            index = index.tiles()

            self.assertEqual(len(index.chunks), 2)
            for c in index.chunks:
                self.assertEqual(c.dtype, np.int64)
                self.assertEqual(c.name, pd_df.index.name)
                self.assertIsInstance(c.index_value.value, IndexValue.Int64Index)

        t = mt.random.rand(10, chunk_size=6)
        index = from_tileable(t, name='new_name')

        self.assertIsInstance(index, Float64Index)
        self.assertEqual(index.dtype, np.float64)
        self.assertEqual(index.name, 'new_name')
        self.assertIsInstance(index.index_value.value, IndexValue.Float64Index)

        index = index.tiles()

        self.assertEqual(len(index.chunks), 2)
        for c in index.chunks:
            self.assertEqual(c.dtype, np.float64)
            self.assertEqual(c.name, 'new_name')
            self.assertIsInstance(c.index_value.value, IndexValue.Float64Index)

    def testFromTensorSerialize(self):
        # test serialization and deserialization
        # pb
        tensor = mt.random.rand(10, 10)
        df = dataframe_from_tensor(tensor)
        df = df.tiles()
        chunk = df.chunks[0]
        serials = self._pb_serial(chunk)
        op, pb = serials[chunk.op, chunk.data]

        self.assertEqual(tuple(pb.index), chunk.index)
        self.assertEqual(pb.key, chunk.key)
        self.assertEqual(tuple(pb.shape), chunk.shape)
        self.assertEqual(int(op.type.split('.', 1)[1]), OperandDef.DATAFRAME_FROM_TENSOR)

        chunk2 = self._pb_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        pd.testing.assert_index_equal(chunk2.index_value.to_pandas(), chunk.index_value.to_pandas())
        pd.testing.assert_index_equal(chunk2.columns_value.to_pandas(), chunk.columns_value.to_pandas())

        # json
        chunk = df.chunks[0]
        serials = self._json_serial(chunk)

        chunk2 = self._json_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        pd.testing.assert_index_equal(chunk2.index_value.to_pandas(), chunk.index_value.to_pandas())
        pd.testing.assert_index_equal(chunk2.columns_value.to_pandas(), chunk.columns_value.to_pandas())

    def testFromTensor(self):
        tensor = mt.random.rand(10, 10, chunk_size=5)
        df = dataframe_from_tensor(tensor)
        self.assertIsInstance(df.index_value._index_value, IndexValue.RangeIndex)
        self.assertEqual(df.op.dtypes[0], tensor.dtype, 'DataFrame converted from tensor have the wrong dtype')

        df = df.tiles()
        self.assertEqual(len(df.chunks), 4)
        self.assertIsInstance(df.chunks[0].index_value._index_value, IndexValue.RangeIndex)
        self.assertIsInstance(df.chunks[0].index_value, IndexValue)

        # test converted from 1-d tensor
        tensor2 = mt.array([1, 2, 3])
        # in fact, tensor3 is (3,1)
        tensor3 = mt.array([tensor2]).T

        df2 = dataframe_from_tensor(tensor2)
        df3 = dataframe_from_tensor(tensor3)
        df2 = df2.tiles()
        df3 = df3.tiles()
        np.testing.assert_equal(df2.chunks[0].index, (0, 0))
        np.testing.assert_equal(df3.chunks[0].index, (0, 0))

        # test converted from scalar
        scalar = mt.array(1)
        np.testing.assert_equal(scalar.ndim, 0)
        with self.assertRaises(TypeError):
            dataframe_from_tensor(scalar)

        # from tensor with given index
        df = dataframe_from_tensor(tensor, index=np.arange(0, 20, 2))
        df = df.tiles()
        pd.testing.assert_index_equal(df.chunks[0].index_value.to_pandas(), pd.Index(np.arange(0, 10, 2)))
        pd.testing.assert_index_equal(df.chunks[1].index_value.to_pandas(), pd.Index(np.arange(0, 10, 2)))
        pd.testing.assert_index_equal(df.chunks[2].index_value.to_pandas(), pd.Index(np.arange(10, 20, 2)))
        pd.testing.assert_index_equal(df.chunks[3].index_value.to_pandas(), pd.Index(np.arange(10, 20, 2)))

        # from tensor with index that is a tensor as well
        df = dataframe_from_tensor(tensor, index=mt.arange(0, 20, 2))
        df = df.tiles()
        self.assertEqual(len(df.chunks[0].inputs), 2)
        self.assertFalse(df.chunks[0].index_value.has_value())

        # from tensor with given columns
        df = dataframe_from_tensor(tensor, columns=list('abcdefghij'))
        df = df.tiles()
        pd.testing.assert_index_equal(df.dtypes.index, pd.Index(list('abcdefghij')))
        pd.testing.assert_index_equal(df.chunks[0].columns_value.to_pandas(), pd.Index(['a', 'b', 'c', 'd', 'e']))
        pd.testing.assert_index_equal(df.chunks[0].dtypes.index, pd.Index(['a', 'b', 'c', 'd', 'e']))
        pd.testing.assert_index_equal(df.chunks[1].columns_value.to_pandas(), pd.Index(['f', 'g', 'h', 'i', 'j']))
        pd.testing.assert_index_equal(df.chunks[1].dtypes.index, pd.Index(['f', 'g', 'h', 'i', 'j']))
        pd.testing.assert_index_equal(df.chunks[2].columns_value.to_pandas(), pd.Index(['a', 'b', 'c', 'd', 'e']))
        pd.testing.assert_index_equal(df.chunks[2].dtypes.index, pd.Index(['a', 'b', 'c', 'd', 'e']))
        pd.testing.assert_index_equal(df.chunks[3].columns_value.to_pandas(), pd.Index(['f', 'g', 'h', 'i', 'j']))
        pd.testing.assert_index_equal(df.chunks[3].dtypes.index, pd.Index(['f', 'g', 'h', 'i', 'j']))

        # test series from tensor
        tensor = mt.random.rand(10, chunk_size=4)
        series = series_from_tensor(tensor, name='a')

        self.assertEqual(series.dtype, tensor.dtype)
        self.assertEqual(series.name, 'a')
        pd.testing.assert_index_equal(series.index_value.to_pandas(), pd.RangeIndex(10))

        series = series.tiles()
        self.assertEqual(len(series.chunks), 3)
        pd.testing.assert_index_equal(series.chunks[0].index_value.to_pandas(), pd.RangeIndex(0, 4))
        self.assertEqual(series.chunks[0].name, 'a')
        pd.testing.assert_index_equal(series.chunks[1].index_value.to_pandas(), pd.RangeIndex(4, 8))
        self.assertEqual(series.chunks[1].name, 'a')
        pd.testing.assert_index_equal(series.chunks[2].index_value.to_pandas(), pd.RangeIndex(8, 10))
        self.assertEqual(series.chunks[2].name, 'a')

        d = OrderedDict([(0, mt.tensor(np.random.rand(4))),
                         (1, mt.tensor(np.random.rand(4)))])
        df = dataframe_from_1d_tileables(d)
        pd.testing.assert_index_equal(df.columns_value.to_pandas(), pd.RangeIndex(2))

        df = df.tiles()

        pd.testing.assert_index_equal(df.chunks[0].index_value.to_pandas(), pd.RangeIndex(4))

        series = series_from_tensor(mt.random.rand(4))
        pd.testing.assert_index_equal(series.index_value.to_pandas(), pd.RangeIndex(4))

        series = series_from_tensor(mt.random.rand(4), index=[1, 2, 3])
        pd.testing.assert_index_equal(series.index_value.to_pandas(), pd.Index([1, 2, 3]))

        series = series_from_tensor(mt.random.rand(4), index=pd.Index([1, 2, 3], name='my_index'))
        pd.testing.assert_index_equal(series.index_value.to_pandas(), pd.Index([1, 2, 3], name='my_index'))
        self.assertEqual(series.index_value.name, 'my_index')

        with self.assertRaises(TypeError):
            series_from_tensor(mt.ones((10, 10)))

        # index has wrong shape
        with self.assertRaises(ValueError):
            dataframe_from_tensor(mt.random.rand(4, 3), index=mt.random.rand(5))

        # columns have wrong shape
        with self.assertRaises(ValueError):
            dataframe_from_tensor(mt.random.rand(4, 3), columns=['a', 'b'])

        # index should be 1-d
        with self.assertRaises(ValueError):
            dataframe_from_tensor(mt.tensor(np.random.rand(3, 2)),
                                  index=mt.tensor(np.random.rand(3, 2)))

        # 1-d tensors should have same shape
        with self.assertRaises(ValueError):
            dataframe_from_1d_tileables(OrderedDict([(0, mt.tensor(np.random.rand(3))),
                                                     (1, mt.tensor(np.random.rand(2)))]))

        # index has wrong shape
        with self.assertRaises(ValueError):
            dataframe_from_1d_tileables({0: mt.tensor(np.random.rand(3))},
                                        index=mt.tensor(np.random.rand(2)))

        # columns have wrong shape
        with self.assertRaises(ValueError):
            dataframe_from_1d_tileables({0: mt.tensor(np.random.rand(3))},
                                        columns=['a', 'b'])

        # index should be 1-d
        with self.assertRaises(ValueError):
            series_from_tensor(mt.random.rand(4), index=mt.random.rand(4, 3))

    def testFromRecords(self):
        dtype = np.dtype([('x', 'int'), ('y', 'double'), ('z', '<U16')])

        tensor = mt.ones((10,), dtype=dtype, chunk_size=3)
        df = from_records(tensor)
        df = df.tiles()

        self.assertEqual(df.chunk_shape, (4, 1))
        self.assertEqual(df.chunks[0].shape, (3, 3))
        self.assertEqual(df.chunks[1].shape, (3, 3))
        self.assertEqual(df.chunks[2].shape, (3, 3))
        self.assertEqual(df.chunks[3].shape, (1, 3))

        self.assertEqual(df.chunks[0].inputs[0].shape, (3,))
        self.assertEqual(df.chunks[1].inputs[0].shape, (3,))
        self.assertEqual(df.chunks[2].inputs[0].shape, (3,))
        self.assertEqual(df.chunks[3].inputs[0].shape, (1,))

        self.assertEqual(df.chunks[0].op.extra_params, {'begin_index': 0, 'end_index': 3})
        self.assertEqual(df.chunks[1].op.extra_params, {'begin_index': 3, 'end_index': 6})
        self.assertEqual(df.chunks[2].op.extra_params, {'begin_index': 6, 'end_index': 9})
        self.assertEqual(df.chunks[3].op.extra_params, {'begin_index': 9, 'end_index': 10})

        names = pd.Index(['x', 'y', 'z'])
        dtypes = pd.Series({'x': np.dtype('int'), 'y': np.dtype('double'), 'z': np.dtype('<U16')})
        for chunk in df.chunks:
            pd.testing.assert_index_equal(chunk.columns_value.to_pandas(), names)
            pd.testing.assert_series_equal(chunk.dtypes, dtypes)

        pd.testing.assert_index_equal(df.chunks[0].index_value.to_pandas(), pd.RangeIndex(0, 3))
        pd.testing.assert_index_equal(df.chunks[1].index_value.to_pandas(), pd.RangeIndex(3, 6))
        pd.testing.assert_index_equal(df.chunks[2].index_value.to_pandas(), pd.RangeIndex(6, 9))
        pd.testing.assert_index_equal(df.chunks[3].index_value.to_pandas(), pd.RangeIndex(9, 10))

    def testReadCSV(self):
        tempdir = tempfile.mkdtemp()
        file_path = os.path.join(tempdir, 'test.csv')
        try:
            df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'], dtype=np.int64)
            df.to_csv(file_path)
            mdf = read_csv(file_path, index_col=0, chunk_bytes=10)
            self.assertIsInstance(mdf.op, DataFrameReadCSV)
            self.assertEqual(mdf.shape[1], 3)
            pd.testing.assert_index_equal(df.columns, mdf.columns_value.to_pandas())

            mdf = mdf.tiles()
            self.assertEqual(len(mdf.chunks), 4)
            index_keys = set()
            for chunk in mdf.chunks:
                index_keys.add(chunk.index_value.key)
                pd.testing.assert_index_equal(df.columns, chunk.columns_value.to_pandas())
                pd.testing.assert_series_equal(df.dtypes, chunk.dtypes)
            self.assertGreater(len(index_keys), 1)
        finally:
            shutil.rmtree(tempdir)

    def testReadSQL(self):
        test_df = pd.DataFrame({'a': np.arange(10).astype(np.int64, copy=False),
                                'b': ['s%d' % i for i in range(10)]})

        with tempfile.TemporaryDirectory() as d:
            table_name = 'test'
            uri = 'sqlite:///' + os.path.join(d, 'test.db')

            test_df.to_sql(table_name, uri, index=False)

            df = read_sql_table(table_name, uri, chunk_size=4)

            self.assertEqual(df.shape, test_df.shape)
            pd.testing.assert_index_equal(df.index_value.to_pandas(), test_df.index)
            pd.testing.assert_series_equal(df.dtypes, test_df.dtypes)

            df = df.tiles()
            self.assertEqual(df.nsplits, ((4, 4, 2), (2,)))
            for c in df.chunks:
                self.assertIsInstance(c.op, DataFrameReadSQL)
                self.assertIsNotNone(c.op.offset)

            with self.assertRaises(NotImplementedError):
                read_sql_table(table_name, uri, chunksize=4, index_col=b'a')
            with self.assertRaises(TypeError):
                read_sql_table(table_name, uri, chunk_size=4, index_col=b'a')
            with self.assertRaises(TypeError):
                read_sql_query('select * from ' + table_name, uri, partition_col='b')

    def testDateRange(self):
        with self.assertRaises(TypeError):
            _ = date_range('2020-1-1', periods='2')

        with self.assertRaises(ValueError):
            _ = date_range('2020-1-1', '2020-1-10', periods=10, freq='D')

        with self.assertRaises(ValueError):
            _ = date_range(pd.NaT, periods=10)

        expected = pd.date_range('2020-1-1', periods=9., name='date')

        dr = date_range('2020-1-1', periods=9., name='date', chunk_size=3)
        self.assertIsInstance(dr, DatetimeIndex)
        self.assertEqual(dr.shape, (9,))
        self.assertEqual(dr.dtype, expected.dtype)
        self.assertIsInstance(dr.index_value.value, IndexValue.DatetimeIndex)
        self.assertEqual(dr.index_value.min_val, expected.min())
        self.assertTrue(dr.index_value.min_val_close)
        self.assertEqual(dr.index_value.max_val, expected.max())
        self.assertTrue(dr.index_value.max_val_close)
        self.assertEqual(dr.index_value.is_unique, expected.is_unique)
        self.assertEqual(dr.index_value.is_monotonic_increasing,
                         expected.is_monotonic_increasing)
        self.assertEqual(dr.name, expected.name)

        dr = dr.tiles()

        for i, c in enumerate(dr.chunks):
            ec = expected[i * 3: (i + 1) * 3]
            self.assertEqual(c.shape, (3,))
            self.assertEqual(c.dtype, ec.dtype)
            self.assertIsInstance(c.index_value.value, IndexValue.DatetimeIndex)
            self.assertEqual(c.index_value.min_val, ec.min())
            self.assertTrue(c.index_value.min_val_close)
            self.assertEqual(c.index_value.max_val, ec.max())
            self.assertTrue(c.index_value.max_val_close)
            self.assertEqual(c.index_value.is_unique, ec.is_unique)
            self.assertEqual(c.index_value.is_monotonic_increasing,
                             ec.is_monotonic_increasing)
            self.assertEqual(c.name, ec.name)
