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

import random

import numpy as np
import pandas as pd

from mars import opcodes
from mars.config import options
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.base import to_gpu, to_cpu, df_reset_index, series_reset_index
from mars.dataframe.operands import ObjectType
from mars.operands import OperandStage
from mars.tests.core import TestBase
from mars.tiles import get_tiled


class Test(TestBase):
    def testToGPU(self):
        # test dataframe
        data = pd.DataFrame(np.random.rand(10, 10), index=np.random.randint(-100, 100, size=(10,)),
                            columns=[np.random.bytes(10) for _ in range(10)])
        df = from_pandas_df(data)
        cdf = to_gpu(df)

        self.assertEqual(df.index_value, cdf.index_value)
        self.assertEqual(df.columns_value, cdf.columns_value)
        self.assertTrue(cdf.op.gpu)
        pd.testing.assert_series_equal(df.dtypes, cdf.dtypes)

        cdf = cdf.tiles()
        df = get_tiled(df)

        self.assertEqual(df.nsplits, cdf.nsplits)
        self.assertEqual(df.chunks[0].index_value, cdf.chunks[0].index_value)
        self.assertEqual(df.chunks[0].columns_value, cdf.chunks[0].columns_value)
        self.assertTrue(cdf.chunks[0].op.gpu)
        pd.testing.assert_series_equal(df.chunks[0].dtypes, cdf.chunks[0].dtypes)

        self.assertIs(cdf, to_gpu(cdf))

        # test series
        sdata = data.iloc[:, 0]
        series = from_pandas_series(sdata)
        cseries = to_gpu(series)

        self.assertEqual(series.index_value, cseries.index_value)
        self.assertTrue(cseries.op.gpu)

        cseries = cseries.tiles()
        series = get_tiled(series)

        self.assertEqual(series.nsplits, cseries.nsplits)
        self.assertEqual(series.chunks[0].index_value, cseries.chunks[0].index_value)
        self.assertTrue(cseries.chunks[0].op.gpu)

        self.assertIs(cseries, to_gpu(cseries))

    def testToCPU(self):
        data = pd.DataFrame(np.random.rand(10, 10), index=np.random.randint(-100, 100, size=(10,)),
                            columns=[np.random.bytes(10) for _ in range(10)])
        df = from_pandas_df(data)
        cdf = to_gpu(df)
        df2 = to_cpu(cdf)

        self.assertEqual(df.index_value, df2.index_value)
        self.assertEqual(df.columns_value, df2.columns_value)
        self.assertFalse(df2.op.gpu)
        pd.testing.assert_series_equal(df.dtypes, df2.dtypes)

        df2 = df2.tiles()
        df = get_tiled(df)

        self.assertEqual(df.nsplits, df2.nsplits)
        self.assertEqual(df.chunks[0].index_value, df2.chunks[0].index_value)
        self.assertEqual(df.chunks[0].columns_value, df2.chunks[0].columns_value)
        self.assertFalse(df2.chunks[0].op.gpu)
        pd.testing.assert_series_equal(df.chunks[0].dtypes, df2.chunks[0].dtypes)

        self.assertIs(df2, to_cpu(df2))

    def testRechunk(self):
        raw = pd.DataFrame(np.random.rand(10, 10))
        df = from_pandas_df(raw, chunk_size=3)
        df2 = df.rechunk(4).tiles()

        self.assertEqual(df2.shape, (10, 10))
        self.assertEqual(len(df2.chunks), 9)

        self.assertEqual(df2.chunks[0].shape, (4, 4))
        pd.testing.assert_index_equal(df2.chunks[0].index_value.to_pandas(), pd.RangeIndex(4))
        pd.testing.assert_index_equal(df2.chunks[0].columns_value.to_pandas(), pd.RangeIndex(4))
        pd.testing.assert_series_equal(df2.chunks[0].dtypes, raw.dtypes[:4])

        self.assertEqual(df2.chunks[2].shape, (4, 2))
        pd.testing.assert_index_equal(df2.chunks[2].index_value.to_pandas(), pd.RangeIndex(4))
        pd.testing.assert_index_equal(df2.chunks[2].columns_value.to_pandas(), pd.RangeIndex(8, 10))
        pd.testing.assert_series_equal(df2.chunks[2].dtypes, raw.dtypes[-2:])

        self.assertEqual(df2.chunks[-1].shape, (2, 2))
        pd.testing.assert_index_equal(df2.chunks[-1].index_value.to_pandas(), pd.RangeIndex(8, 10))
        pd.testing.assert_index_equal(df2.chunks[-1].columns_value.to_pandas(), pd.RangeIndex(8, 10))
        pd.testing.assert_series_equal(df2.chunks[-1].dtypes, raw.dtypes[-2:])

        for c in df2.chunks:
            self.assertEqual(c.shape[1], len(c.dtypes))
            self.assertEqual(len(c.columns_value.to_pandas()), len(c.dtypes))

        columns = [np.random.bytes(10) for _ in range(10)]
        index = np.random.randint(-100, 100, size=(4,))
        raw = pd.DataFrame(np.random.rand(4, 10), index=index, columns=columns)
        df = from_pandas_df(raw, chunk_size=3)
        df2 = df.rechunk(6).tiles()

        self.assertEqual(df2.shape, (4, 10))
        self.assertEqual(len(df2.chunks), 2)

        self.assertEqual(df2.chunks[0].shape, (4, 6))
        pd.testing.assert_index_equal(df2.chunks[0].index_value.to_pandas(), df.index_value.to_pandas())
        pd.testing.assert_index_equal(df2.chunks[0].columns_value.to_pandas(), pd.Index(columns[:6]))
        pd.testing.assert_series_equal(df2.chunks[0].dtypes, raw.dtypes[:6])

        self.assertEqual(df2.chunks[1].shape, (4, 4))
        pd.testing.assert_index_equal(df2.chunks[1].index_value.to_pandas(), df.index_value.to_pandas())
        pd.testing.assert_index_equal(df2.chunks[1].columns_value.to_pandas(), pd.Index(columns[6:]))
        pd.testing.assert_series_equal(df2.chunks[1].dtypes, raw.dtypes[-4:])

        for c in df2.chunks:
            self.assertEqual(c.shape[1], len(c.dtypes))
            self.assertEqual(len(c.columns_value.to_pandas()), len(c.dtypes))

        # test Series rechunk
        series = from_pandas_series(pd.Series(np.random.rand(10,)), chunk_size=3)
        series2 = series.rechunk(4).tiles()

        self.assertEqual(series2.shape, (10,))
        self.assertEqual(len(series2.chunks), 3)
        pd.testing.assert_index_equal(series2.index_value.to_pandas(), pd.RangeIndex(10))

        self.assertEqual(series2.chunk_shape, (3,))
        self.assertEqual(series2.nsplits, ((4, 4, 2), ))
        self.assertEqual(series2.chunks[0].shape, (4,))
        pd.testing.assert_index_equal(series2.chunks[0].index_value.to_pandas(), pd.RangeIndex(4))
        self.assertEqual(series2.chunks[1].shape, (4,))
        pd.testing.assert_index_equal(series2.chunks[1].index_value.to_pandas(), pd.RangeIndex(4, 8))
        self.assertEqual(series2.chunks[2].shape, (2,))
        pd.testing.assert_index_equal(series2.chunks[2].index_value.to_pandas(), pd.RangeIndex(8, 10))

        series2 = series.rechunk(1).tiles()

        self.assertEqual(series2.shape, (10,))
        self.assertEqual(len(series2.chunks), 10)
        pd.testing.assert_index_equal(series2.index_value.to_pandas(), pd.RangeIndex(10))

        self.assertEqual(series2.chunk_shape, (10,))
        self.assertEqual(series2.nsplits, ((1,) * 10, ))
        self.assertEqual(series2.chunks[0].shape, (1,))
        pd.testing.assert_index_equal(series2.chunks[0].index_value.to_pandas(), pd.RangeIndex(1))

        # no need to rechunk
        series2 = series.rechunk(3).tiles()
        series = get_tiled(series)
        self.assertEqual(series2.chunk_shape, series.chunk_shape)
        self.assertEqual(series2.nsplits, series.nsplits)

    def testResetIndex(self):
        data = pd.DataFrame([('bird',    389.0),
                             ('bird',     24.0),
                             ('mammal',   80.5),
                             ('mammal', np.nan)],
                            index=['falcon', 'parrot', 'lion', 'monkey'],
                            columns=('class', 'max_speed'))
        df = df_reset_index(from_pandas_df(data, chunk_size=2))
        r = data.reset_index()

        self.assertEqual(df.shape, (4, 3))
        pd.testing.assert_series_equal(df.dtypes, r.dtypes)

        df2 = df.tiles()

        self.assertEqual(len(df2.chunks), 2)
        self.assertEqual(df2.chunks[0].shape, (2, 3))
        pd.testing.assert_index_equal(df2.chunks[0].index_value.to_pandas(), pd.RangeIndex(2))
        pd.testing.assert_series_equal(df2.chunks[0].dtypes, r.dtypes)
        self.assertEqual(df2.chunks[1].shape, (2, 3))
        pd.testing.assert_index_equal(df2.chunks[1].index_value.to_pandas(), pd.RangeIndex(2, 4))
        pd.testing.assert_series_equal(df2.chunks[1].dtypes, r.dtypes)

        df = df_reset_index(from_pandas_df(data, chunk_size=1), drop=True)
        r = data.reset_index(drop=True)

        self.assertEqual(df.shape, (4, 2))
        pd.testing.assert_series_equal(df.dtypes, r.dtypes)

        df2 = df.tiles()

        self.assertEqual(len(df2.chunks), 8)

        for c in df2.chunks:
            self.assertEqual(c.shape, (1, 1))
            pd.testing.assert_index_equal(c.index_value.to_pandas(), pd.RangeIndex(c.index[0], c.index[0] + 1))
            pd.testing.assert_series_equal(c.dtypes, r.dtypes[c.index[1]: c.index[1] + 1])

        # test Series
        series_data = pd.Series([1, 2, 3, 4], name='foo',
                                index=pd.Index(['a', 'b', 'c', 'd'], name='idx'))
        s = series_reset_index(from_pandas_series(series_data, chunk_size=2))
        r = series_data.reset_index()

        self.assertEqual(s.shape, (4, 2))
        pd.testing.assert_series_equal(s.dtypes, r.dtypes)

        s2 = s.tiles()
        self.assertEqual(len(s2.chunks), 2)
        self.assertEqual(s2.chunks[0].shape, (2, 2))
        pd.testing.assert_index_equal(s2.chunks[0].index_value.to_pandas(), pd.RangeIndex(2))
        self.assertEqual(s2.chunks[1].shape, (2, 2))
        pd.testing.assert_index_equal(s2.chunks[1].index_value.to_pandas(), pd.RangeIndex(2, 4))

    def testFillNA(self):
        df_raw = pd.DataFrame(np.nan, index=range(0, 20), columns=list('ABCDEFGHIJ'))
        for _ in range(20):
            df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)
        value_df_raw = pd.DataFrame(np.random.randint(0, 100, (10, 7)).astype(np.float32),
                                    columns=list('ABCDEFG'))
        series_raw = pd.Series(np.nan, index=range(20))
        for _ in range(3):
            series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)
        value_series_raw = pd.Series(np.random.randint(0, 100, (10,)).astype(np.float32),
                                     index=list('ABCDEFGHIJ'))

        df = from_pandas_df(df_raw)
        series = from_pandas_series(series_raw)

        # when nothing supplied, raise
        with self.assertRaises(ValueError):
            df.fillna()
        # when both values and methods supplied, raises
        with self.assertRaises(ValueError):
            df.fillna(value=1, method='ffill')
        # when call on series, cannot supply DataFrames
        with self.assertRaises(ValueError):
            series.fillna(value=df)
        with self.assertRaises(ValueError):
            series.fillna(value=df_raw)
        with self.assertRaises(NotImplementedError):
            series.fillna(value=series_raw, downcast='infer')
        with self.assertRaises(NotImplementedError):
            series.ffill(limit=1)

        df2 = df.fillna(value_series_raw).tiles()
        self.assertEqual(len(df2.chunks), 1)
        self.assertEqual(df2.chunks[0].shape, df2.shape)
        self.assertIsNone(df2.chunks[0].op.stage)

        series2 = series.fillna(value_series_raw).tiles()
        self.assertEqual(len(series2.chunks), 1)
        self.assertEqual(series2.chunks[0].shape, series2.shape)
        self.assertIsNone(series2.chunks[0].op.stage)

        df = from_pandas_df(df_raw, chunk_size=5)
        df2 = df.fillna(value_series_raw).tiles()
        self.assertEqual(len(df2.chunks), 8)
        self.assertEqual(df2.chunks[0].shape, (5, 5))
        self.assertIsNone(df2.chunks[0].op.stage)

        series = from_pandas_series(series_raw, chunk_size=5)
        series2 = series.fillna(value_series_raw).tiles()
        self.assertEqual(len(series2.chunks), 4)
        self.assertEqual(series2.chunks[0].shape, (5,))
        self.assertIsNone(series2.chunks[0].op.stage)

        df2 = df.ffill(axis='columns').tiles()
        self.assertEqual(len(df2.chunks), 8)
        self.assertEqual(df2.chunks[0].shape, (5, 5))
        self.assertEqual(df2.chunks[0].op.axis, 1)
        self.assertEqual(df2.chunks[0].op.stage, OperandStage.combine)
        self.assertEqual(df2.chunks[0].op.method, 'ffill')
        self.assertIsNone(df2.chunks[0].op.limit)

        series2 = series.bfill().tiles()
        self.assertEqual(len(series2.chunks), 4)
        self.assertEqual(series2.chunks[0].shape, (5,))
        self.assertEqual(series2.chunks[0].op.stage, OperandStage.combine)
        self.assertEqual(series2.chunks[0].op.method, 'bfill')
        self.assertIsNone(series2.chunks[0].op.limit)

        value_df = from_pandas_df(value_df_raw, chunk_size=7)
        value_series = from_pandas_series(value_series_raw, chunk_size=7)

        df2 = df.fillna(value_df).tiles()
        self.assertEqual(df2.shape, df.shape)
        self.assertIsNone(df2.chunks[0].op.stage)

        df2 = df.fillna(value_series).tiles()
        self.assertEqual(df2.shape, df.shape)
        self.assertIsNone(df2.chunks[0].op.stage)

        value_series_raw.index = list(range(10))
        value_series = from_pandas_series(value_series_raw)
        series2 = series.fillna(value_series).tiles()
        self.assertEqual(series2.shape, series.shape)
        self.assertIsNone(series2.chunks[0].op.stage)

    def testDataFrameApply(self):
        cols = [chr(ord('A') + i) for i in range(10)]
        df_raw = pd.DataFrame(dict((c, [i ** 2 for i in range(20)]) for c in cols))

        old_chunk_store_limit = options.chunk_store_limit
        try:
            options.chunk_store_limit = 20

            df = from_pandas_df(df_raw, chunk_size=5)

            r = df.apply('ffill')
            self.assertEqual(r.op._op_type_, opcodes.FILL_NA)

            r = df.apply(np.sqrt).tiles()
            self.assertTrue(all(v == np.dtype('float64') for v in r.dtypes))
            self.assertEqual(r.shape, df.shape)
            self.assertEqual(r.op._op_type_, opcodes.DATAFRAME_APPLY)
            self.assertEqual(r.op.object_type, ObjectType.dataframe)
            self.assertTrue(r.op.elementwise)

            r = df.apply(lambda x: pd.Series([1, 2])).tiles()
            self.assertTrue(all(v == np.dtype('int64') for v in r.dtypes))
            self.assertEqual(r.shape, (np.nan, df.shape[1]))
            self.assertEqual(r.op.object_type, ObjectType.dataframe)
            self.assertEqual(r.chunks[0].shape, (np.nan, 1))
            self.assertEqual(r.chunks[0].inputs[0].shape[0], df_raw.shape[0])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)
            self.assertFalse(r.op.elementwise)

            r = df.apply(np.sum, axis='index').tiles()
            self.assertTrue(np.dtype('int64'), r.dtype)
            self.assertEqual(r.shape, (df.shape[1],))
            self.assertEqual(r.op.object_type, ObjectType.series)
            self.assertEqual(r.chunks[0].shape, (20 // df.shape[0],))
            self.assertEqual(r.chunks[0].inputs[0].shape[0], df_raw.shape[0])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)
            self.assertFalse(r.op.elementwise)

            r = df.apply(np.sum, axis='columns').tiles()
            self.assertTrue(np.dtype('int64'), r.dtype)
            self.assertEqual(r.shape, (df.shape[0],))
            self.assertEqual(r.op.object_type, ObjectType.series)
            self.assertEqual(r.chunks[0].shape, (20 // df.shape[1],))
            self.assertEqual(r.chunks[0].inputs[0].shape[1], df_raw.shape[1])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)
            self.assertFalse(r.op.elementwise)

            r = df.apply(lambda x: pd.Series([1, 2], index=['foo', 'bar']), axis=1).tiles()
            self.assertTrue(all(v == np.dtype('int64') for v in r.dtypes))
            self.assertEqual(r.shape, (df.shape[0], np.nan))
            self.assertEqual(r.op.object_type, ObjectType.dataframe)
            self.assertEqual(r.chunks[0].shape, (20 // df.shape[1], np.nan))
            self.assertEqual(r.chunks[0].inputs[0].shape[1], df_raw.shape[1])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)
            self.assertFalse(r.op.elementwise)

            r = df.apply(lambda x: [1, 2], axis=1, result_type='expand').tiles()
            self.assertTrue(all(v == np.dtype('int64') for v in r.dtypes))
            self.assertEqual(r.shape, (df.shape[0], np.nan))
            self.assertEqual(r.op.object_type, ObjectType.dataframe)
            self.assertEqual(r.chunks[0].shape, (20 // df.shape[1], np.nan))
            self.assertEqual(r.chunks[0].inputs[0].shape[1], df_raw.shape[1])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)
            self.assertFalse(r.op.elementwise)

            r = df.apply(lambda x: list(range(10)), axis=1, result_type='reduce').tiles()
            self.assertTrue(np.dtype('object'), r.dtype)
            self.assertEqual(r.shape, (df.shape[0],))
            self.assertEqual(r.op.object_type, ObjectType.series)
            self.assertEqual(r.chunks[0].shape, (20 // df.shape[1],))
            self.assertEqual(r.chunks[0].inputs[0].shape[1], df_raw.shape[1])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)
            self.assertFalse(r.op.elementwise)

            r = df.apply(lambda x: list(range(10)), axis=1, result_type='broadcast').tiles()
            self.assertTrue(all(v == np.dtype('int64') for v in r.dtypes))
            self.assertEqual(r.shape, (df.shape[0], np.nan))
            self.assertEqual(r.op.object_type, ObjectType.dataframe)
            self.assertEqual(r.chunks[0].shape, (20 // df.shape[1], np.nan))
            self.assertEqual(r.chunks[0].inputs[0].shape[1], df_raw.shape[1])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)
            self.assertFalse(r.op.elementwise)
        finally:
            options.chunk_store_limit = old_chunk_store_limit

    def testSeriesApply(self):
        idxes = [chr(ord('A') + i) for i in range(20)]
        s_raw = pd.Series([i ** 2 for i in range(20)], index=idxes)

        series = from_pandas_series(s_raw, chunk_size=5)

        r = series.apply('add', args=(1,)).tiles()
        self.assertEqual(r.op._op_type_, opcodes.ADD)

        r = series.apply(np.sqrt).tiles()
        self.assertTrue(np.dtype('float64'), r.dtype)
        self.assertEqual(r.shape, series.shape)
        self.assertEqual(r.op._op_type_, opcodes.SERIES_APPLY)
        self.assertEqual(r.op.object_type, ObjectType.series)
        self.assertEqual(r.chunks[0].shape, (5,))
        self.assertEqual(r.chunks[0].inputs[0].shape, (5,))

        r = series.apply('sqrt').tiles()
        self.assertTrue(np.dtype('float64'), r.dtype)
        self.assertEqual(r.shape, series.shape)
        self.assertEqual(r.op._op_type_, opcodes.SERIES_APPLY)
        self.assertEqual(r.op.object_type, ObjectType.series)
        self.assertEqual(r.chunks[0].shape, (5,))
        self.assertEqual(r.chunks[0].inputs[0].shape, (5,))

        r = series.apply(lambda x: [x, x + 1], convert_dtype=False).tiles()
        self.assertTrue(np.dtype('object'), r.dtype)
        self.assertEqual(r.shape, series.shape)
        self.assertEqual(r.op._op_type_, opcodes.SERIES_APPLY)
        self.assertEqual(r.op.object_type, ObjectType.series)
        self.assertEqual(r.chunks[0].shape, (5,))
        self.assertEqual(r.chunks[0].inputs[0].shape, (5,))

    def testTransform(self):
        cols = [chr(ord('A') + i) for i in range(10)]
        df_raw = pd.DataFrame(dict((c, [i ** 2 for i in range(20)]) for c in cols))
        df = from_pandas_df(df_raw, chunk_size=5)

        idxes = [chr(ord('A') + i) for i in range(20)]
        s_raw = pd.Series([i ** 2 for i in range(20)], index=idxes)
        series = from_pandas_series(s_raw, chunk_size=5)

        old_chunk_store_limit = options.chunk_store_limit
        try:
            options.chunk_store_limit = 20

            r = df.transform(lambda x: x + 1).tiles()
            self.assertTrue(all(v == np.dtype('int64') for v in r.dtypes))
            self.assertEqual(r.shape, df.shape)
            self.assertEqual(r.op._op_type_, opcodes.DATAFRAME_TRANSFORM)
            self.assertEqual(r.op.object_type, ObjectType.dataframe)
            self.assertEqual(r.chunks[0].shape, (df.shape[0], 1))
            self.assertEqual(r.chunks[0].inputs[0].shape[0], df_raw.shape[0])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)

            r = series.transform(lambda x: x + 1).tiles()
            self.assertTrue(np.dtype('float64'), r.dtype)
            self.assertEqual(r.shape, series.shape)
            self.assertEqual(r.op._op_type_, opcodes.SERIES_TRANSFORM)
            self.assertEqual(r.op.object_type, ObjectType.series)
            self.assertEqual(r.chunks[0].shape, (5,))
            self.assertEqual(r.chunks[0].inputs[0].shape, (5,))
        finally:
            options.chunk_store_limit = old_chunk_store_limit

    def testStringMethod(self):
        s = pd.Series(['a', 'b', 'c'], name='s')
        series = from_pandas_series(s, chunk_size=2)

        with self.assertRaises(AttributeError):
            _ = series.str.non_exist

        r = series.str.contains('c')
        self.assertEqual(r.dtype, np.bool_)
        self.assertEqual(r.name, s.name)
        pd.testing.assert_index_equal(r.index_value.to_pandas(), s.index)
        self.assertEqual(r.shape, s.shape)

        r = r.tiles()
        for i, c in enumerate(r.chunks):
            self.assertEqual(c.index, (i,))
            self.assertEqual(c.dtype, np.bool_)
            self.assertEqual(c.name, s.name)
            pd.testing.assert_index_equal(c.index_value.to_pandas(),
                                          s.index[i * 2: (i + 1) * 2])
            self.assertEqual(c.shape, (2,) if i == 0 else (1,))

        r = series.str.split(',', expand=True, n=1)
        self.assertEqual(r.op.object_type, ObjectType.dataframe)
        self.assertEqual(r.shape, (3, 2))
        pd.testing.assert_index_equal(r.index_value.to_pandas(), s.index)
        pd.testing.assert_index_equal(r.columns_value.to_pandas(), pd.RangeIndex(2))

        r = r.tiles()
        for i, c in enumerate(r.chunks):
            self.assertEqual(c.index, (i, 0))
            pd.testing.assert_index_equal(c.index_value.to_pandas(),
                                          s.index[i * 2: (i + 1) * 2])
            pd.testing.assert_index_equal(c.columns_value.to_pandas(), pd.RangeIndex(2))
            self.assertEqual(c.shape, (2, 2) if i == 0 else (1, 2))

        with self.assertRaises(TypeError):
            _ = series.str.cat([['1', '2']])

        with self.assertRaises(ValueError):
            _ = series.str.cat(['1', '2'])

        with self.assertRaises(ValueError):
            _ = series.str.cat(',')

        with self.assertRaises(TypeError):
            _ = series.str.cat({'1', '2', '3'})

        r = series.str.cat(sep=',')
        self.assertEqual(r.op.object_type, ObjectType.scalar)
        self.assertEqual(r.dtype, s.dtype)

        r = r.tiles()
        self.assertEqual(len(r.chunks), 1)
        self.assertEqual(r.chunks[0].op.object_type, ObjectType.scalar)
        self.assertEqual(r.chunks[0].dtype, s.dtype)

        r = series.str.extract(r'[ab](\d)', expand=False)
        self.assertEqual(r.op.object_type, ObjectType.series)
        self.assertEqual(r.dtype, s.dtype)

        r = r.tiles()
        for i, c in enumerate(r.chunks):
            self.assertEqual(c.index, (i,))
            self.assertEqual(c.dtype, s.dtype)
            self.assertEqual(c.name, s.name)
            pd.testing.assert_index_equal(c.index_value.to_pandas(),
                                          s.index[i * 2: (i + 1) * 2])
            self.assertEqual(c.shape, (2,) if i == 0 else (1,))

        r = series.str.extract(r'[ab](\d)', expand=True)
        self.assertEqual(r.op.object_type, ObjectType.dataframe)
        self.assertEqual(r.shape, (3, 1))
        pd.testing.assert_index_equal(r.index_value.to_pandas(), s.index)
        pd.testing.assert_index_equal(r.columns_value.to_pandas(), pd.RangeIndex(1))

        r = r.tiles()
        for i, c in enumerate(r.chunks):
            self.assertEqual(c.index, (i, 0))
            pd.testing.assert_index_equal(c.index_value.to_pandas(),
                                          s.index[i * 2: (i + 1) * 2])
            pd.testing.assert_index_equal(c.columns_value.to_pandas(), pd.RangeIndex(1))
            self.assertEqual(c.shape, (2, 1) if i == 0 else (1, 1))

        self.assertIn('lstrip', dir(series.str))

    def testDatetimeMethod(self):
        s = pd.Series([pd.Timestamp('2020-1-1'),
                       pd.Timestamp('2020-2-1'),
                       pd.Timestamp('2020-3-1')],
                      name='ss')
        series = from_pandas_series(s, chunk_size=2)

        r = series.dt.year
        self.assertEqual(r.dtype, s.dt.year.dtype)
        pd.testing.assert_index_equal(r.index_value.to_pandas(), s.index)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.op.object_type, ObjectType.series)
        self.assertEqual(r.name, s.dt.year.name)

        r = r.tiles()
        for i, c in enumerate(r.chunks):
            self.assertEqual(c.index, (i,))
            self.assertEqual(c.dtype, s.dt.year.dtype)
            self.assertEqual(c.op.object_type, ObjectType.series)
            self.assertEqual(r.name, s.dt.year.name)
            pd.testing.assert_index_equal(c.index_value.to_pandas(),
                                          s.index[i * 2: (i + 1) * 2])
            self.assertEqual(c.shape, (2,) if i == 0 else (1,))

        with self.assertRaises(AttributeError):
            _ = series.dt.non_exist

        self.assertIn('ceil', dir(series.dt))
