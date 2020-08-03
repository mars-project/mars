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
from mars.config import options, option_context
from mars.dataframe.core import DATAFRAME_TYPE, SERIES_TYPE, SERIES_CHUNK_TYPE, \
    INDEX_TYPE, CATEGORICAL_TYPE, CATEGORICAL_CHUNK_TYPE
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.datasource.index import from_pandas as from_pandas_index
from mars.dataframe.base import to_gpu, to_cpu, cut, astype
from mars.dataframe.operands import ObjectType
from mars.operands import OperandStage
from mars.tensor.core import TENSOR_TYPE
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
            self.assertEqual(r.op._op_type_, opcodes.APPLY)
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
        self.assertEqual(r.op._op_type_, opcodes.APPLY)
        self.assertEqual(r.op.object_type, ObjectType.series)
        self.assertEqual(r.chunks[0].shape, (5,))
        self.assertEqual(r.chunks[0].inputs[0].shape, (5,))

        r = series.apply('sqrt').tiles()
        self.assertTrue(np.dtype('float64'), r.dtype)
        self.assertEqual(r.shape, series.shape)
        self.assertEqual(r.op._op_type_, opcodes.APPLY)
        self.assertEqual(r.op.object_type, ObjectType.series)
        self.assertEqual(r.chunks[0].shape, (5,))
        self.assertEqual(r.chunks[0].inputs[0].shape, (5,))

        r = series.apply(lambda x: [x, x + 1], convert_dtype=False).tiles()
        self.assertTrue(np.dtype('object'), r.dtype)
        self.assertEqual(r.shape, series.shape)
        self.assertEqual(r.op._op_type_, opcodes.APPLY)
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

        def rename_fn(f, new_name):
            f.__name__ = new_name
            return f

        old_chunk_store_limit = options.chunk_store_limit
        try:
            options.chunk_store_limit = 20

            # DATAFRAME CASES
            # test transform scenarios on data frames
            r = df.transform(lambda x: list(range(len(x)))).tiles()
            self.assertTrue(all(v == np.dtype('int64') for v in r.dtypes))
            self.assertEqual(r.shape, df.shape)
            self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
            self.assertEqual(r.op.object_type, ObjectType.dataframe)
            self.assertEqual(r.chunks[0].shape, (df.shape[0], 20 // df.shape[0]))
            self.assertEqual(r.chunks[0].inputs[0].shape[0], df_raw.shape[0])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)

            r = df.transform(lambda x: list(range(len(x))), axis=1).tiles()
            self.assertTrue(all(v == np.dtype('int64') for v in r.dtypes))
            self.assertEqual(r.shape, df.shape)
            self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
            self.assertEqual(r.op.object_type, ObjectType.dataframe)
            self.assertEqual(r.chunks[0].shape, (20 // df.shape[1], df.shape[1]))
            self.assertEqual(r.chunks[0].inputs[0].shape[1], df_raw.shape[1])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)

            r = df.transform(['cumsum', 'cummax', lambda x: x + 1]).tiles()
            self.assertTrue(all(v == np.dtype('int64') for v in r.dtypes))
            self.assertEqual(r.shape, (df.shape[0], df.shape[1] * 3))
            self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
            self.assertEqual(r.op.object_type, ObjectType.dataframe)
            self.assertEqual(r.chunks[0].shape, (df.shape[0], 20 // df.shape[0] * 3))
            self.assertEqual(r.chunks[0].inputs[0].shape[0], df_raw.shape[0])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)

            r = df.transform({'A': 'cumsum', 'D': ['cumsum', 'cummax'], 'F': lambda x: x + 1}).tiles()
            self.assertTrue(all(v == np.dtype('int64') for v in r.dtypes))
            self.assertEqual(r.shape, (df.shape[0], 4))
            self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
            self.assertEqual(r.op.object_type, ObjectType.dataframe)
            self.assertEqual(r.chunks[0].shape, (df.shape[0], 1))
            self.assertEqual(r.chunks[0].inputs[0].shape[0], df_raw.shape[0])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)

            # test agg scenarios on series
            r = df.transform(lambda x: x.iloc[:-1], _call_agg=True).tiles()
            self.assertTrue(all(v == np.dtype('int64') for v in r.dtypes))
            self.assertEqual(r.shape, (np.nan, df.shape[1]))
            self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
            self.assertEqual(r.op.object_type, ObjectType.dataframe)
            self.assertEqual(r.chunks[0].shape, (np.nan, 1))
            self.assertEqual(r.chunks[0].inputs[0].shape[0], df_raw.shape[0])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)

            r = df.transform(lambda x: x.iloc[:-1], axis=1, _call_agg=True).tiles()
            self.assertTrue(all(v == np.dtype('int64') for v in r.dtypes))
            self.assertEqual(r.shape, (df.shape[0], np.nan))
            self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
            self.assertEqual(r.op.object_type, ObjectType.dataframe)
            self.assertEqual(r.chunks[0].shape, (2, np.nan))
            self.assertEqual(r.chunks[0].inputs[0].shape[1], df_raw.shape[1])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)

            fn_list = [rename_fn(lambda x: x.iloc[1:].reset_index(drop=True), 'f1'),
                       lambda x: x.iloc[:-1].reset_index(drop=True)]
            r = df.transform(fn_list, _call_agg=True).tiles()
            self.assertTrue(all(v == np.dtype('int64') for v in r.dtypes))
            self.assertEqual(r.shape, (np.nan, df.shape[1] * 2))
            self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
            self.assertEqual(r.op.object_type, ObjectType.dataframe)
            self.assertEqual(r.chunks[0].shape, (np.nan, 2))
            self.assertEqual(r.chunks[0].inputs[0].shape[0], df_raw.shape[0])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)

            r = df.transform(lambda x: x.sum(), _call_agg=True).tiles()
            self.assertEqual(r.dtype, np.dtype('int64'))
            self.assertEqual(r.shape, (df.shape[1],))
            self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
            self.assertEqual(r.op.object_type, ObjectType.series)
            self.assertEqual(r.chunks[0].shape, (20 // df.shape[0],))
            self.assertEqual(r.chunks[0].inputs[0].shape[0], df_raw.shape[0])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)

            fn_dict = {
                'A': rename_fn(lambda x: x.iloc[1:].reset_index(drop=True), 'f1'),
                'D': [rename_fn(lambda x: x.iloc[1:].reset_index(drop=True), 'f1'),
                      lambda x: x.iloc[:-1].reset_index(drop=True)],
                'F': lambda x: x.iloc[:-1].reset_index(drop=True),
            }
            r = df.transform(fn_dict, _call_agg=True).tiles()
            self.assertTrue(all(v == np.dtype('int64') for v in r.dtypes))
            self.assertEqual(r.shape, (np.nan, 4))
            self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
            self.assertEqual(r.op.object_type, ObjectType.dataframe)
            self.assertEqual(r.chunks[0].shape, (np.nan, 1))
            self.assertEqual(r.chunks[0].inputs[0].shape[0], df_raw.shape[0])
            self.assertEqual(r.chunks[0].inputs[0].op._op_type_, opcodes.CONCATENATE)

            # SERIES CASES
            # test transform scenarios on series
            r = series.transform(lambda x: x + 1).tiles()
            self.assertTrue(np.dtype('float64'), r.dtype)
            self.assertEqual(r.shape, series.shape)
            self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
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

    def testSeriesIsin(self):
        # one chunk in multiple chunks
        a = from_pandas_series(pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), chunk_size=10)
        b = from_pandas_series(pd.Series([2, 1, 9, 3]), chunk_size=2)

        r = a.isin(b).tiles()
        for i, c in enumerate(r.chunks):
            self.assertEqual(c.index, (i,))
            self.assertEqual(c.dtype, np.dtype('bool'))
            self.assertEqual(c.shape, (10,))
            self.assertEqual(len(c.op.inputs), 2)
            self.assertEqual(c.op.object_type, ObjectType.series)
            self.assertEqual(c.op.inputs[0].index, (i,))
            self.assertEqual(c.op.inputs[0].shape, (10,))
            self.assertEqual(c.op.inputs[1].index, (0,))
            self.assertEqual(c.op.inputs[1].shape, (4,))  # has been rechunked

        # multiple chunk in one chunks
        a = from_pandas_series(pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), chunk_size=2)
        b = from_pandas_series(pd.Series([2, 1, 9, 3]), chunk_size=4)

        r = a.isin(b).tiles()
        for i, c in enumerate(r.chunks):
            self.assertEqual(c.index, (i,))
            self.assertEqual(c.dtype, np.dtype('bool'))
            self.assertEqual(c.shape, (2,))
            self.assertEqual(len(c.op.inputs), 2)
            self.assertEqual(c.op.object_type, ObjectType.series)
            self.assertEqual(c.op.inputs[0].index, (i,))
            self.assertEqual(c.op.inputs[0].shape, (2,))
            self.assertEqual(c.op.inputs[1].index, (0,))
            self.assertEqual(c.op.inputs[1].shape, (4,))

        # multiple chunk in multiple chunks
        a = from_pandas_series(pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), chunk_size=2)
        b = from_pandas_series(pd.Series([2, 1, 9, 3]), chunk_size=2)

        r = a.isin(b).tiles()
        for i, c in enumerate(r.chunks):
            self.assertEqual(c.index, (i,))
            self.assertEqual(c.dtype, np.dtype('bool'))
            self.assertEqual(c.shape, (2,))
            self.assertEqual(len(c.op.inputs), 2)
            self.assertEqual(c.op.object_type, ObjectType.series)
            self.assertEqual(c.op.inputs[0].index, (i,))
            self.assertEqual(c.op.inputs[0].shape, (2,))
            self.assertEqual(c.op.inputs[1].index, (0,))
            self.assertEqual(c.op.inputs[1].shape, (4,))  # has been rechunked

        with self.assertRaises(TypeError):
            _ = a.isin('sth')

    def testDropNA(self):
        # dataframe cases
        df_raw = pd.DataFrame(np.nan, index=range(0, 20), columns=list('ABCDEFGHIJ'))
        for _ in range(30):
            df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)
        for rowid in range(random.randint(1, 5)):
            row = random.randint(0, 19)
            for idx in range(0, 10):
                df_raw.iloc[row, idx] = random.randint(0, 99)

        # not supporting drop with axis=1
        with self.assertRaises(NotImplementedError):
            from_pandas_df(df_raw).dropna(axis=1)

        # only one chunk in columns, can run dropna directly
        r = from_pandas_df(df_raw, chunk_size=(4, 10)).dropna().tiles()
        self.assertEqual(r.shape, (np.nan, 10))
        self.assertEqual(r.nsplits, ((np.nan,) * 5, (10,)))
        for c in r.chunks:
            self.assertIsInstance(c.op, type(r.op))
            self.assertEqual(len(c.inputs), 1)
            self.assertEqual(len(c.inputs[0].inputs), 0)
            self.assertEqual(c.shape, (np.nan, 10))

        # multiple chunks in columns, count() will be called first
        r = from_pandas_df(df_raw, chunk_size=4).dropna().tiles()
        self.assertEqual(r.shape, (np.nan, 10))
        self.assertEqual(r.nsplits, ((np.nan,) * 5, (4, 4, 2)))
        for c in r.chunks:
            self.assertIsInstance(c.op, type(r.op))
            self.assertEqual(len(c.inputs), 2)
            self.assertEqual(len(c.inputs[0].inputs), 0)
            self.assertEqual(c.inputs[1].op.stage, OperandStage.agg)
            self.assertTrue(np.isnan(c.shape[0]))

        # series cases
        series_raw = pd.Series(np.nan, index=range(20))
        for _ in range(10):
            series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)

        r = from_pandas_series(series_raw, chunk_size=4).dropna().tiles()
        self.assertEqual(r.shape, (np.nan,))
        self.assertEqual(r.nsplits, ((np.nan,) * 5,))
        for c in r.chunks:
            self.assertIsInstance(c.op, type(r.op))
            self.assertEqual(len(c.inputs), 1)
            self.assertEqual(len(c.inputs[0].inputs), 0)
            self.assertEqual(c.shape, (np.nan,))

    def testCut(self):
        s = from_pandas_series(pd.Series([1., 2., 3., 4.]), chunk_size=2)

        with self.assertRaises(ValueError):
            _ = cut(s, -1)

        with self.assertRaises(ValueError):
            _ = cut([[1, 2], [3, 4]], 3)

        with self.assertRaises(ValueError):
            _ = cut([], 3)

        r, b = cut(s, [1.5, 2.5], retbins=True)
        self.assertIsInstance(r, SERIES_TYPE)
        self.assertIsInstance(b, TENSOR_TYPE)

        r = r.tiles()

        self.assertEqual(len(r.chunks), 2)
        for c in r.chunks:
            self.assertIsInstance(c, SERIES_CHUNK_TYPE)
            self.assertEqual(c.shape, (2,))

        r = cut(s.to_tensor(), [1.5, 2.5])
        self.assertIsInstance(r, CATEGORICAL_TYPE)
        self.assertEqual(len(r), len(s))
        self.assertIn('Categorical', repr(r))

        r = r.tiles()

        self.assertEqual(len(r.chunks), 2)
        for c in r.chunks:
            self.assertIsInstance(c, CATEGORICAL_CHUNK_TYPE)
            self.assertEqual(c.shape, (2,))
            self.assertEqual(c.ndim, 1)

        # test serialize
        g = r.build_graph(tiled=False)
        g2 = type(g).from_pb(g.to_pb())
        g2 = type(g).from_json(g2.to_json())
        r2 = next(n for n in g2 if isinstance(n, CATEGORICAL_TYPE))
        self.assertEqual(len(r2), len(r))

        r = cut([0, 1, 1, 2], bins=4, labels=False)
        self.assertIsInstance(r, TENSOR_TYPE)
        e = pd.cut([0, 1, 1, 2], bins=4, labels=False)
        self.assertEqual(r.dtype, e.dtype)

    def testAstype(self):
        s = from_pandas_series(pd.Series([1, 2, 1, 2], name='a'), chunk_size=2)
        with self.assertRaises(KeyError):
            astype(s, {'b': 'str'})

        df = from_pandas_df(pd.DataFrame({'a': [1, 2, 1, 2],
                                          'b': ['a', 'b', 'a', 'b']}), chunk_size=2)

        with self.assertRaises(KeyError):
            astype(df, {'c': 'str', 'a': 'str'})

    def testDrop(self):
        # test dataframe drop
        rs = np.random.RandomState(0)
        raw = pd.DataFrame(rs.randint(1000, size=(20, 8)),
                           columns=['c' + str(i + 1) for i in range(8)])

        df = from_pandas_df(raw, chunk_size=3)

        with self.assertRaises(KeyError):
            df.drop(columns=['c9'])
        with self.assertRaises(NotImplementedError):
            df.drop(columns=from_pandas_series(pd.Series(['c9'])))

        columns = ['c2', 'c4', 'c5', 'c6']
        index = [3, 6, 7]
        r = df.drop(columns=columns, index=index)
        self.assertIsInstance(r, DATAFRAME_TYPE)

        # test series drop
        raw = pd.Series(rs.randint(1000, size=(20,)))
        series = from_pandas_series(raw, chunk_size=3)

        r = series.drop(index=index)
        self.assertIsInstance(r, SERIES_TYPE)

        # test index drop
        ser = pd.Series(range(20))
        rs.shuffle(ser)
        raw = pd.Index(ser)

        idx = from_pandas_index(raw)

        r = idx.drop(index)
        self.assertIsInstance(r, INDEX_TYPE)

    def testDropDuplicates(self):
        rs = np.random.RandomState(0)
        raw = pd.DataFrame(rs.randint(1000, size=(20, 7)),
                           columns=['c' + str(i + 1) for i in range(7)])
        raw['c7'] = ['s{}'.format(j) for j in range(20)]

        df = from_pandas_df(raw, chunk_size=10)
        with self.assertRaises(ValueError):
            df.drop_duplicates(method='unknown')
        with self.assertRaises(KeyError):
            df.drop_duplicates(subset='c8')

        # test auto method selection
        self.assertEqual(df.drop_duplicates().tiles().chunks[0].op.method, 'tree')
        # subset size less than chunk_store_limit
        self.assertEqual(df.drop_duplicates(subset=['c1', 'c3']).tiles().chunks[0].op.method, 'subset_tree')
        with option_context({'chunk_store_limit': 5}):
            # subset size greater than chunk_store_limit
            self.assertEqual(df.drop_duplicates(subset=['c1', 'c3']).tiles().chunks[0].op.method, 'tree')
        self.assertEqual(df.drop_duplicates(subset=['c1', 'c7']).tiles().chunks[0].op.method, 'tree')
        self.assertEqual(df['c7'].drop_duplicates().tiles().chunks[0].op.method, 'tree')

        s = df['c7']
        with self.assertRaises(ValueError):
            s.drop_duplicates(method='unknown')

    def testMemoryUsage(self):
        dtypes = ['int64', 'float64', 'complex128', 'object', 'bool']
        data = dict([(t, np.ones(shape=500).astype(t))
                    for t in dtypes])
        raw = pd.DataFrame(data)

        df = from_pandas_df(raw, chunk_size=(500, 2))
        r = df.memory_usage().tiles()

        self.assertIsInstance(r, SERIES_TYPE)
        self.assertEqual(r.shape, (6,))
        self.assertEqual(len(r.chunks), 3)
        self.assertIsNone(r.chunks[0].op.stage)

        df = from_pandas_df(raw, chunk_size=(100, 3))
        r = df.memory_usage(index=True).tiles()

        self.assertIsInstance(r, SERIES_TYPE)
        self.assertEqual(r.shape, (6,))
        self.assertEqual(len(r.chunks), 2)
        self.assertEqual(r.chunks[0].op.stage, OperandStage.reduce)

        r = df.memory_usage(index=False).tiles()

        self.assertIsInstance(r, SERIES_TYPE)
        self.assertEqual(r.shape, (5,))
        self.assertEqual(len(r.chunks), 2)
        self.assertEqual(r.chunks[0].op.stage, OperandStage.reduce)

        raw = pd.Series(np.ones(shape=500).astype('object'), name='s')

        series = from_pandas_series(raw)
        r = series.memory_usage().tiles()

        self.assertIsInstance(r, TENSOR_TYPE)
        self.assertEqual(r.shape, ())
        self.assertEqual(len(r.chunks), 1)
        self.assertIsNone(r.chunks[0].op.stage)

        series = from_pandas_series(raw, chunk_size=100)
        r = series.memory_usage().tiles()

        self.assertIsInstance(r, TENSOR_TYPE)
        self.assertEqual(r.shape, ())
        self.assertEqual(len(r.chunks), 1)
        self.assertEqual(r.chunks[0].op.stage, OperandStage.reduce)
