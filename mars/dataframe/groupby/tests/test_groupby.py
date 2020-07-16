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
from collections import OrderedDict

import mars.dataframe as md
from mars import opcodes
from mars.core import OutputType
from mars.dataframe.core import DataFrameGroupBy, SeriesGroupBy, DataFrame
from mars.dataframe.groupby.core import DataFrameGroupByOperand, DataFrameShuffleProxy
from mars.dataframe.groupby.aggregation import DataFrameGroupByAgg
from mars.dataframe.groupby.getitem import GroupByIndex
from mars.operands import OperandStage
from mars.tests.core import TestBase


class Test(TestBase):
    def testGroupBy(self):
        df = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                           'b': [1, 3, 4, 5, 6, 5, 4, 4, 4]})
        mdf = md.DataFrame(df, chunk_size=2)
        with self.assertRaises(KeyError):
            mdf.groupby('c2')
        with self.assertRaises(KeyError):
            mdf.groupby(['b', 'c2'])

        grouped = mdf.groupby('b')
        self.assertIsInstance(grouped, DataFrameGroupBy)
        self.assertIsInstance(grouped.op, DataFrameGroupByOperand)
        self.assertEqual(list(grouped.key_dtypes.index), ['b'])

        grouped = grouped.tiles()
        self.assertEqual(len(grouped.chunks), 5)
        for chunk in grouped.chunks:
            self.assertIsInstance(chunk.op, DataFrameGroupByOperand)

        series = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
        ms = md.Series(series, chunk_size=3)
        grouped = ms.groupby(lambda x: x + 1)

        self.assertIsInstance(grouped, SeriesGroupBy)
        self.assertIsInstance(grouped.op, DataFrameGroupByOperand)

        grouped = grouped.tiles()
        self.assertEqual(len(grouped.chunks), 3)
        for chunk in grouped.chunks:
            self.assertIsInstance(chunk.op, DataFrameGroupByOperand)

        with self.assertRaises(TypeError):
            ms.groupby(lambda x: x + 1, as_index=False)

    def testGroupByGetItem(self):
        df1 = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                            'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                            'c': list('aabaaddce')})
        mdf = md.DataFrame(df1, chunk_size=3)

        r = mdf.groupby('b')[['a', 'b']].tiles()
        self.assertIsInstance(r, DataFrameGroupBy)
        self.assertIsInstance(r.op, GroupByIndex)
        self.assertEqual(r.selection, ['a', 'b'])
        self.assertEqual(list(r.key_dtypes.index), ['b'])
        self.assertEqual(len(r.chunks), 3)

        r = mdf.groupby('b').a.tiles()
        self.assertIsInstance(r, SeriesGroupBy)
        self.assertIsInstance(r.op, GroupByIndex)
        self.assertEqual(r.name, 'a')
        self.assertEqual(list(r.key_dtypes.index), ['b'])
        self.assertEqual(len(r.chunks), 3)

        with self.assertRaises(IndexError):
            getattr(mdf.groupby('b')[['a', 'b']], 'a')

    def testGroupByAgg(self):
        df = pd.DataFrame({'a': np.random.choice([2, 3, 4], size=(20,)),
                           'b': np.random.choice([2, 3, 4], size=(20,))})
        mdf = md.DataFrame(df, chunk_size=3)
        r = mdf.groupby('a').agg('sum', method='tree')
        self.assertIsInstance(r.op, DataFrameGroupByAgg)
        self.assertIsInstance(r, DataFrame)
        self.assertEqual(r.op.method, 'tree')
        r = r.tiles()
        self.assertEqual(len(r.chunks), 1)
        self.assertEqual(r.chunks[0].op.stage, OperandStage.agg)
        self.assertEqual(len(r.chunks[0].inputs), 1)
        self.assertEqual(len(r.chunks[0].inputs[0].inputs), 2)

        df = pd.DataFrame({'c1': range(10),
                           'c2': np.random.choice(['a', 'b', 'c'], (10,)),
                           'c3': np.random.rand(10)})
        mdf = md.DataFrame(df, chunk_size=2)
        r = mdf.groupby('c2').sum(method='shuffle')

        self.assertIsInstance(r.op, DataFrameGroupByAgg)
        self.assertIsInstance(r, DataFrame)

        r = r.tiles()
        self.assertEqual(len(r.chunks), 5)
        for chunk in r.chunks:
            self.assertIsInstance(chunk.op, DataFrameGroupByAgg)
            self.assertEqual(chunk.op.stage, OperandStage.agg)
            self.assertIsInstance(chunk.inputs[0].op, DataFrameGroupByOperand)
            self.assertEqual(chunk.inputs[0].op.stage, OperandStage.reduce)
            self.assertIsInstance(chunk.inputs[0].inputs[0].op, DataFrameShuffleProxy)
            self.assertIsInstance(chunk.inputs[0].inputs[0].inputs[0].op, DataFrameGroupByOperand)
            self.assertEqual(chunk.inputs[0].inputs[0].inputs[0].op.stage, OperandStage.map)

            agg_chunk = chunk.inputs[0].inputs[0].inputs[0].inputs[0]
            self.assertEqual(agg_chunk.op.stage, OperandStage.map)

        # test unknown method
        with self.assertRaises(ValueError):
            mdf.groupby('c2').sum(method='not_exist')

    def testGroupByApply(self):
        df1 = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                            'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                            'c': list('aabaaddce')})

        def apply_df(df):
            return df.sort_index()

        def apply_series(s):
            return s.sort_index()

        mdf = md.DataFrame(df1, chunk_size=3)

        applied = mdf.groupby('b').apply(apply_df).tiles()
        pd.testing.assert_series_equal(applied.dtypes, df1.dtypes)
        self.assertEqual(applied.shape, (np.nan, 3))
        self.assertEqual(applied.op._op_type_, opcodes.APPLY)
        self.assertEqual(applied.op.output_types[0], OutputType.dataframe)
        self.assertEqual(len(applied.chunks), 3)
        self.assertEqual(applied.chunks[0].shape, (np.nan, 3))
        pd.testing.assert_series_equal(applied.chunks[0].dtypes, df1.dtypes)

        applied = mdf.groupby('b').apply(lambda df: df.a).tiles()
        self.assertEqual(applied.dtype, df1.a.dtype)
        self.assertEqual(applied.shape, (np.nan,))
        self.assertEqual(applied.op._op_type_, opcodes.APPLY)
        self.assertEqual(applied.op.output_types[0], OutputType.series)
        self.assertEqual(len(applied.chunks), 3)
        self.assertEqual(applied.chunks[0].shape, (np.nan,))
        self.assertEqual(applied.chunks[0].dtype, df1.a.dtype)

        applied = mdf.groupby('b').apply(lambda df: df.a.sum()).tiles()
        self.assertEqual(applied.dtype, df1.a.dtype)
        self.assertEqual(applied.shape, (np.nan,))
        self.assertEqual(applied.op._op_type_, opcodes.APPLY)
        self.assertEqual(applied.op.output_types[0], OutputType.series)
        self.assertEqual(len(applied.chunks), 3)
        self.assertEqual(applied.chunks[0].shape, (np.nan,))
        self.assertEqual(applied.chunks[0].dtype, df1.a.dtype)

        series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])

        ms1 = md.Series(series1, chunk_size=3)
        applied = ms1.groupby(lambda x: x % 3).apply(apply_series).tiles()
        self.assertEqual(applied.dtype, series1.dtype)
        self.assertEqual(applied.shape, (np.nan,))
        self.assertEqual(applied.op._op_type_, opcodes.APPLY)
        self.assertEqual(applied.op.output_types[0], OutputType.series)
        self.assertEqual(len(applied.chunks), 3)
        self.assertEqual(applied.chunks[0].shape, (np.nan,))
        self.assertEqual(applied.chunks[0].dtype, series1.dtype)

    def testGroupByTransform(self):
        df1 = pd.DataFrame({
            'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
            'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
            'c': list('aabaaddce'),
            'd': [3, 4, 5, 3, 5, 4, 1, 2, 3],
            'e': [1, 3, 4, 5, 6, 5, 4, 4, 4],
            'f': list('aabaaddce'),
        })

        def transform_df(df):
            return df.sort_index()

        mdf = md.DataFrame(df1, chunk_size=3)

        with self.assertRaises(TypeError):
            mdf.groupby('b').transform(['cummax', 'cumcount'])

        r = mdf.groupby('b').transform(transform_df).tiles()
        self.assertListEqual(r.dtypes.index.tolist(), list('acdef'))
        self.assertEqual(r.shape, (9, 5))
        self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
        self.assertEqual(r.op.output_types[0], OutputType.dataframe)
        self.assertEqual(len(r.chunks), 3)
        self.assertEqual(r.chunks[0].shape, (np.nan, 5))
        self.assertListEqual(r.chunks[0].dtypes.index.tolist(), list('acdef'))

        r = mdf.groupby('b').transform(['cummax', 'cumcount'], _call_agg=True).tiles()
        self.assertEqual(r.shape, (np.nan, 6))
        self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
        self.assertEqual(r.op.output_types[0], OutputType.dataframe)
        self.assertEqual(len(r.chunks), 3)
        self.assertEqual(r.chunks[0].shape, (np.nan, 6))

        agg_dict = OrderedDict([('d', 'cummax'), ('b', 'cumsum')])
        r = mdf.groupby('b').transform(agg_dict, _call_agg=True).tiles()
        self.assertEqual(r.shape, (np.nan, 2))
        self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
        self.assertEqual(r.op.output_types[0], OutputType.dataframe)
        self.assertEqual(len(r.chunks), 3)
        self.assertEqual(r.chunks[0].shape, (np.nan, 2))

        agg_list = ['sum', lambda s: s.sum()]
        r = mdf.groupby('b').transform(agg_list, _call_agg=True).tiles()
        self.assertEqual(r.shape, (np.nan, 10))
        self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
        self.assertEqual(r.op.output_types[0], OutputType.dataframe)
        self.assertEqual(len(r.chunks), 3)
        self.assertEqual(r.chunks[0].shape, (np.nan, 10))

        series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
        ms1 = md.Series(series1, chunk_size=3)

        r = ms1.groupby(lambda x: x % 3).transform(lambda x: x + 1).tiles()
        self.assertEqual(r.dtype, series1.dtype)
        self.assertEqual(r.shape, series1.shape)
        self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
        self.assertEqual(r.op.output_types[0], OutputType.series)
        self.assertEqual(len(r.chunks), 3)
        self.assertEqual(r.chunks[0].shape, (np.nan,))
        self.assertEqual(r.chunks[0].dtype, series1.dtype)

        r = ms1.groupby(lambda x: x % 3).transform('cummax', _call_agg=True).tiles()
        self.assertEqual(r.shape, (np.nan,))
        self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
        self.assertEqual(r.op.output_types[0], OutputType.series)
        self.assertEqual(len(r.chunks), 3)
        self.assertEqual(r.chunks[0].shape, (np.nan,))

        agg_list = ['cummax', 'cumcount']
        r = ms1.groupby(lambda x: x % 3).transform(agg_list, _call_agg=True).tiles()
        self.assertEqual(r.shape, (np.nan, 2))
        self.assertEqual(r.op._op_type_, opcodes.TRANSFORM)
        self.assertEqual(r.op.output_types[0], OutputType.dataframe)
        self.assertEqual(len(r.chunks), 3)
        self.assertEqual(r.chunks[0].shape, (np.nan, 2))

    def testGroupByCum(self):
        df1 = pd.DataFrame({'a': [3, 5, 2, 7, 1, 2, 4, 6, 2, 4],
                            'b': [8, 3, 4, 1, 8, 2, 2, 2, 2, 3],
                            'c': [1, 8, 8, 5, 3, 5, 0, 0, 5, 4]})
        mdf = md.DataFrame(df1, chunk_size=3)

        for fun in ['cummin', 'cummax', 'cumprod', 'cumsum']:
            r = getattr(mdf.groupby('b'), fun)().tiles()
            self.assertEqual(r.op.output_types[0], OutputType.dataframe)
            self.assertEqual(len(r.chunks), 4)
            self.assertEqual(r.shape, (len(df1), 2))
            self.assertEqual(r.chunks[0].shape, (np.nan, 2))
            pd.testing.assert_index_equal(r.chunks[0].columns_value.to_pandas(), pd.Index(['a', 'c']))

            r = getattr(mdf.groupby('b'), fun)(axis=1).tiles()
            self.assertEqual(r.op.output_types[0], OutputType.dataframe)
            self.assertEqual(len(r.chunks), 4)
            self.assertEqual(r.shape, (len(df1), 3))
            self.assertEqual(r.chunks[0].shape, (np.nan, 3))
            pd.testing.assert_index_equal(r.chunks[0].columns_value.to_pandas(), df1.columns)

        r = mdf.groupby('b').cumcount().tiles()
        self.assertEqual(r.op.output_types[0], OutputType.series)
        self.assertEqual(len(r.chunks), 4)
        self.assertEqual(r.shape, (len(df1),))
        self.assertEqual(r.chunks[0].shape, (np.nan,))

        series1 = pd.Series([2, 2, 5, 7, 3, 7, 8, 8, 5, 6])
        ms1 = md.Series(series1, chunk_size=3)

        for fun in ['cummin', 'cummax', 'cumprod', 'cumsum', 'cumcount']:
            r = getattr(ms1.groupby(lambda x: x % 2), fun)().tiles()
            self.assertEqual(r.op.output_types[0], OutputType.series)
            self.assertEqual(len(r.chunks), 4)
            self.assertEqual(r.shape, (len(series1),))
            self.assertEqual(r.chunks[0].shape, (np.nan,))
