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
from collections import OrderedDict

import numpy as np
import pandas as pd
try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

import mars.dataframe as md
from mars.operands import ShuffleProxy
from mars.tests.core import TestBase, ExecutorForTest, assert_groupby_equal, require_cudf
from mars.utils import arrow_array_to_objects


class MockReduction1(md.CustomReduction):
    def agg(self, v1):
        return v1.sum()


class MockReduction2(md.CustomReduction):
    def pre(self, value):
        return value + 1, value * 2

    def agg(self, v1, v2):
        return v1.sum(), v2.min()

    def post(self, v1, v2):
        return v1 + v2


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.executor = ExecutorForTest('numpy')
        self.ctx, self.executor = self._create_test_context(self.executor)
        self.ctx.__enter__()

    def tearDown(self) -> None:
        self.ctx.__exit__(None, None, None)

    def testGroupBy(self):
        df1 = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                            'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                            'c': list('aabaaddce')})
        mdf = md.DataFrame(df1, chunk_size=3)
        grouped = mdf.groupby('b')
        assert_groupby_equal(self.executor.execute_dataframe(grouped, concat=True)[0],
                             df1.groupby('b'))

        df2 = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                            'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                            'c': list('aabaaddce')},
                           index=['i' + str(i) for i in range(9)])
        mdf = md.DataFrame(df2, chunk_size=3)
        grouped = mdf.groupby('b')
        assert_groupby_equal(self.executor.execute_dataframe(grouped, concat=True)[0],
                             df2.groupby('b'))

        # test groupby series
        grouped = mdf.groupby(mdf['b'])
        assert_groupby_equal(self.executor.execute_dataframe(grouped, concat=True)[0],
                             df2.groupby(df2['b']))

        # test groupby multiple series
        grouped = mdf.groupby(by=[mdf['b'], mdf['c']])
        assert_groupby_equal(self.executor.execute_dataframe(grouped, concat=True)[0],
                             df2.groupby(by=[df2['b'], df2['c']]))

        df3 = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                            'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                            'c': list('aabaaddce')},
                           index=pd.MultiIndex.from_tuples([(i % 3, 'i' + str(i)) for i in range(9)]))
        mdf = md.DataFrame(df3, chunk_size=3)
        grouped = mdf.groupby(level=0)
        assert_groupby_equal(self.executor.execute_dataframe(grouped, concat=True)[0],
                             df3.groupby(level=0))

        series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
        ms1 = md.Series(series1, chunk_size=3)
        grouped = ms1.groupby(lambda x: x % 3)
        assert_groupby_equal(self.executor.execute_dataframe(grouped, concat=True)[0],
                             series1.groupby(lambda x: x % 3))

        # test groupby series
        grouped = ms1.groupby(ms1)
        assert_groupby_equal(self.executor.execute_dataframe(grouped, concat=True)[0],
                             series1.groupby(series1))

        series2 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3],
                            index=['i' + str(i) for i in range(9)])
        ms2 = md.Series(series2, chunk_size=3)
        grouped = ms2.groupby(lambda x: int(x[1:]) % 3)
        assert_groupby_equal(self.executor.execute_dataframe(grouped, concat=True)[0],
                             series2.groupby(lambda x: int(x[1:]) % 3))

    def testGroupByGetItem(self):
        df1 = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                            'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                            'c': list('aabaaddce')},
                           index=pd.MultiIndex.from_tuples([(i % 3, 'i' + str(i)) for i in range(9)]))
        mdf = md.DataFrame(df1, chunk_size=3)

        r = mdf.groupby(level=0)[['a', 'b']]
        assert_groupby_equal(self.executor.execute_dataframe(r, concat=True)[0],
                             df1.groupby(level=0)[['a', 'b']], with_selection=True)

        for method in ('tree', 'shuffle'):
            r = mdf.groupby(level=0)[['a', 'b']].sum(method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                          df1.groupby(level=0)[['a', 'b']].sum())

        r = mdf.groupby(level=0)[['a', 'b']].apply(lambda x: x + 1)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                      df1.groupby(level=0)[['a', 'b']].apply(lambda x: x + 1).sort_index())

        r = mdf.groupby('b')[['a', 'b']]
        assert_groupby_equal(self.executor.execute_dataframe(r, concat=True)[0],
                             df1.groupby('b')[['a', 'b']], with_selection=True)

        r = mdf.groupby('b')[['a', 'c']]
        assert_groupby_equal(self.executor.execute_dataframe(r, concat=True)[0],
                             df1.groupby('b')[['a', 'c']], with_selection=True)

        for method in ('tree', 'shuffle'):
            r = mdf.groupby('b')[['a', 'b']].sum(method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                          df1.groupby('b')[['a', 'b']].sum())

            r = mdf.groupby('b')[['a', 'b']].agg(['sum', 'count'], method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                          df1.groupby('b')[['a', 'b']].agg(['sum', 'count']))

            r = mdf.groupby('b')[['a', 'c']].agg(['sum', 'count'], method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                          df1.groupby('b')[['a', 'c']].agg(['sum', 'count']))

        r = mdf.groupby('b')[['a', 'b']].apply(lambda x: x + 1)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                      df1.groupby('b')[['a', 'b']].apply(lambda x: x + 1).sort_index())

        r = mdf.groupby('b')[['a', 'b']].transform(lambda x: x + 1)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                      df1.groupby('b')[['a', 'b']].transform(lambda x: x + 1).sort_index())

        r = mdf.groupby('b')[['a', 'b']].cumsum()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                      df1.groupby('b')[['a', 'b']].cumsum().sort_index())

        r = mdf.groupby('b').a
        assert_groupby_equal(self.executor.execute_dataframe(r, concat=True)[0],
                             df1.groupby('b').a, with_selection=True)

        for method in ('shuffle', 'tree'):
            r = mdf.groupby('b').a.sum(method=method)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                           df1.groupby('b').a.sum())

            r = mdf.groupby('b').a.agg(['sum', 'mean', 'var'], method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                          df1.groupby('b').a.agg(['sum', 'mean', 'var']))

            r = mdf.groupby('b', as_index=False).a.sum(method=method)
            pd.testing.assert_frame_equal(
                self.executor.execute_dataframe(r, concat=True)[0].sort_values('b', ignore_index=True),
                df1.groupby('b', as_index=False).a.sum().sort_values('b', ignore_index=True))

            r = mdf.groupby('b', as_index=False).b.count(method=method)
            pd.testing.assert_frame_equal(
                self.executor.execute_dataframe(r, concat=True)[0].sort_values('b', ignore_index=True),
                df1.groupby('b', as_index=False).b.count().sort_values('b', ignore_index=True))

            r = mdf.groupby('b', as_index=False).b.agg({'cnt': 'count'}, method=method)
            pd.testing.assert_frame_equal(
                self.executor.execute_dataframe(r, concat=True)[0].sort_values('b', ignore_index=True),
                df1.groupby('b', as_index=False).b.agg({'cnt': 'count'}).sort_values('b', ignore_index=True))

        r = mdf.groupby('b').a.apply(lambda x: x + 1)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                       df1.groupby('b').a.apply(lambda x: x + 1).sort_index())

        r = mdf.groupby('b').a.transform(lambda x: x + 1)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                       df1.groupby('b').a.transform(lambda x: x + 1).sort_index())

        r = mdf.groupby('b').a.cumsum()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                       df1.groupby('b').a.cumsum().sort_index())

    def testDataFrameGroupByAgg(self):
        agg_funs = ['std', 'mean', 'var', 'max', 'count', 'size', 'all', 'any', 'skew', 'kurt', 'sem']

        rs = np.random.RandomState(0)
        raw = pd.DataFrame({'c1': np.arange(100).astype(np.int64),
                            'c2': rs.choice(['a', 'b', 'c'], (100,)),
                            'c3': rs.rand(100)})
        mdf = md.DataFrame(raw, chunk_size=13)

        for method in ['tree', 'shuffle']:
            r = mdf.groupby('c2').agg('size', method=method)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                           raw.groupby('c2').agg('size'))

            for agg_fun in agg_funs:
                if agg_fun == 'size':
                    continue
                r = mdf.groupby('c2').agg(agg_fun, method=method)
                pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                              raw.groupby('c2').agg(agg_fun))

            r = mdf.groupby('c2').agg(agg_funs, method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                          raw.groupby('c2').agg(agg_funs))

            agg = OrderedDict([('c1', ['min', 'mean']), ('c3', 'std')])
            r = mdf.groupby('c2').agg(agg, method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                          raw.groupby('c2').agg(agg))

            agg = OrderedDict([('c1', 'min'), ('c3', 'sum')])
            r = mdf.groupby('c2').agg(agg, method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                          raw.groupby('c2').agg(agg))

            r = mdf.groupby('c2').agg({'c1': 'min'}, method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                          raw.groupby('c2').agg({'c1': 'min'}))

            # test groupby series
            r = mdf.groupby(mdf['c2']).sum(method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                          raw.groupby(raw['c2']).sum())

        r = mdf.groupby('c2').size(method='tree')
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       raw.groupby('c2').size())

        # test inserted kurt method
        r = mdf.groupby('c2').kurtosis(method='tree')
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      raw.groupby('c2').kurtosis())

        for agg_fun in agg_funs:
            if agg_fun == 'size' or callable(agg_fun):
                continue
            r = getattr(mdf.groupby('c2'), agg_fun)(method='tree')
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                          getattr(raw.groupby('c2'), agg_fun)())

        for method in ['tree', 'shuffle']:
            # test as_index=False
            r = mdf.groupby('c2', as_index=False).agg('mean', method=method)
            pd.testing.assert_frame_equal(
                self.executor.execute_dataframe(r, concat=True)[0].sort_values('c2', ignore_index=True),
                raw.groupby('c2', as_index=False).agg('mean').sort_values('c2', ignore_index=True))
            self.assertFalse(r.op.groupby_params['as_index'])

        # test as_index=False takes no effect
        r = mdf.groupby(['c1', 'c2'], as_index=False).agg(['mean', 'count'], method='tree')
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      raw.groupby(['c1', 'c2'], as_index=False).agg(['mean', 'count']))
        self.assertTrue(r.op.groupby_params['as_index'])

        r = mdf.groupby('c2').agg(['cumsum', 'cumcount'], method='tree')
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                      raw.groupby('c2').agg(['cumsum', 'cumcount']).sort_index())

        # test auto method
        r = mdf.groupby('c2').agg('prod')
        self.assertEqual(r.op.method, 'auto')
        self.assertTrue(all((not isinstance(c.op, ShuffleProxy)) for c in r.build_graph(tiled=True)))

        r = mdf[['c1', 'c3']].groupby(mdf['c2']).agg(MockReduction2())
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      raw[['c1', 'c3']].groupby(raw['c2']).agg(MockReduction2()))

        r = mdf.groupby('c2').agg(sum_c1=md.NamedAgg('c1', 'sum'), min_c1=md.NamedAgg('c1', 'min'),
                                  mean_c3=md.NamedAgg('c3', 'mean'))
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      raw.groupby('c2').agg(sum_c1=md.NamedAgg('c1', 'sum'),
                                                            min_c1=md.NamedAgg('c1', 'min'),
                                                            mean_c3=md.NamedAgg('c3', 'mean')))

    def testSeriesGroupByAgg(self):
        rs = np.random.RandomState(0)
        series1 = pd.Series(rs.rand(10))
        ms1 = md.Series(series1, chunk_size=3)

        agg_funs = ['std', 'mean', 'var', 'max', 'count', 'size', 'all', 'any', 'skew', 'kurt', 'sem']

        for method in ['tree', 'shuffle']:
            for agg_fun in agg_funs:
                r = ms1.groupby(lambda x: x % 2).agg(agg_fun, method=method)
                pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                               series1.groupby(lambda x: x % 2).agg(agg_fun))

            r = ms1.groupby(lambda x: x % 2).agg(agg_funs, method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                          series1.groupby(lambda x: x % 2).agg(agg_funs))

            # test groupby series
            r = ms1.groupby(ms1).sum(method=method)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                           series1.groupby(series1).sum())

            r = ms1.groupby(ms1).sum(method=method)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                           series1.groupby(series1).sum())

        # test inserted kurt method
        r = ms1.groupby(ms1).kurtosis()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series1.groupby(series1).kurtosis())

        for agg_fun in agg_funs:
            r = getattr(ms1.groupby(lambda x: x % 2), agg_fun)(method='tree')
            pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                           getattr(series1.groupby(lambda x: x % 2), agg_fun)())

        r = ms1.groupby(lambda x: x % 2).agg(['cumsum', 'cumcount'], method='tree')
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                      series1.groupby(lambda x: x % 2).agg(['cumsum', 'cumcount']).sort_index())

        r = ms1.groupby(lambda x: x % 2).agg(MockReduction2(name='custom_r'))
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series1.groupby(lambda x: x % 2).agg(MockReduction2(name='custom_r')))

        r = ms1.groupby(lambda x: x % 2).agg(col_var='var', col_skew='skew')
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      series1.groupby(lambda x: x % 2).agg(col_var='var', col_skew='skew'))

    @require_cudf
    def testGPUGroupByAgg(self):
        rs = np.random.RandomState(0)
        df1 = pd.DataFrame({'a': rs.choice([2, 3, 4], size=(100,)),
                            'b': rs.choice([2, 3, 4], size=(100,))})
        mdf = md.DataFrame(df1, chunk_size=13).to_gpu()

        r = mdf.groupby('a').sum()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].to_pandas(),
                                      df1.groupby('a').sum())

        r = mdf.groupby('a').kurt()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].to_pandas(),
                                      df1.groupby('a').kurt())

        r = mdf.groupby('a').agg(['sum', 'var'])
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].to_pandas(),
                                      df1.groupby('a').agg(['sum', 'var']))

        rs = np.random.RandomState(0)
        idx = pd.Index(np.where(rs.rand(10) > 0.5, 'A', 'B'))
        series1 = pd.Series(rs.rand(10), index=idx)
        ms = md.Series(series1, index=idx, chunk_size=3).to_gpu().to_gpu()

        r = ms.groupby(level=0).sum()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0].to_pandas(),
                                       series1.groupby(level=0).sum())

        r = ms.groupby(level=0).kurt()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0].to_pandas(),
                                       series1.groupby(level=0).kurt())

        r = ms.groupby(level=0).agg(['sum', 'var'])
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].to_pandas(),
                                      series1.groupby(level=0).agg(['sum', 'var']))

    def testGroupByApply(self):
        df1 = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                            'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                            'c': list('aabaaddce')})

        def apply_df(df, ret_series=False):
            df = df.sort_index()
            df.a += df.b
            if len(df.index) > 0:
                if not ret_series:
                    df = df.iloc[:-1, :]
                else:
                    df = df.iloc[-1, :]
            return df

        def apply_series(s, truncate=True):
            s = s.sort_index()
            if truncate and len(s.index) > 0:
                s = s.iloc[:-1]
            return s

        mdf = md.DataFrame(df1, chunk_size=3)

        applied = mdf.groupby('b').apply(lambda df: None)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(applied, concat=True)[0],
                                      df1.groupby('b').apply(lambda df: None))

        applied = mdf.groupby('b').apply(apply_df)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(applied, concat=True)[0].sort_index(),
                                      df1.groupby('b').apply(apply_df).sort_index())

        applied = mdf.groupby('b').apply(apply_df, ret_series=True)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(applied, concat=True)[0].sort_index(),
                                      df1.groupby('b').apply(apply_df, ret_series=True).sort_index())

        applied = mdf.groupby('b').apply(lambda df: df.a, output_type='series')
        pd.testing.assert_series_equal(self.executor.execute_dataframe(applied, concat=True)[0].sort_index(),
                                       df1.groupby('b').apply(lambda df: df.a).sort_index())

        applied = mdf.groupby('b').apply(lambda df: df.a.sum())
        pd.testing.assert_series_equal(self.executor.execute_dataframe(applied, concat=True)[0].sort_index(),
                                       df1.groupby('b').apply(lambda df: df.a.sum()).sort_index())

        series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
        ms1 = md.Series(series1, chunk_size=3)

        applied = ms1.groupby(lambda x: x % 3).apply(lambda df: None)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(applied, concat=True)[0],
                                       series1.groupby(lambda x: x % 3).apply(lambda df: None))

        applied = ms1.groupby(lambda x: x % 3).apply(apply_series)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(applied, concat=True)[0].sort_index(),
                                       series1.groupby(lambda x: x % 3).apply(apply_series).sort_index())

        sindex2 = pd.MultiIndex.from_arrays([list(range(9)), list('ABCDEFGHI')])
        series2 = pd.Series(list('CDECEDABC'), index=sindex2)
        ms2 = md.Series(series2, chunk_size=3)

        applied = ms2.groupby(lambda x: x[0] % 3).apply(apply_series)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(applied, concat=True)[0].sort_index(),
                                       series2.groupby(lambda x: x[0] % 3).apply(apply_series).sort_index())

    def testGroupByTransform(self):
        df1 = pd.DataFrame({
            'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
            'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
            'c': list('aabaaddce'),
            'd': [3, 4, 5, 3, 5, 4, 1, 2, 3],
            'e': [1, 3, 4, 5, 6, 5, 4, 4, 4],
            'f': list('aabaaddce'),
        })

        def transform_series(s, truncate=True):
            s = s.sort_index()
            if truncate and len(s.index) > 1:
                s = s.iloc[:-1].reset_index(drop=True)
            return s

        mdf = md.DataFrame(df1, chunk_size=3)

        r = mdf.groupby('b').transform(transform_series, truncate=False)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                      df1.groupby('b').transform(transform_series, truncate=False).sort_index())

        if pd.__version__ != '1.1.0':
            r = mdf.groupby('b').transform(['cummax', 'cumsum'], _call_agg=True)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                          df1.groupby('b').agg(['cummax', 'cumsum']).sort_index())

            agg_list = ['cummax', 'cumsum']
            r = mdf.groupby('b').transform(agg_list, _call_agg=True)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                          df1.groupby('b').agg(agg_list).sort_index())

            agg_dict = OrderedDict([('d', 'cummax'), ('b', 'cumsum')])
            r = mdf.groupby('b').transform(agg_dict, _call_agg=True)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                          df1.groupby('b').agg(agg_dict).sort_index())

        agg_list = ['sum', lambda s: s.sum()]
        r = mdf.groupby('b').transform(agg_list, _call_agg=True)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                      df1.groupby('b').agg(agg_list).sort_index())

        series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
        ms1 = md.Series(series1, chunk_size=3)

        r = ms1.groupby(lambda x: x % 3).transform(lambda x: x + 1)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                       series1.groupby(lambda x: x % 3).transform(lambda x: x + 1).sort_index())

        r = ms1.groupby(lambda x: x % 3).transform('cummax', _call_agg=True)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                       series1.groupby(lambda x: x % 3).agg('cummax').sort_index())

        agg_list = ['cummax', 'cumcount']
        r = ms1.groupby(lambda x: x % 3).transform(agg_list, _call_agg=True)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                      series1.groupby(lambda x: x % 3).agg(agg_list).sort_index())

    def testGroupByCum(self):
        df1 = pd.DataFrame({'a': [3, 5, 2, 7, 1, 2, 4, 6, 2, 4],
                            'b': [8, 3, 4, 1, 8, 2, 2, 2, 2, 3],
                            'c': [1, 8, 8, 5, 3, 5, 0, 0, 5, 4]})
        mdf = md.DataFrame(df1, chunk_size=3)

        for fun in ['cummin', 'cummax', 'cumprod', 'cumsum']:
            r1 = getattr(mdf.groupby('b'), fun)()
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r1, concat=True)[0].sort_index(),
                                          getattr(df1.groupby('b'), fun)().sort_index())

            r2 = getattr(mdf.groupby('b'), fun)(axis=1)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r2, concat=True)[0].sort_index(),
                                          getattr(df1.groupby('b'), fun)(axis=1).sort_index())

        r3 = mdf.groupby('b').cumcount()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r3, concat=True)[0].sort_index(),
                                       df1.groupby('b').cumcount().sort_index())

        series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
        ms1 = md.Series(series1, chunk_size=3)

        for fun in ['cummin', 'cummax', 'cumprod', 'cumsum', 'cumcount']:
            r1 = getattr(ms1.groupby(lambda x: x % 2), fun)()
            pd.testing.assert_series_equal(self.executor.execute_dataframe(r1, concat=True)[0].sort_index(),
                                           getattr(series1.groupby(lambda x: x % 2), fun)().sort_index())

    @unittest.skipIf(pa is None, 'pyarrow not installed')
    def testGroupbyAggWithArrowDtype(self):
        df1 = pd.DataFrame({'a': [1, 2, 1],
                            'b': ['a', 'b', 'a']})
        mdf = md.DataFrame(df1)
        mdf['b'] = mdf['b'].astype('Arrow[string]')

        r = mdf.groupby('a').count()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = df1.groupby('a').count()
        pd.testing.assert_frame_equal(result, expected)

        r = mdf.groupby('b').count()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = df1.groupby('b').count()
        pd.testing.assert_frame_equal(result, expected)

        series1 = df1['b']
        mseries = md.Series(series1).astype('Arrow[string]')

        r = mseries.groupby(mseries).count()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = series1.groupby(series1).count()
        pd.testing.assert_series_equal(result, expected)

        series2 = series1.copy()
        series2.index = pd.MultiIndex.from_tuples([(0, 1), (2, 3), (4, 5)])
        mseries = md.Series(series2).astype('Arrow[string]')

        r = mseries.groupby(mseries).count()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = series2.groupby(series2).count()
        pd.testing.assert_series_equal(result, expected)

    @unittest.skipIf(pa is None, 'pyarrow not installed')
    def testGroupbyApplyWithArrowDtype(self):
        df1 = pd.DataFrame({'a': [1, 2, 1],
                            'b': ['a', 'b', 'a']})
        mdf = md.DataFrame(df1)
        mdf['b'] = mdf['b'].astype('Arrow[string]')

        applied = mdf.groupby('b').apply(lambda df: df.a.sum())
        result = self.executor.execute_dataframe(applied, concat=True)[0]
        expected = df1.groupby('b').apply(lambda df: df.a.sum())
        pd.testing.assert_series_equal(result, expected)

        series1 = df1['b']
        mseries = md.Series(series1).astype('Arrow[string]')

        applied = mseries.groupby(mseries).apply(lambda s: s)
        result = self.executor.execute_dataframe(applied, concat=True)[0]
        expected = series1.groupby(series1).apply(lambda s: s)
        pd.testing.assert_series_equal(arrow_array_to_objects(result), expected)
