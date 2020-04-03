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

from collections import OrderedDict

import numpy as np
import pandas as pd

import mars.dataframe as md
from mars.tests.core import TestBase, ExecutorForTest, assert_groupby_equal


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.executor = ExecutorForTest()

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

        r = mdf.groupby(level=0)[['a', 'b']].sum()
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

        r = mdf.groupby('b')[['a', 'b']].sum()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df1.groupby('b')[['a', 'b']].sum())

        r = mdf.groupby('b')[['a', 'b']].agg(['sum', 'count'])
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df1.groupby('b')[['a', 'b']].agg(['sum', 'count']))

        r = mdf.groupby('b')[['a', 'c']].agg(['sum', 'count'])
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

        r = mdf.groupby('b').a.sum()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       df1.groupby('b').a.sum())

        r = mdf.groupby('b').a.agg(['sum', 'mean', 'var'])
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df1.groupby('b').a.agg(['sum', 'mean', 'var']))

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
        rs = np.random.RandomState(0)
        df1 = pd.DataFrame({'a': rs.choice([2, 3, 4], size=(100,)),
                            'b': rs.choice([2, 3, 4], size=(100,))})
        mdf = md.DataFrame(df1, chunk_size=3)

        df2 = pd.DataFrame({'c1': np.arange(10).astype(np.int64),
                            'c2': rs.choice(['a', 'b', 'c'], (10,)),
                            'c3': rs.rand(10)})
        mdf2 = md.DataFrame(df2, chunk_size=2)

        for method in ['tree', 'shuffle']:
            r1 = mdf.groupby('a').agg('sum', method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r1, concat=True)[0],
                                          df1.groupby('a').agg('sum'))
            r2 = mdf.groupby('b').agg('min', method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r2, concat=True)[0],
                                          df1.groupby('b').agg('min'))

            r1 = mdf2.groupby('c2').agg('prod', method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r1, concat=True)[0],
                                          df2.groupby('c2').agg('prod'))
            r2 = mdf2.groupby('c2').agg('max', method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r2, concat=True)[0],
                                          df2.groupby('c2').agg('max'))
            r3 = mdf2.groupby('c2').agg('count', method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r3, concat=True)[0],
                                          df2.groupby('c2').agg('count'))
            r4 = mdf2.groupby('c2').agg('mean', method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r4, concat=True)[0],
                                          df2.groupby('c2').agg('mean'))
            r5 = mdf2.groupby('c2').agg('var', method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r5, concat=True)[0],
                                          df2.groupby('c2').agg('var'))
            r6 = mdf2.groupby('c2').agg('std', method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r6, concat=True)[0],
                                          df2.groupby('c2').agg('std'))

            agg = ['std', 'mean', 'var', 'max', 'count']
            r3 = mdf2.groupby('c2').agg(agg, method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r3, concat=True)[0],
                                          df2.groupby('c2').agg(agg))

            agg = OrderedDict([('c1', ['min', 'mean']), ('c3', 'std')])
            r3 = mdf2.groupby('c2').agg(agg, method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r3, concat=True)[0],
                                          df2.groupby('c2').agg(agg))

            agg = OrderedDict([('c1', 'min'), ('c3', 'sum')])
            r3 = mdf2.groupby('c2').agg(agg, method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r3, concat=True)[0],
                                          df2.groupby('c2').agg(agg))

            r3 = mdf2.groupby('c2').agg({'c1': 'min'}, method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r3, concat=True)[0],
                                          df2.groupby('c2').agg({'c1': 'min'}))

        r4 = mdf2.groupby('c2').sum()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r4, concat=True)[0],
                                      df2.groupby('c2').sum())

        r5 = mdf2.groupby('c2').prod()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r5, concat=True)[0],
                                      df2.groupby('c2').prod())

        r6 = mdf2.groupby('c2').min()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r6, concat=True)[0],
                                      df2.groupby('c2').min())

        r7 = mdf2.groupby('c2').max()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r7, concat=True)[0],
                                      df2.groupby('c2').max())

        r8 = mdf2.groupby('c2').count()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r8, concat=True)[0],
                                      df2.groupby('c2').count())

        r9 = mdf2.groupby('c2').mean()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r9, concat=True)[0],
                                      df2.groupby('c2').mean())

        r10 = mdf2.groupby('c2').var()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r10, concat=True)[0],
                                      df2.groupby('c2').var())

        r11 = mdf2.groupby('c2').std()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r11, concat=True)[0],
                                      df2.groupby('c2').std())

        # test as_index=False
        r12 = mdf2.groupby('c2', as_index=False).agg('mean')
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r12, concat=True)[0],
                                      df2.groupby('c2', as_index=False).agg('mean'))
        self.assertFalse(r12.op.groupby_params['as_index'])

        # test as_index=False takes no effect
        r13 = mdf2.groupby(['c1', 'c2'], as_index=False).agg(['mean', 'count'])
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r13, concat=True)[0],
                                      df2.groupby(['c1', 'c2'], as_index=False).agg(['mean', 'count']))
        self.assertTrue(r13.op.groupby_params['as_index'])

        r14 = mdf2.groupby('c2').agg(['cumsum', 'cumcount'])
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r14, concat=True)[0].sort_index(),
                                      df2.groupby('c2').agg(['cumsum', 'cumcount']).sort_index())

    def testSeriesGroupByAgg(self):
        rs = np.random.RandomState(0)
        series1 = pd.Series(rs.rand(10))
        ms1 = md.Series(series1, chunk_size=3)

        for method in ['tree', 'shuffle']:
            r1 = ms1.groupby(lambda x: x % 2).agg('sum', method=method)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(r1, concat=True)[0],
                                           series1.groupby(lambda x: x % 2).agg('sum'))
            r2 = ms1.groupby(lambda x: x % 2).agg('min', method=method)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(r2, concat=True)[0],
                                           series1.groupby(lambda x: x % 2).agg('min'))

            r1 = ms1.groupby(lambda x: x % 2).agg('prod', method=method)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(r1, concat=True)[0],
                                           series1.groupby(lambda x: x % 2).agg('prod'))
            r2 = ms1.groupby(lambda x: x % 2).agg('max', method=method)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(r2, concat=True)[0],
                                           series1.groupby(lambda x: x % 2).agg('max'))
            r3 = ms1.groupby(lambda x: x % 2).agg('count', method=method)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(r3, concat=True)[0],
                                           series1.groupby(lambda x: x % 2).agg('count'))
            r4 = ms1.groupby(lambda x: x % 2).agg('mean', method=method)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(r4, concat=True)[0],
                                           series1.groupby(lambda x: x % 2).agg('mean'))
            r5 = ms1.groupby(lambda x: x % 2).agg('var', method=method)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(r5, concat=True)[0],
                                           series1.groupby(lambda x: x % 2).agg('var'))
            r6 = ms1.groupby(lambda x: x % 2).agg('std', method=method)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(r6, concat=True)[0],
                                           series1.groupby(lambda x: x % 2).agg('std'))

            agg = ['std', 'mean', 'var', 'max', 'count']
            r3 = ms1.groupby(lambda x: x % 2).agg(agg, method=method)
            pd.testing.assert_frame_equal(self.executor.execute_dataframe(r3, concat=True)[0],
                                          series1.groupby(lambda x: x % 2).agg(agg))

        r4 = ms1.groupby(lambda x: x % 2).sum()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r4, concat=True)[0],
                                       series1.groupby(lambda x: x % 2).sum())

        r5 = ms1.groupby(lambda x: x % 2).prod()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r5, concat=True)[0],
                                       series1.groupby(lambda x: x % 2).prod())

        r6 = ms1.groupby(lambda x: x % 2).min()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r6, concat=True)[0],
                                       series1.groupby(lambda x: x % 2).min())

        r7 = ms1.groupby(lambda x: x % 2).max()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r7, concat=True)[0],
                                       series1.groupby(lambda x: x % 2).max())

        r8 = ms1.groupby(lambda x: x % 2).count()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r8, concat=True)[0],
                                       series1.groupby(lambda x: x % 2).count())

        r9 = ms1.groupby(lambda x: x % 2).mean()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r9, concat=True)[0],
                                       series1.groupby(lambda x: x % 2).mean())

        r10 = ms1.groupby(lambda x: x % 2).var()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r10, concat=True)[0],
                                       series1.groupby(lambda x: x % 2).var())

        r11 = ms1.groupby(lambda x: x % 2).std()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r11, concat=True)[0],
                                       series1.groupby(lambda x: x % 2).std())

        r11 = ms1.groupby(lambda x: x % 2).agg(['cumsum', 'cumcount'])
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r11, concat=True)[0].sort_index(),
                                      series1.groupby(lambda x: x % 2).agg(['cumsum', 'cumcount']).sort_index())

    def testGroupByApply(self):
        df1 = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                            'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                            'c': list('aabaaddce')})

        def apply_df(df):
            df = df.sort_index()
            df.a += df.b
            if len(df.index) > 0:
                df = df.iloc[:-1, :]
            return df

        def apply_series(s, truncate=True):
            s = s.sort_index()
            if truncate and len(s.index) > 0:
                s = s.iloc[:-1]
            return s

        mdf = md.DataFrame(df1, chunk_size=3)

        applied = mdf.groupby('b').apply(apply_df)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(applied, concat=True)[0].sort_index(),
                                      df1.groupby('b').apply(apply_df).sort_index())

        applied = mdf.groupby('b').apply(lambda df: df.a)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(applied, concat=True)[0].sort_index(),
                                       df1.groupby('b').apply(lambda df: df.a).sort_index())

        applied = mdf.groupby('b').apply(lambda df: df.a.sum())
        pd.testing.assert_series_equal(self.executor.execute_dataframe(applied, concat=True)[0].sort_index(),
                                       df1.groupby('b').apply(lambda df: df.a.sum()).sort_index())

        series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
        ms1 = md.Series(series1, chunk_size=3)

        applied = ms1.groupby(lambda x: x % 3).apply(apply_series)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(applied, concat=True)[0].sort_index(),
                                       series1.groupby(lambda x: x % 3).apply(apply_series).sort_index())

        sindex2 = pd.MultiIndex.from_arrays([list(range(9)), list('ABCDEFGHI')])
        series2 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3], index=sindex2)
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

        r = mdf.groupby('b').transform(['cummax', 'cumsum'], _call_agg=True)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                      df1.groupby('b').agg(['cummax', 'cumsum']).sort_index())

        agg_list = ['cummax', 'cumsum']
        r = mdf.groupby('b').transform(agg_list, _call_agg=True)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0].sort_index(),
                                      df1.groupby('b').agg(agg_list).sort_index())

        agg_dict = {'d': 'cummax', 'b': 'cumsum'}
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
