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

from mars.core import tile
from mars.dataframe import cut
from mars.dataframe.initializer import DataFrame, Series, Index
from mars.lib.groupby_wrapper import wrapped_groupby
from mars.tests.core import TestBase


def test_dataframe_params():
    raw = pd.DataFrame({'a': [1, 2, 3]})
    df = DataFrame(raw)
    df = df[df['a'] < 2]
    df = tile(df)
    c = df.chunks[0]

    assert any(np.isnan(s) for s in c.params['shape'])
    assert np.isnan(c.params['index_value'].min_val)
    c.params = c.get_params_from_data(raw[raw['a'] < 2])
    # shape and index_value updated
    assert not any(np.isnan(s) for s in c.params['shape'])
    assert not np.isnan(c.params['index_value'].min_val)

    params = c.params.copy()
    params.pop('index', None)
    df.params = params
    assert np.prod(df.shape) > 0
    df.refresh_params()


def test_series_params():
    raw = pd.Series([1, 2, 3], name='a')
    series = Series(raw)
    series = series[series < 2]
    series = tile(series)
    c = series.chunks[0]

    assert any(np.isnan(s) for s in c.params['shape'])
    assert np.isnan(c.params['index_value'].min_val)
    c.params = c.get_params_from_data(raw[raw < 2])
    # shape and index_value updated
    assert not any(np.isnan(s) for s in c.params['shape'])
    assert not np.isnan(c.params['index_value'].min_val)

    params = c.params.copy()
    params.pop('index', None)
    series.params = params
    assert np.prod(series.shape) > 0
    series.refresh_params()


def test_index_params():
    raw = pd.Series([1, 2, 3], name='a')
    series = Series(raw)
    series = series[series < 2]
    index = series.index
    index = tile(index)
    c = index.chunks[0]

    assert any(np.isnan(s) for s in c.params['shape'])
    assert np.isnan(c.params['index_value'].min_val)
    c.params = c.get_params_from_data(raw[raw < 2].index)
    # shape and index_value updated
    assert not any(np.isnan(s) for s in c.params['shape'])
    assert not np.isnan(c.params['index_value'].min_val)

    params = c.params.copy()
    params.pop('index', None)
    index.params = params
    assert np.prod(index.shape) > 0
    index.refresh_params()


def test_categorical_params():
    raw = np.random.rand(10)
    cate = cut(raw, [0.3, 0.5, 0.7])
    cate = tile(cate)
    c = cate.chunks[0]

    c.params = c.get_params_from_data(pd.cut(raw, [0.3, 0.5, 0.7]))
    assert len(c.params['categories_value'].to_pandas()) > 0

    params = c.params.copy()
    params.pop('index', None)
    cate.params = params
    assert len(cate.params['categories_value'].to_pandas()) > 0
    cate.refresh_params()


def test_groupby_params():
    raw = pd.DataFrame({'a': [1, 2, 3]})
    df = DataFrame(raw)
    grouped = df.groupby('a')
    grouped = tile(grouped)
    c = grouped.chunks[0]

    c.params = c.get_params_from_data(wrapped_groupby(raw, by='a'))
    params = c.params.copy()
    params.pop('index', None)
    grouped.params = params

    raw = pd.Series([1, 2, 3], name='a')
    series = Series(raw)
    grouped = series.groupby(level=0)
    grouped = tile(grouped)
    c = grouped.chunks[0]

    c.params = c.get_params_from_data(wrapped_groupby(raw, level=0))
    params = c.params.copy()
    params.pop('index', None)
    grouped.params = params
    grouped.refresh_params()


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.ctx, self.executor = self._create_test_context()

    def testDataFrameDir(self):
        df = DataFrame(pd.DataFrame(np.random.rand(4, 3), columns=list('ABC')))
        dir_result = set(dir(df))
        for c in df.dtypes.index:
            self.assertIn(c, dir_result)

    def testToFrameOrSeries(self):
        raw = pd.Series(np.random.rand(10), name='col')
        series = Series(raw)

        r = series.to_frame()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(raw.to_frame(), result)

        r = series.to_frame(name='new_name')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(raw.to_frame(name='new_name'), result)

        series = series[series > 0.1]
        r = series.to_frame(name='new_name')
        result = self.executor.execute_dataframes([r])[0]
        pd.testing.assert_frame_equal(raw[raw > 0.1].to_frame(name='new_name'), result)

        raw = pd.Index(np.random.rand(10), name='col')
        index = Index(raw)

        r = index.to_frame()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(raw.to_frame(), result)

        r = index.to_frame(index=False)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(raw.to_frame(index=False), result)

        r = index.to_frame(name='new_name')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(raw.to_frame(name='new_name'), result)

        r = index.to_series()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(raw.to_series(), result)

        r = index.to_series(index=pd.RangeIndex(0, 10))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(raw.to_series(index=pd.RangeIndex(0, 10)), result)

        r = index.to_series(name='new_name')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(raw.to_series(name='new_name'), result)

        raw = pd.MultiIndex.from_tuples([('A', 'E'), ('B', 'F'), ('C', 'G')])
        index = Index(raw, tupleize_cols=True)

        r = index.to_frame()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(raw.to_frame(), result)

        with self.assertRaises(TypeError):
            index.to_frame(name='XY')

        with self.assertRaises(ValueError):
            index.to_frame(name=['X', 'Y', 'Z'])

        r = index.to_frame(name=['X', 'Y'])
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(raw.to_frame(name=['X', 'Y']), result)

        r = index.to_series(name='new_name')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(raw.to_series(name='new_name'), result)

    def testKeyValue(self):
        raw = pd.DataFrame(np.random.rand(4, 3), columns=list('ABC'))
        df = DataFrame(raw)

        result = self.executor.execute_dataframe(df.values, concat=True)[0]
        np.testing.assert_array_equal(result, raw.values)

        result = self.executor.execute_dataframe(df.keys(), concat=True)[0]
        pd.testing.assert_index_equal(result, raw.keys())

        raw = pd.Series(np.random.rand(10))
        s = Series(raw)

        result = self.executor.execute_dataframe(s.values, concat=True)[0]
        np.testing.assert_array_equal(result, raw.values)

        result = self.executor.execute_dataframe(s.keys(), concat=True)[0]
        pd.testing.assert_index_equal(result, raw.keys())

        raw = pd.Index(np.random.rand(10))
        idx = Index(raw)

        result = self.executor.execute_dataframe(idx.values, concat=True)[0]
        np.testing.assert_array_equal(result, raw.values)
