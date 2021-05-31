#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

import mars.tensor as mt
import mars.dataframe as md
from mars.core.session import new_session
from mars.config import option_context
from mars.tensor.core import TensorOrder
from mars.tensor.datasource import ArrayDataSource
from mars.tests import new_test_session


test_namedtuple_type = namedtuple('TestNamedTuple', 'a b')


@pytest.fixture(scope='module')
def setup():
    sess = new_test_session(default=True)
    with option_context({'show_progress': False}):
        try:
            yield sess
        finally:
            sess.stop_server()
    

def test_session_async_execute(setup):
    raw_a = np.random.RandomState(0).rand(10, 20)
    a = mt.tensor(raw_a)

    expected = raw_a.sum()
    res = a.sum().to_numpy(wait=False).result()
    assert expected == res
    res = a.sum().execute(wait=False)
    res = res.result().fetch()
    assert expected == res

    raw_df = pd.DataFrame(raw_a)

    expected = raw_df.sum()
    df = md.DataFrame(a)
    res = df.sum().to_pandas(wait=False).result()
    pd.testing.assert_series_equal(expected, res)
    res = df.sum().execute(wait=False)
    res = res.result().fetch()
    pd.testing.assert_series_equal(expected, res)

    t = [df.sum(), a.sum()]
    res = mt.ExecutableTuple(t).to_object(wait=False).result()
    pd.testing.assert_series_equal(raw_df.sum(), res[0])
    assert raw_a.sum() == res[1]
    res = mt.ExecutableTuple(t).execute(wait=False)
    res = res.result().fetch()
    pd.testing.assert_series_equal(raw_df.sum(), res[0])
    assert raw_a.sum() == res[1]


def test_executable_tuple_execute(setup):
    raw_a = np.random.RandomState(0).rand(10, 20)
    a = mt.tensor(raw_a)

    raw_df = pd.DataFrame(raw_a)
    df = md.DataFrame(raw_df)

    tp = test_namedtuple_type(a, df)
    executable_tp = mt.ExecutableTuple(tp)

    assert 'a' in dir(executable_tp)
    assert executable_tp.a is a
    assert test_namedtuple_type.__name__ in repr(executable_tp)
    with pytest.raises(AttributeError):
        getattr(executable_tp, 'c')

    res = mt.ExecutableTuple(tp).execute().fetch()
    assert test_namedtuple_type is type(res)

    np.testing.assert_array_equal(raw_a, res.a)
    pd.testing.assert_frame_equal(raw_df, res.b)


def test_multiple_output_execute(setup):
    data = np.random.random((5, 9))

    # test multiple outputs
    arr1 = mt.tensor(data.copy(), chunk_size=3)
    result = mt.modf(arr1).execute().fetch()
    expected = np.modf(data)

    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])

    # test 1 output
    arr2 = mt.tensor(data.copy(), chunk_size=3)
    result = ((arr2 + 1) * 2).to_numpy()
    expected = (data + 1) * 2

    np.testing.assert_array_equal(result, expected)

    # test multiple outputs, but only execute 1
    arr3 = mt.tensor(data.copy(), chunk_size=3)
    arrs = mt.split(arr3, 3, axis=1)
    result = arrs[0].to_numpy()
    expected = np.split(data, 3, axis=1)[0]

    np.testing.assert_array_equal(result, expected)

    # test multiple outputs, but only execute 1
    data = np.random.randint(0, 10, (5, 5))
    arr3 = (mt.tensor(data) + 1) * 2
    arrs = mt.linalg.qr(arr3)
    result = (arrs[0] + 1).to_numpy()
    expected = np.linalg.qr((data + 1) * 2)[0] + 1

    np.testing.assert_array_almost_equal(result, expected)

    result = (arrs[0] + 2).to_numpy()
    expected = np.linalg.qr((data + 1) * 2)[0] + 2

    np.testing.assert_array_almost_equal(result, expected)

    s = mt.shape(0)

    result = s.execute().fetch()
    expected = np.shape(0)
    assert result == expected


def test_closed_session():
    session = new_test_session(default=True)
    with option_context({'show_progress': False}):
        arr = mt.ones((10, 10))

        result = session.execute(arr)

        np.testing.assert_array_equal(result, np.ones((10, 10)))

        # close session
        session.close()

        with pytest.raises(RuntimeError):
            session.execute(arr)

        with pytest.raises(RuntimeError):
            session.execute(arr + 1)


def test_array_protocol(setup):
    arr = mt.ones((10, 20))

    result = np.asarray(arr)
    np.testing.assert_array_equal(result, np.ones((10, 20)))

    arr2 = mt.ones((10, 20))

    result = np.asarray(arr2, mt.bool_)
    np.testing.assert_array_equal(result, np.ones((10, 20), dtype=np.bool_))

    arr3 = mt.ones((10, 20)).sum()

    result = np.asarray(arr3)
    np.testing.assert_array_equal(result, np.asarray(200))

    arr4 = mt.ones((10, 20)).sum()

    result = np.asarray(arr4, dtype=np.float_)
    np.testing.assert_array_equal(result, np.asarray(200, dtype=np.float_))


def test_without_fuse(setup):
    sess = new_session()

    arr1 = (mt.ones((10, 10), chunk_size=6) + 1) * 2
    r1 = arr1.execute(fuse_enabled=False).fetch()
    arr2 = (mt.ones((10, 10), chunk_size=5) + 1) * 2
    r2 = arr2.execute(fuse_enabled=False).fetch()
    np.testing.assert_array_equal(r1, r2)


def test_fetch_slices(setup):
    arr1 = mt.random.rand(10, 8, chunk_size=3)
    r1 = arr1.execute().fetch()

    r2 = arr1[:2, 3:9].fetch()
    np.testing.assert_array_equal(r2, r1[:2, 3:9])

    r3 = arr1[0].fetch()
    np.testing.assert_array_equal(r3, r1[0])


def test_fetch_dataframe_slices(setup):
    arr1 = mt.random.rand(10, 8, chunk_size=3)
    df1 = md.DataFrame(arr1)
    r1 = df1.execute().fetch()

    r2 = df1.iloc[:, :].fetch()
    pd.testing.assert_frame_equal(r2, r1.iloc[:, :])

    r3 = df1.iloc[1].fetch(extra_config={'check_series_name': False})
    pd.testing.assert_series_equal(r3, r1.iloc[1])

    r4 = df1.iloc[0, 2].fetch()
    assert r4 == r1.iloc[0, 2]

    arr2 = mt.random.rand(10, 3, chunk_size=3)
    df2 = md.DataFrame(arr2)
    r5 = df2.execute().fetch()

    r6 = df2.iloc[:4].fetch(batch_size=3)
    pd.testing.assert_frame_equal(r5.iloc[:4], r6)


def test_repr(setup):
    # test tensor repr
    with np.printoptions(threshold=100):
        arr = np.random.randint(1000, size=(11, 4, 13))

        t = mt.tensor(arr, chunk_size=3)

        result = repr(t.execute())
        expected = repr(arr)
        assert result == expected

    for size in (5, 58, 60, 62, 64):
        pdf = pd.DataFrame(np.random.randint(1000, size=(size, 10)))

        # test DataFrame repr
        df = md.DataFrame(pdf, chunk_size=size//2)

        result = repr(df.execute())
        expected = repr(pdf)
        assert result == expected

        # test DataFrame _repr_html_
        result = df.execute()._repr_html_()
        expected = pdf._repr_html_()
        assert result == expected

        # test Series repr
        ps = pdf[0]
        s = md.Series(ps, chunk_size=size//2)

        result = repr(s.execute())
        expected = repr(ps)
        assert result == expected

    # test Index repr
    pind = pd.date_range('2020-1-1', periods=10)
    ind = md.Index(pind, chunk_size=5)

    assert 'DatetimeIndex' in repr(ind.execute())

    # test groupby repr
    df = md.DataFrame(pd.DataFrame(np.random.rand(100, 3), columns=list('abc')))
    grouped = df.groupby(['a', 'b']).execute()

    assert 'DataFrameGroupBy' in repr(grouped)

    # test Categorical repr
    c = md.qcut(range(5), 3)
    assert 'Categorical' in repr(c)
    assert 'Categorical' in str(c)
    assert repr(c.execute()) == repr(pd.qcut(range(5), 3))


def test_iter(setup):
    raw_data = pd.DataFrame(np.random.randint(1000, size=(20, 10)))
    df = md.DataFrame(raw_data, chunk_size=5)

    for col, series in df.iteritems():
        pd.testing.assert_series_equal(series.execute().fetch(), raw_data[col])

    for i, batch in enumerate(df.iterbatch(batch_size=15)):
        pd.testing.assert_frame_equal(batch, raw_data.iloc[i * 15: (i + 1) * 15])

    i = 0
    for result_row, expect_row in zip(df.iterrows(batch_size=15),
                                      raw_data.iterrows()):
        assert result_row[0] == expect_row[0]
        pd.testing.assert_series_equal(result_row[1], expect_row[1])
        i += 1

    assert i == len(raw_data)

    i = 0
    for result_tup, expect_tup in zip(df.itertuples(batch_size=10),
                                      raw_data.itertuples()):
        assert result_tup == expect_tup
        i += 1

    assert i == len(raw_data)

    raw_data = pd.Series(np.random.randint(1000, size=(20,)))
    s = md.Series(raw_data, chunk_size=5)

    for i, batch in enumerate(s.iterbatch(batch_size=15)):
        pd.testing.assert_series_equal(batch, raw_data.iloc[i * 15: (i + 1) * 15])

    i = 0
    for result_item, expect_item in zip(s.iteritems(batch_size=15),
                                        raw_data.iteritems()):
        assert result_item[0] == expect_item[0]
        assert result_item[1] == expect_item[1]
        i += 1

    assert i == len(raw_data)

    # test to_dict
    assert s.to_dict() == raw_data.to_dict()
