#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

import io
import os
import re
import sys
import tempfile
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

from .. import tensor as mt
from .. import dataframe as md
from .. import remote as mr
from ..config import option_context
from ..deploy.utils import load_service_config_file
from ..session import execute, fetch, fetch_log


test_namedtuple_type = namedtuple("TestNamedTuple", "a b")


@pytest.fixture
def setup():
    from ..deploy.oscar.tests.session import new_test_session

    sess = new_test_session(address="127.0.0.1", init_local=True, default=True)
    with option_context({"show_progress": False}):
        try:
            from .. import __version__ as mars_version

            assert sess.get_cluster_versions() == [mars_version]
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

    expected = raw_df.skew()
    df = md.DataFrame(a)
    res = df.skew().to_pandas(wait=False).result()
    pd.testing.assert_series_equal(expected, res)
    res = df.skew().execute(wait=False)
    res = res.result().fetch()
    pd.testing.assert_series_equal(expected, res)

    t = [df.sum(), a.sum()]
    res = mt.ExecutableTuple(t).to_object(wait=False).result()
    pd.testing.assert_series_equal(raw_df.sum(), res[0])
    assert raw_a.sum() == res[1]
    res = mt.ExecutableTuple(t).execute(wait=False)
    res = fetch(*res.result())
    pd.testing.assert_series_equal(raw_df.sum(), res[0])
    assert raw_a.sum() == res[1]


def test_executable_tuple_execute(setup):
    raw_a = np.random.RandomState(0).rand(10, 20)
    a = mt.tensor(raw_a)

    raw_df = pd.DataFrame(raw_a)
    df = md.DataFrame(raw_df)

    tp = test_namedtuple_type(a, df)
    executable_tp = mt.ExecutableTuple(tp)

    assert "a" in dir(executable_tp)
    assert executable_tp.a is a
    assert test_namedtuple_type.__name__ in repr(executable_tp)
    with pytest.raises(AttributeError):
        getattr(executable_tp, "c")

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
    from ..deploy.oscar.tests.session import new_test_session

    session = new_test_session(default=True)
    with option_context({"show_progress": False}):
        arr = mt.ones((10, 10))
        try:
            result = session.execute(arr)

            np.testing.assert_array_equal(result, np.ones((10, 10)))

            # close session
            session.close()

            with pytest.raises(RuntimeError):
                session.execute(arr)

            with pytest.raises(RuntimeError):
                session.execute(arr + 1)
        finally:
            session.stop_server()


def test_array_protocol(setup):
    arr = mt.ones((10, 20))

    result = np.asarray(arr)
    np.testing.assert_array_equal(result, np.ones((10, 20)))

    arr2 = mt.ones((10, 20))

    result = np.asarray(arr2, mt.bool)
    np.testing.assert_array_equal(result, np.ones((10, 20), dtype=np.bool_))

    arr3 = mt.ones((10, 20)).sum()

    result = np.asarray(arr3)
    np.testing.assert_array_equal(result, np.asarray(200))

    arr4 = mt.ones((10, 20)).sum()

    result = np.asarray(arr4, dtype=np.float_)
    np.testing.assert_array_equal(result, np.asarray(200, dtype=np.float_))


def test_without_fuse(setup):
    arr1 = (mt.ones((10, 10), chunk_size=6) + 1) * 2
    r1 = arr1.execute(fuse_enabled=False).fetch()
    arr2 = (mt.ones((10, 10), chunk_size=5) + 1) * 2
    r2 = arr2.execute(fuse_enabled=False).fetch()
    np.testing.assert_array_equal(r1, r2)


@pytest.mark.ray_dag
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

    r3 = df1.iloc[1].fetch(extra_config={"check_series_name": False})
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
        df = md.DataFrame(pdf, chunk_size=size // 2)

        result = repr(df.execute())
        expected = repr(pdf)
        assert result == expected

        # test DataFrame _repr_html_
        result = df.execute()._repr_html_()
        expected = pdf._repr_html_()
        assert result == expected

        # test Series repr
        ps = pdf[0]
        s = md.Series(ps, chunk_size=size // 2)

        result = repr(s.execute())
        expected = repr(ps)
        assert result == expected

    # test Index repr
    pind = pd.date_range("2020-1-1", periods=10)
    ind = md.Index(pind, chunk_size=5)

    assert "DatetimeIndex" in repr(ind.execute())

    # test groupby repr
    df = md.DataFrame(pd.DataFrame(np.random.rand(100, 3), columns=list("abc")))
    grouped = df.groupby(["a", "b"]).execute()

    assert "DataFrameGroupBy" in repr(grouped)

    # test Categorical repr
    c = md.qcut(range(5), 3)
    assert "Categorical" in repr(c)
    assert "Categorical" in str(c)
    assert repr(c.execute()) == repr(pd.qcut(range(5), 3))


def test_iter(setup):
    raw_data = pd.DataFrame(np.random.randint(1000, size=(20, 10)))
    df = md.DataFrame(raw_data, chunk_size=5)

    for col, series in df.iteritems():
        pd.testing.assert_series_equal(series.execute().fetch(), raw_data[col])

    for i, batch in enumerate(df.iterbatch(batch_size=15)):
        pd.testing.assert_frame_equal(batch, raw_data.iloc[i * 15 : (i + 1) * 15])

    i = 0
    for result_row, expect_row in zip(df.iterrows(batch_size=15), raw_data.iterrows()):
        assert result_row[0] == expect_row[0]
        pd.testing.assert_series_equal(result_row[1], expect_row[1])
        i += 1

    assert i == len(raw_data)

    i = 0
    for result_tup, expect_tup in zip(
        df.itertuples(batch_size=10), raw_data.itertuples()
    ):
        assert result_tup == expect_tup
        i += 1

    assert i == len(raw_data)

    raw_data = pd.Series(np.random.randint(1000, size=(20,)))
    s = md.Series(raw_data, chunk_size=5)

    for i, batch in enumerate(s.iterbatch(batch_size=15)):
        pd.testing.assert_series_equal(batch, raw_data.iloc[i * 15 : (i + 1) * 15])

    i = 0
    for result_item, expect_item in zip(
        s.iteritems(batch_size=15), raw_data.iteritems()
    ):
        assert result_item[0] == expect_item[0]
        assert result_item[1] == expect_item[1]
        i += 1

    assert i == len(raw_data)

    # test to_dict
    assert s.to_dict() == raw_data.to_dict()


CONFIG = """
"@inherits": '@default'
session:
  custom_log_dir: '{custom_log_dir}'
"""


@pytest.fixture
def fetch_log_setup():
    from ..deploy.oscar.tests.session import new_test_session

    with tempfile.TemporaryDirectory() as temp_dir:
        config = io.StringIO(CONFIG.format(custom_log_dir=temp_dir))
        sess = new_test_session(
            default=True, config=load_service_config_file(config), n_cpu=8
        )
        with option_context({"show_progress": False}):
            try:
                yield sess
            finally:
                sess.stop_server()


def test_fetch_log(fetch_log_setup):
    def f():
        print("test")

    r = mr.spawn(f)
    r.execute()

    log = r.fetch_log()
    assert str(log).strip() == "test"

    # test multiple functions
    def f1(size):
        print("f1" * size)
        sys.stdout.flush()

    fs = mr.ExecutableTuple([mr.spawn(f1, 30), mr.spawn(f1, 40)])
    execute(*fs)
    log = fetch_log(*fs, offsets=20, sizes=10)
    assert str(log[0]).strip() == ("f1" * 30)[20:30]
    assert str(log[1]).strip() == ("f1" * 40)[20:30]
    assert len(log[0].offsets) > 0
    assert all(s > 0 for s in log[0].offsets)
    assert len(log[1].offsets) > 0
    assert all(s > 0 for s in log[1].offsets)
    assert len(log[0].chunk_op_keys) > 0

    # test negative offsets
    log = fs.fetch_log(offsets=-20, sizes=10)
    assert str(log[0]).strip() == ("f1" * 30 + os.linesep)[-20:-10]
    assert str(log[1]).strip() == ("f1" * 40 + os.linesep)[-20:-10]
    assert all(s > 0 for s in log[0].offsets) is True
    assert len(log[1].offsets) > 0
    assert all(s > 0 for s in log[1].offsets) is True
    assert len(log[0].chunk_op_keys) > 0

    # test negative offsets which represented in string
    log = fetch_log(*fs, offsets="-0.02K", sizes="0.01K")
    assert str(log[0]).strip() == ("f1" * 30 + os.linesep)[-20:-10]
    assert str(log[1]).strip() == ("f1" * 40 + os.linesep)[-20:-10]
    assert all(s > 0 for s in log[0].offsets) is True
    assert len(log[1].offsets) > 0
    assert all(s > 0 for s in log[1].offsets) is True
    assert len(log[0].chunk_op_keys) > 0

    def test_nested():
        print("level0")
        fr = mr.spawn(f1, 1)
        fr.execute()
        print(fr.fetch_log())

    r = mr.spawn(test_nested)
    r.execute()
    log = str(r.fetch_log())
    assert "level0" in log
    assert "f1" in log

    df = md.DataFrame(mt.random.rand(10, 3), chunk_size=5)

    def df_func(c):
        print("df func")
        return c

    df2 = df.map_chunk(df_func)
    df2.execute()
    log = df2.fetch_log()
    assert "Chunk op key:" in str(log)
    assert "df func" in repr(log)
    assert len(str(df.fetch_log())) == 0

    def test_host(rndf):
        rm = mr.spawn(nested, rndf)
        rm.execute()
        print(rm.fetch_log())

    def nested(_rndf):
        print("log_content")

    ds = [mr.spawn(test_host, n, retry_when_fail=False) for n in np.random.rand(4)]
    xtp = execute(ds)
    for log in fetch_log(xtp):
        assert str(log).strip() == "log_content"

    def test_threaded():
        import threading

        exc_info = None

        def print_fun():
            nonlocal exc_info
            try:
                print("inner")
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                exc_info = sys.exc_info()

        print_thread = threading.Thread(target=print_fun)
        print_thread.start()
        print_thread.join()

        if exc_info is not None:
            raise exc_info[1].with_traceback(exc_info[-1])

        print("after")

    rm = mr.spawn(test_threaded)
    rm.execute()
    logs = str(rm.fetch_log()).strip()
    assert logs == "inner\nafter"


def test_align_series(setup):
    t = np.random.rand(10, 3)
    pdf = pd.DataFrame(t)
    df = md.DataFrame(pdf, chunk_size=(5, 3))
    r = df[0] != df.sort_index()[0].shift(-1)
    expected = pdf[0] != pdf.sort_index()[0].shift(-1)
    pd.testing.assert_series_equal(r.execute().fetch(), expected)


def test_cache_tileable(setup):
    raw = np.random.rand(10, 3)
    t = mt.tensor(raw)
    t.cache = True
    t2 = t + 1
    result = t2.execute().fetch()
    np.testing.assert_array_equal(result, raw + 1)
    np.testing.assert_array_equal(t.fetch(), raw)

    with option_context({"warn_duplicated_execution": True}):
        t = mt.tensor(raw)
        with pytest.warns(
            RuntimeWarning,
            match=re.escape(f"Tileable {repr(t)} has been submitted before"),
        ):
            (t + 1).execute()
            (t + 2).execute()

        # should have no warning
        t = mt.tensor(raw)
        with pytest.raises(BaseException, match="DID NOT WARN"):
            with pytest.warns(
                RuntimeWarning,
                match=re.escape(f"Tileable {repr(t)} has been submitted before"),
            ):
                (t + 1).execute()


@pytest.mark.parametrize("method", ["shuffle", "broadcast", None])
@pytest.mark.parametrize("auto_merge", ["after", "before"])
def test_merge_groupby(setup, method, auto_merge):
    rs = np.random.RandomState(0)
    raw1 = pd.DataFrame({"a": rs.randint(3, size=100), "b": rs.rand(100)})
    raw2 = pd.DataFrame({"a": rs.randint(3, size=10), "c": rs.rand(10)})
    df1 = md.DataFrame(raw1, chunk_size=10).execute()
    df2 = md.DataFrame(raw2, chunk_size=10).execute()
    # do not trigger auto merge
    df3 = df1.merge(
        df2, on="a", auto_merge_threshold=8, method=method, auto_merge=auto_merge
    )
    df4 = df3.groupby("a").sum()

    result = df4.execute().fetch()
    expected = raw1.merge(raw2, on="a").groupby("a").sum()
    pd.testing.assert_frame_equal(result, expected)
