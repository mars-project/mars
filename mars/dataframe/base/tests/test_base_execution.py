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

import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

from ....config import options, option_context
from ....dataframe import DataFrame
from ....tensor import arange, tensor
from ....tensor.random import rand
from ....tests.core import require_cudf
from ....utils import lazy_import, pd_release_version
from ... import eval as mars_eval, cut, qcut, get_dummies
from ...datasource.dataframe import from_pandas as from_pandas_df
from ...datasource.series import from_pandas as from_pandas_series
from ...datasource.index import from_pandas as from_pandas_index
from .. import to_gpu, to_cpu
from ..to_numeric import to_numeric
from ..rebalance import DataFrameRebalance

pytestmark = pytest.mark.pd_compat

cudf = lazy_import("cudf", globals=globals())

_explode_with_ignore_index = pd_release_version[:2] >= (1, 1)


@require_cudf
def test_to_gpu_execution(setup_gpu):
    pdf = pd.DataFrame(np.random.rand(20, 30), index=np.arange(20, 0, -1))
    df = from_pandas_df(pdf, chunk_size=(13, 21))
    cdf = to_gpu(df)

    res = cdf.execute().fetch()
    assert isinstance(res, cudf.DataFrame)
    pd.testing.assert_frame_equal(res.to_pandas(), pdf)

    pseries = pdf.iloc[:, 0]
    series = from_pandas_series(pseries)
    cseries = series.to_gpu()

    res = cseries.execute().fetch()
    assert isinstance(res, cudf.Series)
    pd.testing.assert_series_equal(res.to_pandas(), pseries)


@require_cudf
def test_to_cpu_execution(setup_gpu):
    pdf = pd.DataFrame(np.random.rand(20, 30), index=np.arange(20, 0, -1))
    df = from_pandas_df(pdf, chunk_size=(13, 21))
    cdf = to_gpu(df)
    df2 = to_cpu(cdf)

    res = df2.execute().fetch()
    assert isinstance(res, pd.DataFrame)
    pd.testing.assert_frame_equal(res, pdf)

    pseries = pdf.iloc[:, 0]
    series = from_pandas_series(pseries, chunk_size=(13, 21))
    cseries = to_gpu(series)
    series2 = to_cpu(cseries)

    res = series2.execute().fetch()
    assert isinstance(res, pd.Series)
    pd.testing.assert_series_equal(res, pseries)


def test_rechunk_execution(setup):
    data = pd.DataFrame(np.random.rand(8, 10))
    df = from_pandas_df(pd.DataFrame(data), chunk_size=3)
    df2 = df.rechunk((3, 4))
    res = df2.execute().fetch()
    pd.testing.assert_frame_equal(data, res)

    data = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.random.randint(-100, 100, size=(10,)),
        columns=[np.random.bytes(10) for _ in range(10)],
    )
    df = from_pandas_df(data)
    df2 = df.rechunk(5)
    res = df2.execute().fetch()
    pd.testing.assert_frame_equal(data, res)

    # test Series rechunk execution.
    data = pd.Series(np.random.rand(10))
    series = from_pandas_series(data)
    series2 = series.rechunk(3)
    res = series2.execute().fetch()
    pd.testing.assert_series_equal(data, res)

    series2 = series.rechunk(1)
    res = series2.execute().fetch()
    pd.testing.assert_series_equal(data, res)

    # test index rechunk execution
    data = pd.Index(np.random.rand(10))
    index = from_pandas_index(data)
    index2 = index.rechunk(3)
    res = index2.execute().fetch()
    pd.testing.assert_index_equal(data, res)

    index2 = index.rechunk(1)
    res = index2.execute().fetch()
    pd.testing.assert_index_equal(data, res)

    # test rechunk on mixed typed columns
    data = pd.DataFrame({0: [1, 2], 1: [3, 4], "a": [5, 6]})
    df = from_pandas_df(data)
    df = df.rechunk((2, 2)).rechunk({1: 3})
    res = df.execute().fetch()
    pd.testing.assert_frame_equal(data, res)


def test_series_map_execution(setup):
    raw = pd.Series(np.arange(10))
    s = from_pandas_series(raw, chunk_size=7)

    with pytest.raises(ValueError):
        # cannot infer dtype, the inferred is int,
        # but actually it is float
        # just due to nan
        s.map({5: 10})

    r = s.map({5: 10}, dtype=float)
    result = r.execute().fetch()
    expected = raw.map({5: 10})
    pd.testing.assert_series_equal(result, expected)

    r = s.map({i: 10 + i for i in range(7)}, dtype=float)
    result = r.execute().fetch()
    expected = raw.map({i: 10 + i for i in range(7)})
    pd.testing.assert_series_equal(result, expected)

    r = s.map({5: 10}, dtype=float, na_action="ignore")
    result = r.execute().fetch()
    expected = raw.map({5: 10}, na_action="ignore")
    pd.testing.assert_series_equal(result, expected)

    # dtype can be inferred
    r = s.map({5: 10.0})
    result = r.execute().fetch()
    expected = raw.map({5: 10.0})
    pd.testing.assert_series_equal(result, expected)

    r = s.map(lambda x: x + 1, dtype=int)
    result = r.execute().fetch()
    expected = raw.map(lambda x: x + 1)
    pd.testing.assert_series_equal(result, expected)

    def f(x: int) -> float:
        return x + 1.0

    # dtype can be inferred for function
    r = s.map(f)
    result = r.execute().fetch()
    expected = raw.map(lambda x: x + 1.0)
    pd.testing.assert_series_equal(result, expected)

    def f(x: int):
        return x + 1.0

    # dtype can be inferred for function
    r = s.map(f)
    result = r.execute().fetch()
    expected = raw.map(lambda x: x + 1.0)
    pd.testing.assert_series_equal(result, expected)

    # test arg is a md.Series
    raw2 = pd.Series([10], index=[5])
    s2 = from_pandas_series(raw2)

    r = s.map(s2, dtype=float)
    result = r.execute().fetch()
    expected = raw.map(raw2)
    pd.testing.assert_series_equal(result, expected)

    # test arg is a md.Series, and dtype can be inferred
    raw2 = pd.Series([10.0], index=[5])
    s2 = from_pandas_series(raw2)

    r = s.map(s2)
    result = r.execute().fetch()
    expected = raw.map(raw2)
    pd.testing.assert_series_equal(result, expected)

    # test str
    raw = pd.Series(["a", "b", "c", "d"])
    s = from_pandas_series(raw, chunk_size=2)

    r = s.map({"c": "e"})
    result = r.execute().fetch()
    expected = raw.map({"c": "e"})
    pd.testing.assert_series_equal(result, expected)

    # test map index
    raw = pd.Index(np.random.rand(7))
    idx = from_pandas_index(pd.Index(raw), chunk_size=2)
    r = idx.map(f)
    result = r.execute().fetch()
    expected = raw.map(lambda x: x + 1.0)
    pd.testing.assert_index_equal(result, expected)


def test_describe_execution(setup):
    s_raw = pd.Series(np.random.rand(10))

    # test one chunk
    series = from_pandas_series(s_raw, chunk_size=10)

    r = series.describe()
    result = r.execute().fetch()
    expected = s_raw.describe()
    pd.testing.assert_series_equal(result, expected)

    r = series.describe(percentiles=[])
    result = r.execute().fetch()
    expected = s_raw.describe(percentiles=[])
    pd.testing.assert_series_equal(result, expected)

    # test multi chunks
    series = from_pandas_series(s_raw, chunk_size=3)

    r = series.describe()
    result = r.execute().fetch()
    expected = s_raw.describe()
    pd.testing.assert_series_equal(result, expected)

    r = series.describe(percentiles=[])
    result = r.execute().fetch()
    expected = s_raw.describe(percentiles=[])
    pd.testing.assert_series_equal(result, expected)

    rs = np.random.RandomState(5)
    df_raw = pd.DataFrame(rs.rand(10, 4), columns=list("abcd"))
    df_raw["e"] = rs.randint(100, size=10)

    # test one chunk
    df = from_pandas_df(df_raw, chunk_size=10)

    r = df.describe()
    result = r.execute().fetch()
    expected = df_raw.describe()
    pd.testing.assert_frame_equal(result, expected)

    r = series.describe(percentiles=[], include=np.float64)
    result = r.execute().fetch()
    expected = s_raw.describe(percentiles=[], include=np.float64)
    pd.testing.assert_series_equal(result, expected)

    # test multi chunks
    df = from_pandas_df(df_raw, chunk_size=3)

    r = df.describe()
    result = r.execute().fetch()
    expected = df_raw.describe()
    pd.testing.assert_frame_equal(result, expected)

    r = df.describe(percentiles=[], include=np.float64)
    result = r.execute().fetch()
    expected = df_raw.describe(percentiles=[], include=np.float64)
    pd.testing.assert_frame_equal(result, expected)

    # test skip percentiles
    r = df.describe(percentiles=False, include=np.float64)
    result = r.execute().fetch()
    expected = df_raw.describe(percentiles=[], include=np.float64)
    expected.drop(["50%"], axis=0, inplace=True)
    pd.testing.assert_frame_equal(result, expected)

    with pytest.raises(ValueError):
        df.describe(percentiles=[1.1])

    with pytest.raises(ValueError):
        # duplicated values
        df.describe(percentiles=[0.3, 0.5, 0.3])

    # test input dataframe which has unknown shape
    df = from_pandas_df(df_raw, chunk_size=3)
    df2 = df[df["a"] < 0.5]
    r = df2.describe()

    result = r.execute().fetch()
    expected = df_raw[df_raw["a"] < 0.5].describe()
    pd.testing.assert_frame_equal(result, expected)


def test_data_frame_apply_execute(setup):
    cols = [chr(ord("A") + i) for i in range(10)]
    df_raw = pd.DataFrame(dict((c, [i**2 for i in range(20)]) for c in cols))

    old_chunk_store_limit = options.chunk_store_limit
    try:
        options.chunk_store_limit = 20

        df = from_pandas_df(df_raw, chunk_size=5)

        r = df.apply("ffill")
        result = r.execute().fetch()
        expected = df_raw.apply("ffill")
        pd.testing.assert_frame_equal(result, expected)

        r = df.apply(["sum", "max"])
        result = r.execute().fetch()
        expected = df_raw.apply(["sum", "max"])
        pd.testing.assert_frame_equal(result, expected)

        r = df.apply(np.sqrt)
        result = r.execute().fetch()
        expected = df_raw.apply(np.sqrt)
        pd.testing.assert_frame_equal(result, expected)

        r = df.apply(lambda x: pd.Series([1, 2]))
        result = r.execute().fetch()
        expected = df_raw.apply(lambda x: pd.Series([1, 2]))
        pd.testing.assert_frame_equal(result, expected)

        r = df.apply(np.sum, axis="index")
        result = r.execute().fetch()
        expected = df_raw.apply(np.sum, axis="index")
        pd.testing.assert_series_equal(result, expected)

        r = df.apply(np.sum, axis="columns")
        result = r.execute().fetch()
        expected = df_raw.apply(np.sum, axis="columns")
        pd.testing.assert_series_equal(result, expected)

        r = df.apply(lambda x: [1, 2], axis=1)
        result = r.execute().fetch()
        expected = df_raw.apply(lambda x: [1, 2], axis=1)
        pd.testing.assert_series_equal(result, expected)

        r = df.apply(lambda x: pd.Series([1, 2], index=["foo", "bar"]), axis=1)
        result = r.execute().fetch()
        expected = df_raw.apply(
            lambda x: pd.Series([1, 2], index=["foo", "bar"]), axis=1
        )
        pd.testing.assert_frame_equal(result, expected)

        r = df.apply(lambda x: [1, 2], axis=1, result_type="expand")
        result = r.execute().fetch()
        expected = df_raw.apply(lambda x: [1, 2], axis=1, result_type="expand")
        pd.testing.assert_frame_equal(result, expected)

        r = df.apply(lambda x: list(range(10)), axis=1, result_type="reduce")
        result = r.execute().fetch()
        expected = df_raw.apply(lambda x: list(range(10)), axis=1, result_type="reduce")
        pd.testing.assert_series_equal(result, expected)

        r = df.apply(lambda x: list(range(10)), axis=1, result_type="broadcast")
        result = r.execute().fetch()
        expected = df_raw.apply(
            lambda x: list(range(10)), axis=1, result_type="broadcast"
        )
        pd.testing.assert_frame_equal(result, expected)
    finally:
        options.chunk_store_limit = old_chunk_store_limit


def test_series_apply_execute(setup):
    idxes = [chr(ord("A") + i) for i in range(20)]
    s_raw = pd.Series([i**2 for i in range(20)], index=idxes)

    series = from_pandas_series(s_raw, chunk_size=5)

    r = series.apply("add", args=(1,))
    result = r.execute().fetch()
    expected = s_raw.apply("add", args=(1,))
    pd.testing.assert_series_equal(result, expected)

    r = series.apply(["sum", "max"])
    result = r.execute().fetch()
    expected = s_raw.apply(["sum", "max"])
    pd.testing.assert_series_equal(result, expected)

    r = series.apply(np.sqrt)
    result = r.execute().fetch()
    expected = s_raw.apply(np.sqrt)
    pd.testing.assert_series_equal(result, expected)

    r = series.apply("sqrt")
    result = r.execute().fetch()
    expected = s_raw.apply("sqrt")
    pd.testing.assert_series_equal(result, expected)

    r = series.apply(lambda x: [x, x + 1], convert_dtype=False)
    result = r.execute().fetch()
    expected = s_raw.apply(lambda x: [x, x + 1], convert_dtype=False)
    pd.testing.assert_series_equal(result, expected)

    s_raw2 = pd.Series([np.array([1, 2, 3]), np.array([4, 5, 6])])
    series = from_pandas_series(s_raw2)

    dtypes = pd.Series([np.dtype(float)] * 3)
    r = series.apply(pd.Series, output_type="dataframe", dtypes=dtypes)
    result = r.execute().fetch()
    expected = s_raw2.apply(pd.Series)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_apply_with_arrow_dtype_execution(setup):
    df1 = pd.DataFrame({"a": [1, 2, 1], "b": ["a", "b", "a"]})
    df = from_pandas_df(df1)
    df["b"] = df["b"].astype("Arrow[string]")

    r = df.apply(lambda row: str(row[0]) + row[1], axis=1)
    result = r.execute().fetch()
    expected = df1.apply(lambda row: str(row[0]) + row[1], axis=1)
    pd.testing.assert_series_equal(result, expected)

    s1 = df1["b"]
    s = from_pandas_series(s1)
    s = s.astype("arrow_string")

    r = s.apply(lambda x: x + "_suffix")
    result = r.execute().fetch()
    expected = s1.apply(lambda x: x + "_suffix")
    pd.testing.assert_series_equal(result, expected)


def test_transform_execute(setup):
    cols = [chr(ord("A") + i) for i in range(10)]
    df_raw = pd.DataFrame(dict((c, [i**2 for i in range(20)]) for c in cols))

    idx_vals = [chr(ord("A") + i) for i in range(20)]
    s_raw = pd.Series([i**2 for i in range(20)], index=idx_vals)

    def rename_fn(f, new_name):
        f.__name__ = new_name
        return f

    old_chunk_store_limit = options.chunk_store_limit
    try:
        options.chunk_store_limit = 20

        # DATAFRAME CASES
        df = from_pandas_df(df_raw, chunk_size=5)

        # test transform scenarios on data frames
        r = df.transform(lambda x: list(range(len(x))))
        result = r.execute().fetch()
        expected = df_raw.transform(lambda x: list(range(len(x))))
        pd.testing.assert_frame_equal(result, expected)

        r = df.transform(lambda x: list(range(len(x))), axis=1)
        result = r.execute().fetch()
        expected = df_raw.transform(lambda x: list(range(len(x))), axis=1)
        pd.testing.assert_frame_equal(result, expected)

        r = df.transform(["cumsum", "cummax", lambda x: x + 1])
        result = r.execute().fetch()
        expected = df_raw.transform(["cumsum", "cummax", lambda x: x + 1])
        pd.testing.assert_frame_equal(result, expected)

        fn_dict = OrderedDict(
            [
                ("A", "cumsum"),
                ("D", ["cumsum", "cummax"]),
                ("F", lambda x: x + 1),
            ]
        )
        r = df.transform(fn_dict)
        result = r.execute().fetch()
        expected = df_raw.transform(fn_dict)
        pd.testing.assert_frame_equal(result, expected)

        r = df.transform(lambda x: x.iloc[:-1], _call_agg=True)
        result = r.execute().fetch()
        expected = df_raw.agg(lambda x: x.iloc[:-1])
        pd.testing.assert_frame_equal(result, expected)

        r = df.transform(lambda x: x.iloc[:-1], axis=1, _call_agg=True)
        result = r.execute().fetch()
        expected = df_raw.agg(lambda x: x.iloc[:-1], axis=1)
        pd.testing.assert_frame_equal(result, expected)

        fn_list = [
            rename_fn(lambda x: x.iloc[1:].reset_index(drop=True), "f1"),
            lambda x: x.iloc[:-1].reset_index(drop=True),
        ]
        r = df.transform(fn_list, _call_agg=True)
        result = r.execute().fetch()
        expected = df_raw.agg(fn_list)
        pd.testing.assert_frame_equal(result, expected)

        r = df.transform(lambda x: x.sum(), _call_agg=True)
        result = r.execute().fetch()
        expected = df_raw.agg(lambda x: x.sum())
        pd.testing.assert_series_equal(result, expected)

        fn_dict = OrderedDict(
            [
                ("A", rename_fn(lambda x: x.iloc[1:].reset_index(drop=True), "f1")),
                (
                    "D",
                    [
                        rename_fn(lambda x: x.iloc[1:].reset_index(drop=True), "f1"),
                        lambda x: x.iloc[:-1].reset_index(drop=True),
                    ],
                ),
                ("F", lambda x: x.iloc[:-1].reset_index(drop=True)),
            ]
        )
        r = df.transform(fn_dict, _call_agg=True)
        result = r.execute().fetch()
        expected = df_raw.agg(fn_dict)
        pd.testing.assert_frame_equal(result, expected)

        # SERIES CASES
        series = from_pandas_series(s_raw, chunk_size=5)

        # test transform scenarios on series
        r = series.transform(lambda x: x + 1)
        result = r.execute().fetch()
        expected = s_raw.transform(lambda x: x + 1)
        pd.testing.assert_series_equal(result, expected)

        r = series.transform(["cumsum", lambda x: x + 1])
        result = r.execute().fetch()
        expected = s_raw.transform(["cumsum", lambda x: x + 1])
        pd.testing.assert_frame_equal(result, expected)

        # test transform on string dtype
        df_raw = pd.DataFrame({"col1": ["str"] * 10, "col2": ["string"] * 10})
        df = from_pandas_df(df_raw, chunk_size=3)

        r = df["col1"].transform(lambda x: x + "_suffix")
        result = r.execute().fetch()
        expected = df_raw["col1"].transform(lambda x: x + "_suffix")
        pd.testing.assert_series_equal(result, expected)

        r = df.transform(lambda x: x + "_suffix")
        result = r.execute().fetch()
        expected = df_raw.transform(lambda x: x + "_suffix")
        pd.testing.assert_frame_equal(result, expected)

        r = df["col2"].transform(lambda x: x + "_suffix", dtype=np.dtype("str"))
        result = r.execute().fetch()
        expected = df_raw["col2"].transform(lambda x: x + "_suffix")
        pd.testing.assert_series_equal(result, expected)
    finally:
        options.chunk_store_limit = old_chunk_store_limit


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_transform_with_arrow_dtype_execution(setup):
    raw = pd.DataFrame({"a": [1, 2, 1], "b": ["a", "b", "a"]})
    df = from_pandas_df(raw)
    df["b"] = df["b"].astype("Arrow[string]")

    r = df.transform({"b": lambda x: x + "_suffix"})
    result = r.execute().fetch()
    result["b"] = result["b"].to_numpy()
    expected = raw.transform({"b": lambda x: x + "_suffix"})
    pd.testing.assert_frame_equal(result, expected)

    s1 = raw["b"]
    s = from_pandas_series(s1)
    s = s.astype("arrow_string")

    r = s.transform(lambda x: x + "_suffix")
    result = r.execute().fetch()
    result = pd.Series(result.to_numpy(), name=result.name, index=result.index)
    expected = s1.transform(lambda x: x + "_suffix")
    pd.testing.assert_series_equal(result, expected)


def test_string_method_execution(setup):
    s = pd.Series(["s1,s2", "ef,", "dd", np.nan])
    s2 = pd.concat([s, s, s])

    series = from_pandas_series(s, chunk_size=2)
    series2 = from_pandas_series(s2, chunk_size=2)

    # test getitem
    r = series.str[:3]
    result = r.execute().fetch()
    expected = s.str[:3]
    pd.testing.assert_series_equal(result, expected)

    # test split, expand=False
    r = series.str.split(",", n=2)
    result = r.execute().fetch()
    expected = s.str.split(",", n=2)
    pd.testing.assert_series_equal(result, expected)

    # test split, expand=True
    r = series.str.split(",", expand=True, n=1)
    result = r.execute().fetch()
    expected = s.str.split(",", expand=True, n=1)
    pd.testing.assert_frame_equal(result, expected)

    # test rsplit
    r = series.str.rsplit(",", expand=True, n=1)
    result = r.execute().fetch()
    expected = s.str.rsplit(",", expand=True, n=1)
    pd.testing.assert_frame_equal(result, expected)

    # test cat all data
    r = series2.str.cat(sep="/", na_rep="e")
    result = r.execute().fetch()
    expected = s2.str.cat(sep="/", na_rep="e")
    assert result == expected

    # test cat list
    r = series.str.cat(["a", "b", np.nan, "c"])
    result = r.execute().fetch()
    expected = s.str.cat(["a", "b", np.nan, "c"])
    pd.testing.assert_series_equal(result, expected)

    # test cat series
    r = series.str.cat(series.str.capitalize(), join="outer")
    result = r.execute().fetch()
    expected = s.str.cat(s.str.capitalize(), join="outer")
    pd.testing.assert_series_equal(result, expected)

    # test extractall
    r = series.str.extractall(r"(?P<letter>[ab])(?P<digit>\d)")
    result = r.execute().fetch()
    expected = s.str.extractall(r"(?P<letter>[ab])(?P<digit>\d)")
    pd.testing.assert_frame_equal(result, expected)

    # test extract, expand=False
    r = series.str.extract(r"[ab](\d)", expand=False)
    result = r.execute().fetch()
    expected = s.str.extract(r"[ab](\d)", expand=False)
    pd.testing.assert_series_equal(result, expected)

    # test extract, expand=True
    r = series.str.extract(r"[ab](\d)", expand=True)
    result = r.execute().fetch()
    expected = s.str.extract(r"[ab](\d)", expand=True)
    pd.testing.assert_frame_equal(result, expected)


def test_datetime_method_execution(setup):
    # test datetime
    s = pd.Series([pd.Timestamp("2020-1-1"), pd.Timestamp("2020-2-1"), np.nan])
    series = from_pandas_series(s, chunk_size=2)

    r = series.dt.year
    result = r.execute().fetch()
    expected = s.dt.year
    pd.testing.assert_series_equal(result, expected)

    r = series.dt.strftime("%m-%d-%Y")
    result = r.execute().fetch()
    expected = s.dt.strftime("%m-%d-%Y")
    pd.testing.assert_series_equal(result, expected)

    # test timedelta
    s = pd.Series([pd.Timedelta("1 days"), pd.Timedelta("3 days"), np.nan])
    series = from_pandas_series(s, chunk_size=2)

    r = series.dt.days
    result = r.execute().fetch()
    expected = s.dt.days
    pd.testing.assert_series_equal(result, expected)


def test_isin_execution(setup):
    # one chunk in multiple chunks
    a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = pd.Series([2, 1, 9, 3])
    sa = from_pandas_series(a, chunk_size=10)
    sb = from_pandas_series(b, chunk_size=2)

    result = sa.isin(sb).execute().fetch()
    expected = a.isin(b)
    pd.testing.assert_series_equal(result, expected)

    # multiple chunk in one chunks
    a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = pd.Series([2, 1, 9, 3])
    sa = from_pandas_series(a, chunk_size=2)
    sb = from_pandas_series(b, chunk_size=4)

    result = sa.isin(sb).execute().fetch()
    expected = a.isin(b)
    pd.testing.assert_series_equal(result, expected)

    # multiple chunk in multiple chunks
    a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = pd.Series([2, 1, 9, 3])
    sa = from_pandas_series(a, chunk_size=2)
    sb = from_pandas_series(b, chunk_size=2)

    result = sa.isin(sb).execute().fetch()
    expected = a.isin(b)
    pd.testing.assert_series_equal(result, expected)

    a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = pd.Series([2, 1, 9, 3])
    sa = from_pandas_series(a, chunk_size=2)

    result = sa.isin(sb).execute().fetch()
    expected = a.isin(b)
    pd.testing.assert_series_equal(result, expected)

    a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = np.array([2, 1, 9, 3])
    sa = from_pandas_series(a, chunk_size=2)
    sb = tensor(b, chunk_size=3)

    result = sa.isin(sb).execute().fetch()
    expected = a.isin(b)
    pd.testing.assert_series_equal(result, expected)

    a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = {2, 1, 9, 3}  # set
    sa = from_pandas_series(a, chunk_size=2)

    result = sa.isin(sb).execute().fetch()
    expected = a.isin(b)
    pd.testing.assert_series_equal(result, expected)

    rs = np.random.RandomState(0)
    raw = pd.DataFrame(rs.randint(1000, size=(10, 3)))
    df = from_pandas_df(raw, chunk_size=(5, 2))

    # set
    b = {2, 1, raw[1][0]}
    r = df.isin(b)
    result = r.execute().fetch()
    expected = raw.isin(b)
    pd.testing.assert_frame_equal(result, expected)

    # mars object
    b = tensor([2, 1, raw[1][0]], chunk_size=2)
    r = df.isin(b)
    result = r.execute().fetch()
    expected = raw.isin([2, 1, raw[1][0]])
    pd.testing.assert_frame_equal(result, expected)

    # dict
    b = {1: tensor([2, 1, raw[1][0]], chunk_size=2), 2: [3, 10]}
    r = df.isin(b)
    result = r.execute().fetch()
    expected = raw.isin({1: [2, 1, raw[1][0]], 2: [3, 10]})
    pd.testing.assert_frame_equal(result, expected)


def test_cut_execution(setup):
    session = setup

    rs = np.random.RandomState(0)
    raw = rs.random(15) * 1000
    s = pd.Series(raw, index=[f"i{i}" for i in range(15)])
    bins = [10, 100, 500]
    ii = pd.interval_range(10, 500, 3)
    labels = ["a", "b"]

    t = tensor(raw, chunk_size=4)
    series = from_pandas_series(s, chunk_size=4)
    iii = from_pandas_index(ii, chunk_size=2)

    # cut on Series
    r = cut(series, bins)
    result = r.execute().fetch()
    pd.testing.assert_series_equal(result, pd.cut(s, bins))

    r, b = cut(series, bins, retbins=True)
    r_result = r.execute().fetch()
    b_result = b.execute().fetch()
    r_expected, b_expected = pd.cut(s, bins, retbins=True)
    pd.testing.assert_series_equal(r_result, r_expected)
    np.testing.assert_array_equal(b_result, b_expected)

    # cut on tensor
    r = cut(t, bins)
    # result and expected is array whose dtype is CategoricalDtype
    result = r.execute().fetch()
    expected = pd.cut(raw, bins)
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        np.testing.assert_equal(r, e)

    # one chunk
    r = cut(s, tensor(bins, chunk_size=2), right=False, include_lowest=True)
    result = r.execute().fetch()
    pd.testing.assert_series_equal(
        result, pd.cut(s, bins, right=False, include_lowest=True)
    )

    # test labels
    r = cut(t, bins, labels=labels)
    # result and expected is array whose dtype is CategoricalDtype
    result = r.execute().fetch()
    expected = pd.cut(raw, bins, labels=labels)
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        np.testing.assert_equal(r, e)

    r = cut(t, bins, labels=False)
    # result and expected is array whose dtype is CategoricalDtype
    result = r.execute().fetch()
    expected = pd.cut(raw, bins, labels=False)
    np.testing.assert_array_equal(result, expected)

    # test labels which is tensor
    labels_t = tensor(["a", "b"], chunk_size=1)
    r = cut(raw, bins, labels=labels_t, include_lowest=True)
    # result and expected is array whose dtype is CategoricalDtype
    result = r.execute().fetch()
    expected = pd.cut(raw, bins, labels=labels, include_lowest=True)
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        np.testing.assert_equal(r, e)

    # test labels=False
    r, b = cut(raw, ii, labels=False, retbins=True)
    # result and expected is array whose dtype is CategoricalDtype
    r_result, b_result = session.fetch(*session.execute(r, b))
    r_expected, b_expected = pd.cut(raw, ii, labels=False, retbins=True)
    for r, e in zip(r_result, r_expected):
        np.testing.assert_equal(r, e)
    pd.testing.assert_index_equal(b_result, b_expected)

    # test bins which is md.IntervalIndex
    r, b = cut(series, iii, labels=tensor(labels, chunk_size=1), retbins=True)
    r_result = r.execute().fetch()
    b_result = b.execute().fetch()
    r_expected, b_expected = pd.cut(s, ii, labels=labels, retbins=True)
    pd.testing.assert_series_equal(r_result, r_expected)
    pd.testing.assert_index_equal(b_result, b_expected)

    # test duplicates
    bins2 = [0, 2, 4, 6, 10, 10]
    r, b = cut(s, bins2, labels=False, retbins=True, right=False, duplicates="drop")
    r_result = r.execute().fetch()
    b_result = b.execute().fetch()
    r_expected, b_expected = pd.cut(
        s, bins2, labels=False, retbins=True, right=False, duplicates="drop"
    )
    pd.testing.assert_series_equal(r_result, r_expected)
    np.testing.assert_array_equal(b_result, b_expected)

    # test integer bins
    r = cut(series, 3)
    result = r.execute().fetch()
    pd.testing.assert_series_equal(result, pd.cut(s, 3))

    r, b = cut(series, 3, right=False, retbins=True)
    r_result, b_result = session.fetch(*session.execute(r, b))
    r_expected, b_expected = pd.cut(s, 3, right=False, retbins=True)
    pd.testing.assert_series_equal(r_result, r_expected)
    np.testing.assert_array_equal(b_result, b_expected)

    # test min max same
    s2 = pd.Series([1.1] * 15)
    r = cut(s2, 3)
    result = r.execute().fetch()
    pd.testing.assert_series_equal(result, pd.cut(s2, 3))

    # test inf exist
    s3 = s2.copy()
    s3[-1] = np.inf
    with pytest.raises(ValueError):
        cut(s3, 3).execute()


def test_transpose_execution(setup):
    raw = pd.DataFrame(
        {"a": ["1", "2", "3"], "b": ["5", "-6", "7"], "c": ["1", "2", "3"]}
    )

    # test 1 chunk
    df = from_pandas_df(raw)
    result = df.transpose().execute().fetch()
    pd.testing.assert_frame_equal(result, raw.transpose())

    # test multi chunks
    df = from_pandas_df(raw, chunk_size=2)
    result = df.transpose().execute().fetch()
    pd.testing.assert_frame_equal(result, raw.transpose())

    df = from_pandas_df(raw, chunk_size=2)
    result = df.T.execute().fetch()
    pd.testing.assert_frame_equal(result, raw.transpose())

    # dtypes are varied
    raw = pd.DataFrame({"a": [1.1, 2.2, 3.3], "b": [5, -6, 7], "c": [1, 2, 3]})

    df = from_pandas_df(raw, chunk_size=2)
    result = df.transpose().execute().fetch()
    pd.testing.assert_frame_equal(result, raw.transpose())

    raw = pd.DataFrame({"a": [1.1, 2.2, 3.3], "b": ["5", "-6", "7"]})

    df = from_pandas_df(raw, chunk_size=2)
    result = df.transpose().execute().fetch()
    pd.testing.assert_frame_equal(result, raw.transpose())

    # Transposing from results of other operands
    raw = pd.DataFrame(np.arange(0, 100).reshape(10, 10))
    df = DataFrame(arange(0, 100, chunk_size=5).reshape(10, 10))
    result = df.transpose().execute().fetch()
    pd.testing.assert_frame_equal(result, raw.transpose())

    df = DataFrame(rand(100, 100, chunk_size=10))
    raw = df.to_pandas()
    result = df.transpose().execute().fetch()
    pd.testing.assert_frame_equal(result, raw.transpose())


def test_get_dummies_execution(setup):
    raw = pd.DataFrame(
        {
            "a": [1.1, 2.1, 3.1],
            "b": ["5", "-6", "-7"],
            "c": [1, 2, 3],
            "d": ["2", "3", "4"],
        }
    )
    # test 1 chunk
    df = from_pandas_df(raw)
    r = get_dummies(df)
    pd.testing.assert_frame_equal(r.execute().fetch(), pd.get_dummies(raw))

    # test multi chunks
    df = from_pandas_df(raw, chunk_size=2)
    r = get_dummies(df)
    pd.testing.assert_frame_equal(r.execute().fetch(), pd.get_dummies(raw))

    # test prefix and prefix_sep
    df = from_pandas_df(raw, chunk_size=2)
    r = get_dummies(df, prefix=["col1", "col2"], prefix_sep="_")
    pd.testing.assert_frame_equal(
        r.execute().fetch(),
        pd.get_dummies(raw, prefix=["col1", "col2"], prefix_sep="_"),
    )

    r = get_dummies(df, prefix={"b": "col1", "d": "col2"}, prefix_sep="_")
    pd.testing.assert_frame_equal(
        r.execute().fetch(),
        pd.get_dummies(raw, prefix={"b": "col1", "d": "col2"}, prefix_sep="_"),
    )

    # test dummy_na
    raw = pd.Series(["a", "b", "c", np.nan])
    df = from_pandas_series(raw)
    r = get_dummies(df, dummy_na=False)
    pd.testing.assert_frame_equal(
        r.execute().fetch(), pd.get_dummies(raw, dummy_na=False)
    )

    # test columns
    raw = pd.DataFrame(
        {
            "a": [1.1, 2.1, 3.1],
            "b": ["5", "-6", "-7"],
            "c": [1, 2, 3],
            "d": ["2", "3", "4"],
        }
    )
    df = from_pandas_df(raw, chunk_size=2)
    r = get_dummies(df, columns=["c"])
    pd.testing.assert_frame_equal(
        r.execute().fetch(), pd.get_dummies(raw, columns=["c"])
    )

    r = get_dummies(df, columns=["c", "d"], prefix=["col1", "col2"])
    pd.testing.assert_frame_equal(
        r.execute().fetch(),
        pd.get_dummies(raw, columns=["c", "d"], prefix=["col1", "col2"]),
    )

    # test drop_first
    df = from_pandas_df(raw, chunk_size=2)
    r = get_dummies(df, drop_first=True)
    pd.testing.assert_frame_equal(
        r.execute().fetch(), pd.get_dummies(raw, drop_first=True)
    )

    # test dtype
    df = from_pandas_df(raw, chunk_size=2)
    r = get_dummies(df, dtype=float)
    pd.testing.assert_frame_equal(r.execute().fetch(), pd.get_dummies(raw, dtype=float))

    # test series
    raw = pd.Series([3, 4, 1, 2])
    series = from_pandas_series(raw, chunk_size=2)
    r = get_dummies(series)
    pd.testing.assert_frame_equal(r.execute().fetch(), pd.get_dummies(raw))

    # test other variable
    raw = [3, 4, 1, 2]
    r = get_dummies(raw)
    pd.testing.assert_frame_equal(r.execute().fetch(), pd.get_dummies(raw))

    raw = pd.Series([3, 4, 2, 1])
    r = get_dummies(raw)
    pd.testing.assert_frame_equal(r.execute().fetch(), pd.get_dummies(raw))

    raw = pd.DataFrame(
        {
            "a": [1.1, 2.1, 3.1],
            "b": ["5", "-6", "-7"],
            "c": [1, 2, 3],
            "d": ["2", "3", "4"],
        }
    )
    r = get_dummies(raw)
    pd.testing.assert_frame_equal(r.execute().fetch(), pd.get_dummies(raw))


def test_to_numeric_execution(setup):
    rs = np.random.RandomState(0)
    s = pd.Series(rs.randint(5, size=100))
    s[rs.randint(100)] = np.nan

    # test 1 chunk
    series = from_pandas_series(s)

    r = to_numeric(series)
    pd.testing.assert_series_equal(r.execute().fetch(), pd.to_numeric(s))

    # test multi chunks
    series = from_pandas_series(s, chunk_size=20)

    r = to_numeric(series)
    pd.testing.assert_series_equal(r.execute().fetch(), pd.to_numeric(s))

    # test object dtype
    s = pd.Series(["1.0", 2, -3, "2.0"])
    series = from_pandas_series(s)

    r = to_numeric(series)
    pd.testing.assert_series_equal(r.execute().fetch(), pd.to_numeric(s))

    # test errors and downcast
    s = pd.Series(["appple", 2, -3, "2.0"])
    series = from_pandas_series(s)

    r = to_numeric(series, errors="ignore", downcast="signed")
    pd.testing.assert_series_equal(
        r.execute().fetch(), pd.to_numeric(s, errors="ignore", downcast="signed")
    )

    # test list data
    l = ["1.0", 2, -3, "2.0"]

    r = to_numeric(l)
    np.testing.assert_array_equal(r.execute().fetch(), pd.to_numeric(l))


def test_q_cut_execution(setup):
    rs = np.random.RandomState(0)
    raw = rs.random(15) * 1000
    s = pd.Series(raw, index=[f"i{i}" for i in range(15)])

    series = from_pandas_series(s)
    r = qcut(series, 3)
    result = r.execute().fetch()
    expected = pd.qcut(s, 3)
    pd.testing.assert_series_equal(result, expected)

    r = qcut(s, 3)
    result = r.execute().fetch()
    expected = pd.qcut(s, 3)
    pd.testing.assert_series_equal(result, expected)

    series = from_pandas_series(s)
    r = qcut(series, [0.3, 0.5, 0.7])
    result = r.execute().fetch()
    expected = pd.qcut(s, [0.3, 0.5, 0.7])
    pd.testing.assert_series_equal(result, expected)

    r = qcut(range(5), 3)
    result = r.execute().fetch()
    expected = pd.qcut(range(5), 3)
    assert isinstance(result, type(expected))
    pd.testing.assert_series_equal(pd.Series(result), pd.Series(expected))

    r = qcut(range(5), [0.2, 0.5])
    result = r.execute().fetch()
    expected = pd.qcut(range(5), [0.2, 0.5])
    assert isinstance(result, type(expected))
    pd.testing.assert_series_equal(pd.Series(result), pd.Series(expected))

    r = qcut(range(5), tensor([0.2, 0.5]))
    result = r.execute().fetch()
    expected = pd.qcut(range(5), [0.2, 0.5])
    assert isinstance(result, type(expected))
    pd.testing.assert_series_equal(pd.Series(result), pd.Series(expected))


def test_shift_execution(setup):
    # test dataframe
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        rs.randint(1000, size=(10, 8)), columns=["col" + str(i + 1) for i in range(8)]
    )

    df = from_pandas_df(raw, chunk_size=5)

    for periods in (2, -2, 6, -6):
        for axis in (0, 1):
            for fill_value in (None, 0, 1.0):
                r = df.shift(periods=periods, axis=axis, fill_value=fill_value)

                try:
                    result = r.execute().fetch()
                    expected = raw.shift(
                        periods=periods, axis=axis, fill_value=fill_value
                    )
                    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
                except AssertionError as e:  # pragma: no cover
                    raise AssertionError(
                        f"Failed when periods: {periods}, axis: {axis}, fill_value: {fill_value}"
                    ) from e

    raw2 = raw.copy()
    raw2.index = pd.date_range("2020-1-1", periods=10)
    raw2.columns = pd.date_range("2020-3-1", periods=8)

    df2 = from_pandas_df(raw2, chunk_size=5)

    # test freq not None
    for periods in (2, -2):
        for axis in (0, 1):
            for fill_value in (None, 0, 1.0):
                r = df2.shift(
                    periods=periods, freq="D", axis=axis, fill_value=fill_value
                )

                try:
                    result = r.execute().fetch()
                    expected = raw2.shift(
                        periods=periods, freq="D", axis=axis, fill_value=fill_value
                    )
                    pd.testing.assert_frame_equal(result, expected)
                except AssertionError as e:  # pragma: no cover
                    raise AssertionError(
                        f"Failed when periods: {periods}, axis: {axis}, fill_value: {fill_value}"
                    ) from e

    # test tshift
    r = df2.tshift(periods=1)
    result = r.execute().fetch()
    expected = raw2.tshift(periods=1)
    pd.testing.assert_frame_equal(result, expected)

    with pytest.raises(ValueError):
        _ = df.tshift(periods=1)

    # test series
    s = raw.iloc[:, 0]

    series = from_pandas_series(s, chunk_size=5)
    for periods in (0, 2, -2, 6, -6):
        for fill_value in (None, 0, 1.0):
            r = series.shift(periods=periods, fill_value=fill_value)

            try:
                result = r.execute().fetch()
                expected = s.shift(periods=periods, fill_value=fill_value)
                pd.testing.assert_series_equal(result, expected)
            except AssertionError as e:  # pragma: no cover
                raise AssertionError(
                    f"Failed when periods: {periods}, fill_value: {fill_value}"
                ) from e

    s2 = raw2.iloc[:, 0]

    # test freq not None
    series2 = from_pandas_series(s2, chunk_size=5)
    for periods in (2, -2):
        for fill_value in (None, 0, 1.0):
            r = series2.shift(periods=periods, freq="D", fill_value=fill_value)

            try:
                result = r.execute().fetch()
                expected = s2.shift(periods=periods, freq="D", fill_value=fill_value)
                pd.testing.assert_series_equal(result, expected)
            except AssertionError as e:  # pragma: no cover
                raise AssertionError(
                    f"Failed when periods: {periods}, fill_value: {fill_value}"
                ) from e


def test_diff_execution(setup):
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        rs.randint(1000, size=(10, 8)), columns=["col" + str(i + 1) for i in range(8)]
    )

    raw1 = raw.copy()
    raw1["col4"] = raw1["col4"] < 400

    r = from_pandas_df(raw1, chunk_size=(10, 5)).diff(-1)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw1.diff(-1))

    r = from_pandas_df(raw1, chunk_size=5).diff(-1)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw1.diff(-1))

    r = from_pandas_df(raw, chunk_size=(5, 8)).diff(1, axis=1)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.diff(1, axis=1))

    r = from_pandas_df(raw, chunk_size=5).diff(1, axis=1)
    pd.testing.assert_frame_equal(
        r.execute().fetch(), raw.diff(1, axis=1), check_dtype=False
    )

    # test series
    s = raw.iloc[:, 0]
    s1 = s.copy() < 400

    r = from_pandas_series(s, chunk_size=10).diff(-1)
    pd.testing.assert_series_equal(r.execute().fetch(), s.diff(-1))

    r = from_pandas_series(s, chunk_size=5).diff(-1)
    pd.testing.assert_series_equal(r.execute().fetch(), s.diff(-1))

    r = from_pandas_series(s1, chunk_size=5).diff(1)
    pd.testing.assert_series_equal(r.execute().fetch(), s1.diff(1))


def test_value_counts_execution(setup):
    rs = np.random.RandomState(0)
    s = pd.Series(rs.randint(5, size=100), name="s")
    s[rs.randint(100)] = np.nan

    # test 1 chunk
    series = from_pandas_series(s, chunk_size=100)

    r = series.value_counts()
    pd.testing.assert_series_equal(r.execute().fetch(), s.value_counts())

    r = series.value_counts(bins=5, normalize=True)
    pd.testing.assert_series_equal(
        r.execute().fetch(), s.value_counts(bins=5, normalize=True)
    )

    # test multi chunks
    series = from_pandas_series(s, chunk_size=30)

    r = series.value_counts(method="tree")
    pd.testing.assert_series_equal(r.execute().fetch(), s.value_counts())

    r = series.value_counts(method="tree", normalize=True)
    pd.testing.assert_series_equal(r.execute().fetch(), s.value_counts(normalize=True))

    # test bins and normalize
    r = series.value_counts(method="tree", bins=5, normalize=True)
    pd.testing.assert_series_equal(
        r.execute().fetch(), s.value_counts(bins=5, normalize=True)
    )


def test_astype(setup):
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        rs.randint(1000, size=(20, 8)), columns=["c" + str(i + 1) for i in range(8)]
    )
    # single chunk
    df = from_pandas_df(raw)
    r = df.astype("int32")

    result = r.execute().fetch()
    expected = raw.astype("int32")
    pd.testing.assert_frame_equal(expected, result)

    # multiply chunks
    df = from_pandas_df(raw, chunk_size=6)
    r = df.astype("int32")

    result = r.execute().fetch()
    expected = raw.astype("int32")
    pd.testing.assert_frame_equal(expected, result)

    # dict type
    df = from_pandas_df(raw, chunk_size=5)
    r = df.astype({"c1": "int32", "c2": "float", "c8": "str"})

    result = r.execute().fetch()
    expected = raw.astype({"c1": "int32", "c2": "float", "c8": "str"})
    pd.testing.assert_frame_equal(expected, result)

    # test arrow_string dtype
    df = from_pandas_df(raw, chunk_size=8)
    r = df.astype({"c1": "arrow_string"})

    result = r.execute().fetch()
    expected = raw.astype({"c1": "arrow_string"})
    pd.testing.assert_frame_equal(expected, result)

    # test series
    s = pd.Series(rs.randint(5, size=20))
    series = from_pandas_series(s)
    r = series.astype("int32")

    result = r.execute().fetch()
    expected = s.astype("int32")
    pd.testing.assert_series_equal(result, expected)

    series = from_pandas_series(s, chunk_size=6)
    r = series.astype("arrow_string")

    result = r.execute().fetch()
    expected = s.astype("arrow_string")
    pd.testing.assert_series_equal(result, expected)

    # test index
    raw = pd.Index(rs.randint(5, size=20))
    mix = from_pandas_index(raw)
    r = mix.astype("int32")

    result = r.execute().fetch()
    expected = raw.astype("int32")
    pd.testing.assert_index_equal(result, expected)

    # multiply chunks
    series = from_pandas_series(s, chunk_size=6)
    r = series.astype("str")

    result = r.execute().fetch()
    expected = s.astype("str")
    pd.testing.assert_series_equal(result, expected)

    # test category
    raw = pd.DataFrame(
        rs.randint(3, size=(20, 8)), columns=["c" + str(i + 1) for i in range(8)]
    )

    df = from_pandas_df(raw)
    r = df.astype("category")

    result = r.execute().fetch()
    expected = raw.astype("category")
    pd.testing.assert_frame_equal(expected, result)

    df = from_pandas_df(raw)
    r = df.astype({"c1": "category", "c8": "int32", "c4": "str"})

    result = r.execute().fetch()
    expected = raw.astype({"c1": "category", "c8": "int32", "c4": "str"})
    pd.testing.assert_frame_equal(expected, result)

    df = from_pandas_df(raw, chunk_size=5)
    r = df.astype("category")

    result = r.execute().fetch()
    expected = raw.astype("category")
    pd.testing.assert_frame_equal(expected, result)

    df = from_pandas_df(raw, chunk_size=3)
    r = df.astype({"c1": "category", "c8": "int32", "c4": "str"})

    result = r.execute().fetch()
    expected = raw.astype({"c1": "category", "c8": "int32", "c4": "str"})
    pd.testing.assert_frame_equal(expected, result)

    df = from_pandas_df(raw, chunk_size=6)
    r = df.astype(
        {
            "c1": "category",
            "c5": "float",
            "c2": "int32",
            "c7": pd.CategoricalDtype([1, 3, 4, 2]),
            "c4": pd.CategoricalDtype([1, 3, 2]),
        }
    )
    result = r.execute().fetch()
    expected = raw.astype(
        {
            "c1": "category",
            "c5": "float",
            "c2": "int32",
            "c7": pd.CategoricalDtype([1, 3, 4, 2]),
            "c4": pd.CategoricalDtype([1, 3, 2]),
        }
    )
    pd.testing.assert_frame_equal(expected, result)

    df = from_pandas_df(raw, chunk_size=8)
    r = df.astype({"c2": "category"})
    result = r.execute().fetch()
    expected = raw.astype({"c2": "category"})
    pd.testing.assert_frame_equal(expected, result)

    # test series category
    raw = pd.Series(np.random.choice(["a", "b", "c"], size=(10,)))
    series = from_pandas_series(raw, chunk_size=4)
    result = series.astype("category").execute().fetch()
    expected = raw.astype("category")
    pd.testing.assert_series_equal(expected, result)

    series = from_pandas_series(raw, chunk_size=3)
    result = (
        series.astype(pd.CategoricalDtype(["a", "c", "b"]), copy=False)
        .execute()
        .fetch()
    )
    expected = raw.astype(pd.CategoricalDtype(["a", "c", "b"]), copy=False)
    pd.testing.assert_series_equal(expected, result)

    series = from_pandas_series(raw, chunk_size=6)
    result = series.astype(pd.CategoricalDtype(["a", "c", "b", "d"])).execute().fetch()
    expected = raw.astype(pd.CategoricalDtype(["a", "c", "b", "d"]))
    pd.testing.assert_series_equal(expected, result)


def test_drop(setup):
    # test dataframe drop
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        rs.randint(1000, size=(20, 8)), columns=["c" + str(i + 1) for i in range(8)]
    )

    df = from_pandas_df(raw, chunk_size=3)

    columns = ["c2", "c4", "c5", "c6"]
    index = [3, 6, 7]
    r = df.drop(columns=columns, index=index)
    pd.testing.assert_frame_equal(
        r.execute().fetch(), raw.drop(columns=columns, index=index)
    )

    idx_series = from_pandas_series(pd.Series(index))
    r = df.drop(idx_series)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.drop(pd.Series(index)))

    df.drop(columns, axis=1, inplace=True)
    pd.testing.assert_frame_equal(df.execute().fetch(), raw.drop(columns, axis=1))

    del df["c3"]
    pd.testing.assert_frame_equal(
        df.execute().fetch(), raw.drop(columns + ["c3"], axis=1)
    )

    ps = df.pop("c8")
    pd.testing.assert_frame_equal(
        df.execute().fetch(), raw.drop(columns + ["c3", "c8"], axis=1)
    )
    pd.testing.assert_series_equal(ps.execute().fetch(), raw["c8"])

    # test series drop
    raw = pd.Series(rs.randint(1000, size=(20,)))

    series = from_pandas_series(raw, chunk_size=3)

    r = series.drop(index=index)
    pd.testing.assert_series_equal(r.execute().fetch(), raw.drop(index=index))

    # test index drop
    ser = pd.Series(range(20))
    rs.shuffle(ser)
    raw = pd.Index(ser)

    idx = from_pandas_index(raw)

    r = idx.drop(index)
    pd.testing.assert_index_equal(r.execute().fetch(), raw.drop(index))


def test_melt(setup):
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        rs.randint(1000, size=(20, 8)), columns=["c" + str(i + 1) for i in range(8)]
    )

    df = from_pandas_df(raw, chunk_size=3)

    r = df.melt(id_vars=["c1"], value_vars=["c2", "c4"])
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_values(["c1", "variable"]).reset_index(drop=True),
        raw.melt(id_vars=["c1"], value_vars=["c2", "c4"])
        .sort_values(["c1", "variable"])
        .reset_index(drop=True),
    )


def test_drop_duplicates(setup):
    # test dataframe drop
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        rs.randint(1000, size=(20, 5)),
        columns=["c" + str(i + 1) for i in range(5)],
        index=["i" + str(j) for j in range(20)],
    )
    duplicate_lines = rs.randint(1000, size=5)
    for i in [1, 3, 10, 11, 15]:
        raw.iloc[i] = duplicate_lines

    with option_context({"combine_size": 2}):
        # test dataframe
        for chunk_size in [(8, 3), (20, 5)]:
            df = from_pandas_df(raw, chunk_size=chunk_size)
            if chunk_size[0] < len(raw):
                methods = ["tree", "subset_tree", "shuffle"]
            else:
                # 1 chunk
                methods = [None]
            for method in methods:
                for subset in [None, "c1", ["c1", "c2"]]:
                    for keep in ["first", "last", False]:
                        for ignore_index in [True, False]:
                            try:
                                r = df.drop_duplicates(
                                    method=method,
                                    subset=subset,
                                    keep=keep,
                                    ignore_index=ignore_index,
                                )
                                result = r.execute().fetch()
                                try:
                                    expected = raw.drop_duplicates(
                                        subset=subset,
                                        keep=keep,
                                        ignore_index=ignore_index,
                                    )
                                except TypeError:
                                    # ignore_index is supported in pandas 1.0
                                    expected = raw.drop_duplicates(
                                        subset=subset, keep=keep
                                    )
                                    if ignore_index:
                                        expected.reset_index(drop=True, inplace=True)

                                pd.testing.assert_frame_equal(result, expected)
                            except Exception as e:  # pragma: no cover
                                raise AssertionError(
                                    f"failed when method={method}, subset={subset}, "
                                    f"keep={keep}, ignore_index={ignore_index}"
                                ) from e

        # test series and index
        s = raw["c3"]
        ind = pd.Index(s)

        for tp, obj in [("series", s), ("index", ind)]:
            for chunk_size in [8, 20]:
                to_m = from_pandas_series if tp == "series" else from_pandas_index
                mobj = to_m(obj, chunk_size=chunk_size)
                if chunk_size < len(obj):
                    methods = ["tree", "shuffle"]
                else:
                    # 1 chunk
                    methods = [None]
                for method in methods:
                    for keep in ["first", "last", False]:
                        try:
                            r = mobj.drop_duplicates(method=method, keep=keep)
                            result = r.execute().fetch()
                            expected = obj.drop_duplicates(keep=keep)

                            cmp = (
                                pd.testing.assert_series_equal
                                if tp == "series"
                                else pd.testing.assert_index_equal
                            )
                            cmp(result, expected)
                        except Exception as e:  # pragma: no cover
                            raise AssertionError(
                                f"failed when method={method}, keep={keep}"
                            ) from e

        # test inplace
        series = from_pandas_series(s, chunk_size=11)
        series.drop_duplicates(inplace=True)
        result = series.execute().fetch()
        expected = s.drop_duplicates()
        pd.testing.assert_series_equal(result, expected)


def test_duplicated(setup):
    # test dataframe drop
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        rs.randint(1000, size=(20, 5)),
        columns=["c" + str(i + 1) for i in range(5)],
        index=["i" + str(j) for j in range(20)],
    )
    duplicate_lines = rs.randint(1000, size=5)
    for i in [1, 3, 10, 11, 15]:
        raw.iloc[i] = duplicate_lines

    with option_context({"combine_size": 2}):
        # test dataframe
        for chunk_size in [(8, 3), (20, 5)]:
            df = from_pandas_df(raw, chunk_size=chunk_size)
            if chunk_size[0] < len(raw):
                methods = ["tree", "subset_tree", "shuffle"]
            else:
                # 1 chunk
                methods = [None]
            for method in methods:
                for subset in [None, "c1", ["c1", "c2"]]:
                    for keep in ["first", "last", False]:
                        try:
                            r = df.duplicated(method=method, subset=subset, keep=keep)
                            result = r.execute().fetch()
                            expected = raw.duplicated(subset=subset, keep=keep)
                            pd.testing.assert_series_equal(result, expected)
                        except Exception as e:  # pragma: no cover
                            raise AssertionError(
                                f"failed when method={method}, subset={subset}, "
                                f"keep={keep}"
                            ) from e

        # test series
        s = raw["c3"]

        for tp, obj in [("series", s)]:
            for chunk_size in [8, 20]:
                to_m = from_pandas_series if tp == "series" else from_pandas_index
                mobj = to_m(obj, chunk_size=chunk_size)
                if chunk_size < len(obj):
                    methods = ["tree", "shuffle"]
                else:
                    # 1 chunk
                    methods = [None]
                for method in methods:
                    for keep in ["first", "last", False]:
                        try:
                            r = mobj.duplicated(method=method, keep=keep)
                            result = r.execute().fetch()
                            expected = obj.duplicated(keep=keep)

                            cmp = (
                                pd.testing.assert_series_equal
                                if tp == "series"
                                else pd.testing.assert_index_equal
                            )
                            cmp(result, expected)
                        except Exception as e:  # pragma: no cover
                            raise AssertionError(
                                f"failed when method={method}, keep={keep}"
                            ) from e


def test_memory_usage_execution(setup):
    dtypes = ["int64", "float64", "complex128", "object", "bool"]
    data = dict([(t, np.ones(shape=500).astype(t)) for t in dtypes])
    raw = pd.DataFrame(data)

    df = from_pandas_df(raw, chunk_size=(500, 2))
    r = df.memory_usage(index=False)
    pd.testing.assert_series_equal(r.execute().fetch(), raw.memory_usage(index=False))

    df = from_pandas_df(raw, chunk_size=(500, 2))
    r = df.memory_usage(index=True)
    pd.testing.assert_series_equal(r.execute().fetch(), raw.memory_usage(index=True))

    df = from_pandas_df(raw, chunk_size=(100, 3))
    r = df.memory_usage(index=False)
    pd.testing.assert_series_equal(r.execute().fetch(), raw.memory_usage(index=False))

    r = df.memory_usage(index=True)
    pd.testing.assert_series_equal(r.execute().fetch(), raw.memory_usage(index=True))

    raw = pd.DataFrame(data, index=np.arange(500).astype("object"))

    df = from_pandas_df(raw, chunk_size=(100, 3))
    r = df.memory_usage(index=True)
    pd.testing.assert_series_equal(r.execute().fetch(), raw.memory_usage(index=True))

    raw = pd.Series(np.ones(shape=500).astype("object"), name="s")

    series = from_pandas_series(raw)
    r = series.memory_usage(index=True)
    assert r.execute().fetch() == raw.memory_usage(index=True)

    series = from_pandas_series(raw, chunk_size=100)
    r = series.memory_usage(index=False)
    assert r.execute().fetch() == raw.memory_usage(index=False)

    series = from_pandas_series(raw, chunk_size=100)
    r = series.memory_usage(index=True)
    assert r.execute().fetch() == raw.memory_usage(index=True)

    raw = pd.Series(
        np.ones(shape=500).astype("object"),
        index=np.arange(500).astype("object"),
        name="s",
    )

    series = from_pandas_series(raw, chunk_size=100)
    r = series.memory_usage(index=True)
    assert r.execute().fetch() == raw.memory_usage(index=True)

    raw = pd.Index(np.arange(500), name="s")

    index = from_pandas_index(raw)
    r = index.memory_usage()
    assert r.execute().fetch() == raw.memory_usage()

    index = from_pandas_index(raw, chunk_size=100)
    r = index.memory_usage()
    assert r.execute().fetch() == raw.memory_usage()


def test_select_dtypes_execution(setup):
    raw = pd.DataFrame({"a": np.random.rand(10), "b": np.random.randint(10, size=10)})

    df = from_pandas_df(raw, chunk_size=5)
    r = df.select_dtypes(include=["float64"])

    result = r.execute().fetch()
    expected = raw.select_dtypes(include=["float64"])
    pd.testing.assert_frame_equal(result, expected)


def test_map_chunk_execution(setup):
    raw = pd.DataFrame(np.random.rand(10, 5), columns=[f"col{i}" for i in range(5)])

    df = from_pandas_df(raw, chunk_size=(5, 3))

    def f1(pdf):
        return pdf + 1

    r = df.map_chunk(f1)

    result = r.execute().fetch()
    expected = raw + 1
    pd.testing.assert_frame_equal(result, expected)

    raw_s = raw["col1"]
    series = from_pandas_series(raw_s, chunk_size=5)

    r = series.map_chunk(f1)

    result = r.execute().fetch()
    expected = raw_s + 1
    pd.testing.assert_series_equal(result, expected)

    def f2(pdf):
        return pdf.sum(axis=1)

    df = from_pandas_df(raw, chunk_size=5)
    r = df.map_chunk(f2, output_type="series")

    result = r.execute().fetch()
    expected = raw.sum(axis=1)
    pd.testing.assert_series_equal(result, expected)

    raw = pd.DataFrame({"a": [f"s{i}" for i in range(10)], "b": np.arange(10)})

    df = from_pandas_df(raw, chunk_size=5)

    def f3(pdf):
        return pdf["a"].str.slice(1).astype(int) + pdf["b"]

    with pytest.raises(TypeError):
        r = df.map_chunk(f3)
        _ = r.execute().fetch()

    r = df.map_chunk(f3, output_type="series")
    result = r.execute(extra_config={"check_dtypes": False}).fetch()
    expected = f3(raw)
    pd.testing.assert_series_equal(result, expected)

    def f4(pdf):
        ret = pd.DataFrame(columns=["a", "b"])
        ret["a"] = pdf["a"].str.slice(1).astype(int)
        ret["b"] = pdf["b"]
        return ret

    with pytest.raises(TypeError):
        r = df.map_chunk(f4, output_type="dataframe")
        _ = r.execute().fetch()

    r = df.map_chunk(
        f4,
        output_type="dataframe",
        dtypes=pd.Series([np.dtype(int), raw["b"].dtype], index=["a", "b"]),
    )
    result = r.execute().fetch()
    expected = f4(raw)
    pd.testing.assert_frame_equal(result, expected)

    raw2 = pd.DataFrame({"a": [np.array([1, 2, 3]), np.array([4, 5, 6])]})
    df2 = from_pandas_df(raw2)
    dtypes = pd.Series([np.dtype(float)] * 3)
    r = df2.map_chunk(
        lambda x: x["a"].apply(pd.Series), output_type="dataframe", dtypes=dtypes
    )
    assert r.shape == (np.nan, 3)
    pd.testing.assert_series_equal(r.dtypes, dtypes)
    result = r.execute().fetch()
    expected = raw2.apply(lambda x: x["a"], axis=1, result_type="expand")
    pd.testing.assert_frame_equal(result, expected)

    raw = pd.DataFrame(np.random.rand(10, 5), columns=[f"col{i}" for i in range(5)])

    df = from_pandas_df(raw, chunk_size=(5, 3))

    def f5(pdf, chunk_index):
        return pdf + 1 + chunk_index[0]

    r = df.map_chunk(f5, with_chunk_index=True)

    result = r.execute().fetch()
    expected = (raw + 1).add(np.arange(10) // 5, axis=0)
    pd.testing.assert_frame_equal(result, expected)

    raw_s = raw["col1"]
    series = from_pandas_series(raw_s, chunk_size=5)

    r = series.map_chunk(f5, with_chunk_index=True)

    result = r.execute().fetch()
    expected = raw_s + 1 + np.arange(10) // 5
    pd.testing.assert_series_equal(result, expected)


def test_cartesian_chunk_execution(setup):
    rs = np.random.RandomState(0)
    raw1 = pd.DataFrame({"a": rs.randint(3, size=10), "b": rs.rand(10)})
    raw2 = pd.DataFrame(
        {"c": rs.randint(3, size=10), "d": rs.rand(10), "e": rs.rand(10)}
    )
    df1 = from_pandas_df(raw1, chunk_size=(5, 1))
    df2 = from_pandas_df(raw2, chunk_size=(5, 1))

    def f(c1, c2):
        c1, c2 = c1.copy(), c2.copy()
        c1["x"] = 1
        c2["x"] = 1
        r = c1.merge(c2, on="x")
        r = r[(r["b"] > r["d"]) & (r["b"] < r["e"])]
        return r[["a", "c"]]

    rr = df1.cartesian_chunk(df2, f)

    result = rr.execute().fetch()
    expected = f(raw1, raw2)
    pd.testing.assert_frame_equal(
        result.sort_values(by=["a", "c"]).reset_index(drop=True),
        expected.sort_values(by=["a", "c"]).reset_index(drop=True),
    )

    def f2(c1, c2):
        r = f(c1, c2)
        return r["a"] + r["c"]

    rr = df1.cartesian_chunk(df2, f2)

    result = rr.execute().fetch()
    expected = f2(raw1, raw2)
    pd.testing.assert_series_equal(
        result.sort_values().reset_index(drop=True),
        expected.sort_values().reset_index(drop=True),
    )

    # size_res = setup.executor.execute_dataframe(rr, mock=True)[0][0]
    # assert size_res > 0

    def f3(c1, c2):
        cr = pd.DataFrame()
        cr["a"] = c1.str.slice(1).astype(np.int64)
        cr["x"] = 1
        cr2 = pd.DataFrame()
        cr2["b"] = c2.str.slice(1).astype(np.int64)
        cr2["x"] = 1
        return cr.merge(cr2, on="x")[["a", "b"]]

    s_raw = pd.Series([f"s{i}" for i in range(10)])
    series = from_pandas_series(s_raw, chunk_size=5)

    rr = series.cartesian_chunk(
        series,
        f3,
        output_type="dataframe",
        dtypes=pd.Series([np.dtype(np.int64)] * 2, index=["a", "b"]),
    )

    result = rr.execute().fetch()
    expected = f3(s_raw, s_raw)
    pd.testing.assert_frame_equal(
        result.sort_values(by=["a", "b"]).reset_index(drop=True),
        expected.sort_values(by=["a", "b"]).reset_index(drop=True),
    )

    with pytest.raises(TypeError):
        _ = series.cartesian_chunk(series, f3)

    def f4(c1, c2):
        r = f3(c1, c2)
        return r["a"] + r["b"]

    rr = series.cartesian_chunk(
        series, f4, output_type="series", dtypes=np.dtype(np.int64)
    )

    result = rr.execute().fetch()
    expected = f4(s_raw, s_raw)
    pd.testing.assert_series_equal(
        result.sort_values().reset_index(drop=True),
        expected.sort_values().reset_index(drop=True),
    )


def test_rebalance_execution(setup):
    raw = pd.DataFrame(np.random.rand(10, 3), columns=list("abc"))
    df = from_pandas_df(raw)

    def _expect_count(n):
        def _tile_rebalance(op):
            tileable = yield from op.tile(op)
            assert len(tileable.chunks) == n
            return tileable

        return _tile_rebalance

    r = df.rebalance(num_partitions=3)
    extra_config = {"operand_tile_handlers": {DataFrameRebalance: _expect_count(3)}}
    result = r.execute(extra_config=extra_config).fetch()
    pd.testing.assert_frame_equal(result, raw)

    r = df.rebalance(factor=0.5)
    extra_config = {"operand_tile_handlers": {DataFrameRebalance: _expect_count(1)}}
    result = r.execute(extra_config=extra_config).fetch()
    pd.testing.assert_frame_equal(result, raw)

    # test worker has two cores
    r = df.rebalance()
    extra_config = {"operand_tile_handlers": {DataFrameRebalance: _expect_count(2)}}
    result = r.execute(extra_config=extra_config).fetch()
    pd.testing.assert_frame_equal(result, raw)


def test_stack_execution(setup):
    raw = pd.DataFrame(
        np.random.rand(10, 3), columns=list("abc"), index=[f"s{i}" for i in range(10)]
    )
    for loc in [(5, 1), (8, 2), (1, 0)]:
        raw.iloc[loc] = np.nan
    df = from_pandas_df(raw, chunk_size=(5, 2))

    for dropna in (True, False):
        r = df.stack(dropna=dropna)
        result = r.execute().fetch()
        expected = raw.stack(dropna=dropna)
        pd.testing.assert_series_equal(result, expected)

    cols = pd.MultiIndex.from_tuples([("c1", "cc1"), ("c1", "cc2"), ("c2", "cc3")])
    raw2 = raw.copy()
    raw2.columns = cols
    df = from_pandas_df(raw2, chunk_size=(5, 2))

    for level in [-1, 0, [0, 1]]:
        for dropna in (True, False):
            r = df.stack(level=level, dropna=dropna)
            result = r.execute().fetch()
            expected = raw2.stack(level=level, dropna=dropna)
            assert_method = (
                pd.testing.assert_series_equal
                if expected.ndim == 1
                else pd.testing.assert_frame_equal
            )
            assert_method(result, expected)


@pytest.mark.parametrize(
    "ignore_index", [False, True] if _explode_with_ignore_index else [False]
)
def test_explode_execution(setup, ignore_index):
    explode_kw = {"ignore_index": True} if ignore_index else {}

    raw = pd.DataFrame(
        {
            "a": np.random.rand(10),
            "b": [np.random.rand(random.randint(1, 10)) for _ in range(10)],
            "c": np.random.rand(10),
            "d": np.random.rand(10),
        }
    )
    df = from_pandas_df(raw, chunk_size=(4, 2))
    r = df.explode("b", ignore_index=ignore_index)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.explode("b", **explode_kw))

    series = from_pandas_series(raw.b, chunk_size=4)
    r = series.explode(ignore_index=ignore_index)
    pd.testing.assert_series_equal(r.execute().fetch(), raw.b.explode(**explode_kw))


def test_eval_query_execution(setup):
    rs = np.random.RandomState(0)
    raw = pd.DataFrame({"a": rs.rand(100), "b": rs.rand(100), "c c": rs.rand(100)})
    df = from_pandas_df(raw, chunk_size=(10, 2))

    r = mars_eval('c = df.a * 2 + df["c c"]', target=df)
    pd.testing.assert_frame_equal(
        r.execute().fetch(),
        pd.eval('c = raw.a * 2 + raw["c c"]', engine="python", target=raw),
    )

    r = df.eval("a + b")
    pd.testing.assert_series_equal(r.execute().fetch(), raw.eval("a + b"))

    _val = 5.0  # noqa: F841
    _val_array = [1, 2, 3]  # noqa: F841
    expr = """
    e = -a + b + 1
    f = b + `c c` + @_val + @_val_array[-1]
    """
    r = df.eval(expr)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.eval(expr))

    copied_df = df.copy()
    copied_df.eval("c = a + b", inplace=True)
    pd.testing.assert_frame_equal(copied_df.execute().fetch(), raw.eval("c = a + b"))

    expr = "a > b | a < `c c`"
    r = df.query(expr)
    pd.testing.assert_frame_equal(
        r.execute(extra_config={"check_index_value": False}).fetch(), raw.query(expr)
    )

    expr = "a > b & ~(a < `c c`)"
    r = df.query(expr)
    pd.testing.assert_frame_equal(
        r.execute(extra_config={"check_index_value": False}).fetch(), raw.query(expr)
    )

    expr = "a < b < `c c`"
    r = df.query(expr)
    pd.testing.assert_frame_equal(
        r.execute(extra_config={"check_index_value": False}).fetch(), raw.query(expr)
    )

    expr = "a < 0.5 and a != 0.1 and b != 0.2"
    r = df.query(expr)
    pd.testing.assert_frame_equal(
        r.execute(extra_config={"check_index_value": False}).fetch(), raw.query(expr)
    )

    expr = "(a < 0.5 or a > 0.7) and (b != 0.1 or `c c` > 0.2)"
    r = df.query(expr)
    pd.testing.assert_frame_equal(
        r.execute(extra_config={"check_index_value": False}).fetch(), raw.query(expr)
    )

    copied_df = df.copy()
    copied_df.query("a < b", inplace=True)
    pd.testing.assert_frame_equal(
        copied_df.execute(extra_config={"check_index_value": False}).fetch(),
        raw.query("a < b"),
    )


def test_check_monotonic_execution(setup):
    idx_value = pd.Index(list(range(1000)))

    idx_increase = from_pandas_index(idx_value, chunk_size=100)
    assert idx_increase.is_monotonic_increasing.execute().fetch() is True
    assert idx_increase.is_monotonic_decreasing.execute().fetch() is False

    idx_decrease = from_pandas_index(idx_value[::-1], chunk_size=100)
    assert idx_decrease.is_monotonic_increasing.execute().fetch() is False
    assert idx_decrease.is_monotonic_decreasing.execute().fetch() is True

    idx_mixed = from_pandas_index(
        pd.Index(list(range(500)) + list(range(500))), chunk_size=100
    )
    assert idx_mixed.is_monotonic_increasing.execute().fetch() is False
    assert idx_mixed.is_monotonic_decreasing.execute().fetch() is False

    ser_mixed = from_pandas_series(
        pd.Series(list(range(500)) + list(range(499, 999))), chunk_size=100
    )
    assert ser_mixed.is_monotonic_increasing.execute().fetch() is True
    assert ser_mixed.is_monotonic_decreasing.execute().fetch() is False


def test_pct_change_execution(setup):
    # test dataframe
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        rs.randint(1000, size=(10, 8)),
        columns=["col" + str(i + 1) for i in range(8)],
        index=pd.date_range("2021-1-1", periods=10),
    )

    df = from_pandas_df(raw, chunk_size=5)
    r = df.pct_change()
    result = r.execute().fetch()
    expected = raw.pct_change()
    pd.testing.assert_frame_equal(expected, result)

    df = from_pandas_df(raw, chunk_size=5)
    r = df.pct_change(fill_method=None)
    result = r.execute().fetch()
    expected = raw.pct_change(fill_method=None)
    pd.testing.assert_frame_equal(expected, result)

    df = from_pandas_df(raw, chunk_size=5)
    r = df.pct_change(freq="D")
    result = r.execute().fetch()
    expected = raw.pct_change(freq="D")
    pd.testing.assert_frame_equal(expected, result)
