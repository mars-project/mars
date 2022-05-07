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

import os
import sys

import numpy as np
import pandas as pd
import pytest

from ....tests.core import require_cudf
from ... import DataFrame, Series, ArrowStringDtype


@pytest.mark.parametrize(
    "distinct_opt", ["0"] if sys.platform.lower().startswith("win") else ["0", "1"]
)
def test_sort_values_execution(setup, distinct_opt):
    ns = np.random.RandomState(0)
    os.environ["PSRS_DISTINCT_COL"] = distinct_opt
    df = pd.DataFrame(ns.rand(100, 10), columns=["a" + str(i) for i in range(10)])

    # test one chunk
    mdf = DataFrame(df)
    result = mdf.sort_values("a0").execute().fetch()
    expected = df.sort_values("a0")

    pd.testing.assert_frame_equal(result, expected)

    result = mdf.sort_values(["a6", "a7"], ascending=False).execute().fetch()
    expected = df.sort_values(["a6", "a7"], ascending=False)

    pd.testing.assert_frame_equal(result, expected)

    # test psrs
    mdf = DataFrame(df, chunk_size=10)
    result = mdf.sort_values("a0").execute().fetch()
    expected = df.sort_values("a0")

    pd.testing.assert_frame_equal(result, expected)

    result = mdf.sort_values(["a3", "a4"]).execute().fetch()
    expected = df.sort_values(["a3", "a4"])

    pd.testing.assert_frame_equal(result, expected)

    # test ascending=False
    result = mdf.sort_values(["a0", "a1"], ascending=False).execute().fetch()
    expected = df.sort_values(["a0", "a1"], ascending=False)

    pd.testing.assert_frame_equal(result, expected)

    result = mdf.sort_values(["a7"], ascending=False).execute().fetch()
    expected = df.sort_values(["a7"], ascending=False)

    pd.testing.assert_frame_equal(result, expected)

    # test ascending is a list
    result = (
        mdf.sort_values(["a3", "a4", "a5", "a6"], ascending=[False, True, True, False])
        .execute()
        .fetch()
    )
    expected = df.sort_values(
        ["a3", "a4", "a5", "a6"], ascending=[False, True, True, False]
    )
    pd.testing.assert_frame_equal(result, expected)

    in_df = pd.DataFrame(
        {
            "col1": ns.choice([f"a{i}" for i in range(5)], size=(100,)),
            "col2": ns.choice([f"b{i}" for i in range(5)], size=(100,)),
            "col3": ns.choice([f"c{i}" for i in range(5)], size=(100,)),
            "col4": ns.randint(10, 20, size=(100,)),
        }
    )
    mdf = DataFrame(in_df, chunk_size=10)
    result = (
        mdf.sort_values(
            ["col1", "col4", "col3", "col2"], ascending=[False, False, True, False]
        )
        .execute()
        .fetch()
    )
    expected = in_df.sort_values(
        ["col1", "col4", "col3", "col2"], ascending=[False, False, True, False]
    )
    pd.testing.assert_frame_equal(result, expected)

    # test multiindex
    df2 = df.copy(deep=True)
    df2.columns = pd.MultiIndex.from_product([list("AB"), list("CDEFG")])
    mdf = DataFrame(df2, chunk_size=5)

    result = mdf.sort_values([("A", "C")]).execute().fetch()
    expected = df2.sort_values([("A", "C")])

    pd.testing.assert_frame_equal(result, expected)

    # test rechunk
    mdf = DataFrame(df, chunk_size=3)
    result = mdf.sort_values("a0").execute().fetch()
    expected = df.sort_values("a0")

    pd.testing.assert_frame_equal(result, expected)

    result = mdf.sort_values(["a3", "a4"]).execute().fetch()
    expected = df.sort_values(["a3", "a4"])

    pd.testing.assert_frame_equal(result, expected)

    # test other types
    raw = pd.DataFrame(
        {
            "a": np.random.rand(10),
            "b": np.random.randint(1000, size=10),
            "c": np.random.rand(10),
            "d": [np.random.bytes(10) for _ in range(10)],
            "e": [pd.Timestamp(f"201{i}") for i in range(10)],
            "f": [pd.Timedelta(f"{i} days") for i in range(10)],
        },
    )
    mdf = DataFrame(raw, chunk_size=3)

    for label in raw.columns:
        result = mdf.sort_values(label).execute().fetch()
        expected = raw.sort_values(label)
        pd.testing.assert_frame_equal(result, expected)

    result = mdf.sort_values(["a", "b", "e"], ascending=False).execute().fetch()
    expected = raw.sort_values(["a", "b", "e"], ascending=False)

    pd.testing.assert_frame_equal(result, expected)

    # test nan
    df = pd.DataFrame(
        {
            "col1": ["A", "A", "B", "B", "D", "C"],
            "col2": [2, 1, 9, np.nan, 7, 4],
            "col3": [0, 1, 9, 4, 2, 3],
        }
    )
    mdf = DataFrame(df)
    result = mdf.sort_values(["col2"]).execute().fetch()
    expected = df.sort_values(["col2"])

    pd.testing.assert_frame_equal(result, expected)

    mdf = DataFrame(df, chunk_size=3)
    result = mdf.sort_values(["col2"]).execute().fetch()
    expected = df.sort_values(["col2"])

    pd.testing.assert_frame_equal(result, expected)

    # test None (issue #1885)
    df = pd.DataFrame(np.random.rand(1000, 10))

    df[0][df[0] < 0.5] = "A"
    df[0][df[0] != "A"] = None

    mdf = DataFrame(df)
    result = mdf.sort_values([0, 1]).execute().fetch()
    expected = df.sort_values([0, 1])

    pd.testing.assert_frame_equal(result, expected)

    mdf = DataFrame(df, chunk_size=100)
    result = mdf.sort_values([0, 1]).execute().fetch()
    expected = df.sort_values([0, 1])

    pd.testing.assert_frame_equal(result, expected)

    # test ignore_index
    df = pd.DataFrame(np.random.rand(10, 3), columns=["a" + str(i) for i in range(3)])

    mdf = DataFrame(df, chunk_size=3)
    result = mdf.sort_values(["a0", "a1"], ignore_index=True).execute().fetch()
    try:  # for python3.5
        expected = df.sort_values(["a0", "a1"], ignore_index=True)
    except TypeError:
        expected = df.sort_values(["a0", "a1"])
        expected.index = pd.RangeIndex(len(expected))

    pd.testing.assert_frame_equal(result, expected)

    # test inplace
    mdf = DataFrame(df)
    mdf.sort_values("a0", inplace=True)
    result = mdf.execute().fetch()
    df.sort_values("a0", inplace=True)

    pd.testing.assert_frame_equal(result, df)

    # test unknown shape
    df = pd.DataFrame({"a": list(range(10)), "b": np.random.random(10)})
    mdf = DataFrame(df, chunk_size=4)
    filtered = mdf[mdf["a"] > 2]
    result = filtered.sort_values(by="b").execute().fetch()

    pd.testing.assert_frame_equal(result, df[df["a"] > 2].sort_values(by="b"))

    # test empty dataframe
    df = pd.DataFrame({"a": list(range(10)), "b": np.random.random(10)})
    mdf = DataFrame(df, chunk_size=4)
    filtered = mdf[mdf["b"] > 100]
    result = filtered.sort_values(by="b").execute().fetch()

    pd.testing.assert_frame_equal(result, df[df["b"] > 100].sort_values(by="b"))

    # test chunks with zero length
    df = pd.DataFrame({"a": list(range(10)), "b": np.random.random(10)})
    df.iloc[4:8, 1] = 0

    mdf = DataFrame(df, chunk_size=4)
    filtered = mdf[mdf["b"] != 0]
    result = filtered.sort_values(by="b").execute().fetch()

    pd.testing.assert_frame_equal(result, df[df["b"] != 0].sort_values(by="b"))

    # test Series.sort_values
    raw = pd.Series(np.random.rand(10))
    series = Series(raw)
    result = series.sort_values().execute().fetch()
    expected = raw.sort_values()

    pd.testing.assert_series_equal(result, expected)

    series = Series(raw, chunk_size=3)
    result = series.sort_values().execute().fetch()
    expected = raw.sort_values()

    pd.testing.assert_series_equal(result, expected)

    series = Series(raw, chunk_size=2)
    result = series.sort_values(ascending=False).execute().fetch()
    expected = raw.sort_values(ascending=False)

    pd.testing.assert_series_equal(result, expected)

    # test empty series
    series = pd.Series(list(range(10)), name="a")
    mseries = Series(series, chunk_size=4)
    filtered = mseries[mseries > 100]
    result = filtered.sort_values().execute().fetch()

    pd.testing.assert_series_equal(result, series[series > 100].sort_values())

    # test series with None
    series = pd.Series(np.arange(1000))

    series[series < 500] = "A"
    series[series != "A"] = None

    mseries = Series(series, chunk_size=100)
    result = mseries.sort_values().execute().fetch()
    expected = series.sort_values()
    pd.testing.assert_series_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )

    # test for empty input(#GH 2649)
    pd_df = pd.DataFrame(np.random.rand(10, 3), columns=["col1", "col2", "col3"])
    df = DataFrame(pd_df, chunk_size=4)
    df = df[df["col2"] > 1].execute()
    result = df.sort_values(by="col1").execute().fetch()
    expected = pd_df[pd_df["col2"] > 1].sort_values(by="col1")
    pd.testing.assert_frame_equal(result, expected)

    pd_s = pd.Series(np.random.rand(10))
    s = Series(pd_s, chunk_size=4)
    s = s[s > 1].execute()
    result = s.sort_values().execute().fetch()
    expected = pd_s[pd_s > 1].sort_values()
    pd.testing.assert_series_equal(result, expected)


def test_sort_index_execution(setup):
    raw = pd.DataFrame(np.random.rand(100, 20), index=np.random.rand(100))

    mdf = DataFrame(raw)
    result = mdf.sort_index().execute().fetch()
    expected = raw.sort_index()
    pd.testing.assert_frame_equal(result, expected)

    mdf = DataFrame(raw)
    mdf.sort_index(inplace=True)
    result = mdf.execute().fetch()
    expected = raw.sort_index()
    pd.testing.assert_frame_equal(result, expected)

    mdf = DataFrame(raw, chunk_size=30)
    result = mdf.sort_index().execute().fetch()
    expected = raw.sort_index()
    pd.testing.assert_frame_equal(result, expected)

    mdf = DataFrame(raw, chunk_size=20)
    result = mdf.sort_index(ascending=False).execute().fetch()
    expected = raw.sort_index(ascending=False)
    pd.testing.assert_frame_equal(result, expected)

    mdf = DataFrame(raw, chunk_size=10)
    result = mdf.sort_index(ignore_index=True).execute().fetch()
    try:  # for python3.5
        expected = raw.sort_index(ignore_index=True)
    except TypeError:
        expected = raw.sort_index()
        expected.index = pd.RangeIndex(len(expected))
    pd.testing.assert_frame_equal(result, expected)

    # test axis=1
    raw = pd.DataFrame(np.random.rand(10, 10), columns=np.random.rand(10))

    mdf = DataFrame(raw)
    result = mdf.sort_index(axis=1).execute().fetch()
    expected = raw.sort_index(axis=1)
    pd.testing.assert_frame_equal(result, expected)

    mdf = DataFrame(raw, chunk_size=3)
    result = mdf.sort_index(axis=1).execute().fetch()
    expected = raw.sort_index(axis=1)
    pd.testing.assert_frame_equal(result, expected)

    mdf = DataFrame(raw, chunk_size=4)
    result = mdf.sort_index(axis=1, ascending=False).execute().fetch()
    expected = raw.sort_index(axis=1, ascending=False)
    pd.testing.assert_frame_equal(result, expected)

    mdf = DataFrame(raw, chunk_size=4)

    result = mdf.sort_index(axis=1, ignore_index=True).execute().fetch()
    try:  # for python3.5
        expected = raw.sort_index(axis=1, ignore_index=True)
    except TypeError:
        expected = raw.sort_index(axis=1)
        expected.index = pd.RangeIndex(len(expected))
    pd.testing.assert_frame_equal(result, expected)

    # test series
    raw = pd.Series(np.random.rand(10), index=np.random.rand(10))

    series = Series(raw)
    result = series.sort_index().execute().fetch()
    expected = raw.sort_index()
    pd.testing.assert_series_equal(result, expected)

    series = Series(raw, chunk_size=2)
    result = series.sort_index().execute().fetch()
    expected = raw.sort_index()
    pd.testing.assert_series_equal(result, expected)

    series = Series(raw, chunk_size=3)
    result = series.sort_index(ascending=False).execute().fetch()
    expected = raw.sort_index(ascending=False)
    pd.testing.assert_series_equal(result, expected)


def test_arrow_string_sort_values(setup):
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        {"a": rs.rand(10), "b": [f"s{rs.randint(1000)}" for _ in range(10)]}
    )
    raw["b"] = raw["b"].astype(ArrowStringDtype())
    mdf = DataFrame(raw, chunk_size=3)

    df = mdf.sort_values(by="b")
    result = df.execute().fetch()
    expected = raw.sort_values(by="b")
    pd.testing.assert_frame_equal(result, expected)


@require_cudf
def test_gpu_execution(setup_gpu):
    # test sort_values
    rs = np.random.RandomState(0)
    distinct_opts = ["0"] if sys.platform.lower().startswith("win") else ["0", "1"]
    for add_distinct in distinct_opts:
        os.environ["PSRS_DISTINCT_COL"] = add_distinct

        # test dataframe
        raw = pd.DataFrame(rs.rand(100, 10), columns=["a" + str(i) for i in range(10)])
        mdf = DataFrame(raw, chunk_size=30).to_gpu()

        result = mdf.sort_values(by="a0").execute().fetch()
        expected = raw.sort_values(by="a0")
        pd.testing.assert_frame_equal(result.to_pandas(), expected)

        # test series
        raw = pd.Series(rs.rand(10))
        series = Series(raw).to_gpu()

        result = series.sort_values().execute().fetch()
        expected = raw.sort_values()
        pd.testing.assert_series_equal(result.to_pandas(), expected)

    # test DataFrame.sort_index
    raw = pd.DataFrame(np.random.rand(10, 10), columns=np.random.rand(10))
    mdf = DataFrame(raw).to_gpu()

    result = mdf.sort_index().execute().fetch()
    expected = raw.sort_index()
    pd.testing.assert_frame_equal(result.to_pandas(), expected)

    # test Series.sort_index
    raw = pd.Series(
        np.random.rand(10),
        index=np.random.rand(10),
    )
    series = Series(raw).to_gpu()

    result = series.sort_index().execute().fetch()
    expected = raw.sort_index()
    pd.testing.assert_series_equal(result.to_pandas(), expected)
