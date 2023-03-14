# Copyright 2022 XProbe Inc.
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
import pytest


from .... import dataframe as md


@pytest.fixture
def gen_data1():
    rs = np.random.RandomState(0)
    data_size = 100
    data_dict = {
        "a": rs.randint(0, 10, size=(data_size,)),
        "b": rs.choice(list("abcd"), size=(data_size,)),
        "c": rs.choice(list("abcd"), size=(data_size,)),
    }
    df = pd.DataFrame(data_dict)
    yield df


@pytest.fixture
def gen_data2():
    rs = np.random.RandomState(0)
    data_size = 100
    data_dict = {
        "a": rs.randint(0, 10, size=(data_size,)),
        "b": rs.choice(list("abcd"), size=(data_size,)),
        "c": rs.choice(list("abcd"), size=(data_size,)),
        "d": rs.randint(0, 10, size=(data_size,)),
    }
    df = pd.DataFrame(data_dict)
    yield df


@pytest.fixture
def gen_data3():
    arrays = [
        ["Falcon", "Falcon", "Parrot", "Parrot"],
        ["Captive", "Wild", "Captive", "Wild"],
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=("Animal", "Type"))
    df = pd.DataFrame({"Max Speed": [390.0, 350.0, 30.0, 20.0]}, index=index)
    yield df


def test_groupby_nunique_without_index(setup, gen_data1):
    df = gen_data1
    mdf = md.DataFrame(df, chunk_size=13)
    r1 = mdf.groupby("b", sort=False)[["a"]].nunique(method="tree").execute().fetch()
    r2 = (
        mdf.groupby("b", sort=False)[["a"]]
        .nunique(method="shuffle")
        .execute()
        .fetch()
        .sort_index(level=0)
    )
    r3 = (
        mdf.groupby("b", sort=False)[["a"]]
        .nunique(method="auto")
        .execute()
        .fetch()
        .sort_index(level=0)
    )

    expected = df.groupby("b", sort=False)[["a"]].nunique()
    pd.testing.assert_frame_equal(r1, expected)
    pd.testing.assert_frame_equal(r2, expected.sort_index(level=0))
    pd.testing.assert_frame_equal(r3, expected.sort_index(level=0))


def test_groupby_nunique_with_index(setup, gen_data1):
    df = gen_data1
    mdf = md.DataFrame(df, chunk_size=13)

    r1 = (
        mdf.groupby("b", as_index=False, sort=False)["a"]
        .nunique(method="tree")
        .execute()
        .fetch()
    )
    # shuffle cannot ensure its order
    r2 = (
        mdf.groupby("b", as_index=False, sort=False)["a"]
        .nunique(method="auto")
        .execute()
        .fetch()
        .sort_values(by="b")
        .reset_index(drop=True)
    )
    r3 = (
        mdf.groupby("b", as_index=False, sort=False)["a"]
        .nunique(method="shuffle")
        .execute()
        .fetch()
        .sort_values(by="b")
        .reset_index(drop=True)
    )

    expected = df.groupby("b", as_index=False, sort=False)["a"].nunique()
    pd.testing.assert_frame_equal(r1, expected)
    pd.testing.assert_frame_equal(
        r2, expected.sort_values(by="b").reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        r3, expected.sort_values(by="b").reset_index(drop=True)
    )


def test_groupby_nunique_series(setup, gen_data1):
    df = gen_data1
    mdf = md.DataFrame(df, chunk_size=13)
    # When method = shuffle and output is series, mars has issue about that.
    # Therefore, skip the case.
    r1 = mdf.groupby("b", sort=False)["a"].nunique(method="tree").execute().fetch()
    r2 = (
        mdf.groupby("b", sort=False)["a"]
        .nunique(method="auto")
        .execute()
        .fetch()
        .sort_index(level=0)
    )

    expected = df.groupby("b", sort=False)["a"].nunique()
    pd.testing.assert_series_equal(r1, expected)
    pd.testing.assert_series_equal(r2, expected.sort_index(level=0))


def test_groupby_nunique_frame(setup, gen_data1):
    df = gen_data1
    mdf = md.DataFrame(df, chunk_size=13)

    r1 = mdf.groupby("b", sort=False)["a", "c"].nunique(method="tree").execute().fetch()
    r2 = (
        mdf.groupby("b", sort=False)["a", "c"]
        .nunique(method="auto")
        .execute()
        .fetch()
        .sort_values(by="b")
        .reset_index()
    )
    r3 = (
        mdf.groupby("b", sort=False)["a", "c"]
        .nunique(method="shuffle")
        .execute()
        .fetch()
        .sort_values(by="b")
        .reset_index()
    )

    expected = df.groupby("b", sort=False)["a", "c"].nunique()
    pd.testing.assert_frame_equal(r1, expected)
    pd.testing.assert_frame_equal(r2, expected.sort_values(by="b").reset_index())
    pd.testing.assert_frame_equal(r3, expected.sort_values(by="b").reset_index())


def test_groupby_nunique_with_sort(setup, gen_data1):
    df = gen_data1
    mdf = md.DataFrame(df, chunk_size=13)

    r = mdf.groupby("b", sort=True)["a", "c"].nunique().execute().fetch()

    expected = df.groupby("b", sort=True)["a", "c"].nunique()
    pd.testing.assert_frame_equal(r, expected)

    r = mdf.groupby(["b", "c"], sort=True)["a"].nunique().execute().fetch()
    expected = df.groupby(["b", "c"], sort=True)["a"].nunique()
    pd.testing.assert_series_equal(r, expected)


def test_groupby_nunique_multiindex(setup, gen_data2):
    df = gen_data2
    mdf = md.DataFrame(df, chunk_size=13)

    r1 = (
        mdf.groupby(["b", "c"], sort=False)["a", "d"]
        .nunique(method="tree")
        .execute()
        .fetch()
    )
    r2 = (
        mdf.groupby(["b", "c"], sort=False)["a", "d"]
        .nunique(method="shuffle")
        .execute()
        .fetch()
        .sort_values(by=["b", "c"])
        .reset_index()
    )
    r3 = (
        mdf.groupby(["b", "c"], sort=False)["a", "d"]
        .nunique(method="auto")
        .execute()
        .fetch()
        .sort_values(by=["b", "c"])
        .reset_index()
    )

    expected = df.groupby(["b", "c"], sort=False)["a", "d"].nunique()
    pd.testing.assert_frame_equal(r1, expected)
    pd.testing.assert_frame_equal(r2, expected.sort_values(by=["b", "c"]).reset_index())
    pd.testing.assert_frame_equal(r3, expected.sort_values(by=["b", "c"]).reset_index())


def test_groupby_nunique_level(setup, gen_data1, gen_data3):
    df = gen_data1
    mdf = md.DataFrame(df, chunk_size=13)

    r = (
        mdf.groupby(level=0, as_index=False, sort=False)["a"]
        .nunique()
        .execute()
        .fetch()
    )

    expected = df.groupby(level=0, as_index=False, sort=False)["a"].nunique()
    pd.testing.assert_frame_equal(r, expected)

    r = mdf.groupby(level=0, sort=False)["a"].nunique().execute().fetch()
    expected = df.groupby(level=0, sort=False)["a"].nunique()
    pd.testing.assert_series_equal(r, expected, check_index=False)

    r = mdf.groupby(level=0, sort=False)["a", "b"].nunique().execute().fetch()
    expected = df.groupby(level=0, sort=False)["a", "b"].nunique()
    pd.testing.assert_frame_equal(
        r.reset_index(drop=True), expected.reset_index(drop=True)
    )

    df2 = gen_data3
    mdf2 = md.DataFrame(df2, chunk_size=2)
    r = mdf2.groupby(level="Type", sort=False).nunique().execute().fetch()
    expected = df2.groupby(level="Type", sort=False).nunique()
    pd.testing.assert_frame_equal(r, expected)

    r = mdf2.groupby(level=["Animal", "Type"], sort=False).nunique().execute().fetch()
    expected = df2.groupby(level=["Animal", "Type"], sort=False).nunique()
    pd.testing.assert_frame_equal(r, expected)

    r = mdf2.groupby(level=(0, 1), sort=False).nunique().execute().fetch()
    expected = df2.groupby(level=(0, 1), sort=False).nunique()
    pd.testing.assert_frame_equal(r, expected)

    r = mdf2.groupby(level=["Type", "Animal"]).nunique().execute().fetch()
    expected = df2.groupby(level=["Type", "Animal"]).nunique()
    pd.testing.assert_frame_equal(r, expected)

    r = (
        mdf2.groupby(level=(0, 1), sort=False)
        .nunique(method="shuffle")
        .execute()
        .fetch()
    )
    expected = df2.groupby(level=(0, 1), sort=False).nunique()
    pd.testing.assert_frame_equal(r.sort_index(), expected.sort_index())

    r = mdf2.groupby(level=["Type", "Animal"]).nunique(method="tree").execute().fetch()
    expected = df2.groupby(level=["Type", "Animal"]).nunique()
    pd.testing.assert_frame_equal(r, expected)

    r = mdf2.groupby(level=["Type", "Animal"]).nunique(method="auto").execute().fetch()
    expected = df2.groupby(level=["Type", "Animal"]).nunique()
    pd.testing.assert_frame_equal(r, expected)

    r = (
        mdf2.groupby(level=["Type", "Animal"], sort=False)
        .nunique(method="shuffle")
        .execute()
        .fetch()
    )
    expected = df2.groupby(level=["Type", "Animal"]).nunique()
    pd.testing.assert_frame_equal(r.sort_index(), expected.sort_index())


def test_groupby_agg_nunique(setup, gen_data1):
    df = gen_data1
    mdf = md.DataFrame(df, chunk_size=13)

    r = mdf.groupby(["b", "c"]).agg("nunique").execute().fetch()
    expected = df.groupby(["b", "c"]).agg("nunique")
    pd.testing.assert_frame_equal(r, expected)

    r = mdf.groupby(["b", "c"]).agg(["nunique"], method="tree").execute().fetch()
    expected = df.groupby(["b", "c"]).agg(["nunique"])
    pd.testing.assert_frame_equal(r, expected)

    r = mdf.groupby(["b", "c"]).agg(["nunique"], method="auto").execute().fetch()
    expected = df.groupby(["b", "c"]).agg(["nunique"])
    pd.testing.assert_frame_equal(r, expected)

    r = mdf.groupby(["b", "c"]).agg(["nunique"], method="shuffle").execute().fetch()
    expected = df.groupby(["b", "c"]).agg(["nunique"])
    pd.testing.assert_frame_equal(r, expected)

    r = mdf.groupby(["b", "c"], as_index=False).agg("nunique").execute().fetch()
    expected = df.groupby(["b", "c"], as_index=False).agg("nunique")
    pd.testing.assert_frame_equal(r, expected)

    r = (
        mdf.groupby(["b", "c"], as_index=False, sort=False)
        .agg("nunique")
        .execute()
        .fetch()
    )
    expected = df.groupby(["b", "c"], as_index=False, sort=False).agg("nunique")
    pd.testing.assert_frame_equal(r, expected)

    is_sort = [True, False]
    methods = ["auto", "shuffle", "tree"]
    for sort in is_sort:
        for method in methods:
            r = (
                mdf.groupby("b", sort=sort)
                .agg(["sum", "nunique"], method=method)
                .execute()
                .fetch()
            )
            expected = df.groupby("b", sort=sort).agg(["sum", "nunique"])
            pd.testing.assert_frame_equal(r.sort_index(), expected.sort_index())
