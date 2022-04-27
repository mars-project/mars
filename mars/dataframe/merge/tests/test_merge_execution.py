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

import numpy as np
import pandas as pd
import pytest

from ....core.graph.builder.utils import build_graph
from ...datasource.dataframe import from_pandas
from ...datasource.series import from_pandas as series_from_pandas
from ...utils import sort_dataframe_inplace
from .. import concat, DataFrameConcat, DataFrameMergeAlign


def test_merge(setup):
    df1 = pd.DataFrame(
        np.arange(20).reshape((4, 5)) + 1, columns=["a", "b", "c", "d", "e"]
    )
    df2 = pd.DataFrame(np.arange(20).reshape((5, 4)) + 1, columns=["a", "b", "x", "y"])
    df3 = df1.copy()
    df3.index = pd.RangeIndex(2, 6, name="index")
    df4 = df1.copy()
    df4.index = pd.MultiIndex.from_tuples(
        [(i, i + 1) for i in range(4)], names=["i1", "i2"]
    )

    mdf1 = from_pandas(df1, chunk_size=2)
    mdf2 = from_pandas(df2, chunk_size=2)
    mdf3 = from_pandas(df3, chunk_size=3)
    mdf4 = from_pandas(df4, chunk_size=2)

    # Note [Index of Merge]
    #
    # When `left_index` and `right_index` of `merge` is both false, pandas will generate an RangeIndex to
    # the final result dataframe.
    #
    # We chunked the `left` and `right` dataframe, thus every result chunk will have its own RangeIndex.
    # When they are contenated we don't generate a new RangeIndex for the result, thus we cannot obtain the
    # same index value with pandas. But we guarantee that the content of dataframe is correct.

    # merge on index
    expected0 = df1.merge(df2)
    jdf0 = mdf1.merge(mdf2, auto_merge="none")
    result0 = jdf0.execute().fetch()
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected0, 0), sort_dataframe_inplace(result0, 0)
    )

    # merge on left index and `right_on`
    expected1 = df1.merge(df2, how="left", right_on="x", left_index=True)
    jdf1 = mdf1.merge(
        mdf2, how="left", right_on="x", left_index=True, auto_merge="none"
    )
    result1 = jdf1.execute().fetch()
    expected1.set_index("a_x", inplace=True)
    result1.set_index("a_x", inplace=True)
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected1, 0), sort_dataframe_inplace(result1, 0)
    )

    # merge on `left_on` and right index
    expected2 = df1.merge(df2, how="right", left_on="a", right_index=True)
    jdf2 = mdf1.merge(
        mdf2, how="right", left_on="a", right_index=True, auto_merge="none"
    )
    result2 = jdf2.execute().fetch()
    expected2.set_index("a", inplace=True)
    result2.set_index("a", inplace=True)
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected2, 0), sort_dataframe_inplace(result2, 0)
    )

    # merge on `left_on` and `right_on`
    expected3 = df1.merge(df2, how="left", left_on="a", right_on="x")
    jdf3 = mdf1.merge(mdf2, how="left", left_on="a", right_on="x", auto_merge="none")
    result3 = jdf3.execute().fetch()
    expected3.set_index("a_x", inplace=True)
    result3.set_index("a_x", inplace=True)
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected3, 0), sort_dataframe_inplace(result3, 0)
    )

    # merge on `on`
    expected4 = df1.merge(df2, how="right", on="a")
    jdf4 = mdf1.merge(mdf2, how="right", on="a", auto_merge="none")
    result4 = jdf4.execute().fetch()
    expected4.set_index("a", inplace=True)
    result4.set_index("a", inplace=True)
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected4, 0), sort_dataframe_inplace(result4, 0)
    )

    # merge on multiple columns
    expected5 = df1.merge(df2, how="inner", on=["a", "b"])
    jdf5 = mdf1.merge(mdf2, how="inner", on=["a", "b"], auto_merge="none")
    result5 = jdf5.execute().fetch()
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected5, 0), sort_dataframe_inplace(result5, 0)
    )

    # merge when some on is index
    expected6 = df3.merge(df2, how="inner", left_on="index", right_on="a")
    jdf6 = mdf3.merge(
        mdf2, how="inner", left_on="index", right_on="a", auto_merge="none"
    )
    result6 = jdf6.execute().fetch()
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected6, 0), sort_dataframe_inplace(result6, 0)
    )

    # merge when on is in MultiIndex
    expected7 = df4.merge(df2, how="inner", left_on="i1", right_on="a")
    jdf7 = mdf4.merge(mdf2, how="inner", left_on="i1", right_on="a", auto_merge="none")
    result7 = jdf7.execute().fetch()
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected7, 0), sort_dataframe_inplace(result7, 0)
    )

    mdf5 = from_pandas(df2, chunk_size=4)
    mdf6 = from_pandas(df4, chunk_size=1)
    expected7 = df4.merge(df2, how="inner", left_on="i1", right_on="a")
    jdf7 = mdf6.merge(mdf5, how="inner", left_on="i1", right_on="a", auto_merge="none")
    result7 = jdf7.execute().fetch()
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected7, 0), sort_dataframe_inplace(result7, 0)
    )

    # merge when on is in MultiIndex, and on not in index
    expected8 = df4.merge(df2, how="inner", on=["a", "b"])
    jdf8 = mdf4.merge(mdf2, how="inner", on=["a", "b"], auto_merge="none")
    result8 = jdf8.execute().fetch()
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected8, 0), sort_dataframe_inplace(result8, 0)
    )


def test_join(setup):
    df1 = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]], index=["a1", "a2", "a3"])
    df2 = pd.DataFrame([[1, 2, 3], [1, 5, 6], [7, 8, 9]], index=["a1", "b2", "b3"]) + 1
    df2 = pd.concat([df2, df2 + 1])

    mdf1 = from_pandas(df1, chunk_size=2)
    mdf2 = from_pandas(df2, chunk_size=2)

    # default `how`
    expected0 = df1.join(df2, lsuffix="l_", rsuffix="r_")
    jdf0 = mdf1.join(mdf2, lsuffix="l_", rsuffix="r_", auto_merge="none")
    result0 = jdf0.execute().fetch()
    pd.testing.assert_frame_equal(expected0.sort_index(), result0.sort_index())

    # how = 'left'
    expected1 = df1.join(df2, how="left", lsuffix="l_", rsuffix="r_")
    jdf1 = mdf1.join(mdf2, how="left", lsuffix="l_", rsuffix="r_", auto_merge="none")
    result1 = jdf1.execute().fetch()
    pd.testing.assert_frame_equal(expected1.sort_index(), result1.sort_index())

    # how = 'right'
    expected2 = df1.join(df2, how="right", lsuffix="l_", rsuffix="r_")
    jdf2 = mdf1.join(mdf2, how="right", lsuffix="l_", rsuffix="r_", auto_merge="none")
    result2 = jdf2.execute().fetch()
    pd.testing.assert_frame_equal(expected2.sort_index(), result2.sort_index())

    # how = 'inner'
    expected3 = df1.join(df2, how="inner", lsuffix="l_", rsuffix="r_")
    jdf3 = mdf1.join(mdf2, how="inner", lsuffix="l_", rsuffix="r_", auto_merge="none")
    result3 = jdf3.execute().fetch()
    pd.testing.assert_frame_equal(expected3.sort_index(), result3.sort_index())

    # how = 'outer'
    expected4 = df1.join(df2, how="outer", lsuffix="l_", rsuffix="r_")
    jdf4 = mdf1.join(mdf2, how="outer", lsuffix="l_", rsuffix="r_", auto_merge="none")
    result4 = jdf4.execute().fetch()
    pd.testing.assert_frame_equal(expected4.sort_index(), result4.sort_index())


def test_join_on(setup):
    df1 = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]], columns=["a1", "a2", "a3"])
    df2 = (
        pd.DataFrame([[1, 2, 3], [1, 5, 6], [7, 8, 9]], columns=["a1", "b2", "b3"]) + 1
    )
    df2 = pd.concat([df2, df2 + 1])

    mdf1 = from_pandas(df1, chunk_size=2)
    mdf2 = from_pandas(df2, chunk_size=2)

    expected0 = df1.join(df2, on=None, lsuffix="_l", rsuffix="_r")
    jdf0 = mdf1.join(mdf2, on=None, lsuffix="_l", rsuffix="_r", auto_merge="none")
    result0 = jdf0.execute().fetch()
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected0, 0), sort_dataframe_inplace(result0, 0)
    )

    expected1 = df1.join(df2, how="left", on="a1", lsuffix="_l", rsuffix="_r")
    jdf1 = mdf1.join(
        mdf2, how="left", on="a1", lsuffix="_l", rsuffix="_r", auto_merge="none"
    )
    result1 = jdf1.execute().fetch()

    # Note [Columns of Left Join]
    #
    # I believe we have no chance to obtain the entirely same result with pandas here:
    #
    # Look at the following example:
    #
    # >>> df1
    #     a1  a2  a3
    # 0   1   3   3
    # >>> df2
    #     a1  b2  b3
    # 1   2   6   7
    # >>> df3
    #     a1  b2  b3
    # 1   2   6   7
    # 1   2   6   7
    #
    # >>> df1.merge(df2, how='left', left_on='a1', left_index=False, right_index=True)
    #     a1_x  a2  a3  a1_y  b2  b3
    # 0   1   3   3     2   6   7
    # >>> df1.merge(df3, how='left', left_on='a1', left_index=False, right_index=True)
    #     a1  a1_x  a2  a3  a1_y  b2  b3
    # 0   1     1   3   3     2   6   7
    # 0   1     1   3   3     2   6   7
    #
    # Note that the result of `df1.merge(df3)` has an extra column `a` compared to `df1.merge(df2)`.
    # The value of column `a` is the same of `a1_x`, just because `1` occurs twice in index of `df3`.
    # I haven't invistagated why pandas has such behaviour...
    #
    # We cannot yield the same result with pandas, because, the `df3` is chunked, then some of the
    # result chunk has 6 columns, others may have 7 columns, when concatenated into one DataFrame
    # some cells of column `a` will have value `NaN`, which is different from the result of pandas.
    #
    # But we can guarantee that other effective columns have absolutely same value with pandas.

    columns_to_compare = jdf1.columns_value.to_pandas()

    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected1[columns_to_compare], 0, 1),
        sort_dataframe_inplace(result1[columns_to_compare], 0, 1),
    )

    # Note [Index of Join on EmptyDataFrame]
    #
    # It is tricky that it is non-trivial to get the same `index` result with pandas.
    #
    # Look at the following example:
    #
    # >>> df1
    #    a1  a2  a3
    # 1   4   2   6
    # >>> df2
    #    a1  b2  b3
    # 1   2   6   7
    # 2   8   9  10
    # >>> df3
    # Empty DataFrame
    # Columns: [a1, a2, a3]
    # Index: []
    # >>> df1.join(df2, how='right', on='a2', lsuffix='_l', rsuffix='_r')
    #       a1_l  a2   a3  a1_r  b2  b3
    # 1.0   4.0   2  6.0     8   9  10
    # NaN   NaN   1  NaN     2   6   7
    # >>> df3.join(df2, how='right', on='a2', lsuffix='_l', rsuffix='_r')
    #     a1_l  a2  a3  a1_r  b2  b3
    # 1   NaN   1 NaN     2   6   7
    # 2   NaN   2 NaN     8   9  10
    #
    # When the `left` dataframe is not empty, the mismatched rows in `right` will have index value `NaN`,
    # and the matched rows have index value from `right`. When the `left` dataframe is empty, the mismatched
    # rows have index value from `right`.
    #
    # Since we chunked the `left` dataframe, it is uneasy to obtain the same index value with pandas in the
    # final result dataframe, but we guaranteed that the dataframe content is correctly.

    expected2 = df1.join(df2, how="right", on="a2", lsuffix="_l", rsuffix="_r")
    jdf2 = mdf1.join(
        mdf2, how="right", on="a2", lsuffix="_l", rsuffix="_r", auto_merge="none"
    )
    result2 = jdf2.execute().fetch()

    expected2.set_index("a2", inplace=True)
    result2.set_index("a2", inplace=True)
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected2, 0), sort_dataframe_inplace(result2, 0)
    )

    expected3 = df1.join(df2, how="inner", on="a2", lsuffix="_l", rsuffix="_r")
    jdf3 = mdf1.join(
        mdf2, how="inner", on="a2", lsuffix="_l", rsuffix="_r", auto_merge="none"
    )
    result3 = jdf3.execute().fetch()
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected3, 0), sort_dataframe_inplace(result3, 0)
    )

    expected4 = df1.join(df2, how="outer", on="a2", lsuffix="_l", rsuffix="_r")
    jdf4 = mdf1.join(
        mdf2, how="outer", on="a2", lsuffix="_l", rsuffix="_r", auto_merge="none"
    )
    result4 = jdf4.execute().fetch()

    expected4.set_index("a2", inplace=True)
    result4.set_index("a2", inplace=True)
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected4, 0), sort_dataframe_inplace(result4, 0)
    )


def test_merge_one_chunk(setup):
    df1 = pd.DataFrame(
        {"lkey": ["foo", "bar", "baz", "foo"], "value": [1, 2, 3, 5]},
        index=["a1", "a2", "a3", "a4"],
    )
    df2 = pd.DataFrame(
        {"rkey": ["foo", "bar", "baz", "foo"], "value": [5, 6, 7, 8]},
        index=["a1", "a2", "a3", "a4"],
    )

    # all have one chunk
    mdf1 = from_pandas(df1)
    mdf2 = from_pandas(df2)

    expected = df1.merge(df2, left_on="lkey", right_on="rkey")
    jdf = mdf1.merge(mdf2, left_on="lkey", right_on="rkey", auto_merge="none")
    result = jdf.execute().fetch()

    pd.testing.assert_frame_equal(
        expected.sort_values(by=expected.columns[1]).reset_index(drop=True),
        result.sort_values(by=result.columns[1]).reset_index(drop=True),
    )

    # left have one chunk
    mdf1 = from_pandas(df1)
    mdf2 = from_pandas(df2, chunk_size=2)

    expected = df1.merge(df2, left_on="lkey", right_on="rkey")
    jdf = mdf1.merge(mdf2, left_on="lkey", right_on="rkey", auto_merge="none")
    result = jdf.execute().fetch()

    pd.testing.assert_frame_equal(
        expected.sort_values(by=expected.columns[1]).reset_index(drop=True),
        result.sort_values(by=result.columns[1]).reset_index(drop=True),
    )

    # right have one chunk
    mdf1 = from_pandas(df1, chunk_size=3)
    mdf2 = from_pandas(df2)

    expected = df1.merge(df2, left_on="lkey", right_on="rkey")
    jdf = mdf1.merge(mdf2, left_on="lkey", right_on="rkey", auto_merge="none")
    result = jdf.execute().fetch()

    pd.testing.assert_frame_equal(
        expected.sort_values(by=expected.columns[1]).reset_index(drop=True),
        result.sort_values(by=result.columns[1]).reset_index(drop=True),
    )

    # left have one chunk and how="left", then one chunk tile
    # will result in wrong results, see #GH 2107
    mdf1 = from_pandas(df1, chunk_size=2)
    mdf2 = from_pandas(df2)

    expected = df2.merge(df1, left_on="rkey", right_on="lkey", how="left")
    jdf = mdf2.merge(
        mdf1, left_on="rkey", right_on="lkey", how="left", auto_merge="none"
    )
    result = jdf.execute().fetch()

    pd.testing.assert_frame_equal(
        expected.sort_values(by=expected.columns[1]).reset_index(drop=True),
        result.sort_values(by=result.columns[1]).reset_index(drop=True),
    )


def test_broadcast_merge(setup):
    ns = np.random.RandomState(0)
    # small dataframe
    raw1 = pd.DataFrame(
        {
            "key": ns.randint(0, 10, size=10),
            "value": np.arange(10),
        },
        index=[f"a{i}" for i in range(10)],
    )
    # big dataframe
    raw2 = pd.DataFrame(
        {
            "key": ns.randint(0, 100, size=100),
            "value": np.arange(100, 200),
        },
        index=[f"a{i}" for i in range(100)],
    )

    # test broadcast right and how="inner"
    df1 = from_pandas(raw1, chunk_size=5)
    df2 = from_pandas(raw2, chunk_size=10)
    r = df2.merge(df1, on="key", auto_merge="none", bloom_filter=False)
    # make sure it selects broadcast merge, for broadcast, there must be
    # DataFrameConcat operands
    graph = build_graph([r], tile=True)
    assert any(isinstance(c.op, DataFrameConcat) for c in graph)
    # inner join doesn't need shuffle
    assert all(not isinstance(c.op, DataFrameMergeAlign) for c in graph)

    result = r.execute().fetch()
    expected = raw2.merge(raw1, on="key")

    expected.set_index("key", inplace=True)
    result.set_index("key", inplace=True)
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected, 0), sort_dataframe_inplace(result, 0)
    )

    # test broadcast right and how="left"
    df1 = from_pandas(raw1, chunk_size=5)
    df2 = from_pandas(raw2, chunk_size=10)
    r = df2.merge(df1, on="key", how="left", auto_merge="none", method="broadcast")
    # make sure it selects broadcast merge, for broadcast, there must be
    # DataFrameConcat operands
    graph = build_graph([r], tile=True)
    assert any(isinstance(c.op, DataFrameConcat) for c in graph)
    # left join need shuffle
    assert any(isinstance(c.op, DataFrameMergeAlign) for c in graph)

    result = r.execute().fetch()
    expected = raw2.merge(raw1, on="key", how="left")

    expected.set_index("key", inplace=True)
    result.set_index("key", inplace=True)
    pd.testing.assert_frame_equal(
        expected.sort_values(by=["key", "value_x"]),
        result.sort_values(by=["key", "value_x"]),
    )

    # test broadcast left
    df1 = from_pandas(raw1, chunk_size=5)
    df2 = from_pandas(raw2, chunk_size=10)
    r = df1.merge(df2, on="key", auto_merge="none", bloom_filter=False)
    # make sure it selects broadcast merge, for broadcast, there must be
    # DataFrameConcat operands
    graph = build_graph([r], tile=True)
    assert any(isinstance(c.op, DataFrameConcat) for c in graph)
    # inner join doesn't need shuffle
    assert all(not isinstance(c.op, DataFrameMergeAlign) for c in graph)

    result = r.execute().fetch()
    expected = raw1.merge(raw2, on="key")

    expected.set_index("key", inplace=True)
    result.set_index("key", inplace=True)
    pd.testing.assert_frame_equal(
        sort_dataframe_inplace(expected, 0), sort_dataframe_inplace(result, 0)
    )

    # test broadcast left and how="right"
    df1 = from_pandas(raw1, chunk_size=5)
    df2 = from_pandas(raw2, chunk_size=10)
    r = df1.merge(df2, on="key", how="right", auto_merge="none")
    # make sure it selects broadcast merge, for broadcast, there must be
    # DataFrameConcat operands
    graph = build_graph([r], tile=True)
    assert any(isinstance(c.op, DataFrameConcat) for c in graph)
    # right join need shuffle
    assert any(isinstance(c.op, DataFrameMergeAlign) for c in graph)

    result = r.execute().fetch()
    expected = raw1.merge(raw2, on="key", how="right")

    expected.set_index("key", inplace=True)
    result.set_index("key", inplace=True)
    pd.testing.assert_frame_equal(
        expected.sort_values(by=["key", "value_x"]),
        result.sort_values(by=["key", "value_x"]),
    )


def test_merge_with_bloom_filter(setup):
    ns = np.random.RandomState(0)
    raw_df1 = pd.DataFrame(
        {
            "col1": ns.random(100),
            "col2": ns.randint(0, 10, size=(100,)),
            "col3": ns.randint(0, 10, size=(100,)),
        }
    )
    raw_df2 = pd.DataFrame(
        {
            "col1": ns.random(100),
            "col2": ns.randint(0, 10, size=(100,)),
            "col3": ns.randint(0, 10, size=(100,)),
        }
    )

    df1 = from_pandas(raw_df1, chunk_size=10)
    df2 = from_pandas(raw_df2, chunk_size=15)

    expected = raw_df1.merge(raw_df2, on="col2")

    result = (
        df1.merge(
            df2,
            on="col2",
            bloom_filter={"max_elements": 100, "error_rate": 0.01},
            auto_merge="none",
        )
        .execute()
        .fetch()
    )
    pd.testing.assert_frame_equal(
        expected.sort_values(by=["col1_x", "col2"]).reset_index(drop=True),
        result.sort_values(by=["col1_x", "col2"]).reset_index(drop=True),
    )

    result = (
        df2.merge(df1, on=["col2", "col3"], bloom_filter=True, auto_merge="none")
        .execute()
        .fetch()
    )
    expected = raw_df2.merge(raw_df1, on=["col2", "col3"])
    pd.testing.assert_frame_equal(
        expected.sort_values(by=["col1_x", "col2"]).reset_index(drop=True),
        result.sort_values(by=["col1_x", "col2"]).reset_index(drop=True),
    )

    # on index
    result = df2.merge(df1, bloom_filter=True, auto_merge="none").execute().fetch()
    expected = raw_df2.merge(raw_df1)
    pd.testing.assert_frame_equal(
        expected.sort_index().reset_index(drop=True),
        result.sort_index().reset_index(drop=True),
    )

    # on float column
    result = (
        df2.merge(df1, on="col1", bloom_filter=True, auto_merge="none")
        .execute()
        .fetch()
    )
    expected = raw_df2.merge(raw_df1, on="col1")
    pd.testing.assert_frame_equal(
        expected.sort_values(by=["col1", "col2_x"]).reset_index(drop=True),
        result.sort_values(by=["col1", "col2_x"]).reset_index(drop=True),
    )

    # on float columns
    result = (
        df2.merge(df1, on=["col1", "col2"], bloom_filter=True, auto_merge="none")
        .execute()
        .fetch()
    )
    expected = raw_df2.merge(raw_df1, on=["col1", "col2"])
    pd.testing.assert_frame_equal(
        expected.sort_values(by=["col1", "col2"]).reset_index(drop=True),
        result.sort_values(by=["col1", "col2"]).reset_index(drop=True),
    )

    # multi index
    raw_df3 = raw_df1.copy()
    raw_df3.index = pd.MultiIndex.from_tuples(
        [(i, i + 1) for i in range(100)], names=["i1", "i2"]
    )
    df3 = from_pandas(raw_df3, chunk_size=8)
    result = (
        df3.merge(
            df1, left_on="i1", right_on="col2", bloom_filter=True, auto_merge="none"
        )
        .execute()
        .fetch()
    )
    expected = raw_df3.merge(raw_df1, left_on="i1", right_on="col2")
    pd.testing.assert_frame_equal(
        expected.sort_index().sort_values(by=["col1_x"]).reset_index(drop=True),
        result.sort_index().sort_values(by=["col1_x"]).reset_index(drop=True),
    )

    df4 = from_pandas(raw_df3, chunk_size=20)
    result = (
        df4.merge(
            df1, left_on="i1", right_on="col2", bloom_filter=True, auto_merge="none"
        )
        .execute()
        .fetch()
    )
    expected = raw_df3.merge(raw_df1, left_on="i1", right_on="col2")
    pd.testing.assert_frame_equal(
        expected.sort_index().sort_values(by=["col1_x"]).reset_index(drop=True),
        result.sort_index().sort_values(by=["col1_x"]).reset_index(drop=True),
    )


@pytest.mark.parametrize("auto_merge", ["none", "both", "before", "after"])
def test_merge_on_duplicate_columns(setup, auto_merge):
    raw1 = pd.DataFrame(
        [["foo", 1, "bar"], ["bar", 2, "foo"], ["baz", 3, "foo"]],
        columns=["lkey", "value", "value"],
        index=["a1", "a2", "a3"],
    )
    raw2 = pd.DataFrame(
        {"rkey": ["foo", "bar", "baz", "foo"], "value": [5, 6, 7, 8]},
        index=["a1", "a2", "a3", "a4"],
    )

    df1 = from_pandas(raw1, chunk_size=2)
    df2 = from_pandas(raw2, chunk_size=3)

    r = df1.merge(df2, left_on="lkey", right_on="rkey", auto_merge=auto_merge)
    result = r.execute().fetch()
    expected = raw1.merge(raw2, left_on="lkey", right_on="rkey")
    pd.testing.assert_frame_equal(expected, result)


def test_append_execution(setup):
    df1 = pd.DataFrame(np.random.rand(10, 4), columns=list("ABCD"))
    df2 = pd.DataFrame(np.random.rand(10, 4), columns=list("ABCD"))

    mdf1 = from_pandas(df1, chunk_size=3)
    mdf2 = from_pandas(df2, chunk_size=3)

    adf = mdf1.append(mdf2)
    expected = df1.append(df2)
    result = adf.execute().fetch()
    pd.testing.assert_frame_equal(expected, result)

    adf = mdf1.append(mdf2, ignore_index=True)
    expected = df1.append(df2, ignore_index=True)
    result = adf.execute(extra_config={"check_index_value": False}).fetch()
    pd.testing.assert_frame_equal(expected, result)

    mdf1 = from_pandas(df1, chunk_size=3)
    mdf2 = from_pandas(df2, chunk_size=2)

    adf = mdf1.append(mdf2)
    expected = df1.append(df2)
    result = adf.execute().fetch()
    pd.testing.assert_frame_equal(expected, result)

    adf = mdf1.append(mdf2, ignore_index=True)
    expected = df1.append(df2, ignore_index=True)
    result = adf.execute(extra_config={"check_index_value": False}).fetch()
    pd.testing.assert_frame_equal(expected, result)

    df3 = pd.DataFrame(np.random.rand(8, 4), columns=list("ABCD"))
    mdf3 = from_pandas(df3, chunk_size=3)
    expected = df1.append([df2, df3])
    adf = mdf1.append([mdf2, mdf3])
    result = adf.execute().fetch()
    pd.testing.assert_frame_equal(expected, result)

    adf = mdf1.append(dict(A=1, B=2, C=3, D=4), ignore_index=True)
    expected = df1.append(dict(A=1, B=2, C=3, D=4), ignore_index=True)
    result = adf.execute(extra_config={"check_index_value": False}).fetch()
    pd.testing.assert_frame_equal(expected, result)

    # test for series
    series1 = pd.Series(np.random.rand(10))
    series2 = pd.Series(np.random.rand(10))

    mseries1 = series_from_pandas(series1, chunk_size=3)
    mseries2 = series_from_pandas(series2, chunk_size=3)

    aseries = mseries1.append(mseries2)
    expected = series1.append(series2)
    result = aseries.execute().fetch()
    pd.testing.assert_series_equal(expected, result)

    aseries = mseries1.append(mseries2, ignore_index=True)
    expected = series1.append(series2, ignore_index=True)
    result = aseries.execute(extra_config={"check_index_value": False}).fetch()
    pd.testing.assert_series_equal(expected, result)

    mseries1 = series_from_pandas(series1, chunk_size=3)
    mseries2 = series_from_pandas(series2, chunk_size=2)

    aseries = mseries1.append(mseries2)
    expected = series1.append(series2)
    result = aseries.execute().fetch()
    pd.testing.assert_series_equal(expected, result)

    aseries = mseries1.append(mseries2, ignore_index=True)
    expected = series1.append(series2, ignore_index=True)
    result = aseries.execute(extra_config={"check_index_value": False}).fetch()
    pd.testing.assert_series_equal(expected, result)

    series3 = pd.Series(np.random.rand(4))
    mseries3 = series_from_pandas(series3, chunk_size=2)
    expected = series1.append([series2, series3])
    aseries = mseries1.append([mseries2, mseries3])
    result = aseries.execute().fetch()
    pd.testing.assert_series_equal(expected, result)


def test_concat(setup):
    df1 = pd.DataFrame(np.random.rand(10, 4), columns=list("ABCD"))
    df2 = pd.DataFrame(np.random.rand(10, 4), columns=list("ABCD"))

    mdf1 = from_pandas(df1, chunk_size=3)
    mdf2 = from_pandas(df2, chunk_size=3)

    r = concat([mdf1, mdf2])
    expected = pd.concat([df1, df2])
    result = r.execute().fetch()
    pd.testing.assert_frame_equal(expected, result)

    # test different chunk size and ignore_index=True
    mdf1 = from_pandas(df1, chunk_size=2)
    mdf2 = from_pandas(df2, chunk_size=3)

    r = concat([mdf1, mdf2], ignore_index=True)
    expected = pd.concat([df1, df2], ignore_index=True)
    result = r.execute(extra_config={"check_index_value": False}).fetch()
    pd.testing.assert_frame_equal(expected, result)

    # test axis=1
    mdf1 = from_pandas(df1, chunk_size=2)
    mdf2 = from_pandas(df2, chunk_size=3)

    r = concat([mdf1, mdf2], axis=1)
    expected = pd.concat([df1, df2], axis=1)
    result = r.execute().fetch()
    pd.testing.assert_frame_equal(expected, result)

    # test multiply dataframes
    r = concat([mdf1, mdf2, mdf1])
    expected = pd.concat([df1, df2, df1])
    result = r.execute().fetch()
    pd.testing.assert_frame_equal(expected, result)

    df1 = pd.DataFrame(np.random.rand(10, 4), columns=list("ABCD"))
    df2 = pd.DataFrame(np.random.rand(10, 3), columns=list("ABC"))

    mdf1 = from_pandas(df1, chunk_size=3)
    mdf2 = from_pandas(df2, chunk_size=3)

    # test join=inner
    r = concat([mdf1, mdf2], join="inner")
    expected = pd.concat([df1, df2], join="inner")
    result = r.execute().fetch()
    pd.testing.assert_frame_equal(expected, result)

    # test for series
    series1 = pd.Series(np.random.rand(10))
    series2 = pd.Series(np.random.rand(10))

    mseries1 = series_from_pandas(series1, chunk_size=3)
    mseries2 = series_from_pandas(series2, chunk_size=3)

    r = concat([mseries1, mseries2])
    expected = pd.concat([series1, series2])
    result = r.execute().fetch()
    pd.testing.assert_series_equal(result, expected)

    # test different series and ignore_index
    mseries1 = series_from_pandas(series1, chunk_size=4)
    mseries2 = series_from_pandas(series2, chunk_size=3)

    r = concat([mseries1, mseries2], ignore_index=True)
    expected = pd.concat([series1, series2], ignore_index=True)
    result = r.execute(extra_config={"check_index_value": False}).fetch()
    pd.testing.assert_series_equal(result, expected)

    # test axis=1
    mseries1 = series_from_pandas(series1, chunk_size=3)
    mseries2 = series_from_pandas(series2, chunk_size=3)

    r = concat([mseries1, mseries2], axis=1)
    expected = pd.concat([series1, series2], axis=1)
    result = r.execute(extra_config={"check_shape": False}).fetch()
    pd.testing.assert_frame_equal(result, expected)

    # test merge dataframe and series
    r = concat([mdf1, mseries2], ignore_index=True)
    expected = pd.concat([df1, series2], ignore_index=True)
    result = r.execute(extra_config={"check_index_value": False}).fetch()
    pd.testing.assert_frame_equal(result, expected)

    # test merge series and dataframe
    r = concat([mseries1, mdf2], ignore_index=True)
    expected = pd.concat([series1, df2], ignore_index=True)
    result = r.execute(extra_config={"check_index_value": False}).fetch()
    pd.testing.assert_frame_equal(result, expected)

    # test merge dataframe and series, axis=1
    r = concat([mdf1, mseries2], axis=1)
    expected = pd.concat([df1, series2], axis=1)
    result = r.execute().fetch()
    pd.testing.assert_frame_equal(result, expected)

    # test merge series and dataframe, axis=1
    r = concat([mseries1, mdf2], axis=1)
    expected = pd.concat([series1, df2], axis=1)
    result = r.execute().fetch()
    pd.testing.assert_frame_equal(result, expected)
