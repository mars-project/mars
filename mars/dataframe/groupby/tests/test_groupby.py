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

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from .... import dataframe as md
from .... import opcodes
from ....core import OutputType, tile
from ....core.operand import OperandStage
from ...core import DataFrameGroupBy, SeriesGroupBy, DataFrame
from ..core import DataFrameGroupByOperand, DataFrameShuffleProxy
from ..aggregation import DataFrameGroupByAgg
from ..getitem import GroupByIndex


def test_groupby():
    df = pd.DataFrame(
        {"a": [3, 4, 5, 3, 5, 4, 1, 2, 3], "b": [1, 3, 4, 5, 6, 5, 4, 4, 4]}
    )
    mdf = md.DataFrame(df, chunk_size=2)
    with pytest.raises(KeyError):
        mdf.groupby("c2")
    with pytest.raises(KeyError):
        mdf.groupby(["b", "c2"])

    grouped = mdf.groupby("b")
    assert isinstance(grouped, DataFrameGroupBy)
    assert isinstance(grouped.op, DataFrameGroupByOperand)
    assert list(grouped.key_dtypes.index) == ["b"]

    grouped = tile(grouped)
    assert len(grouped.chunks) == 5
    for chunk in grouped.chunks:
        assert isinstance(chunk.op, DataFrameGroupByOperand)

    series = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
    ms = md.Series(series, chunk_size=3)
    grouped = ms.groupby(lambda x: x + 1)

    assert isinstance(grouped, SeriesGroupBy)
    assert isinstance(grouped.op, DataFrameGroupByOperand)

    grouped = tile(grouped)
    assert len(grouped.chunks) == 3
    for chunk in grouped.chunks:
        assert isinstance(chunk.op, DataFrameGroupByOperand)

    with pytest.raises(TypeError):
        ms.groupby(lambda x: x + 1, as_index=False)


def test_groupby_get_item():
    df1 = pd.DataFrame(
        {
            "a": [3, 4, 5, 3, 5, 4, 1, 2, 3],
            "b": [1, 3, 4, 5, 6, 5, 4, 4, 4],
            "c": list("aabaaddce"),
        }
    )
    mdf = md.DataFrame(df1, chunk_size=3)

    r = tile(mdf.groupby("b")[["a", "b"]])
    assert isinstance(r, DataFrameGroupBy)
    assert isinstance(r.op, GroupByIndex)
    assert r.selection == ["a", "b"]
    assert list(r.key_dtypes.index) == ["b"]
    assert len(r.chunks) == 3

    r = tile(mdf.groupby("b").a)
    assert isinstance(r, SeriesGroupBy)
    assert isinstance(r.op, GroupByIndex)
    assert r.name == "a"
    assert list(r.key_dtypes.index) == ["b"]
    assert len(r.chunks) == 3

    with pytest.raises(IndexError):
        getattr(mdf.groupby("b")[["a", "b"]], "a")


def test_groupby_agg():
    df = pd.DataFrame(
        {
            "a": np.random.choice([2, 3, 4], size=(20,)),
            "b": np.random.choice([2, 3, 4], size=(20,)),
        }
    )
    mdf = md.DataFrame(df, chunk_size=3)
    r = mdf.groupby("a").agg("sum", method="tree")
    assert isinstance(r.op, DataFrameGroupByAgg)
    assert isinstance(r, DataFrame)
    assert r.op.method == "tree"
    r = tile(r)
    assert len(r.chunks) == 1
    assert r.chunks[0].op.stage == OperandStage.agg
    assert len(r.chunks[0].inputs) == 1
    assert len(r.chunks[0].inputs[0].inputs) == 2

    df = pd.DataFrame(
        {
            "c1": range(10),
            "c2": np.random.choice(["a", "b", "c"], (10,)),
            "c3": np.random.rand(10),
        }
    )
    mdf = md.DataFrame(df, chunk_size=2)
    r = mdf.groupby("c2").sum(method="shuffle")

    assert isinstance(r.op, DataFrameGroupByAgg)
    assert isinstance(r, DataFrame)

    r = tile(r)
    assert len(r.chunks) == 5
    for chunk in r.chunks:
        assert isinstance(chunk.op, DataFrameGroupByAgg)
        assert chunk.op.stage == OperandStage.agg
        assert isinstance(chunk.inputs[0].op, DataFrameGroupByOperand)
        assert chunk.inputs[0].op.stage == OperandStage.reduce
        assert isinstance(chunk.inputs[0].inputs[0].op, DataFrameShuffleProxy)
        assert isinstance(
            chunk.inputs[0].inputs[0].inputs[0].op, DataFrameGroupByOperand
        )
        assert chunk.inputs[0].inputs[0].inputs[0].op.stage == OperandStage.map

        agg_chunk = chunk.inputs[0].inputs[0].inputs[0].inputs[0]
        assert agg_chunk.op.stage == OperandStage.map

    # test unknown method
    with pytest.raises(ValueError):
        mdf.groupby("c2").sum(method="not_exist")


def test_groupby_apply():
    df1 = pd.DataFrame(
        {
            "a": [3, 4, 5, 3, 5, 4, 1, 2, 3],
            "b": [1, 3, 4, 5, 6, 5, 4, 4, 4],
            "c": list("aabaaddce"),
        }
    )

    def apply_df(df):
        return df.sort_index()

    def apply_df_with_error(df):
        assert len(df) > 2
        return df.sort_index()

    def apply_series(s):
        return s.sort_index()

    mdf = md.DataFrame(df1, chunk_size=3)

    with pytest.raises(TypeError):
        mdf.groupby("b").apply(apply_df_with_error)

    applied = tile(
        mdf.groupby("b").apply(
            apply_df_with_error, output_type="dataframe", dtypes=df1.dtypes
        )
    )
    pd.testing.assert_series_equal(applied.dtypes, df1.dtypes)
    assert applied.shape == (np.nan, 3)
    assert applied.op._op_type_ == opcodes.APPLY
    assert applied.op.output_types[0] == OutputType.dataframe
    assert len(applied.chunks) == 3
    assert applied.chunks[0].shape == (np.nan, 3)
    pd.testing.assert_series_equal(applied.chunks[0].dtypes, df1.dtypes)

    applied = tile(mdf.groupby("b").apply(apply_df))
    pd.testing.assert_series_equal(applied.dtypes, df1.dtypes)
    assert applied.shape == (np.nan, 3)
    assert applied.op._op_type_ == opcodes.APPLY
    assert applied.op.output_types[0] == OutputType.dataframe
    assert len(applied.chunks) == 3
    assert applied.chunks[0].shape == (np.nan, 3)
    pd.testing.assert_series_equal(applied.chunks[0].dtypes, df1.dtypes)

    applied = tile(mdf.groupby("b").apply(lambda df: df.a))
    assert applied.dtype == df1.a.dtype
    assert applied.shape == (np.nan,)
    assert applied.op._op_type_ == opcodes.APPLY
    assert applied.op.output_types[0] == OutputType.series
    assert len(applied.chunks) == 3
    assert applied.chunks[0].shape == (np.nan,)
    assert applied.chunks[0].dtype == df1.a.dtype

    applied = tile(mdf.groupby("b").apply(lambda df: df.a.sum()))
    assert applied.dtype == df1.a.dtype
    assert applied.shape == (np.nan,)
    assert applied.op._op_type_ == opcodes.APPLY
    assert applied.op.output_types[0] == OutputType.series
    assert len(applied.chunks) == 3
    assert applied.chunks[0].shape == (np.nan,)
    assert applied.chunks[0].dtype == df1.a.dtype

    series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])

    ms1 = md.Series(series1, chunk_size=3)
    applied = tile(ms1.groupby(lambda x: x % 3).apply(apply_series))
    assert applied.dtype == series1.dtype
    assert applied.shape == (np.nan,)
    assert applied.op._op_type_ == opcodes.APPLY
    assert applied.op.output_types[0] == OutputType.series
    assert len(applied.chunks) == 3
    assert applied.chunks[0].shape == (np.nan,)
    assert applied.chunks[0].dtype == series1.dtype


def test_groupby_transform():
    df1 = pd.DataFrame(
        {
            "a": [3, 4, 5, 3, 5, 4, 1, 2, 3],
            "b": [1, 3, 4, 5, 6, 5, 4, 4, 4],
            "c": list("aabaaddce"),
            "d": [3, 4, 5, 3, 5, 4, 1, 2, 3],
            "e": [1, 3, 4, 5, 6, 5, 4, 4, 4],
            "f": list("aabaaddce"),
        }
    )

    def transform_df(df):
        return df.sort_index()

    def transform_df_with_err(df):
        assert len(df) > 2
        return df.sort_index()

    mdf = md.DataFrame(df1, chunk_size=3)

    with pytest.raises(TypeError):
        mdf.groupby("b").transform(["cummax", "cumcount"])

    with pytest.raises(TypeError):
        mdf.groupby("b").transform(transform_df_with_err)

    r = tile(
        mdf.groupby("b").transform(transform_df_with_err, dtypes=df1.dtypes.drop("b"))
    )
    assert r.dtypes.index.tolist() == list("acdef")
    assert r.shape == (9, 5)
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.dataframe
    assert len(r.chunks) == 3
    assert r.chunks[0].shape == (np.nan, 5)
    assert r.chunks[0].dtypes.index.tolist() == list("acdef")

    r = tile(mdf.groupby("b").transform(transform_df))
    assert r.dtypes.index.tolist() == list("acdef")
    assert r.shape == (9, 5)
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.dataframe
    assert len(r.chunks) == 3
    assert r.chunks[0].shape == (np.nan, 5)
    assert r.chunks[0].dtypes.index.tolist() == list("acdef")

    r = tile(mdf.groupby("b").transform(["cummax", "cumcount"], _call_agg=True))
    assert r.shape == (np.nan, 6)
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.dataframe
    assert len(r.chunks) == 3
    assert r.chunks[0].shape == (np.nan, 6)

    agg_dict = OrderedDict([("d", "cummax"), ("b", "cumsum")])
    r = tile(mdf.groupby("b").transform(agg_dict, _call_agg=True))
    assert r.shape == (np.nan, 2)
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.dataframe
    assert len(r.chunks) == 3
    assert r.chunks[0].shape == (np.nan, 2)

    agg_list = ["sum", lambda s: s.sum()]
    r = tile(mdf.groupby("b").transform(agg_list, _call_agg=True))
    assert r.shape == (np.nan, 10)
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.dataframe
    assert len(r.chunks) == 3
    assert r.chunks[0].shape == (np.nan, 10)

    series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
    ms1 = md.Series(series1, chunk_size=3)

    r = tile(ms1.groupby(lambda x: x % 3).transform(lambda x: x + 1))
    assert r.dtype == series1.dtype
    assert r.shape == series1.shape
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.series
    assert len(r.chunks) == 3
    assert r.chunks[0].shape == (np.nan,)
    assert r.chunks[0].dtype == series1.dtype

    r = tile(ms1.groupby(lambda x: x % 3).transform("cummax", _call_agg=True))
    assert r.shape == (np.nan,)
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.series
    assert len(r.chunks) == 3
    assert r.chunks[0].shape == (np.nan,)

    agg_list = ["cummax", "cumcount"]
    r = tile(ms1.groupby(lambda x: x % 3).transform(agg_list, _call_agg=True))
    assert r.shape == (np.nan, 2)
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.dataframe
    assert len(r.chunks) == 3
    assert r.chunks[0].shape == (np.nan, 2)


def test_groupby_cum():
    df1 = pd.DataFrame(
        {
            "a": [3, 5, 2, 7, 1, 2, 4, 6, 2, 4],
            "b": [8, 3, 4, 1, 8, 2, 2, 2, 2, 3],
            "c": [1, 8, 8, 5, 3, 5, 0, 0, 5, 4],
        }
    )
    mdf = md.DataFrame(df1, chunk_size=3)

    for fun in ["cummin", "cummax", "cumprod", "cumsum"]:
        r = tile(getattr(mdf.groupby("b"), fun)())
        assert r.op.output_types[0] == OutputType.dataframe
        assert len(r.chunks) == 4
        assert r.shape == (len(df1), 2)
        assert r.chunks[0].shape == (np.nan, 2)
        pd.testing.assert_index_equal(
            r.chunks[0].columns_value.to_pandas(), pd.Index(["a", "c"])
        )

        r = tile(getattr(mdf.groupby("b"), fun)(axis=1))
        assert r.op.output_types[0] == OutputType.dataframe
        assert len(r.chunks) == 4
        assert r.shape == (len(df1), 3)
        assert r.chunks[0].shape == (np.nan, 3)
        pd.testing.assert_index_equal(
            r.chunks[0].columns_value.to_pandas(), df1.columns
        )

    r = tile(mdf.groupby("b").cumcount())
    assert r.op.output_types[0] == OutputType.series
    assert len(r.chunks) == 4
    assert r.shape == (len(df1),)
    assert r.chunks[0].shape == (np.nan,)

    series1 = pd.Series([2, 2, 5, 7, 3, 7, 8, 8, 5, 6])
    ms1 = md.Series(series1, chunk_size=3)

    for fun in ["cummin", "cummax", "cumprod", "cumsum", "cumcount"]:
        r = tile(getattr(ms1.groupby(lambda x: x % 2), fun)())
        assert r.op.output_types[0] == OutputType.series
        assert len(r.chunks) == 4
        assert r.shape == (len(series1),)
        assert r.chunks[0].shape == (np.nan,)


def test_groupby_fill():
    df1 = pd.DataFrame(
        [
            [1, 1, 10],
            [1, 1, np.nan],
            [1, 1, np.nan],
            [1, 2, np.nan],
            [1, 2, 20],
            [1, 2, np.nan],
            [1, 3, np.nan],
            [1, 3, np.nan],
        ],
        columns=["one", "two", "three"],
    )
    mdf = md.DataFrame(df1, chunk_size=3)

    r = tile(getattr(mdf.groupby(["one", "two"]), "ffill")())
    assert r.op.output_types[0] == OutputType.dataframe
    assert r.shape == (len(df1), 1)
    assert len(r.chunks) == 3
    assert r.chunks[0].shape == (np.nan, 1)
    assert r.dtypes.index.tolist() == ["three"]

    r = tile(getattr(mdf.groupby(["two"]), "bfill")())
    assert r.op.output_types[0] == OutputType.dataframe
    assert r.shape == (len(df1), 2)
    assert len(r.chunks) == 3
    assert r.chunks[0].shape == (np.nan, 2)
    assert r.dtypes.index.tolist() == ["one", "three"]

    r = tile(getattr(mdf.groupby(["two"]), "backfill")())
    assert r.op.output_types[0] == OutputType.dataframe
    assert r.shape == (len(df1), 2)
    assert len(r.chunks) == 3
    assert r.chunks[0].shape == (np.nan, 2)
    assert r.dtypes.index.tolist() == ["one", "three"]

    r = tile(getattr(mdf.groupby(["one"]), "fillna")(5))
    assert r.op.output_types[0] == OutputType.dataframe
    assert r.shape == (len(df1), 2)
    assert len(r.chunks) == 3
    assert r.chunks[0].shape == (np.nan, 2)
    assert r.dtypes.index.tolist() == ["two", "three"]

    s1 = pd.Series([4, 3, 9, np.nan, np.nan, 7, 10, 8, 1, 6])
    ms1 = md.Series(s1, chunk_size=3)
    r = tile(getattr(ms1.groupby(lambda x: x % 2), "ffill")())
    assert r.op.output_types[0] == OutputType.series
    assert len(r.chunks) == 4
    assert r.shape == (len(s1),)
    assert r.chunks[0].shape == (np.nan,)

    r = tile(getattr(ms1.groupby(lambda x: x % 2), "bfill")())
    assert r.op.output_types[0] == OutputType.series
    assert len(r.chunks) == 4
    assert r.shape == (len(s1),)
    assert r.chunks[0].shape == (np.nan,)

    r = tile(getattr(ms1.groupby(lambda x: x % 2), "backfill")())
    assert r.op.output_types[0] == OutputType.series
    assert len(r.chunks) == 4
    assert r.shape == (len(s1),)
    assert r.chunks[0].shape == (np.nan,)

    r = tile(getattr(ms1.groupby(lambda x: x % 2), "fillna")(5))
    assert r.op.output_types[0] == OutputType.series
    assert len(r.chunks) == 4
    assert r.shape == (len(s1),)
    assert r.chunks[0].shape == (np.nan,)

    s1 = pd.Series([4, 3, 9, np.nan, np.nan, 7, 10, 8, 1, 6])
    ms1 = md.Series(s1, chunk_size=3)

    r = tile(getattr(ms1.groupby(lambda x: x % 2), "ffill")())
    assert r.op.output_types[0] == OutputType.series
    assert len(r.chunks) == 4
    assert r.shape == (len(s1),)
    assert r.chunks[0].shape == (np.nan,)

    r = tile(getattr(ms1.groupby(lambda x: x % 2), "bfill")())
    assert r.op.output_types[0] == OutputType.series
    assert len(r.chunks) == 4
    assert r.shape == (len(s1),)
    assert r.chunks[0].shape == (np.nan,)

    r = tile(getattr(ms1.groupby(lambda x: x % 2), "backfill")())
    assert r.op.output_types[0] == OutputType.series
    assert len(r.chunks) == 4
    assert r.shape == (len(s1),)
    assert r.chunks[0].shape == (np.nan,)

    r = tile(getattr(ms1.groupby(lambda x: x % 2), "fillna")(5))
    assert r.op.output_types[0] == OutputType.series
    assert len(r.chunks) == 4
    assert r.shape == (len(s1),)
    assert r.chunks[0].shape == (np.nan,)
