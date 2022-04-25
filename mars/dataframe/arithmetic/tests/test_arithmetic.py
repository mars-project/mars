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

import datetime
import itertools
import operator
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from .... import tensor as mt
from ....core import OutputType, OperandType, tile
from ....core.operand import OperandStage
from ....utils import dataslots
from ...align import DataFrameIndexAlign, DataFrameShuffleProxy
from ...core import IndexValue
from ...datasource.dataframe import from_pandas, DataFrameDataSource
from ...datasource.from_tensor import dataframe_from_tensor
from ...datasource.series import from_pandas as from_pandas_series, SeriesDataSource
from ...utils import hash_dtypes
from ...utils import (
    split_monotonic_index_min_max,
    build_split_idx_to_origin_idx,
    filter_index_value,
)
from .. import (
    DataFrameAbs,
    DataFrameAdd,
    DataFrameSubtract,
    DataFrameMul,
    DataFrameFloorDiv,
    DataFrameTrueDiv,
    DataFrameMod,
    DataFramePower,
    DataFrameEqual,
    DataFrameNotEqual,
    DataFrameGreater,
    DataFrameLess,
    DataFrameGreaterEqual,
    DataFrameLessEqual,
    DataFrameNot,
    DataFrameAnd,
    DataFrameOr,
    DataFrameXor,
)


def comp_func(name, reverse_name):
    def inner(lhs, rhs):
        try:
            return getattr(lhs, name)(rhs)
        except AttributeError:
            return getattr(rhs, reverse_name)(lhs)

    return inner


@dataslots
@dataclass
class FunctionOptions:
    func: Callable
    op: OperandType
    func_name: str
    rfunc_name: str


binary_functions = dict(
    add=FunctionOptions(
        func=operator.add, op=DataFrameAdd, func_name="add", rfunc_name="radd"
    ),
    subtract=FunctionOptions(
        func=operator.sub, op=DataFrameSubtract, func_name="sub", rfunc_name="rsub"
    ),
    multiply=FunctionOptions(
        func=operator.mul, op=DataFrameMul, func_name="mul", rfunc_name="rmul"
    ),
    floordiv=FunctionOptions(
        func=operator.floordiv,
        op=DataFrameFloorDiv,
        func_name="floordiv",
        rfunc_name="rfloordiv",
    ),
    truediv=FunctionOptions(
        func=operator.truediv,
        op=DataFrameTrueDiv,
        func_name="truediv",
        rfunc_name="rtruediv",
    ),
    mod=FunctionOptions(
        func=operator.mod, op=DataFrameMod, func_name="mod", rfunc_name="rmod"
    ),
    power=FunctionOptions(
        func=operator.pow, op=DataFramePower, func_name="pow", rfunc_name="rpow"
    ),
    equal=FunctionOptions(
        func=comp_func("eq", "eq"), op=DataFrameEqual, func_name="eq", rfunc_name="eq"
    ),
    not_equal=FunctionOptions(
        func=comp_func("ne", "ne"),
        op=DataFrameNotEqual,
        func_name="ne",
        rfunc_name="ne",
    ),
    greater=FunctionOptions(
        func=comp_func("gt", "lt"), op=DataFrameGreater, func_name="gt", rfunc_name="lt"
    ),
    less=FunctionOptions(
        func=comp_func("lt", "gt"), op=DataFrameLess, func_name="lt", rfunc_name="gt"
    ),
    greater_equal=FunctionOptions(
        func=comp_func("ge", "le"),
        op=DataFrameGreaterEqual,
        func_name="ge",
        rfunc_name="le",
    ),
    less_equal=FunctionOptions(
        func=comp_func("le", "ge"),
        op=DataFrameLessEqual,
        func_name="le",
        rfunc_name="ge",
    ),
    logical_and=FunctionOptions(
        func=operator.and_, op=DataFrameAnd, func_name="__and__", rfunc_name="and"
    ),
    logical_or=FunctionOptions(
        func=operator.or_, op=DataFrameOr, func_name="__or__", rfunc_name="__ror__"
    ),
    logical_xor=FunctionOptions(
        func=operator.xor, op=DataFrameXor, func_name="__xor__", rfunc_name="__rxor__"
    ),
)


def to_boolean_if_needed(func_name, value, split_value=0.5):
    if func_name in ["__and__", "__or__", "__xor__"]:
        return value > split_value
    else:
        return value


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_without_shuffle(func_name, func_opts):
    # all the axes are monotonic
    # data1 with index split into [0...4], [5...9],
    # columns [3...7], [8...12]
    data1 = pd.DataFrame(
        np.random.rand(10, 10), index=np.arange(10), columns=np.arange(3, 13)
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)
    # data2 with index split into [6...11], [2, 5],
    # columns [4...9], [10, 13]
    data2 = pd.DataFrame(
        np.random.rand(10, 10), index=np.arange(11, 1, -1), columns=np.arange(4, 14)
    )
    data2 = to_boolean_if_needed(func_opts.func_name, data2)
    df2 = from_pandas(data2, chunk_size=6)

    df3 = func_opts.func(df1, df2)

    # test df3's index and columns
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert isinstance(df3.index_value.value, IndexValue.Int64Index)
    pd.testing.assert_index_equal(
        df3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df3.index_value.key != df1.index_value.key
    assert df3.index_value.key != df2.index_value.key
    assert df3.shape[1] == 11  # columns is recorded, so we can get it

    df1, df2, df3 = tile(df1, df2, df3)

    # test df3's index and columns after tiling
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert isinstance(df3.index_value.value, IndexValue.Int64Index)
    pd.testing.assert_index_equal(
        df3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df3.index_value.key != df1.index_value.key
    assert df3.index_value.key != df2.index_value.key
    assert df3.shape[1] == 11  # columns is recorded, so we can get it

    data1_index_min_max = [(0, True, 4, True), (5, True, 9, True)]
    data1_columns_min_max = [[3, True, 7, True], [8, True, 12, True]]
    data2_index_min_max = [(2, True, 5, True), (6, True, 11, True)]
    data2_columns_min_max = [(4, True, 9, True), (10, True, 13, True)]

    left_index_splits, right_index_splits = split_monotonic_index_min_max(
        data1_index_min_max, True, data2_index_min_max, False
    )
    left_columns_splits, right_columns_splits = split_monotonic_index_min_max(
        data1_columns_min_max, True, data2_columns_min_max, True
    )

    left_index_idx_to_original_idx = build_split_idx_to_origin_idx(left_index_splits)
    right_index_idx_to_original_idx = build_split_idx_to_origin_idx(
        right_index_splits, False
    )
    left_columns_idx_to_original_idx = build_split_idx_to_origin_idx(
        left_columns_splits
    )
    right_columns_idx_to_original_idx = build_split_idx_to_origin_idx(
        right_columns_splits
    )

    assert df3.chunk_shape == (7, 7)
    for c in df3.chunks:
        assert isinstance(c.op, func_opts.op)
        assert len(c.inputs) == 2
        # test shape
        idx = c.index
        # test the left side
        assert isinstance(c.inputs[0].op, DataFrameIndexAlign)
        assert c.inputs[0].op.stage == OperandStage.map
        left_row_idx, left_row_inner_idx = left_index_idx_to_original_idx[idx[0]]
        left_col_idx, left_col_inner_idx = left_columns_idx_to_original_idx[idx[1]]
        expect_df1_input = df1.cix[left_row_idx, left_col_idx].data
        assert c.inputs[0].inputs[0] is expect_df1_input
        left_index_min_max = left_index_splits[left_row_idx][left_row_inner_idx]
        assert c.inputs[0].op.index_min == left_index_min_max[0]
        assert c.inputs[0].op.index_min_close == left_index_min_max[1]
        assert c.inputs[0].op.index_max == left_index_min_max[2]
        assert c.inputs[0].op.index_max_close == left_index_min_max[3]
        assert isinstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
        left_column_min_max = left_columns_splits[left_col_idx][left_col_inner_idx]
        assert c.inputs[0].op.column_min == left_column_min_max[0]
        assert c.inputs[0].op.column_min_close == left_column_min_max[1]
        assert c.inputs[0].op.column_max == left_column_min_max[2]
        assert c.inputs[0].op.column_max_close == left_column_min_max[3]
        expect_left_columns = filter_index_value(
            expect_df1_input.columns_value, left_column_min_max, store_data=True
        )
        pd.testing.assert_index_equal(
            c.inputs[0].columns_value.to_pandas(), expect_left_columns.to_pandas()
        )
        pd.testing.assert_index_equal(
            c.inputs[0].dtypes.index, expect_left_columns.to_pandas()
        )
        # test the right side
        assert isinstance(c.inputs[1].op, DataFrameIndexAlign)
        assert c.inputs[1].op.stage == OperandStage.map
        right_row_idx, right_row_inner_idx = right_index_idx_to_original_idx[idx[0]]
        right_col_idx, right_col_inner_idx = right_columns_idx_to_original_idx[idx[1]]
        expect_df2_input = df2.cix[right_row_idx, right_col_idx].data
        assert c.inputs[1].inputs[0] is expect_df2_input
        right_index_min_max = right_index_splits[right_row_idx][right_row_inner_idx]
        assert c.inputs[1].op.index_min == right_index_min_max[0]
        assert c.inputs[1].op.index_min_close == right_index_min_max[1]
        assert c.inputs[1].op.index_max == right_index_min_max[2]
        assert c.inputs[1].op.index_max_close == right_index_min_max[3]
        assert isinstance(c.inputs[1].index_value.to_pandas(), type(data2.index))
        right_column_min_max = right_columns_splits[right_col_idx][right_col_inner_idx]
        assert c.inputs[1].op.column_min == right_column_min_max[0]
        assert c.inputs[1].op.column_min_close == right_column_min_max[1]
        assert c.inputs[1].op.column_max == right_column_min_max[2]
        assert c.inputs[1].op.column_max_close == right_column_min_max[3]
        expect_right_columns = filter_index_value(
            expect_df2_input.columns_value, left_column_min_max, store_data=True
        )
        pd.testing.assert_index_equal(
            c.inputs[1].columns_value.to_pandas(), expect_right_columns.to_pandas()
        )
        pd.testing.assert_index_equal(
            c.inputs[1].dtypes.index, expect_right_columns.to_pandas()
        )


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_dataframe_and_series_with_align_map(func_name, func_opts):
    data1 = pd.DataFrame(
        np.random.rand(10, 10), index=np.arange(10), columns=np.arange(3, 13)
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)
    s1 = df1[3]

    df2 = func_opts.func(df1, s1)
    df1, df2, s1 = tile(df1, df2, s1)

    assert df2.shape == (df1.shape[0], np.nan)
    assert df2.index_value.key == df1.index_value.key

    data1_columns_min_max = [[3, True, 7, True], [8, True, 12, True]]
    data2_index_min_max = [(0, True, 4, True), (5, True, 9, True)]

    left_columns_splits, right_index_splits = split_monotonic_index_min_max(
        data1_columns_min_max, True, data2_index_min_max, True
    )

    left_columns_idx_to_original_idx = build_split_idx_to_origin_idx(
        left_columns_splits
    )
    right_index_idx_to_original_idx = build_split_idx_to_origin_idx(right_index_splits)

    assert df2.chunk_shape == (2, 7)
    for c in df2.chunks:
        assert isinstance(c.op, func_opts.op)
        assert len(c.inputs) == 2
        # test shape
        idx = c.index
        # test the left side (dataframe)
        assert isinstance(c.inputs[0].op, DataFrameIndexAlign)
        assert c.inputs[0].op.stage == OperandStage.map
        left_col_idx, left_col_inner_idx = left_columns_idx_to_original_idx[idx[1]]
        expect_df1_input = df1.cix[idx[0], left_col_idx].data
        assert c.inputs[0].inputs[0] is expect_df1_input
        left_column_min_max = left_columns_splits[left_col_idx][left_col_inner_idx]
        assert c.inputs[0].op.column_min == left_column_min_max[0]
        assert c.inputs[0].op.column_min_close == left_column_min_max[1]
        assert c.inputs[0].op.column_max == left_column_min_max[2]
        assert c.inputs[0].op.column_max_close == left_column_min_max[3]
        expect_left_columns = filter_index_value(
            expect_df1_input.columns_value, left_column_min_max, store_data=True
        )
        pd.testing.assert_index_equal(
            c.inputs[0].columns_value.to_pandas(), expect_left_columns.to_pandas()
        )
        pd.testing.assert_index_equal(
            c.inputs[0].dtypes.index, expect_left_columns.to_pandas()
        )

        # test the right side (series)
        assert isinstance(c.inputs[1].op, DataFrameIndexAlign)
        assert c.inputs[1].op.stage == OperandStage.map
        right_row_idx, right_row_inner_idx = right_index_idx_to_original_idx[idx[1]]
        expect_s1_input = s1.cix[(right_row_idx,)].data
        assert c.inputs[1].inputs[0] is expect_s1_input
        right_index_min_max = right_index_splits[right_row_idx][right_row_inner_idx]
        assert c.inputs[1].op.index_min == right_index_min_max[0]
        assert c.inputs[1].op.index_min_close == right_index_min_max[1]
        assert c.inputs[1].op.index_max == right_index_min_max[2]
        assert c.inputs[1].op.index_max_close == right_index_min_max[3]
        assert isinstance(c.inputs[1].index_value.to_pandas(), type(data1[3].index))


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_dataframe_and_series_identical(func_name, func_opts):
    data1 = pd.DataFrame(
        np.random.rand(10, 10), index=np.arange(10), columns=np.arange(10)
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)
    s1 = from_pandas_series(data1[3], chunk_size=5)

    df2 = func_opts.func(df1, s1)
    df1, df2, s1 = tile(df1, df2, s1)

    assert df2.shape == (10, 10)
    assert df2.index_value.key == df1.index_value.key
    assert df2.columns_value.key == df1.columns_value.key
    assert df2.columns_value.key == s1.index_value.key

    assert df2.chunk_shape == (2, 2)
    for c in df2.chunks:
        assert isinstance(c.op, func_opts.op)
        assert len(c.inputs) == 2
        assert c.shape == (5, 5)
        assert c.index_value.key == df1.cix[c.index].index_value.key
        assert c.index_value.key == df2.cix[c.index].index_value.key
        assert c.columns_value.key == df1.cix[c.index].columns_value.key
        assert c.columns_value.key == df2.cix[c.index].columns_value.key
        pd.testing.assert_index_equal(
            c.columns_value.to_pandas(), df1.cix[c.index].columns_value.to_pandas()
        )
        pd.testing.assert_index_equal(
            c.columns_value.to_pandas(), df2.cix[c.index].columns_value.to_pandas()
        )
        pd.testing.assert_index_equal(
            c.dtypes.index, df1.cix[c.index].columns_value.to_pandas()
        )

        # test the left side
        assert isinstance(c.inputs[0].op, DataFrameDataSource)
        assert c.inputs[0] is df1.cix[c.index].data
        # test the right side
        assert isinstance(c.inputs[1].op, SeriesDataSource)
        assert c.inputs[1] is s1.cix[(c.index[1],)].data


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_dataframe_and_series_with_shuffle(func_name, func_opts):
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[4, 9, 3, 2, 1, 5, 8, 6, 7, 10],
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)
    s1 = from_pandas_series(data1[10], chunk_size=6)

    df2 = func_opts.func(df1, s1)

    # test df2's index and columns
    assert df2.shape == (df1.shape[0], np.nan)
    assert df2.index_value.key == df1.index_value.key
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df2.columns_value.key != df1.columns_value.key

    df1, df2, s1 = tile(df1, df2, s1)

    assert df2.chunk_shape == (2, 2)
    for c in df2.chunks:
        assert isinstance(c.op, func_opts.op)
        assert len(c.inputs) == 2
        idx = c.index
        # test the left side
        assert isinstance(c.inputs[0].op, DataFrameIndexAlign)
        assert c.inputs[0].op.stage == OperandStage.reduce
        expect_dtypes = pd.concat(
            [
                hash_dtypes(ic.inputs[0].op.data.dtypes, 2)[c.index[1]]
                for ic in c.inputs[0].inputs[0].inputs
            ]
        )
        pd.testing.assert_series_equal(c.inputs[0].dtypes, expect_dtypes)
        pd.testing.assert_index_equal(
            c.inputs[0].columns_value.to_pandas(), c.inputs[0].dtypes.index
        )
        pd.testing.assert_index_equal(
            c.inputs[0].index_value.to_pandas(), c.index_value.to_pandas()
        )
        assert isinstance(c.inputs[0].inputs[0].op, DataFrameShuffleProxy)
        for j, ci, ic in zip(
            itertools.count(0), c.inputs[0].inputs[0].inputs, df1.cix[idx[0], :]
        ):
            assert isinstance(ci.op, DataFrameIndexAlign)
            assert ci.op.stage == OperandStage.map
            assert ci.index == (idx[0], j)
            assert ci.op.column_shuffle_size
            shuffle_segments = ci.op.column_shuffle_segments
            expected_shuffle_segments = hash_dtypes(ic.data.dtypes, 2)
            assert len(shuffle_segments) == len(expected_shuffle_segments)
            for ss, ess in zip(shuffle_segments, expected_shuffle_segments):
                pd.testing.assert_series_equal(ss, ess)
            assert ci.inputs[0] is ic.data

        # test the right side
        assert isinstance(c.inputs[1].op, DataFrameIndexAlign)
        assert c.inputs[1].op.stage == OperandStage.reduce
        assert c.inputs[1].op.output_types[0] == OutputType.series
        assert isinstance(c.inputs[1].inputs[0].op, DataFrameShuffleProxy)
        for j, ci, ic in zip(
            itertools.count(0), c.inputs[1].inputs[0].inputs, s1.chunks
        ):
            assert isinstance(ci.op, DataFrameIndexAlign)
            assert ci.op.stage == OperandStage.map
            assert ci.index == (j,)
            assert ci.op.index_shuffle_size
            assert ci.inputs[0] is ic.data

    # make sure shuffle proxies' key are different
    proxy_keys = set()
    for i in range(df2.chunk_shape[0]):
        cs = [c for c in df2.chunks if c.index[0] == i]
        lps = {c.inputs[0].inputs[0].op.key for c in cs}
        assert len(lps) == 1
        proxy_keys.add(lps.pop())
        rps = {c.inputs[1].inputs[0].op.key for c in cs}
        assert len(rps) == 1
        proxy_keys.add(rps.pop())
    assert len(proxy_keys) == df2.chunk_shape[0] + 1


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_series_and_series_with_align_map(func_name, func_opts):
    data1 = pd.DataFrame(
        np.random.rand(10, 10), index=np.arange(10), columns=np.arange(3, 13)
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)

    s1 = df1.iloc[4]
    s2 = df1[3]

    s3 = func_opts.func(s1, s2)

    s1, s2, s3 = tile(s1, s2, s3)

    assert s3.shape == (np.nan,)

    s1_index_min_max = [[3, True, 7, True], [8, True, 12, True]]
    s2_index_min_max = [(0, True, 4, True), (5, True, 9, True)]

    left_index_splits, right_index_splits = split_monotonic_index_min_max(
        s1_index_min_max, True, s2_index_min_max, True
    )

    left_index_idx_to_original_idx = build_split_idx_to_origin_idx(left_index_splits)
    right_index_idx_to_original_idx = build_split_idx_to_origin_idx(right_index_splits)

    assert s3.chunk_shape == (7,)
    for c in s3.chunks:
        assert isinstance(c.op, func_opts.op)
        assert len(c.inputs) == 2
        # test shape
        idx = c.index
        # test the left side (series)
        assert isinstance(c.inputs[0].op, DataFrameIndexAlign)
        assert c.inputs[0].op.stage == OperandStage.map
        left_col_idx, left_col_inner_idx = left_index_idx_to_original_idx[idx[0]]
        expect_s1_input = s1.cix[(left_col_idx,)].data
        assert c.inputs[0].inputs[0] is expect_s1_input
        left_index_min_max = left_index_splits[left_col_idx][left_col_inner_idx]
        assert c.inputs[0].op.index_min == left_index_min_max[0]
        assert c.inputs[0].op.index_min_close == left_index_min_max[1]
        assert c.inputs[0].op.index_max == left_index_min_max[2]
        assert c.inputs[0].op.index_max_close == left_index_min_max[3]
        assert isinstance(
            c.inputs[0].index_value.to_pandas(), type(data1.iloc[4].index)
        )
        expect_left_index = filter_index_value(
            expect_s1_input.index_value, left_index_min_max, store_data=True
        )
        pd.testing.assert_index_equal(
            c.inputs[0].index_value.to_pandas(), expect_left_index.to_pandas()
        )

        # test the right side (series)
        assert isinstance(c.inputs[1].op, DataFrameIndexAlign)
        assert c.inputs[1].op.stage == OperandStage.map
        right_row_idx, right_row_inner_idx = right_index_idx_to_original_idx[idx[0]]
        expect_s2_input = s2.cix[(right_row_idx,)].data
        assert c.inputs[1].inputs[0] is expect_s2_input
        right_index_min_max = right_index_splits[right_row_idx][right_row_inner_idx]
        assert c.inputs[1].op.index_min == right_index_min_max[0]
        assert c.inputs[1].op.index_min_close == right_index_min_max[1]
        assert c.inputs[1].op.index_max == right_index_min_max[2]
        assert c.inputs[1].op.index_max_close == right_index_min_max[3]
        assert isinstance(c.inputs[1].index_value.to_pandas(), type(data1[3].index))
        expect_right_index = filter_index_value(
            expect_s2_input.index_value, right_index_min_max, store_data=True
        )
        pd.testing.assert_index_equal(
            c.inputs[1].index_value.to_pandas(), expect_right_index.to_pandas()
        )


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_series_and_series_identical(func_name, func_opts):
    data1 = pd.DataFrame(
        np.random.rand(10, 10), index=np.arange(10), columns=np.arange(10)
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    s1 = from_pandas_series(data1[1], chunk_size=5)
    s2 = from_pandas_series(data1[3], chunk_size=5)

    s3 = func_opts.func(s1, s2)

    s1, s2, s3 = tile(s1, s2, s3)

    assert s3.shape == (10,)
    assert s3.index_value.key == s1.index_value.key
    assert s3.index_value.key == s2.index_value.key

    assert s3.chunk_shape == (2,)
    for c in s3.chunks:
        assert isinstance(c.op, func_opts.op)
        assert c.op.output_types[0] == OutputType.series
        assert len(c.inputs) == 2
        assert c.shape == (5,)
        assert c.index_value.key == s1.cix[c.index].index_value.key
        assert c.index_value.key == s2.cix[c.index].index_value.key

        # test the left side
        assert isinstance(c.inputs[0].op, SeriesDataSource)
        assert c.inputs[0] is s1.cix[c.index].data
        # test the right side
        assert isinstance(c.inputs[1].op, SeriesDataSource)
        assert c.inputs[1] is s2.cix[c.index].data


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_series_and_series_with_shuffle(func_name, func_opts):
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[4, 9, 3, 2, 1, 5, 8, 6, 7, 10],
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    s1 = from_pandas_series(data1.iloc[4], chunk_size=5)
    s2 = from_pandas_series(data1[10], chunk_size=6)

    s3 = func_opts.func(s1, s2)

    # test s3's index
    assert s3.shape == (np.nan,)
    assert s3.index_value.key != s1.index_value.key
    assert s3.index_value.key != s2.index_value.key
    pd.testing.assert_index_equal(
        s3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )

    s1, s2, s3 = tile(s1, s2, s3)

    assert s3.chunk_shape == (2,)
    for c in s3.chunks:
        assert isinstance(c.op, func_opts.op)
        assert len(c.inputs) == 2
        # test the left side
        assert isinstance(c.inputs[0].op, DataFrameIndexAlign)
        assert c.inputs[0].op.stage == OperandStage.reduce
        assert c.inputs[0].op.output_types[0] == OutputType.series
        assert isinstance(c.inputs[0].inputs[0].op, DataFrameShuffleProxy)
        for j, ci, ic in zip(
            itertools.count(0), c.inputs[0].inputs[0].inputs, s1.chunks
        ):
            assert isinstance(ci.op, DataFrameIndexAlign)
            assert ci.op.stage == OperandStage.map
            assert ci.index == (j,)
            assert ci.op.index_shuffle_size
            assert ci.inputs[0] is ic.data

        # test the right side
        assert isinstance(c.inputs[1].op, DataFrameIndexAlign)
        assert c.inputs[1].op.stage == OperandStage.reduce
        assert c.inputs[1].op.output_types[0] == OutputType.series
        assert isinstance(c.inputs[1].inputs[0].op, DataFrameShuffleProxy)
        for j, ci, ic in zip(
            itertools.count(0), c.inputs[1].inputs[0].inputs, s2.chunks
        ):
            assert isinstance(ci.op, DataFrameIndexAlign)
            assert ci.op.stage == OperandStage.map
            assert ci.index == (j,)
            assert ci.op.index_shuffle_size
            assert ci.inputs[0] is ic.data

    # make sure shuffle proxies' key are different
    proxy_keys = set()
    for c in s3.chunks:
        proxy_keys.add(c.inputs[0].inputs[0].op.key)
        proxy_keys.add(c.inputs[1].inputs[0].op.key)
    assert len(proxy_keys) == 2


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_identical_index_and_columns(func_name, func_opts):
    data1 = pd.DataFrame(np.random.rand(10, 10), columns=np.arange(3, 13))
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)
    data2 = pd.DataFrame(np.random.rand(10, 10), columns=np.arange(3, 13))
    data2 = to_boolean_if_needed(func_opts.func_name, data2)
    df2 = from_pandas(data2, chunk_size=5)

    df3 = func_opts.func(df1, df2)

    # test df3's index and columns
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert isinstance(df3.index_value.value, IndexValue.RangeIndex)
    pd.testing.assert_index_equal(df3.index_value.to_pandas(), pd.RangeIndex(0, 10))
    assert df3.index_value.key == df1.index_value.key
    assert df3.index_value.key == df2.index_value.key
    assert df3.shape == (10, 10)  # columns is recorded, so we can get it

    df1, df2, df3 = tile(df1, df2, df3)

    assert df3.chunk_shape == (2, 2)
    for c in df3.chunks:
        assert isinstance(c.op, func_opts.op)
        assert len(c.inputs) == 2
        assert c.shape == (5, 5)
        assert c.index_value.key == df1.cix[c.index].index_value.key
        assert c.index_value.key == df2.cix[c.index].index_value.key
        assert c.columns_value.key == df1.cix[c.index].columns_value.key
        assert c.columns_value.key == df2.cix[c.index].columns_value.key
        pd.testing.assert_index_equal(
            c.columns_value.to_pandas(), df1.cix[c.index].columns_value.to_pandas()
        )
        pd.testing.assert_index_equal(
            c.columns_value.to_pandas(), df2.cix[c.index].columns_value.to_pandas()
        )
        pd.testing.assert_index_equal(
            c.dtypes.index, df1.cix[c.index].columns_value.to_pandas()
        )

        # test the left side
        assert c.inputs[0] is df1.cix[c.index].data
        # test the right side
        assert c.inputs[1] is df2.cix[c.index].data


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_with_one_shuffle(func_name, func_opts):
    # only 1 axis is monotonic
    # data1 with index split into [0...4], [5...9],
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.arange(10),
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)
    # data2 with index split into [6...11], [2, 5],
    data2 = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.arange(11, 1, -1),
        columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2],
    )
    data2 = to_boolean_if_needed(func_opts.func_name, data2)
    df2 = from_pandas(data2, chunk_size=6)

    df3 = func_opts.func(df1, df2)

    # test df3's index and columns
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert isinstance(df3.index_value.value, IndexValue.Int64Index)
    pd.testing.assert_index_equal(
        df3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df3.index_value.key != df1.index_value.key
    assert df3.index_value.key != df2.index_value.key
    assert df3.shape[1] == 12  # columns is recorded, so we can get it

    df1, df2, df3 = tile(df1, df2, df3)

    data1_index_min_max = [(0, True, 4, True), (5, True, 9, True)]
    data2_index_min_max = [(2, True, 5, True), (6, True, 11, True)]

    left_index_splits, right_index_splits = split_monotonic_index_min_max(
        data1_index_min_max, True, data2_index_min_max, False
    )

    left_index_idx_to_original_idx = build_split_idx_to_origin_idx(left_index_splits)
    right_index_idx_to_original_idx = build_split_idx_to_origin_idx(
        right_index_splits, False
    )

    assert df3.chunk_shape == (7, 2)
    for c in df3.chunks:
        assert isinstance(c.op, func_opts.op)
        assert len(c.inputs) == 2
        idx = c.index
        # test the left side
        assert isinstance(c.inputs[0].op, DataFrameIndexAlign)
        assert c.inputs[0].op.stage == OperandStage.reduce
        expect_dtypes = pd.concat(
            [
                hash_dtypes(ic.inputs[0].op.data.dtypes, 2)[c.index[1]]
                for ic in c.inputs[0].inputs[0].inputs
            ]
        )
        pd.testing.assert_series_equal(c.inputs[0].dtypes, expect_dtypes)
        pd.testing.assert_index_equal(
            c.inputs[0].columns_value.to_pandas(), c.inputs[0].dtypes.index
        )
        assert isinstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
        assert isinstance(c.inputs[0].inputs[0].op, DataFrameShuffleProxy)
        left_row_idx, left_row_inner_idx = left_index_idx_to_original_idx[idx[0]]
        left_index_min_max = left_index_splits[left_row_idx][left_row_inner_idx]
        ics = [ic for ic in df1.chunks if ic.index[0] == left_row_idx]
        for j, ci, ic in zip(itertools.count(0), c.inputs[0].inputs[0].inputs, ics):
            assert isinstance(ci.op, DataFrameIndexAlign)
            assert ci.op.stage == OperandStage.map
            assert ci.index == (idx[0], j)
            assert ci.op.index_min == left_index_min_max[0]
            assert ci.op.index_min_close == left_index_min_max[1]
            assert ci.op.index_max == left_index_min_max[2]
            assert ci.op.index_max_close == left_index_min_max[3]
            assert isinstance(ci.index_value.to_pandas(), type(data1.index))
            assert ci.op.column_shuffle_size
            shuffle_segments = ci.op.column_shuffle_segments
            expected_shuffle_segments = hash_dtypes(ic.data.dtypes, 2)
            assert len(shuffle_segments) == len(expected_shuffle_segments)
            for ss, ess in zip(shuffle_segments, expected_shuffle_segments):
                pd.testing.assert_series_equal(ss, ess)
            assert ci.inputs[0] is ic.data
        # test the right side
        assert isinstance(c.inputs[1].op, DataFrameIndexAlign)
        assert c.inputs[1].op.stage == OperandStage.reduce
        expect_dtypes = pd.concat(
            [
                hash_dtypes(ic.inputs[0].op.data.dtypes, 2)[c.index[1]]
                for ic in c.inputs[1].inputs[0].inputs
            ]
        )
        pd.testing.assert_series_equal(c.inputs[1].dtypes, expect_dtypes)
        pd.testing.assert_index_equal(
            c.inputs[1].columns_value.to_pandas(), c.inputs[1].dtypes.index
        )
        assert isinstance(c.inputs[1].index_value.to_pandas(), type(data1.index))
        assert isinstance(c.inputs[1].inputs[0].op, DataFrameShuffleProxy)
        right_row_idx, right_row_inner_idx = right_index_idx_to_original_idx[idx[0]]
        right_index_min_max = right_index_splits[right_row_idx][right_row_inner_idx]
        ics = [ic for ic in df2.chunks if ic.index[0] == right_row_idx]
        for j, ci, ic in zip(itertools.count(0), c.inputs[1].inputs[0].inputs, ics):
            assert isinstance(ci.op, DataFrameIndexAlign)
            assert ci.op.stage == OperandStage.map
            assert ci.index == (idx[0], j)
            assert ci.op.index_min == right_index_min_max[0]
            assert ci.op.index_min_close == right_index_min_max[1]
            assert ci.op.index_max == right_index_min_max[2]
            assert ci.op.index_max_close == right_index_min_max[3]
            assert ci.op.column_shuffle_size
            shuffle_segments = ci.op.column_shuffle_segments
            expected_shuffle_segments = hash_dtypes(ic.data.dtypes, 2)
            assert len(shuffle_segments) == len(expected_shuffle_segments)
            for ss, ess in zip(shuffle_segments, expected_shuffle_segments):
                pd.testing.assert_series_equal(ss, ess)
            assert ci.inputs[0] is ic.data

    # make sure shuffle proxies' key are different
    proxy_keys = set()
    for i in range(df3.chunk_shape[0]):
        cs = [c for c in df3.chunks if c.index[0] == i]
        lps = {c.inputs[0].inputs[0].op.key for c in cs}
        assert len(lps) == 1
        proxy_keys.add(lps.pop())
        rps = {c.inputs[1].inputs[0].op.key for c in cs}
        assert len(rps) == 1
        proxy_keys.add(rps.pop())
    assert len(proxy_keys) == 2 * df3.chunk_shape[0]


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_with_all_shuffle(func_name, func_opts):
    # no axis is monotonic
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)
    data2 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3],
        columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2],
    )
    data2 = to_boolean_if_needed(func_opts.func_name, data2)
    df2 = from_pandas(data2, chunk_size=6)

    df3 = func_opts.func(df1, df2)

    # test df3's index and columns
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert isinstance(df3.index_value.value, IndexValue.Int64Index)
    pd.testing.assert_index_equal(
        df3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df3.index_value.key != df1.index_value.key
    assert df3.index_value.key != df2.index_value.key
    assert df3.shape[1] == 12  # columns is recorded, so we can get it

    df1, df2, df3 = tile(df1, df2, df3)

    assert df3.chunk_shape == (2, 2)
    proxy_keys = set()
    for c in df3.chunks:
        assert isinstance(c.op, func_opts.op)
        assert len(c.inputs) == 2
        # test left side
        assert isinstance(c.inputs[0].op, DataFrameIndexAlign)
        assert c.inputs[0].op.stage == OperandStage.reduce
        expect_dtypes = pd.concat(
            [
                hash_dtypes(ic.inputs[0].op.data.dtypes, 2)[c.index[1]]
                for ic in c.inputs[0].inputs[0].inputs
                if ic.index[0] == 0
            ]
        )
        pd.testing.assert_series_equal(c.inputs[0].dtypes, expect_dtypes)
        pd.testing.assert_index_equal(
            c.inputs[0].columns_value.to_pandas(), c.inputs[0].dtypes.index
        )
        assert isinstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
        assert isinstance(c.inputs[0].inputs[0].op, DataFrameShuffleProxy)
        proxy_keys.add(c.inputs[0].inputs[0].op.key)
        for ic, ci in zip(c.inputs[0].inputs[0].inputs, df1.chunks):
            assert isinstance(ic.op, DataFrameIndexAlign)
            assert ic.op.stage == OperandStage.map
            assert ic.op.index_shuffle_size == 2
            assert isinstance(ic.index_value.to_pandas(), type(data1.index))
            assert ic.op.column_shuffle_size == 2
            assert ic.columns_value is not None
            shuffle_segments = ic.op.column_shuffle_segments
            expected_shuffle_segments = hash_dtypes(ci.data.dtypes, 2)
            assert len(shuffle_segments) == len(expected_shuffle_segments)
            for ss, ess in zip(shuffle_segments, expected_shuffle_segments):
                pd.testing.assert_series_equal(ss, ess)
            assert ic.inputs[0] is ci.data
        # test right side
        assert isinstance(c.inputs[1].op, DataFrameIndexAlign)
        assert c.inputs[1].op.stage == OperandStage.reduce
        expect_dtypes = pd.concat(
            [
                hash_dtypes(ic.inputs[0].op.data.dtypes, 2)[c.index[1]]
                for ic in c.inputs[1].inputs[0].inputs
                if ic.index[0] == 0
            ]
        )
        pd.testing.assert_series_equal(c.inputs[1].dtypes, expect_dtypes)
        pd.testing.assert_index_equal(
            c.inputs[1].columns_value.to_pandas(), c.inputs[1].dtypes.index
        )
        assert isinstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
        assert isinstance(c.inputs[1].inputs[0].op, DataFrameShuffleProxy)
        proxy_keys.add(c.inputs[1].inputs[0].op.key)
        for ic, ci in zip(c.inputs[1].inputs[0].inputs, df2.chunks):
            assert isinstance(ic.op, DataFrameIndexAlign)
            assert ic.op.stage == OperandStage.map
            assert ic.op.index_shuffle_size == 2
            assert isinstance(ic.index_value.to_pandas(), type(data1.index))
            assert ic.op.column_shuffle_size == 2
            assert ic.columns_value is not None
            shuffle_segments = ic.op.column_shuffle_segments
            expected_shuffle_segments = hash_dtypes(ci.data.dtypes, 2)
            assert len(shuffle_segments) == len(expected_shuffle_segments)
            for ss, ess in zip(shuffle_segments, expected_shuffle_segments):
                pd.testing.assert_series_equal(ss, ess)
            assert ic.inputs[0] is ci.data

    assert len(proxy_keys) == 2

    data4 = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.random.randint(-100, 100, size=(10,)),
        columns=[np.random.bytes(10) for _ in range(10)],
    )
    data4 = to_boolean_if_needed(func_opts.func_name, data4)
    df4 = from_pandas(data4, chunk_size=3)

    data5 = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.random.randint(-100, 100, size=(10,)),
        columns=[np.random.bytes(10) for _ in range(10)],
    )
    data5 = to_boolean_if_needed(func_opts.func_name, data5)
    df5 = from_pandas(data5, chunk_size=3)

    df6 = func_opts.func(df4, df5)

    # test df6's index and columns
    pd.testing.assert_index_equal(
        df6.columns_value.to_pandas(), func_opts.func(data4, data5).columns
    )
    assert isinstance(df6.index_value.value, IndexValue.Int64Index)
    pd.testing.assert_index_equal(
        df6.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df6.index_value.key != df4.index_value.key
    assert df6.index_value.key != df5.index_value.key
    assert df6.shape[1] == 20  # columns is recorded, so we can get it

    df4, df5, df6 = tile(df4, df5, df6)

    assert df6.chunk_shape == (4, 4)
    proxy_keys = set()
    for c in df6.chunks:
        assert isinstance(c.op, func_opts.op)
        assert len(c.inputs) == 2
        # test left side
        assert isinstance(c.inputs[0].op, DataFrameIndexAlign)
        assert c.inputs[0].op.stage == OperandStage.reduce
        expect_dtypes = pd.concat(
            [
                hash_dtypes(ic.inputs[0].op.data.dtypes, 4)[c.index[1]]
                for ic in c.inputs[0].inputs[0].inputs
                if ic.index[0] == 0
            ]
        )
        pd.testing.assert_series_equal(c.inputs[0].dtypes, expect_dtypes)
        pd.testing.assert_index_equal(
            c.inputs[0].columns_value.to_pandas(), c.inputs[0].dtypes.index
        )
        assert isinstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
        assert isinstance(c.inputs[0].inputs[0].op, DataFrameShuffleProxy)
        proxy_keys.add(c.inputs[0].inputs[0].op.key)
        for ic, ci in zip(c.inputs[0].inputs[0].inputs, df4.chunks):
            assert isinstance(ic.op, DataFrameIndexAlign)
            assert ic.op.stage == OperandStage.map
            assert ic.op.index_shuffle_size == 4
            assert isinstance(ic.index_value.to_pandas(), type(data1.index))
            assert ic.op.column_shuffle_size == 4
            assert ic.columns_value is not None
            shuffle_segments = ic.op.column_shuffle_segments
            expected_shuffle_segments = hash_dtypes(ci.data.dtypes, 4)
            assert len(shuffle_segments) == len(expected_shuffle_segments)
            for ss, ess in zip(shuffle_segments, expected_shuffle_segments):
                pd.testing.assert_series_equal(ss, ess)
            assert ic.inputs[0] is ci.data
        # test right side
        assert isinstance(c.inputs[1].op, DataFrameIndexAlign)
        assert c.inputs[1].op.stage == OperandStage.reduce
        expect_dtypes = pd.concat(
            [
                hash_dtypes(ic.inputs[0].op.data.dtypes, 4)[c.index[1]]
                for ic in c.inputs[1].inputs[0].inputs
                if ic.index[0] == 0
            ]
        )
        pd.testing.assert_series_equal(c.inputs[1].dtypes, expect_dtypes)
        pd.testing.assert_index_equal(
            c.inputs[1].columns_value.to_pandas(), c.inputs[1].dtypes.index
        )
        assert isinstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
        assert isinstance(c.inputs[1].inputs[0].op, DataFrameShuffleProxy)
        proxy_keys.add(c.inputs[1].inputs[0].op.key)
        for ic, ci in zip(c.inputs[1].inputs[0].inputs, df5.chunks):
            assert isinstance(ic.op, DataFrameIndexAlign)
            assert ic.op.stage == OperandStage.map
            assert ic.op.index_shuffle_size == 4
            assert isinstance(ic.index_value.to_pandas(), type(data1.index))
            assert ic.op.column_shuffle_size == 4
            assert ic.columns_value is not None
            shuffle_segments = ic.op.column_shuffle_segments
            expected_shuffle_segments = hash_dtypes(ci.data.dtypes, 4)
            assert len(shuffle_segments) == len(expected_shuffle_segments)
            for ss, ess in zip(shuffle_segments, expected_shuffle_segments):
                pd.testing.assert_series_equal(ss, ess)
            assert ic.inputs[0] is ci.data

    assert len(proxy_keys) == 2


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_without_shuffle_and_with_one_chunk(func_name, func_opts):
    # only 1 axis is monotonic
    # data1 with index split into [0...4], [5...9],
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.arange(10),
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=(5, 10))
    # data2 with index split into [6...11], [2, 5],
    data2 = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.arange(11, 1, -1),
        columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2],
    )
    data2 = to_boolean_if_needed(func_opts.func_name, data2)
    df2 = from_pandas(data2, chunk_size=(6, 10))

    df3 = func_opts.func(df1, df2)

    # test df3's index and columns
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert isinstance(df3.index_value.value, IndexValue.Int64Index)
    pd.testing.assert_index_equal(
        df3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df3.index_value.key != df1.index_value.key
    assert df3.index_value.key != df2.index_value.key
    assert df3.shape[1] == 12  # columns is recorded, so we can get it

    df1, df2, df3 = tile(df1, df2, df3)

    data1_index_min_max = [(0, True, 4, True), (5, True, 9, True)]
    data2_index_min_max = [(2, True, 5, True), (6, True, 11, True)]

    left_index_splits, right_index_splits = split_monotonic_index_min_max(
        data1_index_min_max, True, data2_index_min_max, False
    )

    left_index_idx_to_original_idx = build_split_idx_to_origin_idx(left_index_splits)
    right_index_idx_to_original_idx = build_split_idx_to_origin_idx(
        right_index_splits, False
    )

    assert df3.chunk_shape == (7, 1)
    for c in df3.chunks:
        assert isinstance(c.op, func_opts.op)
        assert len(c.inputs) == 2
        # test shape
        idx = c.index
        # test the left side
        assert isinstance(c.inputs[0].op, DataFrameIndexAlign)
        assert c.inputs[0].op.stage == OperandStage.map
        left_row_idx, left_row_inner_idx = left_index_idx_to_original_idx[idx[0]]
        expect_df1_input = df1.cix[left_row_idx, 0].data
        assert c.inputs[0].inputs[0] is expect_df1_input
        left_index_min_max = left_index_splits[left_row_idx][left_row_inner_idx]
        assert c.inputs[0].op.index_min == left_index_min_max[0]
        assert c.inputs[0].op.index_min_close == left_index_min_max[1]
        assert c.inputs[0].op.index_max == left_index_min_max[2]
        assert c.inputs[0].op.index_max_close == left_index_min_max[3]
        assert isinstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
        assert c.inputs[0].op.column_min == expect_df1_input.columns_value.min_val
        assert (
            c.inputs[0].op.column_min_close
            == expect_df1_input.columns_value.min_val_close
        )
        assert c.inputs[0].op.column_max == expect_df1_input.columns_value.max_val
        assert (
            c.inputs[0].op.column_max_close
            == expect_df1_input.columns_value.max_val_close
        )
        expect_left_columns = expect_df1_input.columns_value
        pd.testing.assert_index_equal(
            c.inputs[0].columns_value.to_pandas(), expect_left_columns.to_pandas()
        )
        pd.testing.assert_index_equal(
            c.inputs[0].dtypes.index, expect_left_columns.to_pandas()
        )
        # test the right side
        assert isinstance(c.inputs[1].op, DataFrameIndexAlign)
        assert c.inputs[1].op.stage == OperandStage.map
        right_row_idx, right_row_inner_idx = right_index_idx_to_original_idx[idx[0]]
        expect_df2_input = df2.cix[right_row_idx, 0].data
        assert c.inputs[1].inputs[0] is expect_df2_input
        right_index_min_max = right_index_splits[right_row_idx][right_row_inner_idx]
        assert c.inputs[1].op.index_min == right_index_min_max[0]
        assert c.inputs[1].op.index_min_close == right_index_min_max[1]
        assert c.inputs[1].op.index_max == right_index_min_max[2]
        assert c.inputs[1].op.index_max_close == right_index_min_max[3]
        assert isinstance(c.inputs[1].index_value.to_pandas(), type(data2.index))
        assert c.inputs[1].op.column_min == expect_df2_input.columns_value.min_val
        assert (
            c.inputs[1].op.column_min_close
            == expect_df2_input.columns_value.min_val_close
        )
        assert c.inputs[1].op.column_max == expect_df2_input.columns_value.max_val
        assert (
            c.inputs[1].op.column_max_close
            == expect_df2_input.columns_value.max_val_close
        )
        expect_right_columns = expect_df2_input.columns_value
        pd.testing.assert_index_equal(
            c.inputs[1].columns_value.to_pandas(), expect_right_columns.to_pandas()
        )
        pd.testing.assert_index_equal(
            c.inputs[1].dtypes.index, expect_right_columns.to_pandas()
        )


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_both_one_chunk(func_name, func_opts):
    # no axis is monotonic, but 1 chunk for all axes
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=10)
    data2 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3],
        columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2],
    )
    data2 = to_boolean_if_needed(func_opts.func_name, data2)
    df2 = from_pandas(data2, chunk_size=10)

    df3 = func_opts.func(df1, df2)

    # test df3's index and columns
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert isinstance(df3.index_value.value, IndexValue.Int64Index)
    pd.testing.assert_index_equal(
        df3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df3.index_value.key != df1.index_value.key
    assert df3.index_value.key != df2.index_value.key
    assert df3.shape[1] == 12  # columns is recorded, so we can get it

    df1, df2, df3 = tile(df1, df2, df3)

    assert df3.chunk_shape == (1, 1)
    for c in df3.chunks:
        assert isinstance(c.op, func_opts.op)
        assert len(c.inputs) == 2
        # test the left side
        assert c.inputs[0] is df1.chunks[0].data
        # test the right side
        assert c.inputs[1] is df2.chunks[0].data


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_with_shuffle_and_one_chunk(func_name, func_opts):
    # no axis is monotonic
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=(5, 10))
    data2 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3],
        columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2],
    )
    data2 = to_boolean_if_needed(func_opts.func_name, data2)
    df2 = from_pandas(data2, chunk_size=(6, 10))

    df3 = func_opts.func(df1, df2)

    # test df3's index and columns
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert isinstance(df3.index_value.value, IndexValue.Int64Index)
    pd.testing.assert_index_equal(
        df3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df3.index_value.key != df1.index_value.key
    assert df3.index_value.key != df2.index_value.key
    assert df3.shape[1] == 12  # columns is recorded, so we can get it

    df1, df2, df3 = tile(df1, df2, df3)

    assert df3.chunk_shape == (2, 1)
    proxy_keys = set()
    for c in df3.chunks:
        assert isinstance(c.op, func_opts.op)
        assert len(c.inputs) == 2
        # test left side
        assert isinstance(c.inputs[0].op, DataFrameIndexAlign)
        assert c.inputs[0].op.stage == OperandStage.reduce
        expect_dtypes = pd.concat(
            [
                ic.inputs[0].op.data.dtypes
                for ic in c.inputs[0].inputs[0].inputs
                if ic.index[0] == 0
            ]
        )
        pd.testing.assert_series_equal(c.inputs[0].dtypes, expect_dtypes)
        pd.testing.assert_index_equal(
            c.inputs[0].columns_value.to_pandas(), c.inputs[0].dtypes.index
        )
        assert isinstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
        assert isinstance(c.inputs[0].inputs[0].op, DataFrameShuffleProxy)
        proxy_keys.add(c.inputs[0].inputs[0].op.key)
        for ic, ci in zip(c.inputs[0].inputs[0].inputs, df1.chunks):
            assert isinstance(ic.op, DataFrameIndexAlign)
            assert ic.op.stage == OperandStage.map
            assert ic.op.index_shuffle_size == 2
            assert isinstance(ic.index_value.to_pandas(), type(data1.index))
            assert ic.op.column_min == ci.columns_value.min_val
            assert ic.op.column_min_close == ci.columns_value.min_val_close
            assert ic.op.column_max == ci.columns_value.max_val
            assert ic.op.column_max_close == ci.columns_value.max_val_close
            assert ic.op.column_shuffle_size is None
            assert ic.columns_value is not None
            assert ic.inputs[0] is ci.data
        # test right side
        assert isinstance(c.inputs[1].op, DataFrameIndexAlign)
        assert c.inputs[1].op.stage == OperandStage.reduce
        expect_dtypes = pd.concat(
            [
                ic.inputs[0].op.data.dtypes
                for ic in c.inputs[1].inputs[0].inputs
                if ic.index[0] == 0
            ]
        )
        pd.testing.assert_series_equal(c.inputs[1].dtypes, expect_dtypes)
        pd.testing.assert_index_equal(
            c.inputs[1].columns_value.to_pandas(), c.inputs[1].dtypes.index
        )
        assert isinstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
        assert isinstance(c.inputs[1].inputs[0].op, DataFrameShuffleProxy)
        proxy_keys.add(c.inputs[1].inputs[0].op.key)
        for ic, ci in zip(c.inputs[1].inputs[0].inputs, df2.chunks):
            assert isinstance(ic.op, DataFrameIndexAlign)
            assert ic.op.stage == OperandStage.map
            assert ic.op.index_shuffle_size == 2
            assert isinstance(ic.index_value.to_pandas(), type(data1.index))
            assert ic.op.column_shuffle_size is None
            assert ic.op.column_min == ci.columns_value.min_val
            assert ic.op.column_min_close == ci.columns_value.min_val_close
            assert ic.op.column_max == ci.columns_value.max_val
            assert ic.op.column_max_close == ci.columns_value.max_val_close
            assert ic.op.column_shuffle_size is None
            assert ic.columns_value is not None
            assert ic.inputs[0] is ci.data

    assert len(proxy_keys) == 2


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_on_same_dataframe(func_name, func_opts):
    data = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.random.randint(-100, 100, size=(10,)),
        columns=[np.random.bytes(10) for _ in range(10)],
    )
    data = to_boolean_if_needed(func_opts.func_name, data)
    df = from_pandas(data, chunk_size=3)
    df2 = func_opts.func(df, df)

    # test df2's index and columns
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), func_opts.func(data, data).columns
    )
    assert isinstance(df2.index_value.value, IndexValue.Int64Index)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df2.index_value.key == df.index_value.key
    assert df2.columns_value.key == df.columns_value.key
    assert df2.shape[1] == 10

    df, df2 = tile(df, df2)

    assert df2.chunk_shape == df.chunk_shape
    for c in df2.chunks:
        assert isinstance(c.op, func_opts.op)
        assert len(c.inputs) == 2
        # test the left side
        assert c.inputs[0] is df.cix[c.index].data
        # test the right side
        assert c.inputs[1] is df.cix[c.index].data


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_dataframe_and_scalar(func_name, func_opts):
    if func_opts.func_name in ["__and__", "__or__", "__xor__"]:
        # bitwise logical operators doesn\'t support floating point scalars
        return

    data = pd.DataFrame(
        np.random.rand(10, 10), index=np.arange(10), columns=np.arange(3, 13)
    )
    df = from_pandas(data, chunk_size=5)
    # test operator with scalar
    result = func_opts.func(df, 1)
    result2 = getattr(df, func_opts.func_name)(1)

    # test reverse operator with scalar
    result3 = getattr(df, func_opts.rfunc_name)(1)
    result4 = func_opts.func(df, 1)
    result5 = func_opts.func(1, df)

    expected = func_opts.func(data, 2)
    pd.testing.assert_series_equal(result.dtypes, expected.dtypes)

    pd.testing.assert_index_equal(result.columns_value.to_pandas(), data.columns)
    assert isinstance(result.index_value.value, IndexValue.Int64Index)

    pd.testing.assert_index_equal(result2.columns_value.to_pandas(), data.columns)
    assert isinstance(result2.index_value.value, IndexValue.Int64Index)

    pd.testing.assert_index_equal(result3.columns_value.to_pandas(), data.columns)
    assert isinstance(result3.index_value.value, IndexValue.Int64Index)

    pd.testing.assert_index_equal(result4.columns_value.to_pandas(), data.columns)
    assert isinstance(result4.index_value.value, IndexValue.Int64Index)

    pd.testing.assert_index_equal(result5.columns_value.to_pandas(), data.columns)
    assert isinstance(result5.index_value.value, IndexValue.Int64Index)

    if "builtin_function_or_method" not in str(type(func_opts.func)):
        # skip NotImplemented test for comparison function
        return

    # test NotImplemented, use other's rfunc instead
    class TestRFunc:
        pass

    setattr(TestRFunc, f"__{func_opts.rfunc_name}__", lambda *_: 1)
    other = TestRFunc()
    ret = func_opts.func(df, other)
    assert ret == 1


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_series_and_scalar(func_name, func_opts):
    if func_opts.func_name in ["__and__", "__or__", "__xor__"]:
        # bitwise logical operators doesn\'t support floating point scalars
        return

    data = pd.Series(range(10), index=[1, 3, 4, 2, 9, 10, 33, 23, 999, 123])
    s1 = from_pandas_series(data, chunk_size=3)
    r = getattr(s1, func_opts.func_name)(456)
    s1, r = tile(s1, r)

    assert r.index_value.key == s1.index_value.key
    assert r.chunk_shape == s1.chunk_shape
    assert r.dtype == getattr(data, func_opts.func_name)(456).dtype

    for cr in r.chunks:
        cs = s1.cix[cr.index]
        assert cr.index_value.key == cs.index_value.key
        assert isinstance(cr.op, func_opts.op)
        assert len(cr.inputs) == 1
        assert isinstance(cr.inputs[0].op, SeriesDataSource)
        assert cr.op.rhs == 456

    if "builtin_function_or_method" not in str(type(func_opts.func)):
        # skip rfunc test for comparison function
        return

    s1 = from_pandas_series(data, chunk_size=3)
    r = getattr(s1, func_opts.rfunc_name)(789)
    s1, r = tile(s1, r)

    assert r.index_value.key == s1.index_value.key
    assert r.chunk_shape == s1.chunk_shape

    for cr in r.chunks:
        cs = s1.cix[cr.index]
        assert cr.index_value.key == cs.index_value.key
        assert isinstance(cr.op, func_opts.op)
        assert len(cr.inputs) == 1
        assert isinstance(cr.inputs[0].op, SeriesDataSource)
        assert cr.op.lhs == 789


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_check_inputs(func_name, func_opts):
    data = pd.DataFrame(np.random.rand(10, 3))
    data = to_boolean_if_needed(func_opts.func_name, data)
    df = from_pandas(data)

    with pytest.raises(ValueError):
        _ = df + np.random.rand(5, 3)

    with pytest.raises(ValueError):
        _ = df + np.random.rand(10)

    with pytest.raises(ValueError):
        _ = df + np.random.rand(10, 3, 2)

    data = pd.Series(np.random.rand(10))
    series = from_pandas_series(data)

    with pytest.raises(ValueError):
        _ = series + np.random.rand(5, 3)

    with pytest.raises(ValueError):
        _ = series + np.random.rand(5)


def test_abs():
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    df1 = from_pandas(data1, chunk_size=(5, 10))

    df2 = df1.abs()

    # test df2's index and columns
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), df1.columns_value.to_pandas()
    )
    assert isinstance(df2.index_value.value, IndexValue.Int64Index)
    assert df2.shape == (10, 10)

    df1, df2 = tile(df1, df2)

    assert df2.chunk_shape == (2, 1)
    for c2, c1 in zip(df2.chunks, df1.chunks):
        assert isinstance(c2.op, DataFrameAbs)
        assert len(c2.inputs) == 1
        # compare with input chunks
        assert c2.index == c1.index
        pd.testing.assert_index_equal(
            c2.columns_value.to_pandas(), c1.columns_value.to_pandas()
        )
        pd.testing.assert_index_equal(
            c2.index_value.to_pandas(), c1.index_value.to_pandas()
        )


def test_not():
    data1 = pd.DataFrame(
        np.random.rand(10, 10) > 0.5,
        index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    df1 = from_pandas(data1, chunk_size=(5, 10))

    df2 = ~df1

    # test df2's index and columns
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), df1.columns_value.to_pandas()
    )
    assert isinstance(df2.index_value.value, IndexValue.Int64Index)
    assert df2.shape == (10, 10)

    df1, df2 = tile(df1, df2)

    assert df2.chunk_shape == (2, 1)
    for c2, c1 in zip(df2.chunks, df1.chunks):
        assert isinstance(c2.op, DataFrameNot)
        assert len(c2.inputs) == 1
        # compare with input chunks
        assert c2.index == c1.index
        pd.testing.assert_index_equal(
            c2.columns_value.to_pandas(), c1.columns_value.to_pandas()
        )
        pd.testing.assert_index_equal(
            c2.index_value.to_pandas(), c1.index_value.to_pandas()
        )


def test_arithmetic_lazy_chunk_meta():
    df = dataframe_from_tensor(mt.random.rand(10, 3, chunk_size=3))
    df2 = df + 1
    df2 = tile(df2)

    chunk = df2.chunks[0].data
    assert chunk._FIELD_VALUES.get("_dtypes") is None
    pd.testing.assert_series_equal(chunk.dtypes, df.dtypes)
    assert chunk._FIELD_VALUES.get("_index_value") is None
    pd.testing.assert_index_equal(chunk.index_value.to_pandas(), pd.RangeIndex(3))
    assert chunk._FIELD_VALUES.get("_columns_value") is None
    pd.testing.assert_index_equal(chunk.columns_value.to_pandas(), pd.RangeIndex(3))


def test_datetime_arithmetic():
    data1 = (
        pd.Series([pd.Timedelta(days=d) for d in range(10)]) + datetime.datetime.now()
    )
    s1 = from_pandas_series(data1)

    assert (s1 + pd.Timedelta(days=10)).dtype == (data1 + pd.Timedelta(days=10)).dtype
    assert (s1 + datetime.timedelta(days=10)).dtype == (
        data1 + datetime.timedelta(days=10)
    ).dtype
    assert (s1 - pd.Timestamp.now()).dtype == (data1 - pd.Timestamp.now()).dtype
    assert (s1 - datetime.datetime.now()).dtype == (
        data1 - datetime.datetime.now()
    ).dtype
