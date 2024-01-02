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

import functools
import operator
from functools import reduce
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from .... import dataframe as md
from ....core import tile
from ....core.operand import OperandStage
from ....tensor import Tensor
from ...core import DataFrame, IndexValue, Series, OutputType
from ...datasource.series import from_pandas as from_pandas_series
from ...datasource.dataframe import from_pandas as from_pandas_df
from ...merge import DataFrameConcat
from .. import (
    DataFrameSum,
    DataFrameProd,
    DataFrameMin,
    DataFrameMax,
    DataFrameCount,
    DataFrameMean,
    DataFrameVar,
    DataFrameAll,
    DataFrameAny,
    DataFrameSkew,
    DataFrameKurtosis,
    DataFrameSem,
    DataFrameAggregate,
    DataFrameCummin,
    DataFrameCummax,
    DataFrameCumprod,
    DataFrameCumsum,
    DataFrameNunique,
    CustomReduction,
)
from ..aggregation import where_function
from ..core import ReductionCompiler

pytestmark = pytest.mark.pd_compat


class FunctionOptions(NamedTuple):
    has_skipna: bool = True
    has_numeric_only: bool = True
    has_bool_only: bool = False


reduction_functions = [
    ("sum", DataFrameSum, FunctionOptions()),
    ("prod", DataFrameProd, FunctionOptions()),
    ("min", DataFrameMin, FunctionOptions()),
    ("max", DataFrameMax, FunctionOptions()),
    ("count", DataFrameCount, FunctionOptions(has_skipna=False)),
    ("mean", DataFrameMean, FunctionOptions()),
    ("var", DataFrameVar, FunctionOptions()),
    ("skew", DataFrameSkew, FunctionOptions()),
    ("kurt", DataFrameKurtosis, FunctionOptions()),
    ("sem", DataFrameSem, FunctionOptions()),
    ("all", DataFrameAll, FunctionOptions(has_numeric_only=False, has_bool_only=True)),
    ("any", DataFrameAny, FunctionOptions(has_numeric_only=False, has_bool_only=True)),
]


@pytest.mark.parametrize("func_name,op,func_opts", reduction_functions)
def test_series_reduction(func_name, op, func_opts: FunctionOptions):
    data = pd.Series(range(20), index=[str(i) for i in range(20)])
    series = getattr(from_pandas_series(data, chunk_size=3), func_name)()

    assert isinstance(series, Tensor)
    assert isinstance(series.op, op)
    assert series.shape == ()

    series = tile(series)

    assert len(series.chunks) == 1
    assert isinstance(series.chunks[0].op, DataFrameAggregate)
    assert isinstance(series.chunks[0].inputs[0].op, DataFrameConcat)
    assert len(series.chunks[0].inputs[0].inputs) == 2

    data = pd.Series(np.random.rand(25), name="a")
    if func_opts.has_skipna:
        kwargs = dict(axis="index", skipna=False)
    else:
        kwargs = dict()
    series = getattr(from_pandas_series(data, chunk_size=7), func_name)(**kwargs)

    assert isinstance(series, Tensor)
    assert series.shape == ()

    series = tile(series)

    assert len(series.chunks) == 1
    assert isinstance(series.chunks[0].op, DataFrameAggregate)
    assert isinstance(series.chunks[0].inputs[0].op, DataFrameConcat)
    assert len(series.chunks[0].inputs[0].inputs) == 4


@pytest.mark.parametrize("func_name,op,func_opts", reduction_functions)
def test_dataframe_reduction(func_name, op, func_opts: FunctionOptions):
    data = pd.DataFrame(
        {"a": list(range(20)), "b": list(range(20, 0, -1))},
        index=[str(i) for i in range(20)],
    )
    reduction_df = getattr(from_pandas_df(data, chunk_size=3), func_name)()

    assert isinstance(reduction_df, Series)
    assert isinstance(reduction_df.op, op)
    assert isinstance(reduction_df.index_value._index_value, IndexValue.Index)
    assert reduction_df.shape == (2,)

    reduction_df = tile(reduction_df)

    assert len(reduction_df.chunks) == 1
    assert isinstance(reduction_df.chunks[0].op, DataFrameAggregate)
    assert isinstance(reduction_df.chunks[0].inputs[0].op, DataFrameConcat)
    assert len(reduction_df.chunks[0].inputs[0].inputs) == 2

    data = pd.DataFrame(np.random.rand(20, 10))
    reduction_df = getattr(from_pandas_df(data, chunk_size=3), func_name)()

    assert isinstance(reduction_df, Series)
    assert isinstance(
        reduction_df.index_value._index_value,
        (IndexValue.RangeIndex, IndexValue.Int64Index),
    )
    assert reduction_df.shape == (10,)

    reduction_df = tile(reduction_df)

    assert len(reduction_df.chunks) == 4
    assert reduction_df.nsplits == ((3, 3, 3, 1),)
    assert isinstance(reduction_df.chunks[0].op, DataFrameAggregate)
    assert isinstance(reduction_df.chunks[0].inputs[0].op, DataFrameConcat)
    assert len(reduction_df.chunks[0].inputs[0].inputs) == 2

    data = pd.DataFrame(np.random.rand(20, 20), index=[str(i) for i in range(20)])
    reduction_df = getattr(from_pandas_df(data, chunk_size=4), func_name)(
        axis="columns"
    )

    assert reduction_df.shape == (20,)

    reduction_df = tile(reduction_df)

    assert len(reduction_df.chunks) == 5
    assert reduction_df.nsplits == ((4,) * 5,)
    assert isinstance(reduction_df.chunks[0].op, DataFrameAggregate)
    assert isinstance(reduction_df.chunks[0].inputs[0].op, DataFrameConcat)
    assert len(reduction_df.chunks[0].inputs[0].inputs) == 2

    with pytest.raises(NotImplementedError):
        getattr(from_pandas_df(data, chunk_size=3), func_name)(level=0, axis=1)


cum_reduction_functions = [
    ("cummin", DataFrameCummin, FunctionOptions()),
    ("cummax", DataFrameCummax, FunctionOptions()),
    ("cumprod", DataFrameCumprod, FunctionOptions()),
    ("cumsum", DataFrameCumsum, FunctionOptions()),
]


@pytest.mark.parametrize("func_name,op,func_opts", cum_reduction_functions)
def test_cum_series_reduction(func_name, op, func_opts: FunctionOptions):
    data = pd.Series({"a": list(range(20))}, index=[str(i) for i in range(20)])
    series = getattr(from_pandas_series(data, chunk_size=3), func_name)()

    assert isinstance(series, Series)
    assert series.shape == (20,)

    series = tile(series)

    assert len(series.chunks) == 7
    assert isinstance(series.chunks[0].op, op)
    assert series.chunks[0].op.stage == OperandStage.combine
    assert isinstance(series.chunks[-1].inputs[-1].op, op)
    assert series.chunks[-1].inputs[-1].op.stage == OperandStage.map
    assert len(series.chunks[-1].inputs) == 7

    data = pd.Series(np.random.rand(25), name="a")
    if func_opts.has_skipna:
        kwargs = dict(axis="index", skipna=False)
    else:
        kwargs = dict()
    series = getattr(from_pandas_series(data, chunk_size=7), func_name)(**kwargs)

    assert isinstance(series, Series)
    assert series.shape == (25,)

    series = tile(series)

    assert len(series.chunks) == 4
    assert isinstance(series.chunks[0].op, op)
    assert series.chunks[0].op.stage == OperandStage.combine
    assert isinstance(series.chunks[-1].inputs[-1].op, op)
    assert series.chunks[-1].inputs[-1].op.stage == OperandStage.map
    assert len(series.chunks[-1].inputs) == 4


@pytest.mark.parametrize("func_name,op,func_opts", cum_reduction_functions)
def test_cum_dataframe_reduction(func_name, op, func_opts: FunctionOptions):
    data = pd.DataFrame(
        {"a": list(range(20)), "b": list(range(20, 0, -1))},
        index=[str(i) for i in range(20)],
    )
    reduction_df = getattr(from_pandas_df(data, chunk_size=3), func_name)()

    assert isinstance(reduction_df, DataFrame)
    assert isinstance(reduction_df.index_value._index_value, IndexValue.Index)
    assert reduction_df.shape == (20, 2)

    reduction_df = tile(reduction_df)

    assert len(reduction_df.chunks) == 7
    assert isinstance(reduction_df.chunks[0].op, op)
    assert reduction_df.chunks[0].op.stage == OperandStage.combine
    assert isinstance(reduction_df.chunks[-1].inputs[-1].op, op)
    assert reduction_df.chunks[-1].inputs[-1].op.stage == OperandStage.map
    assert len(reduction_df.chunks[-1].inputs) == 7

    data = pd.DataFrame(np.random.rand(20, 10))
    reduction_df = getattr(from_pandas_df(data, chunk_size=3), func_name)()

    assert isinstance(reduction_df, DataFrame)
    assert isinstance(reduction_df.index_value._index_value, IndexValue.RangeIndex)
    assert reduction_df.shape == (20, 10)

    reduction_df = tile(reduction_df)

    assert len(reduction_df.chunks) == 28
    assert reduction_df.nsplits == ((3, 3, 3, 3, 3, 3, 2), (3, 3, 3, 1))
    assert reduction_df.chunks[0].op.stage == OperandStage.combine
    assert isinstance(reduction_df.chunks[-1].inputs[-1].op, op)
    assert reduction_df.chunks[-1].inputs[-1].op.stage == OperandStage.map
    assert len(reduction_df.chunks[-1].inputs) == 7


def test_nunique():
    data = pd.DataFrame(
        np.random.randint(0, 6, size=(20, 10)),
        columns=["c" + str(i) for i in range(10)],
    )
    df = from_pandas_df(data, chunk_size=3)
    result = df.nunique()

    assert result.shape == (10,)
    assert result.op.output_types[0] == OutputType.series
    assert isinstance(result.op, DataFrameNunique)

    tiled = tile(result)
    assert tiled.shape == (10,)
    assert len(tiled.chunks) == 4
    assert tiled.nsplits == ((3, 3, 3, 1),)
    assert tiled.chunks[0].op.stage == OperandStage.agg
    assert isinstance(tiled.chunks[0].op, DataFrameAggregate)

    data2 = data.copy()
    df2 = from_pandas_df(data2, chunk_size=3)
    result2 = df2.nunique(axis=1)

    assert result2.shape == (20,)
    assert result2.op.output_types[0] == OutputType.series
    assert isinstance(result2.op, DataFrameNunique)

    tiled = tile(result2)
    assert tiled.shape == (20,)
    assert len(tiled.chunks) == 7
    assert tiled.nsplits == ((3, 3, 3, 3, 3, 3, 2),)
    assert tiled.chunks[0].op.stage == OperandStage.agg
    assert isinstance(tiled.chunks[0].op, DataFrameAggregate)


def test_dataframe_aggregate():
    data = pd.DataFrame(np.random.rand(20, 19))
    agg_funcs = [
        "sum",
        "min",
        "max",
        "mean",
        "var",
        "std",
        "all",
        "any",
        "skew",
        "kurt",
        "sem",
    ]

    df = from_pandas_df(data)
    result = tile(df.agg(agg_funcs))
    assert len(result.chunks) == 1
    assert result.shape == (len(agg_funcs), data.shape[1])
    assert list(result.columns_value.to_pandas()) == list(range(19))
    assert list(result.index_value.to_pandas()) == agg_funcs
    assert result.op.output_types[0] == OutputType.dataframe
    assert result.op.func == agg_funcs

    df = from_pandas_df(data, chunk_size=(3, 4))

    result = tile(df.agg("sum"))
    assert len(result.chunks) == 5
    assert result.shape == (data.shape[1],)
    assert list(result.index_value.to_pandas()) == list(range(data.shape[1]))
    assert result.op.output_types[0] == OutputType.series
    assert result.op.func == ["sum"]
    agg_chunk = result.chunks[0]
    assert agg_chunk.shape == (4,)
    assert list(agg_chunk.index_value.to_pandas()) == list(range(4))
    assert agg_chunk.op.stage == OperandStage.agg

    result = tile(df.agg("sum", axis=1))
    assert len(result.chunks) == 7
    assert result.shape == (data.shape[0],)
    assert list(result.index_value.to_pandas()) == list(range(data.shape[0]))
    assert result.op.output_types[0] == OutputType.series
    agg_chunk = result.chunks[0]
    assert agg_chunk.shape == (3,)
    assert list(agg_chunk.index_value.to_pandas()) == list(range(3))
    assert agg_chunk.op.stage == OperandStage.agg

    result = tile(df.agg("var", axis=1))
    assert len(result.chunks) == 7
    assert result.shape == (data.shape[0],)
    assert list(result.index_value.to_pandas()) == list(range(data.shape[0]))
    assert result.op.output_types[0] == OutputType.series
    assert result.op.func == ["var"]
    agg_chunk = result.chunks[0]
    assert agg_chunk.shape == (3,)
    assert list(agg_chunk.index_value.to_pandas()) == list(range(3))
    assert agg_chunk.op.stage == OperandStage.agg

    result = tile(df.agg(agg_funcs))
    assert len(result.chunks) == 5
    assert result.shape == (len(agg_funcs), data.shape[1])
    assert list(result.columns_value.to_pandas()) == list(range(data.shape[1]))
    assert list(result.index_value.to_pandas()) == agg_funcs
    assert result.op.output_types[0] == OutputType.dataframe
    assert result.op.func == agg_funcs
    agg_chunk = result.chunks[0]
    assert agg_chunk.shape == (len(agg_funcs), 4)
    assert list(agg_chunk.columns_value.to_pandas()) == list(range(4))
    assert list(agg_chunk.index_value.to_pandas()) == agg_funcs
    assert agg_chunk.op.stage == OperandStage.agg

    result = tile(df.agg(agg_funcs, axis=1))
    assert len(result.chunks) == 7
    assert result.shape == (data.shape[0], len(agg_funcs))
    assert list(result.columns_value.to_pandas()) == agg_funcs
    assert list(result.index_value.to_pandas()) == list(range(data.shape[0]))
    assert result.op.output_types[0] == OutputType.dataframe
    assert result.op.func == agg_funcs
    agg_chunk = result.chunks[0]
    assert agg_chunk.shape == (3, len(agg_funcs))
    assert list(agg_chunk.columns_value.to_pandas()) == agg_funcs
    assert list(agg_chunk.index_value.to_pandas()) == list(range(3))
    assert agg_chunk.op.stage == OperandStage.agg

    dict_fun = {0: "sum", 2: ["var", "max"], 9: ["mean", "var", "std"]}
    all_cols = set(
        reduce(
            operator.add, [[v] if isinstance(v, str) else v for v in dict_fun.values()]
        )
    )
    result = tile(df.agg(dict_fun))
    assert len(result.chunks) == 2
    assert result.shape == (len(all_cols), len(dict_fun))
    assert set(result.columns_value.to_pandas()) == set(dict_fun.keys())
    assert set(result.index_value.to_pandas()) == all_cols
    assert result.op.output_types[0] == OutputType.dataframe
    assert result.op.func[0] == [dict_fun[0]]
    assert result.op.func[2] == dict_fun[2]
    agg_chunk = result.chunks[0]
    assert agg_chunk.shape == (len(all_cols), 2)
    assert list(agg_chunk.columns_value.to_pandas()) == [0, 2]
    assert set(agg_chunk.index_value.to_pandas()) == all_cols
    assert agg_chunk.op.stage == OperandStage.agg

    with pytest.raises(TypeError):
        df.agg(sum_0="sum", mean_0="mean")
    with pytest.raises(NotImplementedError):
        df.agg({0: ["sum", "min", "var"], 9: ["mean", "var", "std"]}, axis=1)


def test_series_aggregate():
    data = pd.Series(np.random.rand(20), index=[str(i) for i in range(20)], name="a")
    agg_funcs = [
        "sum",
        "min",
        "max",
        "mean",
        "var",
        "std",
        "all",
        "any",
        "skew",
        "kurt",
        "sem",
    ]

    series = from_pandas_series(data)

    result = tile(series.agg(agg_funcs))
    assert len(result.chunks) == 1
    assert result.shape == (len(agg_funcs),)
    assert list(result.index_value.to_pandas()) == agg_funcs
    assert result.op.output_types[0] == OutputType.series
    assert result.op.func == agg_funcs

    series = from_pandas_series(data, chunk_size=3)

    result = tile(series.agg("sum"))
    assert len(result.chunks) == 1
    assert result.shape == ()
    assert result.op.output_types[0] == OutputType.scalar
    agg_chunk = result.chunks[0]
    assert agg_chunk.shape == ()
    assert agg_chunk.op.stage == OperandStage.agg

    result = tile(series.agg(agg_funcs))
    assert len(result.chunks) == 1
    assert result.shape == (len(agg_funcs),)
    assert list(result.index_value.to_pandas()) == agg_funcs
    assert result.op.output_types[0] == OutputType.series
    assert result.op.func == agg_funcs
    agg_chunk = result.chunks[0]
    assert agg_chunk.shape == (len(agg_funcs),)
    assert list(agg_chunk.index_value.to_pandas()) == agg_funcs
    assert agg_chunk.op.stage == OperandStage.agg

    with pytest.raises(TypeError):
        series.agg(sum_0=(0, "sum"), mean_0=(0, "mean"))


def test_compile_function():
    compiler = ReductionCompiler()
    ms = md.Series([1, 2, 3])
    # no Mars objects inside closures
    with pytest.raises(ValueError):
        compiler.add_function(functools.partial(lambda x: (x + ms).sum()), ndim=2)
    # function should return a Mars object
    with pytest.raises(ValueError):
        compiler.add_function(lambda x: x is not None, ndim=2)
    # function should perform some sort of reduction in dimensionality
    with pytest.raises(ValueError):
        compiler.add_function(lambda x: x, ndim=2)
    # function should only contain acceptable operands
    with pytest.raises(ValueError):
        compiler.add_function(lambda x: x.sort_values().max(), ndim=1)
    with pytest.raises(ValueError):
        compiler.add_function(lambda x: x.max().shift(1), ndim=2)

    # test agg for all data
    for ndim in [1, 2]:
        compiler = ReductionCompiler(store_source=True)
        compiler.add_function(lambda x: (x ** 2).count() + 1, ndim=ndim)
        result = compiler.compile()
        # check pre_funcs
        assert len(result.pre_funcs) == 1
        assert "pow" in result.pre_funcs[0].func.__source__
        # check agg_funcs
        assert len(result.agg_funcs) == 1
        assert result.agg_funcs[0].map_func_name == "count"
        assert result.agg_funcs[0].agg_func_name == "sum"
        # check post_funcs
        assert len(result.post_funcs) == 1
        assert result.post_funcs[0].func_name == "<lambda>"
        assert "add" in result.post_funcs[0].func.__source__

        compiler.add_function(
            lambda x: -x.prod() ** 2 + (1 + (x ** 2).count()), ndim=ndim
        )
        result = compiler.compile()
        # check pre_funcs
        assert len(result.pre_funcs) == 2
        assert (
            "pow" in result.pre_funcs[0].func.__source__
            or "pow" in result.pre_funcs[1].func.__source__
        )
        assert (
            "pow" not in result.pre_funcs[0].func.__source__
            or "pow" not in result.pre_funcs[1].func.__source__
        )
        # check agg_funcs
        assert len(result.agg_funcs) == 2
        assert set(result.agg_funcs[i].map_func_name for i in range(2)) == {
            "count",
            "prod",
        }
        assert set(result.agg_funcs[i].agg_func_name for i in range(2)) == {
            "sum",
            "prod",
        }
        # check post_funcs
        assert len(result.post_funcs) == 2
        assert result.post_funcs[0].func_name == "<lambda_0>"
        assert "add" in result.post_funcs[0].func.__source__
        assert "add" in result.post_funcs[1].func.__source__

        compiler = ReductionCompiler(store_source=True)
        compiler.add_function(
            lambda x: where_function(x.all(), x.count(), 0), ndim=ndim
        )
        result = compiler.compile()
        # check pre_funcs
        assert len(result.pre_funcs) == 1
        assert result.pre_funcs[0].input_key == result.pre_funcs[0].output_key
        # check agg_funcs
        assert len(result.agg_funcs) == 2
        assert set(result.agg_funcs[i].map_func_name for i in range(2)) == {
            "all",
            "count",
        }
        assert set(result.agg_funcs[i].agg_func_name for i in range(2)) == {
            "sum",
            "all",
        }
        # check post_funcs
        assert len(result.post_funcs) == 1
        if ndim == 1:
            assert "np.where" in result.post_funcs[0].func.__source__
        else:
            assert "np.where" not in result.post_funcs[0].func.__source__
            assert ".where" in result.post_funcs[0].func.__source__

        # check boolean expressions
        compiler = ReductionCompiler(store_source=True)
        compiler.add_function(lambda x: (x == "1").sum(), ndim=ndim)
        result = compiler.compile()
        # check pre_funcs
        assert len(result.pre_funcs) == 1
        assert "eq" in result.pre_funcs[0].func.__source__
        # check agg_funcs
        assert len(result.agg_funcs) == 1
        assert result.agg_funcs[0].map_func_name == "sum"
        assert result.agg_funcs[0].agg_func_name == "sum"

    # test agg for specific columns
    compiler = ReductionCompiler(store_source=True)
    compiler.add_function(lambda x: 1 + x.sum(), ndim=2, cols=["a", "b"])
    compiler.add_function(lambda x: -1 + x.sum(), ndim=2, cols=["b", "c"])
    result = compiler.compile()
    # check pre_funcs
    assert len(result.pre_funcs) == 1
    assert set(result.pre_funcs[0].columns) == set("abc")
    # check agg_funcs
    assert len(result.agg_funcs) == 1
    assert result.agg_funcs[0].map_func_name == "sum"
    assert result.agg_funcs[0].agg_func_name == "sum"
    # check post_funcs
    assert len(result.post_funcs) == 2
    assert set("".join(sorted(result.post_funcs[i].columns)) for i in range(2)) == {
        "ab",
        "bc",
    }

    # test agg for multiple columns
    compiler = ReductionCompiler(store_source=True)
    compiler.add_function(lambda x: x.sum(), ndim=2, cols=["a"])
    compiler.add_function(lambda x: x.sum(), ndim=2, cols=["b"])
    compiler.add_function(lambda x: x.min(), ndim=2, cols=["c"])
    result = compiler.compile()
    # check pre_funcs
    assert len(result.pre_funcs) == 1
    assert set(result.pre_funcs[0].columns) == set("abc")
    # check agg_funcs
    assert len(result.agg_funcs) == 2
    assert result.agg_funcs[0].map_func_name == "sum"
    assert result.agg_funcs[0].agg_func_name == "sum"
    # check post_funcs
    assert len(result.post_funcs) == 2
    assert set(result.post_funcs[0].columns) == set("ab")


def test_custom_aggregation():
    class MockReduction1(CustomReduction):
        def agg(self, v1):
            return v1.sum()

    class MockReduction2(CustomReduction):
        def pre(self, value):
            return value + 1, value ** 2

        def agg(self, v1, v2):
            return v1.sum(), v2.prod()

        def post(self, v1, v2):
            return v1 + v2

    for ndim in [1, 2]:
        compiler = ReductionCompiler()
        compiler.add_function(MockReduction1(), ndim=ndim)
        result = compiler.compile()
        # check agg_funcs
        assert len(result.agg_funcs) == 1
        assert result.agg_funcs[0].map_func_name == "custom_reduction"
        assert result.agg_funcs[0].agg_func_name == "custom_reduction"
        assert isinstance(result.agg_funcs[0].custom_reduction, MockReduction1)
        assert result.agg_funcs[0].output_limit == 1

        compiler = ReductionCompiler()
        compiler.add_function(MockReduction2(), ndim=ndim)
        result = compiler.compile()
        # check agg_funcs
        assert len(result.agg_funcs) == 1
        assert result.agg_funcs[0].map_func_name == "custom_reduction"
        assert result.agg_funcs[0].agg_func_name == "custom_reduction"
        assert isinstance(result.agg_funcs[0].custom_reduction, MockReduction2)
        assert result.agg_funcs[0].output_limit == 2
