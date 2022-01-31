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

from ..... import dataframe as md
from ..... import execute, fetch
from .....core import enter_mode, TileableGraph, TileableGraphBuilder
from .....dataframe.base.eval import DataFrameEval
from .. import optimize


@enter_mode(build=True)
def test_arithmetic_query(setup):
    raw = pd.DataFrame(np.random.rand(100, 10), columns=list("ABCDEFGHIJ"))
    raw2 = pd.DataFrame(np.random.rand(100, 5), columns=list("ABCDE"))

    # does not support heterogeneous sources
    df1 = md.DataFrame(raw, chunk_size=10)
    df2 = md.DataFrame(raw2, chunk_size=10)
    df3 = -(df1["A"] + df2["B"])
    graph = TileableGraph([df3.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    assert records.get_optimization_result(df3.data) is None

    # does not support customized args in arithmetic
    df1 = md.DataFrame(raw, chunk_size=10)
    df3 = (-df1["A"]).add(df1["B"], fill_value=0.0)
    graph = TileableGraph([df3.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    assert records.get_optimization_result(df3.data) is None

    # does not support non-string headers
    df1 = md.DataFrame(np.random.rand(100, 5))
    df2 = df1[0] + df1[1]
    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    assert records.get_optimization_result(df2.data) is None

    df1 = md.DataFrame(raw, chunk_size=10)
    df2 = -df1["A"] + df1["B"] * 5
    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2.op.expr == "(-(`A`)) + ((`B`) * (5))"

    pd.testing.assert_series_equal(df2.execute().fetch(), -raw["A"] + raw["B"] * 5)

    raw = pd.DataFrame(np.random.rand(100, 10), columns=list("ABCDEFGHIJ"))
    df1 = md.DataFrame(raw, chunk_size=10)
    df2 = -df1["A"] + df1["B"] * 5 + 3 * df1["C"]
    graph = TileableGraph([df1["A"].data, df2.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2.op.expr == "((-(`A`)) + ((`B`) * (5))) + ((3) * (`C`))"

    r_df2, _r_col_a = fetch(execute(df2, df1["A"]))
    pd.testing.assert_series_equal(r_df2, -raw["A"] + raw["B"] * 5 + 3 * raw["C"])


@enter_mode(build=True)
def test_bool_eval_to_query(setup):
    raw = pd.DataFrame(np.random.rand(100, 10), columns=list("ABCDEFGHIJ"))

    # does not support non-eval inputs
    df1 = md.DataFrame(raw, chunk_size=10)
    df2 = df1[(df1["A"] * 5).astype(bool)]
    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    assert records.get_optimization_result(df2.data) is None

    df1 = md.DataFrame(raw, chunk_size=10)
    df2 = df1[(df1["A"] > 0.5) & (df1["C"] < 0.5)]
    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df2 = records.get_optimization_result(df2.data)
    assert isinstance(opt_df2.op, DataFrameEval)
    assert opt_df2.op.is_query
    assert opt_df2.op.expr == "((`A`) > (0.5)) & ((`C`) < (0.5))"

    pd.testing.assert_frame_equal(
        df2.execute().fetch(), raw[(raw["A"] > 0.5) & (raw["C"] < 0.5)]
    )

    raw = pd.DataFrame(np.random.rand(100, 10), columns=list("ABCDEFGHIJ"))
    df1 = md.DataFrame(raw, chunk_size=10)
    df2 = df1[(df1["A"] > 0.5) & (df1["C"] < 0.5)] + 1
    assert isinstance(opt_df2.op, DataFrameEval)
    assert opt_df2.op.is_query

    r_df2, _r_col_a = fetch(execute(df2, df1["A"]))
    pd.testing.assert_frame_equal(r_df2, raw[(raw["A"] > 0.5) & (raw["C"] < 0.5)] + 1)

    raw = pd.DataFrame(
        {
            "a": np.arange(100),
            "b": [pd.Timestamp("2022-1-1") + pd.Timedelta(days=i) for i in range(100)],
        }
    )
    df1 = md.DataFrame(raw, chunk_size=10)
    df2 = df1[df1.b < pd.Timestamp("2022-3-20")]
    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2.op.expr == "(`b`) < (Timestamp('2022-03-20 00:00:00'))"

    r_df2 = fetch(execute(df2))
    pd.testing.assert_frame_equal(r_df2, raw[raw.b < pd.Timestamp("2022-3-20")])
