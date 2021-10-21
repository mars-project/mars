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
import tempfile

import pandas as pd
import numpy as np
import pytest

from ..... import dataframe as md
from .....core import TileableGraph, TileableGraphBuilder, enter_mode
from .....dataframe.indexing.iloc import DataFrameIlocGetItem, SeriesIlocGetItem
from .. import optimize


@pytest.fixture(scope="module")
def prepare_data():
    rs = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "a": rs.randint(10, size=100),
            "b": rs.rand(100),
            "c": rs.choice(list("abc"), size=100),
        }
    )

    with tempfile.TemporaryDirectory() as tempdir:
        yield tempdir, df


def _execute_iloc(*_):  # pragma: no cover
    raise ValueError("cannot run iloc")


_iloc_operand_executors = {
    DataFrameIlocGetItem: _execute_iloc,
    SeriesIlocGetItem: _execute_iloc,
}


@enter_mode(build=True)
def test_read_csv_head(prepare_data, setup):
    tempdir, pdf = prepare_data
    file_path = os.path.join(tempdir, "test.csv")
    pdf.to_csv(file_path, index=False)

    size = os.stat(file_path).st_size / 2
    df1 = md.read_csv(file_path, chunk_bytes=size)
    df2 = df1.head(5)
    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    assert records.get_optimization_result(df1.data) is None
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2.op.nrows == 5
    assert len(graph) == 1
    assert opt_df2 in graph.results

    result = df2.execute(
        extra_config={"operand_executors": _iloc_operand_executors}
    ).fetch()
    expected = pdf.head(5)
    pd.testing.assert_frame_equal(result, expected)

    # test multiple head
    df3 = df1.head(10)
    graph = TileableGraph([df2.data, df3.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df1 = records.get_optimization_result(df1.data)
    assert opt_df1 is not None
    assert opt_df1.op.nrows == 10
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2 is not None
    assert graph.predecessors(opt_df2)[0] is opt_df1
    assert opt_df2.inputs[0] is opt_df1
    opt_df3 = records.get_optimization_result(df3.data)
    assert opt_df3 is not None
    assert graph.predecessors(opt_df3)[0] is opt_df1
    assert opt_df3.inputs[0] is opt_df1

    # test head with successor
    df1 = md.read_csv(file_path, chunk_bytes=size)
    df2 = df1.head(5)
    df3 = df2 + 1
    graph = TileableGraph([df3.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    assert records.get_optimization_result(df1.data) is None
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2.op.nrows == 5
    assert len(graph) == 2


@enter_mode(build=True)
def test_read_parquet_head(prepare_data, setup):
    tempdir, pdf = prepare_data
    dirname = os.path.join(tempdir, "test_parquet")
    os.makedirs(dirname)
    for i in range(3):
        file_path = os.path.join(dirname, f"test{i}.parquet")
        pdf[i * 40 : (i + 1) * 40].to_parquet(file_path, index=False)

    df1 = md.read_parquet(dirname)
    df2 = df1.head(5)
    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    assert records.get_optimization_result(df1.data) is None
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2.op.nrows == 5
    assert len(graph) == 1
    assert opt_df2 in graph.results

    result = df2.execute(
        extra_config={"operand_executors": _iloc_operand_executors}
    ).fetch()
    expected = pdf.head(5)
    pd.testing.assert_frame_equal(result, expected)


@enter_mode(build=True)
def test_sort_head(prepare_data, setup):
    _, pdf = prepare_data

    df1 = md.DataFrame(pdf, chunk_size=20)
    df1 = df1.sort_values(by="b")
    df2 = df1.head(10)
    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    assert records.get_optimization_result(df1.data) is None
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2.op.nrows == 10
    assert len(graph) == 2
    assert opt_df2 in graph.results

    result = df2.execute(
        extra_config={"operand_executors": _iloc_operand_executors}
    ).fetch()
    expected = pdf.sort_values(by="b").head(10)
    pd.testing.assert_frame_equal(result, expected)

    pdf2 = pdf.copy()
    pdf2.set_index("b", inplace=True)
    df1 = md.DataFrame(pdf2, chunk_size=20)
    df1 = df1.sort_index()
    df2 = df1.head(10)
    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    assert records.get_optimization_result(df1.data) is None
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2.op.nrows == 10
    assert len(graph) == 2
    assert opt_df2 in graph.results

    result = df2.execute(
        extra_config={"operand_executors": _iloc_operand_executors}
    ).fetch()
    expected = pdf2.sort_index().head(10)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("chunk_size", [5, 10])
@enter_mode(build=True)
def test_value_counts_head(prepare_data, setup, chunk_size):
    _, pdf = prepare_data
    df = md.DataFrame(pdf, chunk_size=chunk_size)

    df1 = df["a"].value_counts(method="tree")
    df2 = df1.head(3)
    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    assert records.get_optimization_result(df1.data) is None
    opt_df2 = records.get_optimization_result(df2.data)
    assert opt_df2.op.nrows == 3
    assert len(graph) == 3
    assert opt_df2 in graph.results

    result = df2.execute(
        extra_config={"operand_executors": _iloc_operand_executors}
    ).fetch()
    expected = pdf["a"].value_counts().head(3)
    pd.testing.assert_series_equal(result, expected)


@enter_mode(build=True)
def test_no_head(prepare_data):
    tempdir, pdf = prepare_data
    file_path = os.path.join(tempdir, "test.csv")
    pdf.to_csv(file_path, index=False)

    size = os.stat(file_path).st_size / 2
    df1 = md.read_csv(file_path, chunk_bytes=size)
    df2 = df1.iloc[1:10]

    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    assert records.get_optimization_result(df1.data) is None
    assert records.get_optimization_result(df2.data) is None

    df2 = df1.head(3)
    df3 = df1 + 1

    graph = TileableGraph([df2.data, df3.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    assert records.get_optimization_result(df1.data) is None
    assert records.get_optimization_result(df2.data) is None
    assert records.get_optimization_result(df3.data) is None
