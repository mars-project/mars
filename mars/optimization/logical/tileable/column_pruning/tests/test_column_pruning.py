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

import os
import tempfile

import pandas as pd
import pytest

from ...... import dataframe as md
from ...... import tensor as mt
from ......dataframe.arithmetic import DataFrameMul
from ......dataframe.base.eval import DataFrameEval
from ......dataframe.base.isin import DataFrameIsin
from ......dataframe.core import DataFrameData, SeriesData, DataFrameGroupByData
from ......dataframe.datasource.dataframe import DataFrameDataSource
from ......dataframe.datasource.read_csv import DataFrameReadCSV
from ......dataframe.datasource.read_parquet import DataFrameReadParquet
from ......dataframe.groupby.aggregation import DataFrameGroupByAgg
from ......dataframe.groupby.core import DataFrameGroupByOperand
from ......dataframe.indexing.getitem import DataFrameIndex
from ......dataframe.indexing.setitem import DataFrameSetitem
from ......dataframe.merge import DataFrameMerge
from ......optimization.logical.tileable import optimize
from ......tensor.core import TensorData
from ......tensor.datasource import ArrayDataSource


@pytest.fixture()
def gen_data1():
    with tempfile.TemporaryDirectory() as tempdir:
        df = pd.DataFrame(
            {
                "c1": [3, 4, 5, 3, 5, 4, 1, 2, 3],
                "c2": [1, 3, 4, 5, 6, 5, 4, 4, 4],
                "c3": list("aabaaddce"),
                "c4": list("abaaaddce"),
            }
        )

        df2 = pd.DataFrame(
            {
                "c1": [3, 3, 4, 5, 6, 5, 4, 4, 4],
                "c2": [1, 3, 4, 5, 6, 5, 4, 4, 4],
                "c3": list("aabaaddce"),
                "c4": list("abaaaddce"),
            }
        )
        file_path = os.path.join(tempdir, "test.csv")
        file_path2 = os.path.join(tempdir, "test2.csv")

        df.to_csv(file_path, index=False)
        df2.to_csv(file_path2, index=False)
        yield file_path, file_path2


@pytest.fixture()
def gen_data2():
    with tempfile.TemporaryDirectory() as tempdir:
        df = pd.DataFrame(
            {
                "c1": [3, 4, 5, 3, 5, 4, 1, 2, 3],
                "c2": [1, 3, 4, 5, 6, 5, 4, 4, 4],
                "c3": [1, 3, 4, 1, 1, 9, 4, 4, 4],
                "c4": [3, 0, 5, 3, 5, 4, 1, 2, 10],
            }
        )

        df2 = pd.DataFrame(
            {
                "cc1": [3, 4, 5, 3, 5, 4, 1, 2, 3],
                "cc2": [1, 6, 4, 5, 6, 5, 4, 4, 4],
                "cc3": [1, 3, 4, 1, 1, 9, 4, 8, 4],
                "cc4": [3, 0, 5, 3, 5, 4, 1, 2, 10],
            }
        )

        file_path = os.path.join(tempdir, "test.pq")
        file_path2 = os.path.join(tempdir, "test2.pq")
        df.to_parquet(file_path)
        df2.to_parquet(file_path2)
        yield file_path, file_path2


def test_groupby(setup, gen_data2):
    # no column pruning
    file_path, file_path2 = gen_data2
    df1 = md.read_parquet(file_path)
    df2 = md.read_parquet(file_path2)
    m = df1.merge(df2, left_on="c1", right_on="cc1")
    g = m.groupby(["c1"])

    graph = g.build_graph()
    optimize(graph)

    assert len(graph.result_tileables) == 1
    groupby_data = graph.result_tileables[0]
    assert isinstance(groupby_data, DataFrameGroupByData)
    assert isinstance(groupby_data.op, DataFrameGroupByOperand)
    assert len(groupby_data.dtypes) == 8

    assert len(groupby_data.inputs) == 1
    merge_data = groupby_data.inputs[0]
    assert isinstance(merge_data, DataFrameData)
    assert isinstance(merge_data.op, DataFrameMerge)
    assert len(groupby_data.dtypes) == 8

    assert len(merge_data.inputs) == 2
    left_data = merge_data.inputs[0]
    right_data = merge_data.inputs[1]
    assert isinstance(left_data, DataFrameData)
    assert isinstance(left_data.op, DataFrameReadParquet)
    assert len(left_data.dtypes) == 4
    assert isinstance(right_data, DataFrameData)
    assert isinstance(right_data.op, DataFrameReadParquet)
    assert len(right_data.dtypes) == 4


def test_tensor(setup):
    t = mt.tensor((1, 2, 3))
    s = md.DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)}).isin(t)

    graph = s.build_graph()
    optimize(graph)

    assert len(graph.result_tileables) == 1
    isin_data = graph.result_tileables[0]
    assert isinstance(isin_data, DataFrameData)
    assert isinstance(isin_data.op, DataFrameIsin)
    assert len(isin_data.dtypes) == 2

    assert len(isin_data.inputs) == 2
    df_data = isin_data.inputs[0]
    assert isinstance(df_data, DataFrameData)
    assert isinstance(df_data.op, DataFrameDataSource)
    assert len(df_data.dtypes) == 2

    tensor_data = isin_data.inputs[1]
    assert isinstance(tensor_data, TensorData)
    assert isinstance(tensor_data.op, ArrayDataSource)


def test_groupby_agg(setup, gen_data1):
    file_path, _ = gen_data1

    df1 = md.read_csv(file_path)
    c = df1.groupby("c1")["c2"].sum()

    graph = c.build_graph()
    optimize(graph)
    groupby_agg_node = graph.result_tileables[0]
    assert isinstance(groupby_agg_node, SeriesData)
    assert isinstance(groupby_agg_node.op, DataFrameGroupByAgg)
    assert type(groupby_agg_node.op) is DataFrameGroupByAgg
    assert groupby_agg_node.name == "c2"

    groupby_agg_node_preds = graph.predecessors(groupby_agg_node)
    assert len(groupby_agg_node_preds) == 1
    read_csv_node = groupby_agg_node_preds[0]
    assert isinstance(read_csv_node, DataFrameData)
    assert isinstance(read_csv_node.op, DataFrameReadCSV)
    assert len(read_csv_node.op.usecols) == 2
    assert len({"c1", "c2"} ^ set(read_csv_node.op.usecols)) == 0

    raw = pd.read_csv(file_path)
    pd_res = raw.groupby("c1")["c2"].sum()
    r = c.execute().fetch()
    pd.testing.assert_series_equal(r, pd_res)


def test_merge_and_getitem(setup, gen_data1):
    file_path, file_path2 = gen_data1
    df1 = md.read_csv(file_path)
    df2 = md.read_csv(file_path2, names=["c1", "c2", "cc3", "cc4"], header=0)
    r = df1.merge(df2)["c1"]

    graph = r.build_graph()
    optimize(graph)

    index_node = graph.result_tileables[0]
    assert isinstance(index_node.op, DataFrameIndex)
    assert index_node.name == "c1"

    assert len(graph.predecessors(index_node)) == 1
    merge_node = graph.predecessors(index_node)[0]
    assert type(merge_node.op) is DataFrameMerge

    read_csv_node_left, read_csv_node_right = graph.predecessors(merge_node)
    assert type(read_csv_node_left.op) is DataFrameReadCSV
    assert type(read_csv_node_right.op) is DataFrameReadCSV
    assert len(read_csv_node_left.op.usecols) == 2
    assert len(read_csv_node_right.op.usecols) == 2
    assert set(read_csv_node_left.op.usecols) == {"c1", "c2"}
    assert set(read_csv_node_right.op.usecols) == {"c1", "c2"}

    r = r.execute().fetch()
    raw1 = pd.read_csv(file_path)
    raw2 = pd.read_csv(file_path2, names=["c1", "c2", "cc3", "cc4"], header=0)
    expected = raw1.merge(raw2)["c1"]
    pd.testing.assert_series_equal(r, expected)


def test_merge_on_one_column(setup, gen_data1):
    file_path, file_path2 = gen_data1
    df1 = md.read_csv(file_path)
    df2 = md.read_csv(file_path2)
    c = df1.merge(df2, left_on="c1", right_on="c1")["c1"]

    graph = c.build_graph()
    optimize(graph)

    index_node = graph.result_tileables[0]
    assert type(index_node.op) is DataFrameIndex

    index_node_preds = graph.predecessors(index_node)
    assert len(index_node_preds) == 1

    merge_node = index_node_preds[0]
    assert type(merge_node.op) is DataFrameMerge

    merge_node_preds = graph.predecessors(merge_node)
    assert len(merge_node_preds) == 2

    read_csv_node = merge_node_preds[0]
    read_csv_op = read_csv_node.op
    assert type(read_csv_op) is DataFrameReadCSV
    assert len(read_csv_op.usecols) == 1
    assert read_csv_op.usecols == ["c1"]

    r = c.execute().fetch()
    raw1 = pd.read_csv(file_path)
    raw2 = pd.read_csv(file_path2)
    expected = raw1.merge(raw2, left_on="c1", right_on="c1")["c1"]
    pd.testing.assert_series_equal(r, expected)


def test_merge_on_two_columns(setup, gen_data1):
    file_path, file_path2 = gen_data1
    df1 = md.read_csv(file_path)
    df2 = md.read_csv(file_path2)
    c = df1.merge(df2, left_on=["c1", "c2"], right_on=["c1", "c2"])[["c1", "c2"]]

    graph = c.build_graph()
    optimize(graph)

    index_node = graph.result_tileables[0]
    assert type(index_node.op) is DataFrameIndex
    assert len(index_node.op.col_names) == 2

    merge_node = graph.predecessors(index_node)[0]
    read_csv_node = graph.predecessors(merge_node)[0]
    assert type(read_csv_node.op) is DataFrameReadCSV

    use_cols = read_csv_node.op.usecols
    assert len(use_cols) == 2
    assert set(use_cols) & {"c1", "c2"} == {"c1", "c2"}
    assert len(set(use_cols) ^ {"c1", "c2"}) == 0

    r = c.execute().fetch()
    raw1 = pd.read_csv(file_path)
    raw2 = pd.read_csv(file_path2)
    expected = raw1.merge(raw2, left_on=["c1", "c2"], right_on=["c1", "c2"])[
        ["c1", "c2"]
    ]
    pd.testing.assert_frame_equal(r, expected)


def test_groupby_agg_then_merge(setup, gen_data1):
    file_path, file_path2 = gen_data1
    df1 = md.read_csv(file_path)
    df2 = md.read_csv(file_path2)
    r_group_res = df1.groupby(["c1"])[["c2"]].sum()
    c = df2.merge(r_group_res, left_on=["c2"], right_on=["c2"])[["c1", "c3"]]
    graph = c.build_graph()
    optimize(graph)
    r = c.execute().fetch()

    raw1 = pd.read_csv(file_path)
    raw2 = pd.read_csv(file_path2)
    group_res = raw1.groupby(["c1"])[["c2"]].sum()
    expected = raw2.merge(group_res, left_on=["c2"], right_on=["c2"])[["c1", "c3"]]
    pd.testing.assert_frame_equal(r, expected)

    index_node = graph.result_tileables[0]
    assert type(index_node.op) is DataFrameIndex

    merge_node = graph.predecessors(index_node)[0]
    merge_node_preds = graph.predecessors(merge_node)

    df2_node = [n for n in merge_node_preds if type(n.op) is DataFrameReadCSV][0]
    assert set(df2_node.op.usecols) == {"c1", "c2", "c3"}

    df1_node = [
        n
        for n in graph._nodes
        if type(n.op) is DataFrameReadCSV and n.op.path == file_path
    ][0]
    assert type(df1_node.op) is DataFrameReadCSV
    assert set(df1_node.op.usecols) == {"c1", "c2"}


def test_merge_then_groupby_apply(setup, gen_data2):
    file_path, file_path2 = gen_data2
    df1 = md.read_parquet(file_path)
    df2 = md.read_parquet(file_path2)

    c = (
        (
            ((df1 + 1) * 2).merge(df2, left_on=["c1", "c3"], right_on=["cc2", "cc4"])[
                ["c1", "cc4"]
            ]
            * 2
        )
        .groupby(["cc4"])
        .apply(lambda x: x / x.sum())
    )
    graph = c.build_graph()
    optimize(graph)
    r = c.execute().fetch()

    raw1 = pd.read_parquet(file_path)
    raw2 = pd.read_parquet(file_path2)
    expected = (
        (
            ((raw1 + 1) * 2).merge(raw2, left_on=["c1", "c3"], right_on=["cc2", "cc4"])[
                ["c1", "cc4"]
            ]
            * 2
        )
        .groupby(["cc4"])
        .apply(lambda x: x / x.sum())
    )
    pd.testing.assert_frame_equal(r, expected)

    read_parquet_nodes = [n for n in graph._nodes if type(n.op) is DataFrameReadParquet]
    assert len(read_parquet_nodes) == 2

    for n in read_parquet_nodes:
        assert len(n.op.get_columns()) == 2

    merge_node = [n for n in graph._nodes if type(n.op) is DataFrameMerge][0]
    merge_node_preds = graph.predecessors(merge_node)
    assert len(merge_node_preds) == 2

    inserted_node = [n for n in merge_node_preds if type(n.op) is DataFrameIndex][0]
    assert len(inserted_node.op.col_names) == 2
    assert set(inserted_node.op.col_names) == {"c1", "c3"}

    mul_node = graph.predecessors(inserted_node)[0]
    assert type(mul_node.op) is DataFrameMul
    assert set(mul_node.dtypes.index.tolist()) == {"c1", "c3"}


def test_two_merges(setup, gen_data2):
    file_path, file_path2 = gen_data2
    df1 = md.read_parquet(file_path)
    df2 = md.read_parquet(file_path2)
    c = (
        (df1 + 1)
        .merge((df2 + 2), left_on=["c2", "c3"], right_on=["cc1", "cc4"])[
            ["c2", "c4", "cc1", "cc2"]
        ]
        .merge(df2, left_on=["cc1"], right_on=["cc3"])
    )
    graph = c.build_graph()
    optimize(graph)
    r = c.execute().fetch()

    raw1 = pd.read_parquet(file_path)
    raw2 = pd.read_parquet(file_path2)

    expected = (
        (raw1 + 1)
        .merge((raw2 + 2), left_on=["c2", "c3"], right_on=["cc1", "cc4"])[
            ["c2", "c4", "cc1", "cc2"]
        ]
        .merge(raw2, left_on=["cc1"], right_on=["cc3"])
    )
    pd.testing.assert_frame_equal(r, expected)

    parquet_nodes = [n for n in graph._nodes if type(n.op) is DataFrameReadParquet]
    assert len(parquet_nodes) == 2

    # df1 read parquet push down
    df1_node = [n for n in parquet_nodes if n.op.path == file_path][0]
    assert set(df1_node.op.get_columns()) == {"c2", "c3", "c4"}

    # df2 read parquet not push down since it needs all the columns
    df2_node = [n for n in parquet_nodes if n.op.path == file_path2][0]
    assert df2_node.op.columns is None

    # prove that inserted nodes take effect
    inserted_nodes = [n for n in graph._nodes if type(n.op) is DataFrameIndex]
    assert len(inserted_nodes) == 3

    index_after_merge_node = [
        n for n in inserted_nodes if type(graph.predecessors(n)[0].op) is DataFrameMerge
    ][0]
    assert set(index_after_merge_node.op.col_names) == {"c2", "c4", "cc1", "cc2"}


def test_two_groupby_aggs_with_multi_index(setup, gen_data2):
    file_path, _ = gen_data2
    df = md.read_parquet(file_path)
    c = (
        (df * 2)
        .groupby(["c2", "c3"])
        .apply(lambda x: x["c1"].sum() / x["c2"].mean())
        .reset_index()
        .groupby("c3")
        .agg(["min", "max"])
    )
    graph = c.build_graph()
    optimize(graph)
    r = c.execute().fetch()

    raw = pd.read_parquet(file_path)
    expected = (
        (raw * 2)
        .groupby(["c2", "c3"])
        .apply(lambda x: x["c1"].sum() / x["c2"].mean())
        .reset_index()
        .groupby("c3")
        .agg(["min", "max"])
    )
    pd.testing.assert_frame_equal(r, expected)

    apply_node = [n for n in graph._nodes if type(n.op) is DataFrameGroupByAgg][0]
    assert set(apply_node.columns.index_value._index_value._data) == {
        (0, "min"),
        (0, "max"),
        ("c2", "max"),
        ("c2", "min"),
    }

    # apply cannot push down
    read_parquet_node = [
        n
        for n in graph._nodes
        if type(n.op) is DataFrameReadParquet and n.op.path == file_path
    ][0]
    assert read_parquet_node.op.get_columns() is None


def test_merge_and_get_col_with_suffix(setup, gen_data1):
    file_path, file_path2 = gen_data1
    left = md.read_csv(file_path)
    right = md.read_csv(file_path2)
    r = left.merge(right, on="c1")[["c3_x"]]

    graph = r.build_graph()
    optimize(graph)

    index_node = graph.result_tileables[0]
    assert isinstance(index_node.op, DataFrameIndex)
    assert index_node.op.col_names == ["c3_x"]

    assert len(graph.predecessors(index_node)) == 1
    merge_node = graph.predecessors(index_node)[0]
    assert type(merge_node.op) is DataFrameMerge

    read_csv_node_left, read_csv_node_right = graph.predecessors(merge_node)
    assert type(read_csv_node_left.op) is DataFrameReadCSV
    assert type(read_csv_node_right.op) is DataFrameReadCSV
    assert len(read_csv_node_left.op.usecols) == 2
    assert len(read_csv_node_right.op.usecols) == 2
    assert set(read_csv_node_left.op.usecols) == {"c1", "c3"}
    assert set(read_csv_node_right.op.usecols) == {"c1", "c3"}

    r = r.execute().fetch()
    raw1 = pd.read_csv(file_path)
    raw2 = pd.read_csv(file_path2)
    expected = raw1.merge(raw2, on="c1")[["c3_x"]]
    pd.testing.assert_frame_equal(r, expected)


def test_getitem_with_mask(setup, gen_data1):
    """
    Getitem with mask shouldn't prune any column.
    """
    file_path, file_path2 = gen_data1
    df = md.read_csv(file_path)
    df2 = md.read_csv(file_path2)

    df = df[df2["c1"] > 3]
    r = df.groupby(by="c1", as_index=False).sum()["c2"]

    graph = r.build_graph()
    optimize(graph)

    index_node = graph.result_tileables[0]
    assert isinstance(index_node.op, DataFrameIndex)
    assert index_node.name == "c2"

    assert len(graph.predecessors(index_node)) == 1
    gb_node = graph.predecessors(index_node)[0]
    assert isinstance(gb_node.op, DataFrameGroupByAgg)
    assert set(gb_node.dtypes.index) == {"c1", "c2"}

    assert len(graph.predecessors(gb_node)) == 1
    index_node_2 = graph.predecessors(gb_node)[0]
    isinstance(index_node_2.op, DataFrameIndex)
    assert set(index_node_2.dtypes.index) == {"c1", "c2"}

    assert len(graph.predecessors(index_node_2)) == 1
    index_node_3 = graph.predecessors(index_node_2)[0]
    isinstance(index_node_3.op, DataFrameIndex)
    assert set(index_node_3.dtypes.index) == {"c1", "c2", "c3", "c4"}

    assert len(graph.predecessors(index_node_3)) == 2
    read_csv_node, eval_node = graph.predecessors(index_node_3)
    assert isinstance(read_csv_node.op, DataFrameReadCSV)
    assert isinstance(eval_node.op, DataFrameEval)
    assert read_csv_node.op.usecols is None  # all the columns.
    assert eval_node.name == "c1"

    assert len(graph.predecessors(eval_node)) == 1
    read_csv_node_2 = graph.predecessors(eval_node)[0]
    assert isinstance(read_csv_node_2.op, DataFrameReadCSV)
    assert read_csv_node_2.op.usecols == ["c1"]

    raw1 = pd.read_csv(file_path)
    raw2 = pd.read_csv(file_path2)
    raw1 = raw1[raw2["c1"] > 3]
    expected = raw1.groupby(by="c1", as_index=False).sum()["c2"]
    pd.testing.assert_series_equal(
        r.execute(extra_config={"check_series_name": False}).fetch(), expected
    )


def test_setitem(setup, gen_data1):
    """
    The output of DataFrameSetitem should preserve the column being set so that tile can work
    correctly.
    """
    file_path, file_path2 = gen_data1
    df = md.read_csv(file_path)
    df2 = md.read_csv(file_path2)

    df["c5"] = df2["c1"]
    r = df.groupby(by="c1", as_index=False).sum()["c2"]

    graph = r.build_graph()
    optimize(graph)

    index_node = graph.result_tileables[0]
    assert isinstance(index_node.op, DataFrameIndex)
    assert index_node.name == "c2"

    assert len(graph.predecessors(index_node)) == 1
    gb_node = graph.predecessors(index_node)[0]
    assert isinstance(gb_node.op, DataFrameGroupByAgg)
    assert set(gb_node.dtypes.index) == {"c1", "c2"}

    assert len(graph.predecessors(gb_node)) == 1
    index_node_2 = graph.predecessors(gb_node)[0]
    isinstance(index_node_2.op, DataFrameIndex)
    assert set(index_node_2.dtypes.index) == {"c1", "c2"}

    assert len(graph.predecessors(index_node_2)) == 1
    setitem_node = graph.predecessors(index_node_2)[0]
    isinstance(setitem_node.op, DataFrameSetitem)
    assert set(setitem_node.dtypes.index) == {"c1", "c2", "c5"}

    assert len(graph.predecessors(setitem_node)) == 2
    read_csv_node, index_node_3 = graph.predecessors(setitem_node)
    assert isinstance(read_csv_node.op, DataFrameReadCSV)
    assert isinstance(index_node_3.op, DataFrameIndex)
    assert set(read_csv_node.op.usecols) == {"c1", "c2"}
    assert index_node_3.name == "c1"

    assert len(graph.predecessors(index_node_3)) == 1
    read_csv_node_2 = graph.predecessors(index_node_3)[0]
    assert isinstance(read_csv_node_2.op, DataFrameReadCSV)
    assert read_csv_node_2.op.usecols == ["c1"]

    raw1 = pd.read_csv(file_path)
    raw2 = pd.read_csv(file_path2)
    raw1["c5"] = raw2["c1"]
    expected = raw1.groupby(by="c1", as_index=False).sum()["c2"]
    pd.testing.assert_series_equal(r.execute().fetch(), expected)
