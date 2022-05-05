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

import pytest

import numpy as np

from .... import dataframe as md
from .... import tensor as mt
from ....tests.core import flaky
from ....utils import to_str
from .. import DAG, GraphContainsCycleError


def test_dag():
    r"""
    1 --- 4
    2 --- 6
      \  /
       5
     /
    3
    """

    dag = DAG()
    [dag.add_node(i) for i in range(1, 7)]
    dag.add_edge(1, 4)
    dag.add_edge(2, 6)
    dag.add_edge(2, 5)
    dag.add_edge(5, 6)
    dag.add_edge(3, 5)

    with pytest.raises(KeyError):
        dag.add_edge(1, 10)
    with pytest.raises(KeyError):
        dag.add_edge(10, 1)

    assert set(dag[2]) == {5, 6}
    assert list(dag.topological_iter()) == [3, 2, 5, 6, 1, 4]

    assert list(dag.dfs()) == [3, 2, 5, 6, 1, 4]
    assert list(dag.bfs()) == [1, 2, 3, 4, 5, 6]

    dag.add_edge(6, 1)
    dag.add_edge(1, 2)

    with pytest.raises(KeyError):
        for _ in dag.iter_predecessors(-1):
            pass

    with pytest.raises(KeyError):
        for _ in dag.iter_successors(-1):
            pass

    with pytest.raises(GraphContainsCycleError):
        _ = list(dag.topological_iter())

    dag.remove_edge(2, 5)
    assert dag.has_successor(2, 5) is False
    with pytest.raises(KeyError):
        dag.remove_edge(2, 5)

    rev_dag = dag.build_reversed()
    for n in dag:
        assert n in rev_dag
        assert (
            all(rev_dag.has_successor(n, pred) for pred in dag.predecessors(n)) is True
        )

    undigraph = dag.build_undirected()
    for n in dag:
        assert n in undigraph
        assert (
            all(undigraph.has_predecessor(pred, n) for pred in dag.predecessors(n))
            is True
        )
        assert (
            all(undigraph.has_successor(n, pred) for pred in dag.predecessors(n))
            is True
        )

    dag_copy = dag.copy()
    for n in dag:
        assert n in dag_copy
        assert (
            all(dag_copy.has_successor(pred, n) for pred in dag_copy.predecessors(n))
            is True
        )


@flaky(max_runs=3)
def test_to_dot():
    arr = mt.random.randint(10, size=(10, 8), chunk_size=4)
    arr_add = mt.random.randint(10, size=(10, 8), chunk_size=4)
    arr2 = arr + arr_add
    graph = arr2.build_graph(fuse_enabled=False, tile=True)

    dot = to_str(graph.to_dot(trunc_key=5))
    assert all(to_str(n.key)[:5] in dot for n in graph) is True


def test_tileable_graph_logic_key():
    # Tensor
    t1 = mt.random.randint(10, size=(10, 8), chunk_size=4)
    t2 = mt.random.randint(10, size=(10, 8), chunk_size=5)
    graph1 = (t1 + t2).build_graph(tile=False)
    tt1 = mt.random.randint(10, size=(10, 8), chunk_size=4)
    tt2 = mt.random.randint(10, size=(10, 8), chunk_size=5)
    graph2 = (tt1 + tt2).build_graph(tile=False)
    assert graph1.logic_key == graph2.logic_key
    t3 = mt.random.randint(10, size=(10, 8), chunk_size=6)
    tt3 = mt.random.randint(10, size=(10, 8), chunk_size=6)
    graph3 = (t1 + t3).build_graph(tile=False)
    graph4 = (t1 + tt3).build_graph(tile=False)
    assert graph1.logic_key != graph3.logic_key
    assert graph3.logic_key == graph4.logic_key
    t4 = mt.random.randint(10, size=(10, 8))
    graph5 = (t1 + t4).build_graph(tile=False)
    assert graph1.logic_key != graph5.logic_key

    # Series
    s1 = md.Series([1, 3, 5, mt.nan, 6, 8])
    s2 = md.Series(np.random.randn(1000), chunk_size=100)
    graph1 = (s1 + s2).build_graph(tile=False)
    ss1 = md.Series([1, 3, 5, mt.nan, 6, 8])
    ss2 = md.Series(np.random.randn(1000), chunk_size=100)
    graph2 = (ss1 + ss2).build_graph(tile=False)
    assert graph1.logic_key == graph2.logic_key
    s3 = md.Series(np.random.randn(1000), chunk_size=200)
    ss3 = md.Series(np.random.randn(1000), chunk_size=200)
    graph3 = (s1 + s3).build_graph(tile=False)
    graph4 = (s1 + ss3).build_graph(tile=False)
    assert graph1.logic_key != graph3.logic_key
    assert graph3.logic_key == graph4.logic_key
    s4 = md.Series(np.random.randn(1000))
    graph5 = (s1 + s4).build_graph(tile=False)
    assert graph1.logic_key != graph5.logic_key

    # DataFrame
    df1 = md.DataFrame(
        np.random.randint(0, 100, size=(100_000, 4)), columns=list("ABCD"), chunk_size=5
    )
    df2 = md.DataFrame(
        np.random.randint(0, 100, size=(100_000, 4)), columns=list("ABCD"), chunk_size=4
    )
    graph1 = (df1 + df2).build_graph(tile=False)
    ddf1 = md.DataFrame(
        np.random.randint(0, 100, size=(100_000, 4)), columns=list("ABCD"), chunk_size=5
    )
    ddf2 = md.DataFrame(
        np.random.randint(0, 100, size=(100_000, 4)), columns=list("ABCD"), chunk_size=4
    )
    graph2 = (ddf1 + ddf2).build_graph(tile=False)
    assert graph1.logic_key == graph2.logic_key
    df3 = md.DataFrame(
        np.random.randint(0, 100, size=(100_000, 4)), columns=list("ABCD"), chunk_size=3
    )
    ddf3 = md.DataFrame(
        np.random.randint(0, 100, size=(100_000, 4)), columns=list("ABCD"), chunk_size=3
    )
    graph3 = (df1 + df3).build_graph(tile=False)
    graph4 = (df1 + ddf3).build_graph(tile=False)
    assert graph1.logic_key != graph3.logic_key
    assert graph3.logic_key == graph4.logic_key
    df5 = md.DataFrame(
        np.random.randint(0, 100, size=(100_000, 4)), columns=list("ABCD")
    )
    graph5 = (df1 + df5).build_graph(tile=False)
    assert graph1.logic_key != graph5.logic_key
    graph6 = df1.describe().build_graph(tile=False)
    graph7 = df2.describe().build_graph(tile=False)
    assert graph6.logic_key != graph7.logic_key
    graph8 = df1.apply(lambda x: x.max() - x.min()).build_graph(tile=False)
    graph9 = df2.apply(lambda x: x.max() - x.min()).build_graph(tile=False)
    assert graph8.logic_key != graph9.logic_key
    pieces1 = [df1[:3], df1[3:7], df1[7:]]
    graph10 = md.concat(pieces1).build_graph(tile=False)
    pieces2 = [df2[:3], df2[3:7], df2[7:]]
    graph11 = md.concat(pieces2).build_graph(tile=False)
    assert graph10.logic_key != graph11.logic_key
    graph12 = md.merge(df1, df2, on="A", how="left").build_graph(tile=False)
    graph13 = md.merge(df1, df3, on="A", how="left").build_graph(tile=False)
    assert graph12.logic_key != graph13.logic_key
    graph14 = df2.groupby("A").sum().build_graph(tile=False)
    graph15 = df3.groupby("A").sum().build_graph(tile=False)
    assert graph14.logic_key != graph15.logic_key
