# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

import mars.tensor as mt
from mars.core.graph import DAG, GraphContainsCycleError
from mars.core.graph.builder import _build_graph


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
        assert all(rev_dag.has_successor(n, pred)
                   for pred in dag.predecessors(n)) is True

    undigraph = dag.build_undirected()
    for n in dag:
        assert n in undigraph
        assert all(undigraph.has_predecessor(pred, n)
                   for pred in dag.predecessors(n)) is True
        assert all(undigraph.has_successor(n, pred)
                   for pred in dag.predecessors(n)) is True

    dag_copy = dag.copy()
    for n in dag:
        assert n in dag_copy
        assert all(dag_copy.has_successor(pred, n)
                   for pred in dag_copy.predecessors(n)) is True


def test_to_dot():
    arr = mt.random.randint(10, size=(10, 8), chunk_size=4)
    arr_add = mt.random.randint(10, size=(10, 8), chunk_size=4)
    arr2 = arr + arr_add
    graph = _build_graph([arr2], fuse_enabled=False, tiled=True)

    dot = str(graph.to_dot(trunc_key=5))
    assert all(str(n.op.key)[5] in dot for n in graph) is True
