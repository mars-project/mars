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
import itertools
import pytest


from ..core import OptimizationRule
from .... import tensor as mt
from .... import dataframe as md


class _MockRule(OptimizationRule):
    def apply(self) -> bool:
        pass

    def replace_subgraph(self, graph, nodes_to_remove, new_results=None):
        self._replace_subgraph(graph, nodes_to_remove, new_results)


def test_replace_tileable_subgraph():
    """
    Original Graph:
     s1 ---> c1 ---> v1 ---> v4 ----> v6(output) <--- v5 <--- c5 <--- s5
                     |        ^
                     |        |
                     V        |
                     v3 ------|
                     ^
                     |
     s2 ---> c2 ---> v2

    Target Graph:
      s1 ---> c1 ---> v1 ---> v7 ----> v8(output) <--- v5 <--- c5 <--- s5
                              ^
                              |
              s2 ---> c2 ---> v2

    The nodes [v3, v4, v6] will be removed.
    Subgraph only contains [v7, v8]
    """
    s1 = mt.random.randint(0, 100, size=(5, 4))
    v1 = md.DataFrame(s1, columns=list("ABCD"), chunk_size=5)
    s2 = mt.random.randint(0, 100, size=(5, 4))
    v2 = md.DataFrame(s2, columns=list("ABCD"), chunk_size=5)
    v3 = v1.add(v2)
    v4 = v3.add(v1)
    s5 = mt.random.randint(0, 100, size=(5, 4))
    v5 = md.DataFrame(s5, columns=list("ABCD"), chunk_size=4)
    v6 = v5.sub(v4)
    g1 = v6.build_graph()
    v7 = v1.sub(v2)
    v8 = v7.add(v5)
    v8._key = v6.key
    v8.outputs[0]._key = v6.key
    g2 = v8.build_graph()
    # Here we use a trick way to construct the subgraph for test only
    key_to_node = dict()
    for node in g2.iter_nodes():
        key_to_node[node.key] = node
    for key, node in key_to_node.items():
        if key != v7.key and key != v8.key:
            g2.remove_node(node)
    r = _MockRule(g1, None, None)
    for node in g1.iter_nodes():
        key_to_node[node.key] = node

    c1 = g1.successors(key_to_node[s1.key])[0]
    c2 = g1.successors(key_to_node[s2.key])[0]
    c5 = g1.successors(key_to_node[s5.key])[0]

    new_results = [v8.outputs[0]]
    r.replace_subgraph(g2, {key_to_node[op.key] for op in [v3, v4, v6]}, new_results)
    assert g1.results == new_results
    for node in g1.iter_nodes():
        if node.key == v8.key:
            key_to_node[v8.key] = node
            break
    expected_nodes = {s1, c1, v1, s2, c2, v2, s5, c5, v5, v7, v8}
    assert set(g1) == {key_to_node[n.key] for n in expected_nodes}

    expected_edges = {
        s1: [c1],
        c1: [v1],
        v1: [v7],
        s2: [c2],
        c2: [v2],
        v2: [v7],
        s5: [c5],
        c5: [v5],
        v5: [v8],
        v7: [v8],
        v8: [],
    }
    for pred, successors in expected_edges.items():
        pred_node = key_to_node[pred.key]
        assert g1.count_successors(pred_node) == len(successors)
        for successor in successors:
            assert g1.has_successor(pred_node, key_to_node[successor.key])


def test_replace_null_subgraph():
    """
    Original Graph:
     s1 ---> c1 ---> v1 ---> v3(out) <--- v2 <--- c2 <--- s2

    Target Graph:
      c1 ---> v1 ---> v3(out)  <--- v2 <--- c2

    The nodes [s1, s2] will be removed.
    Subgraph is None
    """
    s1 = mt.random.randint(0, 100, size=(10, 4))
    v1 = md.DataFrame(s1, columns=list("ABCD"), chunk_size=5)
    s2 = mt.random.randint(0, 100, size=(10, 4))
    v2 = md.DataFrame(s2, columns=list("ABCD"), chunk_size=5)
    v3 = v1.add(v2)
    g1 = v3.build_graph()
    key_to_node = {node.key: node for node in g1.iter_nodes()}
    c1 = g1.successors(key_to_node[s1.key])[0]
    c2 = g1.successors(key_to_node[s2.key])[0]
    r = _MockRule(g1, None, None)
    expected_results = [v3.outputs[0]]

    # delete c5 s5 will fail
    with pytest.raises(ValueError):
        r.replace_subgraph(
            None, {key_to_node[op.key] for op in [s1, s2]}, [v2.outputs[0]]
        )

    assert g1.results == expected_results
    assert set(g1) == {key_to_node[n.key] for n in {s1, c1, v1, s2, c2, v2, v3}}
    expected_edges = {
        s1: [c1],
        c1: [v1],
        v1: [v3],
        s2: [c2],
        c2: [v2],
        v2: [v3],
        v3: [],
    }
    for pred, successors in expected_edges.items():
        pred_node = key_to_node[pred.key]
        assert g1.count_successors(pred_node) == len(successors)
        for successor in successors:
            assert g1.has_successor(pred_node, key_to_node[successor.key])

    c1.inputs.clear()
    c2.inputs.clear()
    r.replace_subgraph(
        None,
        {key_to_node[op.key] for op in [s1, s2]}
    )
    assert g1.results == expected_results
    assert set(g1) == {key_to_node[n.key] for n in {c1, v1, c2, v2, v3}}
    expected_edges = {
        c1: [v1],
        v1: [v3],
        c2: [v2],
        v2: [v3],
        v3: [],
    }
    for pred, successors in expected_edges.items():
        pred_node = key_to_node[pred.key]
        assert g1.count_successors(pred_node) == len(successors)
        for successor in successors:
            assert g1.has_successor(pred_node, key_to_node[successor.key])


def test_replace_subgraph_without_removing_nodes():
    """
    Original Graph:
     s1 ---> c1 ---> v1 ---> v4 <--- v2 <--- c2 <--- s2

    Target Graph:
      s1 ---> c1 ---> v1 ---> v4 <--- v2 <--- c2 <--- s2
      s3 ---> c3 ---> v3

    Nothing will be removed.
    Subgraph only contains [s3, c3, v3]
    """
    s1 = mt.random.randint(0, 100, size=(10, 4))
    v1 = md.DataFrame(s1, columns=list("ABCD"), chunk_size=5)
    s2 = mt.random.randint(0, 100, size=(10, 4))
    v2 = md.DataFrame(s2, columns=list("ABCD"), chunk_size=5)
    v4 = v1.add(v2)
    g1 = v4.build_graph()

    s3 = mt.random.randint(0, 100, size=(10, 4))
    v3 = md.DataFrame(s3, columns=list("ABCD"), chunk_size=5)
    g2 = v3.build_graph()
    key_to_node = {
        node.key: node for node in itertools.chain(g1.iter_nodes(), g2.iter_nodes())
    }
    expected_results = [v4.outputs[0]]
    c1 = g1.successors(key_to_node[s1.key])[0]
    c2 = g1.successors(key_to_node[s2.key])[0]
    c3 = g2.successors(key_to_node[s3.key])[0]
    r = _MockRule(g1, None, None)
    r.replace_subgraph(g2, None)
    assert g1.results == expected_results
    assert set(g1) == {
        key_to_node[n.key] for n in {s1, c1, v1, s2, c2, v2, s3, c3, v3, v4}
    }
    expected_edges = {
        s1: [c1],
        c1: [v1],
        v1: [v4],
        s2: [c2],
        c2: [v2],
        v2: [v4],
        s3: [c3],
        c3: [v3],
        v3: [],
        v4: [],
    }
    for pred, successors in expected_edges.items():
        pred_node = key_to_node[pred.key]
        assert g1.count_successors(pred_node) == len(successors)
        for successor in successors:
            assert g1.has_successor(pred_node, key_to_node[successor.key])
