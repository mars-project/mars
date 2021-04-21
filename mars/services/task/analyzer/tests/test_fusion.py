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

from mars.core import ChunkGraph
from mars.services.task.analyzer.fusion import Fusion
from mars.tensor.arithmetic import TensorTreeAdd


def test_fuse():
    """
    test compose in build graph and optimize
    """

    r"""
    graph(@: node, #: composed_node):

    @ --> @ --> @   ========>    #
    """
    chunks = [TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data
              for n in range(3)]
    graph = ChunkGraph([])
    for c in chunks:
        graph.add_node(c)
    graph.add_edge(chunks[0], chunks[1])
    graph.add_edge(chunks[1], chunks[2])

    graph2 = graph.copy()
    graph2._result_chunks = [chunks[2]]
    _, fused_nodes = Fusion(graph2).fuse()
    assert fused_nodes[0].composed == chunks[:3]

    # make the middle one as result chunk, thus the graph cannot be composed
    graph3 = graph.copy()
    graph3._result_chunks = [chunks[1]]
    _, fused_nodes = Fusion(graph3).fuse()
    assert fused_nodes[0].composed == chunks[:2]

    r"""
    graph(@: node, #: composed_node):

    @             @              @       @
      \         /                  \   /
        @ --> @       ========>      #
      /         \                  /   \
    @             @              @       @
    """
    chunks = [TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data
              for n in range(6)]
    graph = ChunkGraph([chunks[4], chunks[5]])
    for c in chunks:
        graph.add_node(c)

    chunks[2].op._inputs = [chunks[0], chunks[1]]
    chunks[3].op._inputs = [chunks[2]]
    chunks[4].op._inputs = [chunks[3]]
    chunks[5].op._inputs = [chunks[3]]

    graph.add_edge(chunks[0], chunks[2])
    graph.add_edge(chunks[1], chunks[2])
    graph.add_edge(chunks[2], chunks[3])
    graph.add_edge(chunks[3], chunks[4])
    graph.add_edge(chunks[3], chunks[5])

    _, fused_nodes = Fusion(graph).fuse()
    assert fused_nodes[0].composed == chunks[2:4]

    # to make sure the predecessors and successors of compose are right
    # 0 and 1's successors must be composed
    assert fused_nodes[0] in graph.successors(chunks[0])
    assert fused_nodes[0] in graph.successors(chunks[1])
    # check composed's inputs
    assert chunks[0] in fused_nodes[0].inputs
    assert chunks[1] in fused_nodes[0].inputs
    # check composed's predecessors
    assert chunks[0] in graph.predecessors(fused_nodes[0])
    assert chunks[1] in graph.predecessors(fused_nodes[0])
    # check 4 and 5's inputs
    assert fused_nodes[0] in graph.successors(fused_nodes[0])[0].inputs
    assert fused_nodes[0] in graph.successors(fused_nodes[0])[0].inputs
    # check 4 and 5's predecessors
    assert fused_nodes[0] in graph.predecessors(chunks[4])
    assert fused_nodes[0] in graph.predecessors(chunks[5])
