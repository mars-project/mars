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

from ....core import ChunkGraph
from ....tensor.arithmetic import TensorTreeAdd
from ....tensor.indexing import TensorSlice
from ..numexpr import NumexprRuntimeOptimizer


def test_numexpr():
    r"""
        graph(@: node, S: Slice Chunk, #: fused_node):

        @                   @              @             @
          \               /                  \         /
            @ --> @ --> S      ========>       # --> S
          /               \                  /         \
        @                   @              @             @

        fuse stopped at S, because numexpr don't support Slice op
        """
    chunks = [TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data
              for n in range(6)]
    chunk_slice = TensorSlice().new_chunk([None], None).data
    graph = ChunkGraph([chunks[4], chunks[5]])
    list(map(graph.add_node, chunks[:6]))
    graph.add_node(chunk_slice)
    graph.add_edge(chunks[0], chunks[2])
    graph.add_edge(chunks[1], chunks[2])
    graph.add_edge(chunks[2], chunks[3])
    graph.add_edge(chunks[3], chunk_slice)
    graph.add_edge(chunk_slice, chunks[4])
    graph.add_edge(chunk_slice, chunks[5])

    optimizer = NumexprRuntimeOptimizer(graph)
    _, fused_nodes = optimizer.optimize()
    assert fused_nodes[0].composed == chunks[2:4]

    r"""
            graph(@: node, S: Slice Chunk, #: fused_node):

            @ --> @ --> S --> @  ========>  # --> S --> @

        fuse stopped at S, because numexpr don't support Slice op
        """
    chunks = [TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data
              for n in range(4)]
    graph = ChunkGraph([chunks[2]])
    list(map(graph.add_node, chunks[:3]))
    graph.add_node(chunk_slice)
    graph.add_edge(chunks[0], chunks[1])
    graph.add_edge(chunks[1], chunk_slice)
    graph.add_edge(chunk_slice, chunks[2])

    optimizer = NumexprRuntimeOptimizer(graph)
    _, fused_nodes = optimizer.optimize()
    assert fused_nodes[0].composed == chunks[:2]
    assert len(fused_nodes) == 1

    r"""
        graph(@: node, S: Slice Chunk, #: fused_node):

        @ --> @ --> S --> @ --> @   ========>  # --> S --> #

    fuse stopped at S, because numexpr don't support Slice op
    """
    chunks = [TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data
              for n in range(4)]
    graph = ChunkGraph([chunks[3]])
    list(map(graph.add_node, chunks[:4]))
    graph.add_node(chunk_slice)
    graph.add_edge(chunks[0], chunks[1])
    graph.add_edge(chunks[1], chunk_slice)
    graph.add_edge(chunk_slice, chunks[2])
    graph.add_edge(chunks[2], chunks[3])

    optimizer = NumexprRuntimeOptimizer(graph)
    _, fused_nodes = optimizer.optimize()
    assert fused_nodes[0].composed == chunks[:2]
    assert fused_nodes[1].composed == chunks[2:4]
