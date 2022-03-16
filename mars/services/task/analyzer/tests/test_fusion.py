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

from .....core import ChunkGraph
from .....tensor.arithmetic import TensorTreeAdd
from ..fusion import Coloring


def test_simple_coloring():
    # graph: https://user-images.githubusercontent.com/357506/132340029-b595afcf-3cec-44cb-b1c3-aac379e2e607.png
    chunks = [
        TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data for n in range(8)
    ]
    graph = ChunkGraph([chunks[3], chunks[7]])
    for c in chunks:
        graph.add_node(c)
    chunks[2].op._inputs = [chunks[0], chunks[1]]
    graph.add_edge(chunks[0], chunks[2])
    graph.add_edge(chunks[1], chunks[2])
    chunks[3].op._inputs = [chunks[2]]
    graph.add_edge(chunks[2], chunks[3])
    chunks[6].op._inputs = [chunks[4], chunks[5]]
    graph.add_edge(chunks[4], chunks[6])
    graph.add_edge(chunks[5], chunks[6])
    chunks[7].op._inputs = [chunks[6]]
    graph.add_edge(chunks[6], chunks[7])

    all_bands = [("127.0.0.1", "0"), ("127.0.0.1", "1")]
    chunk_to_bands = {
        chunks[0]: all_bands[0],
        chunks[1]: all_bands[0],
        chunks[4]: all_bands[1],
        chunks[5]: all_bands[1],
    }

    # allocate node 0, 1 with band 0, node 4, 5 with band 1
    coloring = Coloring(graph, all_bands, chunk_to_bands)
    chunk_to_colors = coloring.color()
    assert len(set(chunk_to_colors.values())) == 2
    assert (
        chunk_to_colors[chunks[0]]
        == chunk_to_colors[chunks[1]]
        == chunk_to_colors[chunks[2]]
        == chunk_to_colors[chunks[3]]
    )
    assert (
        chunk_to_colors[chunks[4]]
        == chunk_to_colors[chunks[5]]
        == chunk_to_colors[chunks[6]]
        == chunk_to_colors[chunks[7]]
    )

    # initial nodes all have different colors
    coloring = Coloring(graph, all_bands, chunk_to_bands, initial_same_color_num=1)
    chunk_to_colors = coloring.color()
    assert len(set(chunk_to_colors.values())) == 6
    assert (
        len(
            {
                chunk_to_colors[chunks[0]],
                chunk_to_colors[chunks[1]],
                chunk_to_colors[chunks[2]],
            }
        )
        == 3
    )
    assert chunk_to_colors[chunks[2]] == chunk_to_colors[chunks[3]]


def test_coloring_with_gpu_attr():
    chunks = [
        TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data for n in range(8)
    ]
    graph = ChunkGraph([chunks[3], chunks[7]])
    for c in chunks:
        graph.add_node(c)

    # two lines, one line can be fused as one task,
    # the other cannot, because gpu attributes are different
    chunks[0].op.gpu = True
    chunks[1].op.gpu = True
    chunks[1].op._inputs = [chunks[0]]
    graph.add_edge(chunks[0], chunks[1])
    chunks[2].op._inputs = [chunks[1]]
    graph.add_edge(chunks[1], chunks[2])
    chunks[3].op._inputs = [chunks[2]]
    graph.add_edge(chunks[2], chunks[3])
    chunks[5].op._inputs = [chunks[4]]
    graph.add_edge(chunks[4], chunks[5])
    chunks[6].op._inputs = [chunks[5]]
    graph.add_edge(chunks[5], chunks[6])
    chunks[7].op._inputs = [chunks[6]]
    graph.add_edge(chunks[6], chunks[7])

    all_bands = [("127.0.0.1", "0"), ("127.0.0.1", "1")]
    chunk_to_bands = {
        chunks[0]: all_bands[0],
        chunks[4]: all_bands[1],
    }

    coloring = Coloring(graph, all_bands, chunk_to_bands)
    chunk_to_colors = coloring.color()
    assert len(set(chunk_to_colors.values())) == 3
    assert chunk_to_colors[chunks[0]] == chunk_to_colors[chunks[1]]
    assert chunk_to_colors[chunks[2]] == chunk_to_colors[chunks[3]]
    assert (
        chunk_to_colors[chunks[4]]
        == chunk_to_colors[chunks[5]]
        == chunk_to_colors[chunks[6]]
        == chunk_to_colors[chunks[7]]
    )


def test_complex_coloring():
    # graph: https://user-images.githubusercontent.com/357506/132340055-f08106dd-b507-4e24-bc79-8364d6e1ef79.png
    chunks = [
        TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data
        for n in range(13)
    ]
    graph = ChunkGraph([chunks[7], chunks[12]])
    for c in chunks:
        graph.add_node(c)
    chunks[2].op._inputs = [chunks[0], chunks[1]]
    graph.add_edge(chunks[0], chunks[2])
    graph.add_edge(chunks[1], chunks[2])
    chunks[10].op._inputs = [chunks[8], chunks[9]]
    graph.add_edge(chunks[8], chunks[10])
    graph.add_edge(chunks[9], chunks[10])
    chunks[3].op._inputs = [chunks[2]]
    graph.add_edge(chunks[2], chunks[3])
    chunks[4].op._inputs = [chunks[3]]
    graph.add_edge(chunks[3], chunks[4])
    chunks[5].op._inputs = [chunks[2], chunks[10]]
    graph.add_edge(chunks[2], chunks[5])
    graph.add_edge(chunks[10], chunks[5])
    chunks[6].op._inputs = [chunks[5]]
    graph.add_edge(chunks[5], chunks[6])
    chunks[7].op._inputs = [chunks[4], chunks[6]]
    graph.add_edge(chunks[4], chunks[7])
    graph.add_edge(chunks[6], chunks[7])
    chunks[11].op._inputs = [chunks[10]]
    graph.add_edge(chunks[10], chunks[11])
    chunks[12].op._inputs = [chunks[6], chunks[11]]
    graph.add_edge(chunks[6], chunks[12])
    graph.add_edge(chunks[11], chunks[12])

    all_bands = [("127.0.0.1", "0"), ("127.0.0.1", "1")]
    chunk_to_bands = {
        chunks[0]: all_bands[0],
        chunks[1]: all_bands[0],
        chunks[8]: all_bands[1],
        chunks[9]: all_bands[1],
    }
    # allocate node 0, 1 with band 0, node 8, 9 with band 1
    coloring = Coloring(graph, all_bands, chunk_to_bands)
    chunk_to_colors = coloring.color()
    assert len(set(chunk_to_colors.values())) == 7
    assert (
        chunk_to_colors[chunks[0]]
        == chunk_to_colors[chunks[1]]
        == chunk_to_colors[chunks[2]]
    )
    assert chunk_to_colors[chunks[3]] == chunk_to_colors[chunks[4]]
    assert chunk_to_colors[chunks[5]] == chunk_to_colors[chunks[6]]
    assert (
        chunk_to_colors[chunks[8]]
        == chunk_to_colors[chunks[9]]
        == chunk_to_colors[chunks[10]]
    )
    assert (
        len(
            {
                chunk_to_colors[chunks[0]],
                chunk_to_colors[chunks[3]],
                chunk_to_colors[chunks[5]],
                chunk_to_colors[chunks[7]],
                chunk_to_colors[chunks[8]],
                chunk_to_colors[chunks[11]],
                chunk_to_colors[chunks[12]],
            }
        )
        == 7
    )


def test_coloring_broadcaster():
    chunks = [
        TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data for n in range(3)
    ]
    graph = ChunkGraph([chunks[2]])
    for c in chunks:
        graph.add_node(c)
    chunks[1].op._inputs = [chunks[0]]
    graph.add_edge(chunks[0], chunks[1])
    chunks[2].op._inputs = [chunks[0]]
    graph.add_edge(chunks[0], chunks[2])

    all_bands = [("127.0.0.1", "0"), ("127.0.0.1", "1")]
    chunk_to_bands = {
        chunks[0]: all_bands[0],
    }

    coloring = Coloring(graph, all_bands, chunk_to_bands)
    chunk_to_colors = coloring.color()
    assert len(set(chunk_to_colors.values())) == 1
    assert (
        chunk_to_colors[chunks[0]]
        == chunk_to_colors[chunks[1]]
        == chunk_to_colors[chunks[2]]
    )
    coloring = Coloring(
        graph, all_bands, chunk_to_bands, as_broadcaster_successor_num=1
    )
    chunk_to_colors = coloring.color()
    assert len(set(chunk_to_colors.values())) == 3
    assert (
        len(
            {
                chunk_to_colors[chunks[0]],
                chunk_to_colors[chunks[1]],
                chunk_to_colors[chunks[2]],
            }
        )
        == 3
    )
