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

from .... import tensor as mt
from ....core import (
    enter_mode,
    TileableGraph,
    TileableGraphBuilder,
    ChunkGraphBuilder,
    TileContext,
)
from ..cupy import CupyRuntimeOptimizer


@enter_mode(build=True)
def test_cupy():
    t1 = mt.ones((100, 50), chunk_size=50, gpu=True)
    t2 = mt.ones(50, chunk_size=50, gpu=True)
    t = (t1 - t2) / mt.sqrt(t2 * (1 - t2) * len(t2))

    graph = TileableGraph([t.data])
    next(TileableGraphBuilder(graph).build())
    context = TileContext()
    chunk_graph_builder = ChunkGraphBuilder(
        graph, fuse_enabled=False, tile_context=context
    )
    chunk_graph = next(chunk_graph_builder.build())

    CupyRuntimeOptimizer(chunk_graph).optimize()
    assert any(n.op.__class__.__name__ == "TensorCpFuseChunk" for n in chunk_graph)
