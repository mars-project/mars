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


from ..... import dataframe as md
from ..... import tensor as mt
from .....core.graph import TileableGraph, TileableGraphBuilder, ChunkGraphBuilder
from ..analyzer import GraphAnalyzer
from ..assigner import GraphAssigner


def test_assigner():
    num_rows = 10000
    df1 = md.DataFrame(
        mt.random.rand(num_rows, 4, chunk_size=10), columns=list("abcd")
    )
    df2 = md.DataFrame(
        mt.random.rand(num_rows, 4, chunk_size=10), columns=list("abcd")
    )
    merged_df = df1.merge(df2, left_on="a", right_on="a")
    graph = TileableGraph([merged_df.data])
    next(TileableGraphBuilder(graph).build())
    chunk_graph = next(ChunkGraphBuilder(graph, fuse_enabled=False).build())
    start_ops = list(GraphAnalyzer._iter_start_ops(chunk_graph))
    band_slots = {(f"worker-{i}", "numa-0"): 16 for i in range(50)}
    current_assign = {}
    assigner = GraphAssigner(chunk_graph, start_ops, band_slots)
    assigned_result = assigner.assign(current_assign)
    assert len(assigned_result) == 2000
