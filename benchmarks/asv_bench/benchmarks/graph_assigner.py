# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

import random
import tracemalloc

import mars.tensor as mt
import mars.dataframe as md
from mars.core.graph import TileableGraph, TileableGraphBuilder, ChunkGraphBuilder
from mars.resource import Resource
from mars.services.task.analyzer import GraphAnalyzer
from mars.services.task.analyzer.assigner import GraphAssigner


class ChunkGraphAssignerSuite:
    """
    Benchmark that times performance of chunk graph assigner
    """

    repeat = 10

    def setup(self):
        tracemalloc.start()
        random.seed()

        num_rows = 10000
        df1 = md.DataFrame(
            mt.random.rand(num_rows, 4, chunk_size=10), columns=list("abcd")
        )
        df2 = md.DataFrame(
            mt.random.rand(num_rows, 4, chunk_size=10), columns=list("abcd")
        )
        merged_df = df1.merge(
            df2, left_on="a", right_on="a", auto_merge="none", bloom_filter=False
        )
        graph = TileableGraph([merged_df.data])
        next(TileableGraphBuilder(graph).build())
        self.chunk_graph = next(ChunkGraphBuilder(graph, fuse_enabled=False).build())
        self.mem_size, self.mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    def time_assigner(self):
        start_ops = list(GraphAnalyzer._iter_start_ops(self.chunk_graph))
        band_resource = {
            (f"worker-{i}", "numa-0"): Resource(num_cpus=16) for i in range(50)
        }
        current_assign = {}
        assigner = GraphAssigner(self.chunk_graph, start_ops, band_resource)
        assigned_result = assigner.assign(current_assign)
        assert len(assigned_result) == len(start_ops)

    def peakmem_setup(self):
        """peakmem includes the memory used by setup.
        Peakmem benchmarks measure the maximum amount of RAM used by a
        function. However, this maximum also includes the memory used
        by ``setup`` (as of asv 0.2.1; see [1]_)
        Measuring an empty peakmem function might allow us to disambiguate
        between the memory used by setup and the memory used by slic (see
        ``peakmem_slic_basic``, below).
        References
        ----------
        .. [1]: https://asv.readthedocs.io/en/stable/writing_benchmarks.html#peak-memory
        """
        pass

    def mem_chunk_graph(self):
        return self.chunk_graph

    def track_traced_mem_size(self):
        return self.mem_size

    def track_traced_mem_peak(self):
        return self.mem_peak
