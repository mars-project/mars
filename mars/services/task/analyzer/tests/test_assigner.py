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

import numpy as np

from .....config import Config
from .....core import ChunkGraph
from .....tensor.random import TensorRand
from .....tensor.arithmetic import TensorAdd
from .....tensor.fetch import TensorFetch
from .....resource import Resource
from ...core import Task
from ..analyzer import GraphAnalyzer
from ..assigner import GraphAssigner


def test_assigner_with_fetch_inputs():
    band_num = 8
    all_bands = [(f"address_{i}", "numa-0") for i in range(band_num)]
    inputs = [
        TensorFetch(key=str(i), source_key=str(i), dtype=np.dtype(int)).new_chunk([])
        for i in range(band_num)
    ]
    no_fetch_inputs = [TensorRand(i).new_chunk([]) for i in range(4)]
    results = [TensorAdd(lhs=inp, rhs=1).new_chunk([inp]) for inp in inputs]
    cur_assigns = dict(
        (fetch_chunk.op.key, band[0][0])
        for fetch_chunk, band in zip(reversed(inputs), all_bands)
    )

    chunk_graph = ChunkGraph()
    for fetch_chunk, add_chunk in zip(inputs, results):
        chunk_graph.add_node(fetch_chunk)
        chunk_graph.add_node(add_chunk)
        chunk_graph.add_edge(fetch_chunk, add_chunk)
    for inp in no_fetch_inputs:
        results.append(inp)
        chunk_graph.add_node(inp)
    chunk_graph.results = results

    band_resource = dict((band, Resource(num_cpus=1)) for band in all_bands)

    task = Task("mock_task", "mock_session")
    analyzer = GraphAnalyzer(chunk_graph, band_resource, task, Config(), dict())
    subtask_graph = analyzer.gen_subtask_graph(cur_assigns)

    assigner = GraphAssigner(
        chunk_graph, list(GraphAnalyzer._iter_start_ops(chunk_graph)), band_resource
    )
    assigns = assigner.assign(cur_assigns)
    key_to_assign = dict((c.key, band) for c, band in assigns.items())
    for subtask in subtask_graph:
        input_chunks = list(subtask.chunk_graph.iter_indep())
        if all(isinstance(inp.op, TensorFetch) for inp in input_chunks):
            # all inputs are fetch, expect band should be None
            assert subtask.expect_band is None
        else:
            # if subtask has truly initial chunks, expect band should be
            # same as assign results
            for inp in input_chunks:
                if not isinstance(inp.op, TensorFetch):
                    assert subtask.expect_band == key_to_assign[inp.key]
