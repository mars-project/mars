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

from ..... import dataframe as md
from ..... import tensor as mt
from .....config import Config
from .....core.operand.shuffle import ShuffleFetchType, ShuffleProxy
from .....resource import Resource
from ...core import Task
from ..analyzer import GraphAnalyzer


t1 = mt.random.RandomState(0).rand(31, 27, chunk_size=10)
t2 = t1.reshape(27, 31)
t2.op.extra_params["_reshape_with_shuffle"] = True
df1 = md.DataFrame(t1, columns=[f"c{i}" for i in range(t1.shape[1])])
df2 = df1.groupby(["c1"]).apply(lambda pdf: pdf.sum())


@pytest.mark.parametrize("tileable", [df1.describe(), df2, t2])
@pytest.mark.parametrize("fuse", [True, False])
def test_shuffle_graph(tileable, fuse):
    # can't test df.groupby and mt.bincount, those chunk graph build depend on ctx.get_chunks_meta/get_chunks_result
    chunk_graph = tileable.build_graph(tile=True)
    assert len(chunk_graph) > 0
    all_bands = [(f"address_{i}", "numa-0") for i in range(5)]
    band_resource = dict((band, Resource(num_cpus=1)) for band in all_bands)
    task = Task("mock_task", "mock_session", fuse_enabled=fuse)
    analyzer = GraphAnalyzer(
        chunk_graph,
        band_resource,
        task,
        Config(),
        dict(),
        shuffle_fetch_type=ShuffleFetchType.FETCH_BY_INDEX,
    )
    subtask_graph = analyzer.gen_subtask_graph()
    proxy_subtasks = []
    for subtask in subtask_graph:
        for c in subtask.chunk_graph.results:
            if isinstance(c.op, ShuffleProxy):
                assert len(subtask.chunk_graph.results) == 1
                proxy_subtasks.append(subtask)
    proxy_chunks = [
        c
        for subtask in proxy_subtasks
        for c in chunk_graph
        if subtask.chunk_graph.results[0].key == c.key
    ]
    assert len(proxy_subtasks) == len(proxy_chunks)
    assert len(proxy_subtasks) > 0
    assert len(proxy_subtasks) == len(subtask_graph.get_shuffle_proxy_subtasks())
    for proxy_chunk, proxy_subtask in zip(proxy_chunks, proxy_subtasks):
        reducer_subtasks = subtask_graph.successors(proxy_subtask)
        for reducer_subtask in reducer_subtasks:
            start_chunks = list(reducer_subtask.chunk_graph.iter_indep())
            assert len(start_chunks) == 1
            assert (
                start_chunks[0].op.shuffle_fetch_type == ShuffleFetchType.FETCH_BY_INDEX
            )
        reducer_chunks = chunk_graph.successors(proxy_chunk)
        # single reducer may have multiple output chunks, see `PSRSShuffle._execute_reduce
        if len(reducer_subtasks) != len(reducer_chunks):
            assert len(reducer_subtasks) == len(set(c.op for c in reducer_chunks))
        mapper_subtasks = subtask_graph.predecessors(proxy_subtask)
        for mapper_subtask in mapper_subtasks:
            assert len(mapper_subtask.chunk_graph.results) == 1
        mapper_chunks = chunk_graph.predecessors(proxy_chunk)
        assert len(mapper_subtasks) == len(mapper_chunks)
