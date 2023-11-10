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

import os
import tempfile

import pandas as pd
import pytest

from ..... import dataframe as md
from .....core import (
    enter_mode,
    TileableGraph,
    TileableGraphBuilder,
    ChunkGraphBuilder,
    TileContext,
)
from .. import optimize


@pytest.fixture(scope="module")
def gen_data1():
    with tempfile.TemporaryDirectory() as tempdir:
        df = pd.DataFrame(
            {
                "a": [3, 4, 5, 3, 5, 4, 1, 2, 3],
                "b": [1, 3, 4, 5, 6, 5, 4, 4, 4],
                "c": list("aabaaddce"),
                "d": list("abaaaddce"),
            }
        )
        yield df, tempdir


@enter_mode(build=True)
def test_read_csv_head(gen_data1):
    pdf, tempdir = gen_data1
    file_path = os.path.join(tempdir, "test.csv")
    pdf.to_csv(file_path)

    df1 = md.read_csv(file_path)
    df2 = df1.head(5)
    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    context = TileContext()
    chunk_graph_builder = ChunkGraphBuilder(
        graph, fuse_enabled=False, tile_context=context
    )
    chunk_graph = next(chunk_graph_builder.build())
    chunk1 = context[df1.data].chunks[0]
    chunk2 = context[df2.data].chunks[0]
    records = optimize(chunk_graph)
    assert records.get_optimization_result(chunk1) is None
    opt_chunk2 = records.get_optimization_result(chunk2)
    assert opt_chunk2.op.nrows == 5
    assert len(chunk_graph) == 1
    assert opt_chunk2 in chunk_graph.results
