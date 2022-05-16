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
from .....core import TileableGraph, TileableGraphBuilder, ChunkGraphBuilder
from .....core.operand.shuffle import ShuffleType
from .....resource import Resource
from ...core import Task
from ..analyzer import GraphAnalyzer


t1 = mt.random.RandomState(0).rand(31, 27, chunk_size=10)
df1 = md.DataFrame(t1, columns=[f"c{i}" for i in range(t1.shape[1])])
df2 = df1.groupby("c1").apply(lambda pdf: None)


@pytest.mark.parametrize(
    "tileable", [df2, df1.describe(), t1.reshape(27, 31), mt.bincount(mt.arange(5, 10))]
)
@pytest.mark.parametrize("fuse", [True, False])
def test_shuffle_graph(tileable, fuse):
    tileable_graph = next(TileableGraphBuilder(TileableGraph([tileable])).build())
    chunk_graph = next(ChunkGraphBuilder(tileable_graph).build())
    all_bands = [(f"address_{i}", "numa-0") for i in range(5)]
    band_resource = dict((band, Resource(num_cpus=1)) for band in all_bands)
    task = Task("mock_task", "mock_session", fuse_enabled=fuse)
    config = Config()
    config.register_option("shuffle_type", ShuffleType.PUSH)
    analyzer = GraphAnalyzer(chunk_graph, band_resource, task, config, dict())
    analyzer.gen_subtask_graph()
