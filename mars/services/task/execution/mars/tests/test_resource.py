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
import pytest

from typing import Dict, Any

from ...... import dataframe as md
from ...... import tensor as mt
from .... import Task
from ......config import Config
from ......core import Tileable, TileableGraph, ChunkGraphBuilder
from ......resource import Resource
from ....analyzer import GraphAnalyzer
from ..resource import ResourceEvaluator, register_resource_evaluator, DefaultEvaluator
from ..stage import TaskStageProcessor


@register_resource_evaluator
class MockedEvaluator(ResourceEvaluator):
    name = "mock"

    def __init__(self, config, **kwargs):
        self._config = config

    @classmethod
    async def create(cls, config: Dict[str, Any], **kwargs) -> "ResourceEvaluator":
        return cls(config, **kwargs)

    async def evaluate(self, stage_processor: "TaskStageProcessor"):
        pass

    async def report(self):
        pass


def _build_chunk_graph(tileable_graph: TileableGraph):
    return next(ChunkGraphBuilder(tileable_graph).build())


async def _gen_stage_processor(t):
    tileable_graph = t.build_graph(tile=False)
    chunk_graph = _build_chunk_graph(tileable_graph)
    bands = [(f"address_{i}", "numa-0") for i in range(4)]
    band_resource = dict((band, Resource(num_cpus=1)) for band in bands)
    task = Task("mock_task", "mock_session", tileable_graph)
    analyzer = GraphAnalyzer(chunk_graph, band_resource, task, Config(), dict())
    subtask_graph = analyzer.gen_subtask_graph()
    stage_processor = TaskStageProcessor(
        "stage_id", task, chunk_graph, subtask_graph, bands, None, None, None
    )
    return stage_processor


async def _test_default_evaluator(config: Dict[str, Any], t: Tileable):
    resource_evaluator = await ResourceEvaluator.create(config)
    assert resource_evaluator is not None
    assert isinstance(resource_evaluator, DefaultEvaluator)
    stage_processor = await _gen_stage_processor(t)
    await resource_evaluator.evaluate(stage_processor)
    for subtask in stage_processor.subtask_graph.iter_nodes():
        is_gpu = any(c.op.gpu for c in subtask.chunk_graph)
        assert (
            subtask.required_resource == Resource(num_gpus=1)
            if is_gpu
            else Resource(num_cpus=1)
        )
    assert await resource_evaluator.report() is None


@pytest.mark.asyncio
async def test_resource_evaluator():
    # test mocked resource evaluator
    resource_evaluator = await ResourceEvaluator.create({"resource_evaluator": "mock"})
    assert resource_evaluator is not None
    assert isinstance(resource_evaluator, MockedEvaluator)

    # test default resource evaluator
    t = mt.ones((10, 10), chunk_size=5) + 1
    await _test_default_evaluator({}, t)
    await _test_default_evaluator({"resource_evaluator": "default"}, t)
    df = md.DataFrame(
        np.random.randint(0, 100, size=(100_000, 4)),
        columns=list("ABCD"),
        chunk_size=1000,
    )
    df = df[df["A"] > 50]
    await _test_default_evaluator({}, df)
