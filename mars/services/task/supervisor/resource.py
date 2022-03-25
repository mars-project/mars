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

from ....resource import Resource
from .stage import TaskStageProcessor


class ResourceEvaluator:
    """
    Evaluate and initialize the required resource of subtasks by different
    configurations, e.g. fixed value by default, or some recommended values
    through external services like an HBO service which could calculate a
    accurate value by the running history of tasks.
    """

    def __init__(self, stage_processor: TaskStageProcessor):
        self._stage_processor = stage_processor
        self._subtask_graph = stage_processor.subtask_graph

    def evaluate(self):
        """Here we could implement different acquisitions by state processor
        configurations.
        """
        for subtask in self._subtask_graph.iter_nodes():
            is_gpu = any(c.op.gpu for c in subtask.chunk_graph)
            subtask.required_resource = (
                Resource(num_gpus=1) if is_gpu else Resource(num_cpus=1)
            )
