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

from abc import ABC, abstractmethod
from typing import Dict, Type, Any

from .....resource import Resource


_name_to_resource_evaluator: Dict[str, Type["ResourceEvaluator"]] = {}


def register_resource_evaluator(evaluator_cls: Type["ResourceEvaluator"]):
    _name_to_resource_evaluator[evaluator_cls.name] = evaluator_cls
    return evaluator_cls


def init_default_resource_for_subtask(subtask_graph: "SubtaskGraph"):  # noqa: F821
    for subtask in subtask_graph.iter_nodes():
        is_gpu = any(c.op.gpu for c in subtask.chunk_graph)
        subtask.required_resource = (
            Resource(num_gpus=1) if is_gpu else Resource(num_cpus=1)
        )


class ResourceEvaluator(ABC):
    """
    Resource evaluator is used to estimate and set resources required by
    subtasks. It can be an internal service or an external service. If it
    is an internal service, we can set default of adjustable resources for
    subtasks. If it is an external service, we should report the running
    result of the task to the external service, so that it can accurately
    predict the required resources of subtasks based on the historical
    running information, we call it HBO.

    Best practice
    ----------
    You can follow the steps below to add a new resource evaluator:
        * Inherit `ResourceEvaluator` and implement `create`, `evaluate`
          and `report` methods. The `create` method is to create a new
          resource evaluator instance. The `evaluate` method is to estimate
          and set required resources for the subtasks of a task stage. And
          this method must be implemented. The `report` method is to report
          the running information and result of the task. And this method
          does not have to be implemented.

        * Add default configs of the new evaluator needed in `base_config.xml`
          or its descendant files.

        * Set the `resource_evaluator` to choose a resource evaluator in
          `base_config.xml` when running a mars job.
    """

    name = None

    @classmethod
    @abstractmethod
    async def create(cls, config: Dict[str, Any], **kwargs) -> "ResourceEvaluator":
        name = config.get("resource_evaluator", "default")
        evaluator_config = config.get(name, {})
        evaluator_cls = _name_to_resource_evaluator[name]
        return await evaluator_cls.create(evaluator_config, **kwargs)

    @abstractmethod
    async def evaluate(self, stage_processor: "TaskStageProcessor"):  # noqa: F821
        """Called before executing a task stage."""

    @abstractmethod
    async def report(self):
        """Called after executing a task."""


@register_resource_evaluator
class DefaultEvaluator(ResourceEvaluator):
    name = "default"

    @classmethod
    async def create(cls, config, **kwargs) -> "ResourceEvaluator":
        return cls()

    async def evaluate(self, stage_processor: "TaskStageProcessor"):  # noqa: F821
        init_default_resource_for_subtask(stage_processor.subtask_graph)

    async def report(self):
        pass
