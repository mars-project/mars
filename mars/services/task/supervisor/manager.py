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

import asyncio
import importlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Type, Union

from .... import oscar as mo
from ....core import TileableGraph, TileableType, enter_mode, TileContext
from ....core.operand import Fetch
from ...subtask import SubtaskResult, SubtaskGraph
from ..config import task_options
from ..core import Task, new_task_id, TaskStatus
from ..errors import TaskNotExist
from .preprocessor import TaskPreprocessor
from .processor import TaskProcessor
from .task import TaskProcessorActor

logger = logging.getLogger(__name__)


class TaskConfigurationActor(mo.Actor):
    def __init__(
        self,
        task_conf: Dict[str, Any],
        execution_config: Dict[str, Any],
        task_processor_cls: Type[TaskProcessor] = None,
        task_preprocessor_cls: Type[TaskPreprocessor] = None,
    ):
        for name, value in task_conf.items():
            setattr(task_options, name, value)
        self._execution_config = execution_config
        self._task_processor_cls = task_processor_cls
        self._task_preprocessor_cls = task_preprocessor_cls

    def get_config(self):
        return {
            "task_options": task_options,
            "execution_config": self._execution_config,
            "task_processor_cls": self._task_processor_cls,
            "task_preprocessor_cls": self._task_preprocessor_cls,
        }


@dataclass
class ResultTileableInfo:
    tileable: TileableType
    processor_ref: Union[TaskProcessorActor, mo.ActorRef]


class TaskManagerActor(mo.Actor):
    _task_name_to_parent_task_id: Dict[str, str]
    _task_name_to_task_ids: Dict[str, List[str]]

    _task_id_to_processor_ref: Dict[str, Union[TaskProcessorActor, mo.ActorRef]]
    _tileable_key_to_info: Dict[str, List[ResultTileableInfo]]

    def __init__(self, session_id: str):
        self._session_id = session_id

        self._config = None
        self._execution_config = None
        self._task_processor_cls = None
        self._task_preprocessor_cls = None
        self._last_idle_time = None

        self._task_name_to_parent_task_id = dict()
        self._task_name_to_task_ids = defaultdict(list)

        self._task_id_to_processor_ref = dict()
        self._tileable_key_to_info = defaultdict(list)

    async def __post_create__(self):
        # get config
        configuration_ref = await mo.actor_ref(
            TaskConfigurationActor.default_uid(), address=self.address
        )
        task_conf = await configuration_ref.get_config()
        (
            self._config,
            self._execution_config,
            self._task_processor_cls,
            self._task_preprocessor_cls,
        ) = (
            task_conf["task_options"],
            task_conf["execution_config"],
            task_conf["task_processor_cls"],
            task_conf["task_preprocessor_cls"],
        )
        self._task_preprocessor_cls = self._get_task_preprocessor_cls()

    async def __pre_destroy__(self):
        for processor_ref in self._task_id_to_processor_ref.values():
            await processor_ref.destroy()

    @staticmethod
    def gen_uid(session_id):
        return f"{session_id}_task_manager"

    @enter_mode(kernel=True)
    async def submit_tileable_graph(
        self,
        graph: TileableGraph,
        task_name: str = None,
        fuse_enabled: bool = None,
        extra_config: dict = None,
    ) -> str:
        self._last_idle_time = None
        if task_name is None:
            # new task without task name
            task_id = task_name = new_task_id()
            parent_task_id = new_task_id()
        elif task_name in self._task_name_to_parent_task_id:
            # task with the same name submitted before
            parent_task_id = self._task_name_to_parent_task_id[task_name]
            task_id = new_task_id()
        else:
            # new task with task_name
            task_id = new_task_id()
            parent_task_id = new_task_id()

        uid = TaskProcessorActor.gen_uid(self._session_id, parent_task_id)
        if task_name not in self._task_name_to_parent_task_id:
            # gen main task which mean each submission from user
            processor_ref = await mo.create_actor(
                TaskProcessorActor,
                self._session_id,
                parent_task_id,
                task_name=task_name,
                task_processor_cls=self._task_processor_cls,
                address=self.address,
                uid=uid,
            )
            self._task_name_to_parent_task_id[task_name] = parent_task_id
        else:
            processor_ref = await mo.actor_ref(mo.ActorRef(self.address, uid))
        self._task_name_to_task_ids[task_name].append(task_id)
        self._task_id_to_processor_ref[task_id] = processor_ref

        if fuse_enabled is None:
            fuse_enabled = self._config.fuse_enabled
        # gen task
        task = Task(
            task_id,
            self._session_id,
            graph,
            task_name,
            parent_task_id=parent_task_id,
            fuse_enabled=fuse_enabled,
            extra_config=extra_config,
        )
        # gen task processor
        tiled_context = await self._gen_tiled_context(graph)
        await processor_ref.add_task(
            task,
            tiled_context,
            self._config,
            self._execution_config,
            self._task_preprocessor_cls,
        )

        for tileable in graph.result_tileables:
            info = ResultTileableInfo(tileable=tileable, processor_ref=processor_ref)
            self._tileable_key_to_info[tileable.key].append(info)

        return task_id

    async def get_subtask_graphs(self, task_id: str) -> List[SubtaskGraph]:
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f"Task {task_id} does not exist")

        return processor_ref.get_subtask_graphs(task_id)

    async def get_tileable_graph_dict_by_task_id(self, task_id: str):
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:
            raise TaskNotExist(f"Task {task_id} does not exist")

        res = await processor_ref.get_tileable_graph_as_dict()
        return res

    async def get_tileable_details(self, task_id):
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:
            raise TaskNotExist(f"Task {task_id} does not exist")

        return await processor_ref.get_tileable_details()

    async def get_tileable_subtasks(self, task_id, tileable_id, with_input_output):
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:
            raise TaskNotExist(f"Task {task_id} does not exist")

        return await processor_ref.get_tileable_subtasks(tileable_id, with_input_output)

    async def _gen_tiled_context(self, graph: TileableGraph) -> TileContext:
        # process graph, add fetch node to tiled context
        tiled_context = TileContext()
        for tileable in graph:
            if isinstance(tileable.op, Fetch) and tileable.is_coarse():
                info = self._tileable_key_to_info[tileable.key][-1]
                tiled_context[tileable] = await info.processor_ref.get_result_tileable(
                    tileable.key
                )
        return tiled_context

    def _get_task_preprocessor_cls(self):
        if self._task_preprocessor_cls is not None:
            assert isinstance(self._task_preprocessor_cls, str)
            module, name = self._task_preprocessor_cls.rsplit(".", 1)
            return getattr(importlib.import_module(module), name)
        else:
            return TaskPreprocessor

    async def wait_task(self, task_id: str, timeout: int = None):
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f"Task {task_id} does not exist")

        return processor_ref.wait(timeout)

    async def cancel_task(self, task_id: str):
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f"Task {task_id} does not exist")

        yield processor_ref.cancel()

    async def get_task_results(self, progress: bool = False):
        if not self._task_id_to_processor_ref:
            raise mo.Return([])

        results = yield asyncio.gather(
            *[ref.result() for ref in self._task_id_to_processor_ref.values()]
        )

        if progress:
            task_to_result = {res.task_id: res for res in results}

            progress_task_ids = []
            for res in results:
                if res.status != TaskStatus.terminated:
                    progress_task_ids.append(res.task_id)
                else:
                    res.progress = 1.0

            progresses = yield asyncio.gather(
                *[
                    self._task_id_to_processor_ref[task_id].progress()
                    for task_id in progress_task_ids
                ]
            )
            for task_id, progress in zip(progress_task_ids, progresses):
                task_to_result[task_id].progress = progress

        raise mo.Return(results)

    async def get_task_result(self, task_id: str):
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f"Task {task_id} does not exist")

        return await processor_ref.result()

    async def get_task_result_tileables(self, task_id: str):
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f"Task {task_id} does not exist")

        return await processor_ref.get_result_tileables()

    async def set_subtask_result(self, subtask_result: SubtaskResult):
        task_id = subtask_result.task_id
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:  # pragma: no cover
            # raise TaskNotExist(f'Task {task_id} does not exist')
            logger.warning(
                "Current task is finished, got stale result %s  for subtask %s "
                "which may be speculative execution from previous tasks, just ignore it.",
                subtask_result.subtask_id,
                subtask_result,
            )
            return

        yield processor_ref.set_subtask_result(subtask_result)

    @mo.extensible
    async def get_task_progress(self, task_id: str) -> float:
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f"Task {task_id} does not exist")

        return await processor_ref.progress()

    async def get_last_idle_time(self):
        if self._last_idle_time is None:
            for processor_ref in self._task_id_to_processor_ref.values():
                if not await processor_ref.is_done():
                    break
            else:
                self._last_idle_time = time.time()
        return self._last_idle_time
