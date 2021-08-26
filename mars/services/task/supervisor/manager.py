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
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Type, Union, Optional

from .... import oscar as mo
from ....core import TileableGraph, TileableType, enter_mode
from ....core.context import set_context
from ....core.operand import Fetch
from ...cluster.api import ClusterAPI
from ...context import ThreadedServiceContext
from ...lifecycle.api import LifecycleAPI
from ...meta import MetaAPI
from ...scheduling import SchedulingAPI
from ...subtask import SubtaskResult
from ..config import task_options
from ..core import Task, new_task_id, TaskStatus
from ..errors import TaskNotExist
from .preprocessor import TaskPreprocessor
from .processor import TaskProcessorActor


class TaskConfigurationActor(mo.Actor):
    def __init__(self,
                 task_conf: Dict[str, Any],
                 task_preprocessor_cls: Type[TaskPreprocessor] = None):
        for name, value in task_conf.items():
            setattr(task_options, name, value)
        self._task_preprocessor_cls = task_preprocessor_cls

    def get_config(self):
        return {
            'task_options': task_options,
            'task_preprocessor_cls': self._task_preprocessor_cls
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

    _cluster_api: Optional[ClusterAPI]
    _meta_api: Optional[MetaAPI]
    _lifecycle_api: Optional[LifecycleAPI]

    def __init__(self, session_id: str):
        self._session_id = session_id

        self._config = None
        self._task_preprocessor_cls = None
        self._last_idle_time = None

        self._task_name_to_parent_task_id = dict()
        self._task_name_to_task_ids = defaultdict(list)

        self._task_id_to_processor_ref = dict()
        self._tileable_key_to_info = defaultdict(list)

        self._cluster_api = None
        self._meta_api = None
        self._lifecycle_api = None
        self._scheduling_api = None

    async def __post_create__(self):
        self._cluster_api = await ClusterAPI.create(self.address)
        self._scheduling_api = await SchedulingAPI.create(self._session_id, self.address)
        self._meta_api = await MetaAPI.create(self._session_id, self.address)
        self._lifecycle_api = await LifecycleAPI.create(
            self._session_id, self.address)

        # get config
        configuration_ref = await mo.actor_ref(
            TaskConfigurationActor.default_uid(),
            address=self.address)
        task_conf = await configuration_ref.get_config()
        self._config, self._task_preprocessor_cls = \
            task_conf['task_options'], task_conf['task_preprocessor_cls']
        self._task_preprocessor_cls = self._get_task_preprocessor_cls()

        # init context
        await self._init_context()

    async def __pre_destroy__(self):
        for processor_ref in self._task_id_to_processor_ref.values():
            await processor_ref.destroy()

    async def _init_context(self):
        loop = asyncio.get_running_loop()
        context = ThreadedServiceContext(
            self._session_id, self.address, self.address, loop=loop)
        await context.init()
        set_context(context)

    @staticmethod
    def gen_uid(session_id):
        return f'{session_id}_task_manager'

    @enter_mode(kernel=True)
    async def submit_tileable_graph(self,
                                    graph: TileableGraph,
                                    task_name: str = None,
                                    fuse_enabled: bool = None,
                                    extra_config: dict = None) -> str:
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
                TaskProcessorActor, self._session_id, parent_task_id,
                task_name=task_name, address=self.address, uid=uid)
            self._task_name_to_parent_task_id[task_name] = parent_task_id
        else:
            processor_ref = await mo.actor_ref(mo.ActorRef(self.address, uid))
        self._task_name_to_task_ids[task_name].append(task_id)
        self._task_id_to_processor_ref[task_id] = processor_ref

        if fuse_enabled is None:
            fuse_enabled = self._config.fuse_enabled
        # gen task
        task = Task(task_id, self._session_id,
                    graph, task_name,
                    parent_task_id=parent_task_id,
                    fuse_enabled=fuse_enabled,
                    extra_config=extra_config)
        # gen task processor
        tiled_context = await self._gen_tiled_context(graph)
        await processor_ref.add_task(
            task, tiled_context, self._config, self._task_preprocessor_cls)

        for tileable in graph.result_tileables:
            info = ResultTileableInfo(tileable=tileable,
                                      processor_ref=processor_ref)
            self._tileable_key_to_info[tileable.key].append(info)

        return task_id

    async def get_tileable_graph_dict_by_task_id(self, task_id):
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:
            raise TaskNotExist(f'Task {task_id} does not exist')

        res = await processor_ref.get_tileable_graph_as_dict()
        return res

    async def get_tileable_details(self, task_id):
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:
            raise TaskNotExist(f'Task {task_id} does not exist')

        return await processor_ref.get_tileable_details()

    async def _gen_tiled_context(self, graph: TileableGraph) -> \
            Dict[TileableType, TileableType]:
        # process graph, add fetch node to tiled context
        tiled_context = dict()
        for tileable in graph:
            if isinstance(tileable.op, Fetch) and tileable.is_coarse():
                info = self._tileable_key_to_info[tileable.key][-1]
                tiled_context[tileable] = \
                    await info.processor_ref.get_result_tileable(tileable.key)
        return tiled_context

    def _get_task_preprocessor_cls(self):
        if self._task_preprocessor_cls is not None:
            assert isinstance(self._task_preprocessor_cls, str)
            module, name = self._task_preprocessor_cls.rsplit('.', 1)
            return getattr(importlib.import_module(module), name)
        else:
            return TaskPreprocessor

    async def wait_task(self,
                        task_id: str,
                        timeout: int = None):
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f'Task {task_id} does not exist')

        return processor_ref.wait(timeout)

    async def cancel_task(self, task_id: str):
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f'Task {task_id} does not exist')

        yield processor_ref.cancel()

    async def get_task_results(self, progress: bool = False):
        if not self._task_id_to_processor_ref:
            raise mo.Return([])

        results = yield asyncio.gather(*[
            ref.result() for ref in self._task_id_to_processor_ref.values()
        ])

        if progress:
            task_to_result = {res.task_id: res for res in results}

            progress_task_ids = []
            for res in results:
                if res.status != TaskStatus.terminated:
                    progress_task_ids.append(res.task_id)
                else:
                    res.progress = 1.0

            progresses = yield asyncio.gather(*[
                self._task_id_to_processor_ref[task_id].progress()
                for task_id in progress_task_ids
            ])
            for task_id, progress in zip(progress_task_ids, progresses):
                task_to_result[task_id].progress = progress

        raise mo.Return(results)

    async def get_task_result(self, task_id: str):
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f'Task {task_id} does not exist')

        return await processor_ref.result()

    async def get_task_result_tileables(self, task_id: str):
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f'Task {task_id} does not exist')

        return await processor_ref.get_result_tileables()

    async def set_subtask_result(self, subtask_result: SubtaskResult):
        task_id = subtask_result.task_id
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f'Task {task_id} does not exist')

        yield processor_ref.set_subtask_result(subtask_result)

    @mo.extensible
    async def get_task_progress(self, task_id: str) -> float:
        try:
            processor_ref = self._task_id_to_processor_ref[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f'Task {task_id} does not exist')

        return await processor_ref.progress()

    async def get_last_idle_time(self):
        if self._last_idle_time is None:
            for processor_ref in self._task_id_to_processor_ref.values():
                if not await processor_ref.is_done():
                    break
            else:
                self._last_idle_time = time.time()
        return self._last_idle_time
