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

import asyncio
import dataclasses
import importlib
import logging
from typing import Any, Dict, Optional, Set, Type, List

from .... import oscar as mo
from ....config import Config
from ....core import TileContext
from ....core.operand import Fetch
from ....typing import TileableType
from ....utils import build_fetch
from ...subtask import SubtaskResult, SubtaskStatus, SubtaskGraph
from ..core import Task, TaskStatus
from ..execution.api import TaskExecutor
from .preprocessor import TaskPreprocessor
from .processor import TaskProcessor

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _TileableStageInfo:
    progress: float
    subtask_ids: Set[str]


@dataclasses.dataclass
class _TileableDetailInfo:
    progress: float
    subtask_count: int
    status: int
    properties: Dict[str, Any]


class _TaskInfoProcessorMixin:
    _task_id_to_processor: Dict[str, TaskProcessor]
    _tileable_to_details_cache: Dict[TileableType, _TileableDetailInfo]

    def _init_cache(self):
        try:
            return self._tileable_to_details_cache
        except AttributeError:
            cache = self._tileable_to_details_cache = dict()
            return cache

    def _get_all_subtask_results(self) -> Dict[str, SubtaskResult]:
        subtask_results = dict()
        for processor in self._task_id_to_processor.values():
            for stage in processor.stage_processors:
                for subtask, result in stage.subtask_results.items():
                    subtask_results[subtask.subtask_id] = result
                for subtask, result in stage.subtask_snapshots.items():
                    if subtask.subtask_id in subtask_results:
                        continue
                    subtask_results[subtask.subtask_id] = result
        return subtask_results

    def _get_tileable_infos(self) -> Dict[TileableType, _TileableDetailInfo]:
        cache = self._init_cache()

        tileable_to_stage_infos: Dict[TileableType, List[_TileableStageInfo]] = dict()
        for processor in self._task_id_to_processor.values():
            tile_context = processor.tile_context
            for tileable, infos in tile_context.get_tileable_tile_infos().items():
                tileable_to_stage_infos[tileable] = []
                if tileable in cache:
                    # cached
                    continue
                for info in infos:
                    chunks = [
                        c for c in info.generated_chunks if not isinstance(c.op, Fetch)
                    ]
                    try:
                        subtask_ids = {
                            st.subtask_id for st in processor.get_subtasks(chunks)
                        }
                    except KeyError:  # pragma: no cover
                        subtask_ids = None
                    stage_info = _TileableStageInfo(
                        progress=info.tile_progress, subtask_ids=subtask_ids
                    )
                    tileable_to_stage_infos[tileable].append(stage_info)

        tileable_to_defails = dict()
        subtask_id_to_results = self._get_all_subtask_results()
        for tileable, infos in tileable_to_stage_infos.items():
            if tileable in cache:
                # cached
                tileable_to_defails[tileable] = cache[tileable]
                continue

            statuses = set()
            progress = 0.0 if not isinstance(tileable.op, Fetch) else 1.0
            n_subtask = 0
            for stage_info in infos:
                tile_progress = stage_info.progress
                stage_progress = 0.0
                if stage_info.subtask_ids is None:
                    continue
                for subtask_id in stage_info.subtask_ids:
                    try:
                        result = subtask_id_to_results[subtask_id]
                        stage_progress += result.progress * tile_progress
                        statuses.add(result.status)
                    except KeyError:
                        # pending
                        statuses.add(SubtaskStatus.pending)
                n_subtask += len(stage_info.subtask_ids)
                if stage_info.subtask_ids:
                    progress += stage_progress / len(stage_info.subtask_ids)
                else:
                    progress += tile_progress

            # calc status
            if (not statuses or statuses == {SubtaskStatus.succeeded}) and abs(
                progress - 1.0
            ) < 1e-3:
                status = SubtaskStatus.succeeded
            elif statuses == {SubtaskStatus.cancelled}:
                status = SubtaskStatus.cancelled
            elif statuses == {SubtaskStatus.pending}:
                status = SubtaskStatus.pending
            elif SubtaskStatus.errored in statuses:
                status = SubtaskStatus.errored
            else:
                status = SubtaskStatus.running

            props = tileable.op.to_kv(
                exclude_fields=("_key", "_id"), accept_value_types=(int, float, str)
            )
            info = _TileableDetailInfo(
                progress=progress,
                subtask_count=n_subtask,
                status=status.value,
                properties=props,
            )
            tileable_to_defails[tileable] = info
            if status.is_done and tileable not in cache:
                cache[tileable] = info

        return tileable_to_defails

    async def get_tileable_details(self):
        tileable_to_details = yield asyncio.to_thread(self._get_tileable_infos)
        raise mo.Return(
            {
                t.key: {
                    "progress": info.progress,
                    "subtaskCount": info.subtask_count,
                    "status": info.status,
                    "properties": info.properties,
                }
                for t, info in tileable_to_details.items()
            }
        )

    def _get_tileable_graph_as_dict(self):
        processor = list(self._task_id_to_processor.values())[-1]
        tileable_graph = processor.tileable_graph

        node_list = []
        edge_list = []

        visited = set()

        for chunk in tileable_graph:
            if chunk.key in visited:  # pragma: no cover
                continue
            visited.add(chunk.key)

            node_name = str(chunk.op)

            node_list.append({"tileableId": chunk.key, "tileableName": node_name})
            for inp, is_pure_dep in zip(chunk.inputs, chunk.op.pure_depends):
                if inp not in tileable_graph:  # pragma: no cover
                    continue
                edge_list.append(
                    {
                        "fromTileableId": inp.key,
                        "toTileableId": chunk.key,
                        "linkType": 1 if is_pure_dep else 0,
                    }
                )

        graph_dict = {"tileables": node_list, "dependencies": edge_list}
        return graph_dict

    async def get_tileable_graph_as_dict(self):
        return await asyncio.to_thread(self._get_tileable_graph_as_dict)

    def _get_tileable_subtasks(self, tileable_id: str, with_input_output: bool):
        returned_subtasks = dict()
        subtask_id_to_types = dict()

        subtask_details = dict()
        subtask_graph = subtask_results = subtask_snapshots = None
        for processor in self._task_id_to_processor.values():
            tileable_to_subtasks = processor.get_tileable_to_subtasks()
            tileable_id_to_tileable = processor.tileable_id_to_tileable
            for stage in processor.stage_processors:
                if tileable_id in tileable_id_to_tileable:
                    tileable = tileable_id_to_tileable[tileable_id]
                    returned_subtasks = {
                        subtask.subtask_id: subtask
                        for subtask in tileable_to_subtasks[tileable]
                    }
                    subtask_graph = stage.subtask_graph
                    subtask_results = stage.subtask_results
                    subtask_snapshots = stage.subtask_snapshots
                    break
            if returned_subtasks:
                break

        if subtask_graph is None:  # pragma: no cover
            return {}

        if with_input_output:
            for subtask in list(returned_subtasks.values()):
                for pred in subtask_graph.iter_predecessors(subtask):
                    if pred.subtask_id in returned_subtasks:  # pragma: no cover
                        continue
                    returned_subtasks[pred.subtask_id] = pred
                    subtask_id_to_types[pred.subtask_id] = "Input"
                for succ in subtask_graph.iter_successors(subtask):
                    if succ.subtask_id in returned_subtasks:  # pragma: no cover
                        continue
                    returned_subtasks[succ.subtask_id] = succ
                    subtask_id_to_types[succ.subtask_id] = "Output"

        for subtask in returned_subtasks.values():
            subtask_result = subtask_results.get(
                subtask,
                subtask_snapshots.get(
                    subtask,
                    SubtaskResult(
                        progress=0.0,
                        status=SubtaskStatus.pending,
                        stage_id=subtask.stage_id,
                    ),
                ),
            )
            subtask_details[subtask.subtask_id] = {
                "name": subtask.subtask_name,
                "status": subtask_result.status.value,
                "progress": subtask_result.progress,
                "nodeType": subtask_id_to_types.get(subtask.subtask_id, "Calculation"),
            }

        for subtask in returned_subtasks.values():
            pred_ids = []
            for pred in subtask_graph.iter_predecessors(subtask):
                if pred.subtask_id in returned_subtasks:
                    pred_ids.append(pred.subtask_id)
            subtask_details[subtask.subtask_id]["fromSubtaskIds"] = pred_ids
        return subtask_details

    async def get_tileable_subtasks(self, tileable_id: str, with_input_output: bool):
        return await asyncio.to_thread(
            self._get_tileable_subtasks, tileable_id, with_input_output
        )


class TaskProcessorActor(mo.Actor, _TaskInfoProcessorMixin):
    _task_id_to_processor: Dict[str, TaskProcessor]
    _cur_processor: Optional[TaskProcessor]

    def __init__(
        self,
        session_id: str,
        task_id: str,
        task_name: str = None,
        task_processor_cls: Type[TaskPreprocessor] = None,
    ):
        self.session_id = session_id
        self.task_id = task_id
        self.task_name = task_name

        self._task_processor_cls = self._get_task_processor_cls(task_processor_cls)
        self._task_id_to_processor = dict()
        self._cur_processor = None

    @classmethod
    def gen_uid(cls, session_id: str, task_id: str):
        return f"task_processor_{session_id}_{task_id}"

    async def add_task(
        self,
        task: Task,
        tiled_context: TileContext,
        config: Config,
        execution_config: Dict,
        task_preprocessor_cls: Type[TaskPreprocessor],
    ):
        task_preprocessor = task_preprocessor_cls(
            task, tiled_context=tiled_context, config=config
        )
        task_executor = await TaskExecutor.create(
            execution_config,
            task=task,
            session_id=self.session_id,
            address=self.address,
            tile_context=task_preprocessor.tile_context,
        )
        processor = self._task_processor_cls(
            task,
            task_preprocessor,
            task_executor,
        )
        self._task_id_to_processor[task.task_id] = processor

        # tell self to start running
        await self.ref().start.tell()

    @classmethod
    def _get_task_processor_cls(cls, task_processor_cls):
        if task_processor_cls is not None:  # pragma: no cover
            assert isinstance(task_processor_cls, str)
            module, name = task_processor_cls.rsplit(".", 1)
            return getattr(importlib.import_module(module), name)
        else:
            return TaskProcessor

    def _get_unprocessed_task_processor(self):
        for processor in self._task_id_to_processor.values():
            if processor.result.status == TaskStatus.pending:
                return processor

    async def start(self):
        if self._cur_processor is not None:  # pragma: no cover
            # some processor is running
            return

        processor = self._get_unprocessed_task_processor()
        if processor is None:  # pragma: no cover
            return
        self._cur_processor = processor
        try:
            yield processor.run()
        finally:
            self._cur_processor = None

    async def wait(self, timeout: int = None):
        fs = [
            asyncio.ensure_future(processor.done.wait())
            for processor in self._task_id_to_processor.values()
        ]

        _, pending = yield asyncio.wait(fs, timeout=timeout)
        if not pending:
            raise mo.Return(self.result())
        else:
            _ = [fut.cancel() for fut in pending]

    async def cancel(self):
        if self._cur_processor:
            await self._cur_processor.cancel()

    def result(self):
        terminated_result = None
        for processor in self._task_id_to_processor.values():
            if processor.result.status != TaskStatus.terminated:
                return processor.result
            else:
                terminated_result = processor.result
        return terminated_result

    async def progress(self):
        processor_progresses = [
            await processor.get_progress()
            for processor in self._task_id_to_processor.values()
        ]
        return sum(processor_progresses) / len(processor_progresses)

    def get_result_tileables(self):
        processor = list(self._task_id_to_processor.values())[-1]
        tileable_graph = processor.tileable_graph
        result = []
        for result_tileable in tileable_graph.result_tileables:
            tiled = processor.get_tiled(result_tileable)
            result.append(build_fetch(tiled))
        return result

    def get_subtask_graphs(self, task_id: str) -> List[SubtaskGraph]:
        return [
            stage_processor.subtask_graph
            for stage_processor in self._task_id_to_processor[task_id].stage_processors
        ]

    def get_result_tileable(self, tileable_key: str):
        processor = list(self._task_id_to_processor.values())[-1]
        tileable_graph = processor.tileable_graph
        for result_tileable in tileable_graph.result_tileables:
            if result_tileable.key == tileable_key:
                tiled = processor.get_tiled(result_tileable)
                return build_fetch(tiled)
        raise KeyError(f"Tileable {tileable_key} does not exist")  # pragma: no cover

    async def set_subtask_result(self, subtask_result: SubtaskResult):
        logger.debug(
            "Set subtask %s with result %s.", subtask_result.subtask_id, subtask_result
        )
        if self._cur_processor is not None:
            await self._cur_processor.set_subtask_result(subtask_result)

    def is_done(self) -> bool:
        for processor in self._task_id_to_processor.values():
            if not processor.is_done():
                return False
        return True
