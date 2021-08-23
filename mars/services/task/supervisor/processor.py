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
import itertools
import operator
import sys
import time
from collections import defaultdict
from functools import reduce, wraps
from typing import Callable, Coroutine, Dict, Iterator, \
    List, Optional, Set, Type, Union

from .... import oscar as mo
from ....config import Config
from ....core import ChunkGraph, TileableGraph
from ....core.operand import Fetch, FetchShuffle, MapReduceOperand, \
    ShuffleProxy, OperandStage
from ....optimization.logical import OptimizationRecords
from ....typing import TileableType, BandType
from ....utils import build_fetch
from ...cluster.api import ClusterAPI
from ...lifecycle.api import LifecycleAPI
from ...meta.api import MetaAPI
from ...scheduling import SchedulingAPI
from ...subtask import Subtask, SubtaskResult, SubtaskStatus, SubtaskGraph
from ..core import Task, TaskResult, TaskStatus, new_task_id
from .preprocessor import TaskPreprocessor
from .stage import TaskStageProcessor


def _record_error(func: Union[Callable, Coroutine]):
    assert asyncio.iscoroutinefunction(func)

    @wraps(func)
    async def inner(processor: "TaskProcessor", *args, **kwargs):
        try:
            return await func(processor, *args, **kwargs)
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except  # pragma: no cover
            processor._err_infos.append(sys.exc_info())
            raise

    return inner


class TaskProcessor:
    stage_processors: List[TaskStageProcessor]
    cur_stage_processor: Optional[TaskStageProcessor]

    def __init__(self,
                 task: Task,
                 preprocessor: TaskPreprocessor,
                 # APIs
                 cluster_api: ClusterAPI,
                 lifecycle_api: LifecycleAPI,
                 scheduling_api: SchedulingAPI,
                 meta_api: MetaAPI):
        self._task = task
        self._preprocessor = preprocessor

        # APIs
        self._cluster_api = cluster_api
        self._lifecycle_api = lifecycle_api
        self._scheduling_api = scheduling_api
        self._meta_api = meta_api

        self.result = TaskResult(
            task_id=task.task_id, session_id=task.session_id, start_time=time.time(),
            status=TaskStatus.pending)
        self.done = asyncio.Event()
        self.stage_processors = []
        self.cur_stage_processor = None

        self._err_infos = []
        self._chunk_graph_iter = None
        self._raw_tile_context = preprocessor.tile_context.copy()
        self._lifecycle_processed_tileables = set()

    @property
    def preprocessor(self):
        return self._preprocessor

    @property
    def tileable_graph(self):
        return self._preprocessor.tileable_graph

    @property
    def tile_context(self):
        return self._preprocessor.tile_context

    def get_tiled(self, tileable: TileableType):
        return self._preprocessor.get_tiled(tileable)

    @_record_error
    async def optimize(self) -> TileableGraph:
        # optimization, run it in executor,
        # since optimization may be a CPU intensive operation
        return await asyncio.to_thread(self._preprocessor.optimize)

    @_record_error
    async def incref_fetch_tileables(self):
        # incref fetch tileables in tileable graph to prevent them from deleting
        to_incref_tileable_keys = [
            tileable.op.source_key for tileable in self.tileable_graph
            if isinstance(tileable.op, Fetch) and tileable in self._raw_tile_context]
        await self._lifecycle_api.incref_tileables(to_incref_tileable_keys)

    @_record_error
    async def decref_fetch_tileables(self):
        fetch_tileable_keys = [
            tileable.op.source_key for tileable in self.tileable_graph
            if isinstance(tileable.op, Fetch) and tileable in self._raw_tile_context]
        await self._lifecycle_api.decref_tileables(fetch_tileable_keys)

    @_record_error
    async def incref_result_tileables(self):
        processed = self._lifecycle_processed_tileables
        # track and incref result tileables if tiled
        tracks = [], []
        for result_tileable in self.tileable_graph.result_tileables:
            if result_tileable in processed:  # pragma: no cover
                continue
            try:
                tiled_tileable = self._preprocessor.get_tiled(result_tileable)
                tracks[0].append(result_tileable.key)
                tracks[1].append(self._lifecycle_api.track.delay(
                    result_tileable.key, [c.key for c in tiled_tileable.chunks]))
                processed.add(result_tileable)
            except KeyError:
                # not tiled, skip
                pass
        if tracks:
            await self._lifecycle_api.track.batch(*tracks[1])
            await self._lifecycle_api.incref_tileables(tracks[0])

    @_record_error
    async def decref_result_tileables(self):
        await self._lifecycle_api.decref_tileables(
            [t.key for t in self._lifecycle_processed_tileables])

    @_record_error
    async def incref_stage(self, stage_processor: "TaskStageProcessor"):
        subtask_graph = stage_processor.subtask_graph
        incref_chunk_keys = []
        for subtask in subtask_graph:
            # for subtask has successors, incref number of successors
            n = subtask_graph.count_successors(subtask)
            for c in subtask.chunk_graph.results:
                incref_chunk_keys.extend([c.key] * n)
            # process reducer, since mapper will generate sub keys
            # we incref (main_key, sub_key) for reducer
            for chunk in subtask.chunk_graph:
                if isinstance(chunk.op, MapReduceOperand) and \
                        chunk.op.stage == OperandStage.reduce:
                    # reducer
                    data_keys = chunk.op.get_dependent_data_keys()
                    incref_chunk_keys.extend(data_keys)
                    # main key incref as well, to ensure existence of meta
                    incref_chunk_keys.extend([key[0] for key in data_keys])
        result_chunks = stage_processor.chunk_graph.result_chunks
        incref_chunk_keys.extend([c.key for c in result_chunks])
        await self._lifecycle_api.incref_chunks(incref_chunk_keys)

    @classmethod
    def _get_decref_stage_chunk_keys(cls,
                                     stage_processor: "TaskStageProcessor") -> List[str]:
        decref_chunk_keys = []
        error_or_cancelled = stage_processor.error_or_cancelled()
        if stage_processor.subtask_graph:
            subtask_graph = stage_processor.subtask_graph
            if error_or_cancelled:
                # error or cancel, rollback incref for subtask results
                for subtask in subtask_graph:
                    if stage_processor.subtask_results.get(subtask):
                        continue
                    # if subtask not executed, rollback incref of predecessors
                    for inp_subtask in subtask_graph.predecessors(subtask):
                        for result_chunk in inp_subtask.chunk_graph.results:
                            # for reducer chunk, decref mapper chunks
                            if isinstance(result_chunk.op, ShuffleProxy):
                                for chunk in subtask.chunk_graph:
                                    if isinstance(chunk.op, MapReduceOperand):
                                        data_keys = chunk.op.get_dependent_data_keys()
                                        decref_chunk_keys.extend(data_keys)
                                        decref_chunk_keys.extend(
                                            [key[0] for key in data_keys])
                        decref_chunk_keys.extend(
                            [c.key for c in inp_subtask.chunk_graph.results])
            # decref result of chunk graphs
            decref_chunk_keys.extend(
                [c.key for c in stage_processor.chunk_graph.results])
        return decref_chunk_keys

    @mo.extensible
    @_record_error
    async def decref_stage(self, stage_processor: "TaskStageProcessor"):
        decref_chunk_keys = self._get_decref_stage_chunk_keys(stage_processor)
        await self._lifecycle_api.decref_chunks(decref_chunk_keys)

    @decref_stage.batch
    @_record_error
    async def decref_stage(self, args_list, kwargs_list):
        decref_chunk_keys = []
        for args, kwargs in zip(args_list, kwargs_list):
            decref_chunk_keys.extend(
                self._get_decref_stage_chunk_keys(*args, **kwargs))
        await self._lifecycle_api.decref_chunks(decref_chunk_keys)

    async def _get_next_chunk_graph(self,
                                    chunk_graph_iter: Iterator[ChunkGraph]) \
            -> Optional[ChunkGraph]:
        def next_chunk_graph():
            try:
                return next(chunk_graph_iter)
            except StopIteration:
                return

        fut = asyncio.to_thread(next_chunk_graph)
        chunk_graph = await fut
        return chunk_graph

    async def _get_available_band_slots(self) -> Dict[BandType, int]:
        async for bands in self._cluster_api.watch_all_bands():
            if bands:
                return bands

    def _init_chunk_graph_iter(self, tileable_graph: TileableGraph):
        if self._chunk_graph_iter is None:
            self._chunk_graph_iter = iter(self._preprocessor.tile(tileable_graph))

    def _get_chunk_optimization_records(self) -> OptimizationRecords:
        if self._preprocessor.chunk_optimization_records_list:
            return self._preprocessor.chunk_optimization_records_list[-1]

    def _get_tileable_to_subtasks(self, subtask_graph: SubtaskGraph) \
            -> Dict[TileableType, List[Subtask]]:
        tileable_to_chunks = defaultdict(set)
        chunk_to_subtasks = dict()

        tileable_graph = self._preprocessor.tileable_graph
        tile_ctx = self._preprocessor.tile_context

        for tileable in tileable_graph:
            if tileable not in tile_ctx:
                continue
            for chunk in tile_ctx[tileable].chunks:
                tileable_to_chunks[tileable].add(chunk.key)
                # register chunk mapping for tiled terminals
                chunk_to_subtasks[chunk.key] = set()

        for subtask in subtask_graph:
            for chunk in subtask.chunk_graph:
                # for every non-fuse chunks (including fused),
                # register subtasks if needed
                if isinstance(chunk.op, (FetchShuffle, Fetch)) or \
                        chunk.key not in chunk_to_subtasks:
                    continue
                chunk_to_subtasks[chunk.key].add(subtask)

        tileable_to_subtasks = dict()
        # collect subtasks for tileables
        for tileable, chunk_keys in tileable_to_chunks.items():
            tileable_to_subtasks[tileable] = list(reduce(
                operator.or_,
                [chunk_to_subtasks[chunk_key] for chunk_key in chunk_keys]
            ))
        return tileable_to_subtasks

    @_record_error
    async def get_next_stage_processor(self) \
            -> Optional[TaskStageProcessor]:
        tileable_graph = self._preprocessor.tileable_graph
        self._init_chunk_graph_iter(tileable_graph)

        chunk_graph = await self._get_next_chunk_graph(self._chunk_graph_iter)
        if chunk_graph is None:
            # tile finished
            self._preprocessor.done = True
            return

        # gen subtask graph
        available_bands = await self._get_available_band_slots()
        subtask_graph = await asyncio.to_thread(
            self._preprocessor.analyze, chunk_graph, available_bands)
        tileable_to_subtasks = await asyncio.to_thread(
            self._get_tileable_to_subtasks, subtask_graph)
        stage_processor = TaskStageProcessor(
            new_task_id(), self._task, chunk_graph, subtask_graph,
            list(available_bands), tileable_to_subtasks,
            self._get_chunk_optimization_records(),
            self._scheduling_api, self._meta_api)
        return stage_processor

    @_record_error
    async def schedule(self, stage_processor: TaskStageProcessor):
        await stage_processor.run()

    def gen_result(self):
        self.result.status = TaskStatus.terminated
        self.result.end_time = time.time()
        for stage_processor in self.stage_processors:
            if stage_processor.result.error is not None:
                err = stage_processor.result.error
                tb = stage_processor.result.traceback
                self._err_infos.append((type(err), err, tb))
        if self._err_infos:
            # grab the last error
            _, err, tb = self._err_infos[-1]
            self.result.error = err
            self.result.traceback = tb

    def finish(self):
        self.done.set()

    def is_done(self) -> bool:
        return self.done.is_set()


class TaskProcessorActor(mo.Actor):
    _task_id_to_processor: Dict[str, TaskProcessor]
    _processed_task_ids: Set[str]
    _cur_processor: Optional[TaskProcessor]

    def __init__(self,
                 session_id: str,
                 task_id: str,
                 task_name: str = None):
        self.session_id = session_id
        self.task_id = task_id
        self.task_name = task_name

        self._task_id_to_processor = dict()
        self._cur_processor = None

        self._cluster_api = None
        self._meta_api = None
        self._lifecycle_api = None
        self._scheduling_api = None

    async def __post_create__(self):
        self._cluster_api = await ClusterAPI.create(self.address)
        self._scheduling_api = await SchedulingAPI.create(self.session_id, self.address)
        self._meta_api = await MetaAPI.create(self.session_id, self.address)
        self._lifecycle_api = await LifecycleAPI.create(
            self.session_id, self.address)

    @classmethod
    def gen_uid(cls, session_id: str, task_id: str):
        return f'task_processor_{session_id}_{task_id}'

    async def add_task(self,
                       task: Task,
                       tiled_context: Dict[TileableType, TileableType],
                       config: Config,
                       task_preprocessor_cls: Type[TaskPreprocessor]):
        task_preprocessor = task_preprocessor_cls(
            task, tiled_context=tiled_context, config=config)
        processor = TaskProcessor(task, task_preprocessor,
                                  self._cluster_api, self._lifecycle_api,
                                  self._scheduling_api, self._meta_api)
        self._task_id_to_processor[task.task_id] = processor

        # tell self to start running
        await self.ref().start.tell()

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
        processor.result.status = TaskStatus.running

        incref_fetch, incref_result = False, False
        try:
            # optimization
            yield processor.optimize()
            # incref fetch tileables to ensure fetch data not deleted
            yield processor.incref_fetch_tileables()
            incref_fetch = True
            while True:
                stage_processor = yield processor.get_next_stage_processor()
                if stage_processor is None:
                    break
                # track and incref result tileables
                yield processor.incref_result_tileables()
                incref_result = True
                # incref stage
                yield processor.incref_stage(stage_processor)
                # schedule stage
                processor.stage_processors.append(stage_processor)
                processor.cur_stage_processor = stage_processor
                yield processor.schedule(stage_processor)
                if stage_processor.error_or_cancelled():
                    break
        finally:
            processor.gen_result()
            processor.cur_stage_processor = None
            try:
                # clean ups
                decrefs = []
                error_or_cancelled = False
                for stage_processor in processor.stage_processors:
                    if stage_processor.error_or_cancelled():
                        error_or_cancelled = True
                    decrefs.append(processor.decref_stage.delay(stage_processor))
                yield processor.decref_stage.batch(*decrefs)
                if incref_fetch:
                    # revert fetch incref
                    yield processor.decref_fetch_tileables()
                if incref_result and error_or_cancelled:
                    # revert result incref if error or cancelled
                    yield processor.decref_result_tileables()
            finally:
                processor.finish()
                self._cur_processor = None

    async def wait(self, timeout: int = None):
        fs = [processor.done.wait() for processor
              in self._task_id_to_processor.values()]

        _, pending = yield asyncio.wait(fs, timeout=timeout)
        if not pending:
            raise mo.Return(self.result())

    async def cancel(self):
        if self._cur_processor:
            if not self._cur_processor.cur_stage_processor:
                # still in preprocess
                self._cur_processor.preprocessor.cancel()
            else:
                # otherwise, already in stages, cancel current running stage
                await self._cur_processor.cur_stage_processor.cancel()

    def result(self):
        terminated_result = None
        for processor in self._task_id_to_processor.values():
            if processor.result.status != TaskStatus.terminated:
                return processor.result
            else:
                terminated_result = processor.result
        return terminated_result

    def progress(self):
        tiled_percentage = 0.0
        i = 0
        for processor in self._task_id_to_processor.values():
            # get tileable proportion that is tiled
            tileable_graph = processor.tileable_graph
            tileable_context = processor.tile_context
            tiled_percentage += len(tileable_context) / len(tileable_graph)
            i += 1
        tiled_percentage /= i

        # get progress of stages
        subtask_progress = 0.0
        n_stage = 0
        stage_processors = itertools.chain(
            *(processor.stage_processors for processor
              in self._task_id_to_processor.values()))
        for stage_processor in stage_processors:
            if stage_processor.subtask_graph is None:  # pragma: no cover
                # generating subtask
                continue
            n_subtask = len(stage_processor.subtask_graph)
            if n_subtask == 0:  # pragma: no cover
                continue
            progress = sum(result.progress for result
                           in stage_processor.subtask_results.values())
            progress += sum(result.progress for subtask_key, result
                            in stage_processor.subtask_temp_results.items()
                            if subtask_key not in stage_processor.subtask_results)
            subtask_progress += progress / n_subtask
            n_stage += 1
        if n_stage > 0:
            subtask_progress /= n_stage

        return subtask_progress * tiled_percentage

    def get_result_tileables(self):
        processor = list(self._task_id_to_processor.values())[-1]
        tileable_graph = processor.tileable_graph
        result = []
        for result_tileable in tileable_graph.result_tileables:
            tiled = processor.get_tiled(result_tileable)
            result.append(build_fetch(tiled))
        return result

    def get_tileable_graph_as_dict(self):
        processor = list(self._task_id_to_processor.values())[-1]
        tileable_graph = processor.tileable_graph

        node_list = []
        edge_list = []

        visited = set()

        for chunk in tileable_graph:
            if chunk.key in visited:
                continue
            visited.add(chunk.key)

            node_name = str(chunk.op)

            node_list.append({
                'tileableId': chunk.key,
                'tileableName': node_name
            })
            for inp, is_pure_dep in zip(chunk.inputs, chunk.op.pure_depends):
                if inp not in tileable_graph:  # pragma: no cover
                    continue
                edge_list.append({
                    'fromTileableId': inp.key,
                    'toTileableId': chunk.key,
                    'linkType': 1 if is_pure_dep else 0,
                })

        graph_dict = {
            'tileables': node_list,
            'dependencies': edge_list
        }
        return graph_dict

    def get_tileable_details(self):
        tileable_to_subtasks = dict()
        subtask_results = dict()
        default_result = SubtaskResult(progress=0.0, status=SubtaskStatus.pending)

        for processor in self._task_id_to_processor.values():
            for stage in processor.stage_processors:
                tileable_to_subtasks.update(stage.tileable_to_subtasks)

                for subtask, result in stage.subtask_results.items():
                    subtask_results[subtask.subtask_id] = result
                for subtask, result in stage.subtask_temp_results.items():
                    if subtask.subtask_id in subtask_results:
                        continue
                    subtask_results[subtask.subtask_id] = result

        tileable_infos = dict()
        for tileable, subtasks in tileable_to_subtasks.items():
            results = [subtask_results.get(subtask.subtask_id, default_result)
                       for subtask in subtasks]

            # calc progress
            if not results:  # pragma: no cover
                progress = 1.0
            else:
                progress = 1.0 * sum(result.progress for result in results) / len(results)

            # calc status
            statuses = set(result.status for result in results)
            if not results or statuses == {SubtaskStatus.succeeded}:
                status = SubtaskStatus.succeeded
            elif statuses == {SubtaskStatus.cancelled}:
                status = SubtaskStatus.cancelled
            elif statuses == {SubtaskStatus.pending}:
                status = SubtaskStatus.pending
            elif SubtaskStatus.errored in statuses:
                status = SubtaskStatus.errored
            else:
                status = SubtaskStatus.running

            tileable_infos[tileable.key] = {
                'progress': progress,
                'subtaskCount': len(results),
                'status': status.value,
            }
        return tileable_infos

    def get_result_tileable(self, tileable_key: str):
        processor = list(self._task_id_to_processor.values())[-1]
        tileable_graph = processor.tileable_graph
        for result_tileable in tileable_graph.result_tileables:
            if result_tileable.key == tileable_key:
                tiled = processor.get_tiled(result_tileable)
                return build_fetch(tiled)
        raise KeyError(f'Tileable {tileable_key} does not exist')  # pragma: no cover

    async def _decref_input_subtasks(self,
                                     subtask: Subtask,
                                     subtask_graph: SubtaskGraph):
        decref_chunk_keys = []
        for in_subtask in subtask_graph.iter_predecessors(subtask):
            for result_chunk in in_subtask.chunk_graph.results:
                # for reducer chunk, decref mapper chunks
                if isinstance(result_chunk.op, ShuffleProxy):
                    for chunk in subtask.chunk_graph:
                        if isinstance(chunk.op, MapReduceOperand) and \
                                chunk.op.stage == OperandStage.reduce:
                            data_keys = chunk.op.get_dependent_data_keys()
                            decref_chunk_keys.extend(data_keys)
                            # decref main key as well
                            decref_chunk_keys.extend([key[0] for key in data_keys])
                decref_chunk_keys.append(result_chunk.key)
        await self._lifecycle_api.decref_chunks(decref_chunk_keys)

    async def set_subtask_result(self, subtask_result: SubtaskResult):
        stage_processor = self._cur_processor.cur_stage_processor
        subtask = stage_processor.subtask_id_to_subtask[subtask_result.subtask_id]

        prev_result = stage_processor.subtask_results.get(subtask)
        if prev_result:
            # set before
            return

        stage_processor.subtask_temp_results[subtask] = subtask_result
        if subtask_result.status.is_done:
            try:
                # Since every worker will call supervisor to set subtask result,
                # we need to release actor lock to make `decref_chunks` parallel to avoid blocking
                # other `set_subtask_result` calls.
                yield self._decref_input_subtasks(subtask, stage_processor.subtask_graph)
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except  # pragma: no cover
                _, err, tb = sys.exc_info()
                if subtask_result.status not in (SubtaskStatus.errored, SubtaskStatus.cancelled):
                    subtask_result.status = SubtaskStatus.errored
                    subtask_result.error = err
                    subtask_result.traceback = tb
            await stage_processor.set_subtask_result(subtask_result)

    def is_done(self) -> bool:
        for processor in self._task_id_to_processor.values():
            if not processor.is_done():
                return False
        return True
