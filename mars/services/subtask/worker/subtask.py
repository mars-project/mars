# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
import concurrent.futures as futures
import importlib
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, List, Optional, Type, Union

from .... import oscar as mo
from ....core import ChunkGraph, OperandType
from ....core.context import get_context, set_context
from ....core.operand import Fetch, FetchShuffle, \
    MapReduceOperand, VirtualOperand, OperandStage, execute
from ....lib.aio import alru_cache
from ....oscar.backends.allocate_strategy import IdleLabel
from ....optimization.physical import optimize
from ...context import ThreadedServiceContext
from ...core import BandType
from ...meta.api import MetaAPI
from ...storage import StorageAPI
from ...task import TaskAPI, task_options
from ..core import Subtask, SubtaskStatus, SubtaskResult
from ..errors import SlotOccupiedAlready

logger = logging.getLogger(__name__)

SubtaskRunnerRef = Union["SubtaskRunnerActor", mo.ActorRef]


class SubtaskManagerActor(mo.Actor):
    def __init__(self, subtask_processor_cls: Type):
        # specify subtask process class
        # for test purpose
        self._subtask_processor_cls = subtask_processor_cls
        self._cluster_api = None

    async def __post_create__(self):
        from ...cluster.api import ClusterAPI
        self._cluster_api = await ClusterAPI.create(self.address)

        band_to_slots = await self._cluster_api.get_bands()
        supervisor_address = (await self._cluster_api.get_supervisors())[0]
        for band, n_slot in band_to_slots.items():
            await self._create_band_runner_actors(band[1], n_slot, supervisor_address)

    async def _create_band_runner_actors(self, band_name: str, n_slots: int,
                                         supervisor_address: str):
        strategy = IdleLabel(band_name, 'subtask_runner')
        band = (self.address, band_name)
        for slot_id in range(n_slots):
            await mo.create_actor(
                SubtaskRunnerActor,
                supervisor_address, band,
                subtask_processor_cls=self._subtask_processor_cls,
                uid=SubtaskRunnerActor.gen_uid(band_name, slot_id),
                address=self.address,
                allocate_strategy=strategy)


class DataStore(dict):
    def __getattr__(self, attr):
        ctx = get_context()
        return getattr(ctx, attr)


class SubtaskProcessor:
    _chunk_graph: ChunkGraph

    def __init__(self,
                 subtask: Subtask,
                 storage_api: StorageAPI,
                 meta_api: MetaAPI,
                 band: BandType,
                 supervisor_address: str,
                 engines: List[str] = None):
        self.subtask = subtask
        self._session_id = self.subtask.session_id
        self._chunk_graph = subtask.chunk_graph
        self._band = band
        self._supervisor_address = supervisor_address
        self._engines = engines if engines is not None else \
            task_options.runtime_engines

        # result
        self.result = SubtaskResult(
            subtask_id=subtask.subtask_id,
            session_id=subtask.session_id,
            task_id=subtask.task_id,
            status=SubtaskStatus.pending,
            progress=0.0)

        # status and intermediate states
        # operand progress, from op key to progress
        self._op_progress: Dict[str, float] = defaultdict(lambda: 0.0)
        # temp data store that holds chunk data during computation
        self._datastore = DataStore()
        # chunk key to real data keys
        self._chunk_key_to_data_keys = dict()

        # other service APIs
        self._storage_api = storage_api
        self._meta_api = meta_api

    @property
    def status(self):
        return self.result.status

    @property
    def subtask_id(self):
        return self.subtask.subtask_id

    def _gen_chunk_key_to_data_keys(self):
        for chunk in self._chunk_graph:
            if chunk.key in self._chunk_key_to_data_keys:
                continue
            if not isinstance(chunk.op, FetchShuffle):
                self._chunk_key_to_data_keys[chunk.key] = [chunk.key]
            else:
                keys = []
                for succ in self._chunk_graph.iter_successors(chunk):
                    if isinstance(succ.op, MapReduceOperand) and \
                            succ.op.stage == OperandStage.reduce:
                        for key in succ.op.get_dependent_data_keys():
                            if key not in keys:
                                keys.append(key)
                self._chunk_key_to_data_keys[chunk.key] = keys

    async def _load_input_data(self):
        keys = []
        gets = []
        fetches = []
        for chunk in self._chunk_graph.iter_indep():
            if isinstance(chunk.op, Fetch):
                keys.append(chunk.key)
                gets.append(self._storage_api.get.delay(chunk.key))
                fetches.append(self._storage_api.fetch.delay(chunk.key))
            elif isinstance(chunk.op, FetchShuffle):
                for key in self._chunk_key_to_data_keys[chunk.key]:
                    keys.append(key)
                    gets.append(self._storage_api.get.delay(key, error='ignore'))
                    fetches.append(self._storage_api.fetch.delay(key, error='ignore'))
        if keys:
            logger.info(f'Start getting input data keys: {keys}, '
                        f'subtask id: {self.subtask.subtask_id}')
            await self._storage_api.fetch.batch(*fetches)
            inputs = await self._storage_api.get.batch(*gets)
            self._datastore.update({key: get for key, get in zip(keys, inputs) if get is not None})
            logger.info(f'Finish getting input data keys: {keys}, '
                        f'subtask id: {self.subtask.subtask_id}')

    @staticmethod
    @alru_cache(cache_exceptions=False)
    async def _get_task_api(supervisor_address: str, session_id: str) -> TaskAPI:
        return await TaskAPI.create(session_id, supervisor_address)

    @staticmethod
    async def notify_task_manager_result(supervisor_address: str,
                                         result: SubtaskResult):
        task_api = await SubtaskProcessor._get_task_api(
            supervisor_address, result.session_id)
        # notify task service
        await task_api.set_subtask_result(result)

    def _init_ref_counts(self) -> Dict[str, int]:
        chunk_graph = self._chunk_graph
        ref_counts = defaultdict(lambda: 0)
        # set 1 for result chunks
        for result_chunk in chunk_graph.result_chunks:
            ref_counts[result_chunk.key] += 1
        # iter graph to set ref counts
        for chunk in chunk_graph:
            ref_counts[chunk.key] += chunk_graph.count_successors(chunk)
        return ref_counts

    def _execute_operand(self,
                         ctx: Dict[str, Any],
                         op: OperandType):  # noqa: R0201  # pylint: disable=no-self-use
        return execute(ctx, op)

    async def done(self):
        if self.result.status == SubtaskStatus.running:
            self.result.status = SubtaskStatus.succeeded
        self.result.progress = 1.0

    async def run(self):
        self.result.status = SubtaskStatus.running

        try:
            loop = asyncio.get_running_loop()
            executor = futures.ThreadPoolExecutor(1)

            chunk_graph = optimize(self._chunk_graph, self._engines)
            self._gen_chunk_key_to_data_keys()
            ref_counts = self._init_ref_counts()

            report_progress = asyncio.create_task(
                self.report_progress_periodically())

            await self._load_input_data()

            # from data_key to results
            for chunk in chunk_graph.topological_iter():
                if chunk.key not in self._datastore:
                    # since `op.execute` may be a time-consuming operation,
                    # we make it run in a thread pool to not block current thread.
                    logger.info(f'Start executing operand: {chunk.op},'
                                f'chunk: {chunk}, subtask id: {self.subtask.subtask_id}')
                    future = loop.run_in_executor(executor, self._execute_operand,
                                                  self._datastore, chunk.op)
                    try:
                        await future
                        logger.info(f'Finish executing operand: {chunk.op},'
                                    f'chunk: {chunk}, subtask id: {self.subtask.subtask_id}')
                    except asyncio.CancelledError:
                        logger.info(f'Receive cancel instruction for operand: {chunk.op},'
                                    f'chunk: {chunk}, subtask id: {self.subtask.subtask_id}')
                        # wait for this computation to finish
                        await loop.run_in_executor(None, executor.shutdown)
                        # if cancelled, stop next computation,
                        logger.info(f'Cancelled operand: {chunk.op}, chunk: {chunk}, '
                                    f'subtask id: {self.subtask.subtask_id}')
                        self.result.status = SubtaskStatus.cancelled
                        raise
                    self._op_progress[chunk.op.key] = 1.0
                else:
                    self._op_progress[chunk.op.key] += 1.0
                for inp in chunk.inputs:
                    ref_counts[inp.key] -= 1
                    if ref_counts[inp.key] == 0:
                        # ref count reaches 0, remove it
                        for key in self._chunk_key_to_data_keys[inp.key]:
                            del self._datastore[key]

            # skip virtual operands for result chunks
            result_chunks = [c for c in chunk_graph.result_chunks
                             if not isinstance(c.op, VirtualOperand)]

            # store data into storage
            data_key_to_puts = defaultdict(list)
            stored_keys = []
            for result_chunk in result_chunks:
                data_key = result_chunk.key
                if data_key in self._datastore:
                    # non shuffle op
                    stored_keys.append(data_key)
                    result_data = self._datastore[data_key]
                    # update meta
                    if not isinstance(result_data, tuple):
                        result_chunk.params = result_chunk.get_params_from_data(result_data)

                    put = self._storage_api.put.delay(data_key, result_data)
                    data_key_to_puts[data_key].append(put)
                else:
                    assert isinstance(result_chunk.op, MapReduceOperand)
                    keys = [store_key for store_key in self._datastore
                            if isinstance(store_key, tuple) and store_key[0] == data_key]
                    for key in keys:
                        stored_keys.append(key)
                        result_data = self._datastore[key]
                        put = self._storage_api.put.delay(key, result_data)
                        data_key_to_puts[data_key].append(put)
            logger.info(f'Start putting data keys: {stored_keys}, '
                        f'subtask id: {self.subtask.subtask_id}')
            puts = list(chain(*data_key_to_puts.values()))
            data_key_to_store_size = defaultdict(lambda: 0)
            data_key_to_memory_size = defaultdict(lambda: 0)
            if puts:
                put_infos = asyncio.create_task(self._storage_api.put.batch(*puts))
                try:
                    store_infos = await put_infos
                    store_infos_iter = iter(store_infos)
                    for data_key, puts in data_key_to_puts.items():
                        for _ in puts:
                            store_info = next(store_infos_iter)
                            data_key_to_store_size[data_key] += store_info.store_size
                            data_key_to_memory_size[data_key] += store_info.memory_size
                    logger.info(f'Finish putting data keys: {stored_keys}, '
                                f'subtask id: {self.subtask.subtask_id}')
                except asyncio.CancelledError:
                    logger.info(f'Cancelling put data keys: {stored_keys}, '
                                f'subtask id: {self.subtask.subtask_id}')
                    put_infos.cancel()

                    logger.info(f'Cancelled put data keys: {stored_keys}, '
                                f'subtask id: {self.subtask.subtask_id}')
                    self.result.status = SubtaskStatus.cancelled
                    raise

            # clear data
            self._datastore = dict()

            # store meta
            set_chunk_metas = []
            memory_sizes = []
            for result_chunk in chunk_graph.result_chunks:
                store_size = data_key_to_store_size[result_chunk.key]
                memory_size = data_key_to_memory_size[result_chunk.key]
                memory_sizes.append(memory_size)
                set_chunk_metas.append(
                    self._meta_api.set_chunk_meta.delay(
                        result_chunk, memory_size=memory_size,
                        store_size=store_size, bands=[self._band]))
            logger.info(f'Start storing chunk metas for data keys: {stored_keys}, '
                        f'subtask id: {self.subtask.subtask_id}')
            if set_chunk_metas:
                set_chunks_meta = asyncio.create_task(
                    self._meta_api.set_chunk_meta.batch(*set_chunk_metas))
                try:
                    await set_chunks_meta
                    logger.info(f'Finish store chunk metas for data keys: {stored_keys}, '
                                f'subtask id: {self.subtask.subtask_id}')
                except asyncio.CancelledError:
                    logger.info(f'Cancelling store chunk metas for data keys: {stored_keys}, '
                                f'subtask id: {self.subtask.subtask_id}')
                    set_chunks_meta.cancel()

                    # remote stored data
                    deletes = []
                    for data_key in stored_keys:
                        deletes.append(self._storage_api.delete.delay(data_key))
                    await self._storage_api.delete.batch(*deletes)

                    self.result.status = SubtaskStatus.cancelled
                    logger.info(f'Cancelled store chunk metas for data keys: {stored_keys}, '
                                f'subtask id: {self.subtask.subtask_id}')
                    raise

            # set result data size
            self.result.data_size = sum(memory_sizes)
        except asyncio.CancelledError:
            self.result.status = SubtaskStatus.cancelled
            self.result.progress = 1.0
            raise
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except
            self.result.status = SubtaskStatus.errored
            self.result.progress = 1.0
            _, self.result.error, self.result.traceback = sys.exc_info()
            await self.done()
            raise
        finally:
            pass

        await self.done()
        report_progress.cancel()
        try:
            await report_progress
        except asyncio.CancelledError:
            pass
        return self.result

    async def report_progress_periodically(self, interval=.5, eps=0.001):
        last_progress = self.result.progress
        while not self.result.status.is_done:
            size = len(self._chunk_graph)
            progress = sum(self._op_progress.values()) / size
            assert progress <= 1
            self.result.progress = progress
            if abs(last_progress - progress) >= eps:
                # report progress
                if not self.result.status.is_done:
                    fut = self.notify_task_manager_result(
                        self._supervisor_address, self.result)
                    if fut:
                        await fut
            await asyncio.sleep(interval)
            last_progress = progress


@dataclass
class _SubtaskRunningInfo:
    task: asyncio.Task
    processor: SubtaskProcessor = None


class SubtaskRunnerActor(mo.Actor):
    @classmethod
    def gen_uid(cls, band_name: str, slot_id: int):
        return f'slot_{band_name}_{slot_id}_subtask_runner'

    def __init__(self,
                 supervisor_address: str,
                 band: BandType,
                 subtask_processor_cls: Type = None):
        self._supervisor_address = supervisor_address
        self._band = band
        self._subtask_info: Optional[_SubtaskRunningInfo] = None
        self._subtask_processor_cls = \
            self._get_subtask_process_cls(subtask_processor_cls)

    async def _init_context(self, session_id: str):
        loop = asyncio.get_running_loop()
        context = ThreadedServiceContext(
            session_id, self._supervisor_address,
            self.address, loop)
        await context.init()
        set_context(context)

    @classmethod
    def _get_subtask_process_cls(cls, subtask_processor_cls):
        if subtask_processor_cls is None:
            return SubtaskProcessor
        else:
            assert isinstance(subtask_processor_cls, str)
            module, class_name = subtask_processor_cls.rsplit('.', 1)
            return getattr(importlib.import_module(module), class_name)

    async def _init_subtask_processor(self, subtask: Subtask) -> SubtaskProcessor:
        # storage API
        storage_api = await StorageAPI.create(
            subtask.session_id, self.address)
        # meta API
        meta_api = await MetaAPI.create(
            subtask.session_id, self._supervisor_address)
        # init context
        await self._init_context(subtask.session_id)

        processor_cls = self._subtask_processor_cls
        return processor_cls(subtask, storage_api, meta_api,
                             self._band, self._supervisor_address)

    async def _run_subtask(self, subtask: Subtask):
        processor = await self._init_subtask_processor(subtask)
        self._subtask_info.processor = processor
        return await processor.run()

    async def run_subtask(self, subtask: Subtask):
        if not self.is_runner_free():  # pragma: no cover
            # current subtask is still running
            raise SlotOccupiedAlready(
                f'There is subtask(id: '
                f'{self._subtask_info.processor.subtask_id}) running, '
                f'cannot run another subtask')

        logger.info(f'Start to run subtask: {subtask.subtask_id}')
        aio_task = asyncio.create_task(self._run_subtask(subtask))
        self._subtask_info = _SubtaskRunningInfo(task=aio_task)
        return aio_task

    def is_runner_free(self):
        return self._subtask_info is None or \
            getattr(self._subtask_info, 'processor', None) is None or \
            self._subtask_info.processor.status.is_done

    async def wait_subtask(self):
        await self._subtask_info.task

    async def get_subtask_result(self) -> SubtaskResult:
        return self._subtask_info.processor.result

    async def cancel_subtask(self):
        if self._subtask_info is None:
            return

        logger.info(f'Cancelling subtask: '
                    f'{self._subtask_info.processor.subtask_id}')
        aio_task = self._subtask_info.task
        aio_task.cancel()

        async def waiter():
            try:
                await aio_task
            except asyncio.CancelledError:
                pass

        # return asyncio task to not block current actor
        return waiter()
