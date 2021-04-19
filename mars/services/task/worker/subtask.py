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
import logging
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Dict, Union, Optional

from .... import oscar as mo
from ....core import ChunkGraph
from ....core.operand import Fetch, FetchShuffle, \
    MapReduceOperand, VirtualOperand, OperandStage
from ....lib.aio import alru_cache
from ....oscar.backends.allocate_strategy import IdleLabel
from ...core import BandType
from ...meta.api import MetaAPI
from ...storage.api import StorageAPI
from ..supervisor.task_manager import TaskManagerActor
from ..core import Subtask, SubTaskStatus, SubtaskResult
from ..errors import SlotOccupiedAlready


logger = logging.getLogger(__name__)

_SubTaskRunnerType = Union["SubtaskRunnerActor", mo.ActorRef]


class SubtaskManagerActor(mo.Actor):
    def __init__(self):
        self._cluster_api = None

    async def __post_create__(self):
        from ...cluster.api import ClusterAPI

        self._cluster_api = await ClusterAPI.create(self.address)
        await self._create_band_subtask_managers()

    async def _create_band_subtask_managers(self):
        band_to_slots = await self._cluster_api.get_bands()
        supervisor_address = (await self._cluster_api.get_supervisors())[0]
        for band, n_slot in band_to_slots.items():
            await mo.create_actor(BandSubtaskManagerActor, supervisor_address,
                                  n_slot, band[1], address=self.address,
                                  uid=BandSubtaskManagerActor.gen_uid(band[1]))


class BandSubtaskManagerActor(mo.Actor):
    """
    Manage subtask runner slots for a band.
    """
    def __init__(self,
                 supervisor_address: str,
                 n_slots: int,
                 band: str = 'numa-0'):
        self._supervisor_address = supervisor_address
        self._n_slots = n_slots
        self._band = band

        self._subtask_runner_slots = list()
        self._free_slots = set()
        self._running = asyncio.Semaphore(self._n_slots)
        self._cancelled_subtask_runners: Dict[_SubTaskRunnerType, asyncio.Event] = dict()

    async def __post_create__(self):
        strategy = IdleLabel(self._band, 'subtask_runner')
        band = (self.address, self._band)
        for _ in range(self._n_slots):
            runner = await mo.create_actor(
                SubtaskRunnerActor,
                self._supervisor_address, band, self.ref(),
                uid=SubtaskRunnerActor.default_uid(),
                address=self.address,
                allocate_strategy=strategy)
            self._subtask_runner_slots.append(runner)
            self._free_slots.add(runner)

    @classmethod
    def gen_uid(cls, band: str = 'numa-0'):
        return f'{band}_subtask_manager'

    def register_slot(self, subtask_runner: _SubTaskRunnerType):
        # when subtask runner created, it will notify subtask manager,
        # normally, no particular operation needed,
        # because subtask runner is created by manager itself,
        # however, when subtask runner is forced to cancel via kill_actor,
        # cancelling event etc will be handled
        if subtask_runner in self._cancelled_subtask_runners:
            self._cancelled_subtask_runners.pop(subtask_runner).set()

    def mark_slot_free(self, subtask_runner: _SubTaskRunnerType):
        self._free_slots.add(subtask_runner)
        self._running.release()

    def is_slot_free(self, subtask_runner: _SubTaskRunnerType):
        return subtask_runner in self._free_slots

    async def _free_slot(self,
                         subtask_runner: _SubTaskRunnerType,
                         timeout=5):
        # otherwise, call subtask runner to cancel
        cancel_subtask = asyncio.create_task(subtask_runner.cancel_subtask())
        try:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            cancel_subtask.add_done_callback(lambda f: future.set_result(None))
            await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            # timeout, force to cancel subtask by killing actor

            # get subtask result and set it to cancelled
            result = await subtask_runner.get_subtask_result()
            result.status = SubTaskStatus.cancelled

            event = self._cancelled_subtask_runners[subtask_runner] = asyncio.Event()
            await mo.kill_actor(subtask_runner)
            # when subtask runner recovered, this event will be set
            await event.wait()

            # since subtask runner is forced to get killed,
            # it cannot notify task manager no more,
            # so notify task manager here instead.
            fut = SubtaskProcessor.notify_task_manager_result(
                self._supervisor_address, result)
            if fut:
                await fut

    async def free_slot(self,
                        subtask_runner: _SubTaskRunnerType,
                        timeout=5):
        if self.is_slot_free(subtask_runner):
            # slot is available, no action needed
            return
        # return coroutine to release Actor lock

        yield self._free_slot(subtask_runner, timeout)

        # succeeded, mark slot free
        self.mark_slot_free(subtask_runner)

    def get_all_slots(self) -> List[_SubTaskRunnerType]:
        return self._subtask_runner_slots

    async def _get_free_slot(self) -> _SubTaskRunnerType:
        await self._running.acquire()
        return self._free_slots.pop()

    async def get_free_slot(self):
        # return coroutine to release Actor lock.
        return self._get_free_slot()


class SubtaskProcessor:
    _chunk_graph: ChunkGraph

    def __init__(self,
                 subtask: Subtask,
                 storage_api: StorageAPI,
                 meta_api: MetaAPI,
                 band: BandType,
                 supervisor_address: str):
        self.subtask = subtask
        self._session_id = self.subtask.session_id
        self._chunk_graph = subtask.chunk_graph
        self._band = band
        self._supervisor_address = supervisor_address

        # result
        self.result = SubtaskResult(
            subtask_id=subtask.subtask_id,
            session_id=subtask.session_id,
            task_id=subtask.task_id,
            status=SubTaskStatus.pending,
            progress=0.0)

        # status and intermediate states
        # operand progress, from op key to progress
        self._op_progress: Dict[str, float] = defaultdict(lambda: 0.0)
        # temp data store that holds chunk data during computation
        self._datastore = dict()
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
        prefetches = []
        for chunk in self._chunk_graph.iter_indep():
            if isinstance(chunk.op, Fetch):
                keys.append(chunk.key)
                gets.append(self._storage_api.get.delay(chunk.key))
                prefetches.append(self._storage_api.prefetch.delay(chunk.key))
            elif isinstance(chunk.op, FetchShuffle):
                for key in self._chunk_key_to_data_keys[chunk.key]:
                    keys.append(key)
                    gets.append(self._storage_api.get.delay(key))
                    prefetches.append(self._storage_api.prefetch.delay(key))
        if keys:
            logger.info(f'Start getting input data keys: {keys}, '
                        f'subtask id: {self.subtask.subtask_id}')
            await self._storage_api.prefetch.batch(*prefetches)
            inputs = await self._storage_api.get.batch(*gets)
            self._datastore.update({key: get for key, get in zip(keys, inputs)})
            logger.info(f'Finish getting input data keys: {keys}, '
                        f'subtask id: {self.subtask.subtask_id}')

    @staticmethod
    @alru_cache
    async def _get_task_manager(supervisor_address: str, uid: str) -> mo.ActorRef:
        return await mo.actor_ref(supervisor_address, uid)

    @staticmethod
    async def notify_task_manager_result(supervisor_address: str,
                                         result: SubtaskResult):
        task_manager = await SubtaskProcessor._get_task_manager(
            supervisor_address, TaskManagerActor.gen_uid(result.session_id))
        # notify task manger
        await task_manager.set_subtask_result(result)

    async def cancel(self):
        self.result.status = SubTaskStatus.cancelled
        self.result.progress = 1.0
        # notify task manager
        fut = self.notify_task_manager_result(
            self._supervisor_address, self.result)
        if fut:
            await fut

    async def done(self):
        if self.result.status == SubTaskStatus.running:
            self.result.status = SubTaskStatus.succeeded
        self.result.progress = 1.0
        fut = self.notify_task_manager_result(
            self._supervisor_address, self.result)
        if fut:
            await fut

    @contextmanager
    def _catch_error(self):
        try:
            yield
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except  # pragma: no cover
            logger.exception(f"Error when executing subtask: {self.subtask_id}")
            _, err, tb = sys.exc_info()
            self.result.status = SubTaskStatus.errored
            self.result.error = err
            self.result.traceback = tb

    def _init_ref_counts(self):
        chunk_graph = self._chunk_graph
        ref_counts = defaultdict(lambda: 0)
        # set 1 for result chunks
        for result_chunk in chunk_graph.result_chunks:
            ref_counts[result_chunk] = 1
        # iter graph to set ref counts
        for chunk in chunk_graph:
            ref_counts[chunk] += chunk_graph.count_successors(chunk)
        return ref_counts

    async def run(self):
        self.result.status = SubTaskStatus.running

        with self._catch_error():
            loop = asyncio.get_running_loop()
            executor = futures.ThreadPoolExecutor(1)

            chunk_graph = self._chunk_graph
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
                    future = loop.run_in_executor(executor, chunk.op.execute,
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
                        await self.cancel()
                        logger.info(f'Cancelled operand: {chunk.op}, chunk: {chunk}, '
                                    f'subtask id: {self.subtask.subtask_id}')
                        return
                    self._op_progress[chunk.op.key] = 1.0
                else:
                    self._op_progress[chunk.op.key] += 1.0
                for inp in chunk.inputs:
                    ref_counts[inp] -= 1
                    if ref_counts[inp] == 0:
                        # ref count reaches 0, remove it
                        for key in self._chunk_key_to_data_keys[inp.key]:
                            del self._datastore[key]

            # skip virtual operands for result chunks
            result_chunks = [c for c in chunk_graph.result_chunks
                             if not isinstance(c.op, VirtualOperand)]

            # store data into storage
            puts = []
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

                    puts.append(
                        self._storage_api.put.delay(data_key, result_data))
                else:
                    assert isinstance(result_chunk.op, MapReduceOperand)
                    keys = [store_key for store_key in self._datastore
                            if isinstance(store_key, tuple) and store_key[0] == data_key]
                    for key in keys:
                        stored_keys.append(key)
                        result_data = self._datastore[key]
                        puts.append(self._storage_api.put.delay(key, result_data))
            logger.info(f'Start putting data keys: {stored_keys}, '
                        f'subtask id: {self.subtask.subtask_id}')
            put_infos = asyncio.create_task(self._storage_api.put.batch(*puts))
            try:
                store_infos = await put_infos
                logger.info(f'Finish putting data keys: {stored_keys}, '
                            f'subtask id: {self.subtask.subtask_id}')
            except asyncio.CancelledError:
                logger.info(f'Cancelling put data keys: {stored_keys}, '
                            f'subtask id: {self.subtask.subtask_id}')
                put_infos.cancel()
                await self.cancel()
                logger.info(f'Cancelled put data keys: {stored_keys}, '
                            f'subtask id: {self.subtask.subtask_id}')
                return

            # clear data
            self._datastore = dict()

            # store meta
            set_chunk_metas = []
            memory_sizes = []
            for result_chunk, store_info in \
                    zip(chunk_graph.result_chunks, store_infos):
                store_size = store_info.store_size
                memory_size = store_info.memory_size
                memory_sizes.append(memory_size)
                set_chunk_metas.append(
                    self._meta_api.set_chunk_meta.delay(
                        result_chunk, memory_size=memory_size,
                        store_size=store_size, bands=[self._band]))
            logger.info(f'Start storing chunk metas for data keys: {stored_keys}, '
                        f'subtask id: {self.subtask.subtask_id}')
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

                await self.cancel()
                logger.info(f'Cancelled store chunk metas for data keys: {stored_keys}, '
                            f'subtask id: {self.subtask.subtask_id}')
                return

            # set result data size
            self.result.data_size = sum(memory_sizes)

        await self.done()
        await report_progress

    async def report_progress_periodically(self, interval=.5, eps=0.001):
        last_progress = self.result.progress
        while not self.result.status.is_done:
            size = len(self._chunk_graph)
            progress = sum(self._op_progress.values()) / size
            assert progress <= 1
            self.result.progress = progress
            if abs(last_progress - progress) >= eps:
                # report progress
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
    def __init__(self,
                 supervisor_address: str,
                 band: BandType,
                 subtask_manager: Union[BandSubtaskManagerActor, mo.ActorRef]):
        self._supervisor_address = supervisor_address
        self._band = band
        self._subtask_info: Optional[_SubtaskRunningInfo] = None
        self._subtask_manager = subtask_manager

    async def __post_create__(self):
        await self._subtask_manager.register_slot(self.ref())

    async def _init_subtask_processor(self, subtask: Subtask) -> SubtaskProcessor:
        # storage API
        storage_api = await StorageAPI.create(
            subtask.session_id, self.address)
        # meta API
        meta_api = await MetaAPI.create(
            subtask.session_id, self._supervisor_address)

        return SubtaskProcessor(subtask, storage_api, meta_api,
                                self._band, self._supervisor_address)

    async def _run_subtask(self, subtask: Subtask):
        processor = await self._init_subtask_processor(subtask)
        self._subtask_info.processor = processor
        try:
            await processor.run()
        finally:
            # release slot after notifying task manager,
            # make sure the subtasks that have higher priorities
            # have been enqueued so that they can be scheduled first.
            await self._subtask_manager.mark_slot_free(self.ref())

    async def run_subtask(self, subtask: Subtask):
        if self._subtask_info is not None and \
                getattr(self._subtask_info, 'processor', None) is not None and \
                not self._subtask_info.processor.status.is_done:
            # current subtask is still running
            raise SlotOccupiedAlready(
                f'There is subtask(id: '
                f'{self._subtask_info.processor.subtask_id}) running, '
                f'cannot run another subtask')

        logger.info(f'Start to run subtask: {subtask.subtask_id}')
        aio_task = asyncio.create_task(self._run_subtask(subtask))
        self._subtask_info = _SubtaskRunningInfo(task=aio_task)
        return aio_task

    async def wait_subtask(self):
        await self._subtask_info.task

    async def get_subtask_result(self) -> SubtaskResult:
        return self._subtask_info.processor.result

    async def cancel_subtask(self):
        logger.info(f'Cancelling subtask: '
                    f'{self._subtask_info.processor.subtask_id}')
        aio_task = self._subtask_info.task
        aio_task.cancel()
        # return asyncio task to not block current actor
        return aio_task
