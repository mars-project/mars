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
import concurrent.futures as futures
import logging
import sys
from collections import defaultdict
from itertools import chain
from typing import Any, Dict, List, Optional, Type

from .... import oscar as mo
from ....core import ChunkGraph, OperandType
from ....core.context import get_context, set_context
from ....core.operand import Fetch, FetchShuffle, \
    MapReduceOperand, VirtualOperand, OperandStage, execute
from ....lib.aio import alru_cache
from ....optimization.physical import optimize
from ...context import ThreadedServiceContext
from ...core import BandType
from ...meta.api import MetaAPI
from ...storage import StorageAPI
from ...session import SessionAPI
from ...task import TaskAPI, task_options
from ..core import Subtask, SubtaskStatus, SubtaskResult

logger = logging.getLogger(__name__)


class DataStore(dict):
    def __getattr__(self, attr):
        ctx = get_context()
        return getattr(ctx, attr)


class SubtaskProcessor:
    _chunk_graph: ChunkGraph

    def __init__(self,
                 subtask: Subtask,
                 session_api: SessionAPI,
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
        self.is_done = asyncio.Event()

        # status and intermediate states
        # operand progress, from op key to progress
        self._op_progress: Dict[str, float] = defaultdict(lambda: 0.0)
        # temp data store that holds chunk data during computation
        self._datastore = DataStore()
        # chunk key to real data keys
        self._chunk_key_to_data_keys = dict()

        # other service APIs
        self._session_api = session_api
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
        return keys

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

    async def _async_execute_operand(self,
                                     loop,
                                     executor,
                                     ctx: Dict[str, Any],
                                     op: OperandType):
        return loop.run_in_executor(executor, self._execute_operand,
                                    ctx, op)

    def _execute_operand(self,
                         ctx: Dict[str, Any],
                         op: OperandType):  # noqa: R0201  # pylint: disable=no-self-use
        return execute(ctx, op)

    async def _execute_graph(self, chunk_graph: ChunkGraph):
        loop = asyncio.get_running_loop()
        executor = futures.ThreadPoolExecutor(1)
        ref_counts = self._init_ref_counts()

        # from data_key to results
        for chunk in chunk_graph.topological_iter():
            if chunk.key not in self._datastore:
                # since `op.execute` may be a time-consuming operation,
                # we make it run in a thread pool to not block current thread.
                logger.info(f'Start executing operand: {chunk.op},'
                            f'chunk: {chunk}, subtask id: {self.subtask.subtask_id}')
                future = await self._async_execute_operand(loop, executor,
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

    async def _unpin_data(self, data_keys):
        # unpin input keys
        unpins = []
        for key in data_keys:
            if isinstance(key, tuple):
                # a tuple key means it's a shuffle key,
                # some shuffle data is None and not stored in storage
                unpins.append(self._storage_api.unpin.delay(key, error='ignore'))
            else:
                unpins.append(self._storage_api.unpin.delay(key))
        await self._storage_api.unpin.batch(*unpins)

    async def _store_data(self, chunk_graph: ChunkGraph):
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
        return stored_keys, data_key_to_store_size, data_key_to_memory_size

    async def _store_meta(self,
                          chunk_graph: ChunkGraph,
                          stored_keys: List,
                          data_key_to_store_size: Dict,
                          data_key_to_memory_size: Dict):
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

    async def done(self):
        if self.result.status == SubtaskStatus.running:
            self.result.status = SubtaskStatus.succeeded
        self.result.progress = 1.0
        self.is_done.set()

    async def run(self):
        self.result.status = SubtaskStatus.running
        input_keys = None
        try:
            chunk_graph = optimize(self._chunk_graph, self._engines)
            self._gen_chunk_key_to_data_keys()
            report_progress = asyncio.create_task(
                self.report_progress_periodically())

            # load inputs data
            input_keys = await self._load_input_data()
            try:
                # execute chunk graph
                await self._execute_graph(chunk_graph)
            finally:
                # unpin inputs data
                await self._unpin_data(input_keys)
            # store results data
            stored_keys, store_sizes, memory_sizes = await self._store_data(chunk_graph)
            # store meta
            await self._store_meta(chunk_graph, stored_keys, store_sizes, memory_sizes)
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
            if input_keys is not None:
                await self._unpin_data(input_keys)

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


class SubtaskProcessorActor(mo.Actor):
    _session_api: Optional[SessionAPI]
    _storage_api: Optional[StorageAPI]
    _meta_api: Optional[MetaAPI]
    _processor: Optional[SubtaskProcessor]
    _last_processor: Optional[SubtaskProcessor]
    _running_aio_task: Optional[asyncio.Task]

    def __init__(self,
                 session_id: str,
                 band: BandType,
                 supervisor_address: str,
                 subtask_processor_cls: Type[SubtaskProcessor]):
        self._session_id = session_id
        self._band = band
        self._supervisor_address = supervisor_address
        self._subtask_processor_cls = subtask_processor_cls

        # current processor
        self._processor = None
        self._last_processor = None
        self._running_aio_task = None

        self._session_api = None
        self._storage_api = None
        self._meta_api = None

    @classmethod
    def gen_uid(cls, session_id: str):
        return f'{session_id}_subtask_processor'

    async def __post_create__(self):
        coros = [
            SessionAPI.create(self._supervisor_address),
            StorageAPI.create(self._session_id, self.address),
            MetaAPI.create(self._session_id, self._supervisor_address)]
        coros = [asyncio.ensure_future(coro) for coro in coros]
        await asyncio.gather(*coros)
        self._session_api, self._storage_api, self._meta_api = \
            [coro.result() for coro in coros]

    async def _init_context(self, session_id: str):
        loop = asyncio.get_running_loop()
        context = ThreadedServiceContext(
            session_id, self._supervisor_address,
            self.address, loop)
        await context.init()
        set_context(context)

    async def run(self, subtask: Subtask):
        logger.info(f'Start to run subtask: {subtask.subtask_id}')

        assert subtask.session_id == self._session_id

        # init context
        await self._init_context(self._session_id)
        processor = self._subtask_processor_cls(
            subtask, self._session_api, self._storage_api, self._meta_api,
            self._band, self._supervisor_address)
        self._processor = self._last_processor = processor
        self._running_aio_task = asyncio.create_task(processor.run())
        try:
            result = yield self._running_aio_task
            raise mo.Return(result)
        finally:
            self._processor = self._running_aio_task = None

    async def wait(self):
        return self._processor.is_done.wait()

    async def result(self):
        return self._last_processor.result

    async def cancel(self):
        logger.info(f'Cancelling subtask: '
                    f'{self._processor.subtask_id}')

        aio_task = self._running_aio_task
        aio_task.cancel()

        async def waiter():
            try:
                await aio_task
            except asyncio.CancelledError:
                pass

        # return asyncio task to not block current actor
        return waiter()

    def get_running_subtask_id(self):
        return self._processor.subtask_id
