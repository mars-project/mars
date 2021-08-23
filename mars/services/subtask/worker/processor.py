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
import logging
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type

from .... import oscar as mo
from ....core import ChunkGraph, OperandType, enter_mode
from ....core.context import get_context, set_context
from ....core.operand import Fetch, FetchShuffle, \
    MapReduceOperand, VirtualOperand, execute
from ....optimization.physical import optimize
from ....typing import BandType
from ....utils import get_chunk_key_to_data_keys
from ...context import ThreadedServiceContext
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
    _chunk_key_to_data_keys: Dict[str, List[str]]

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
        self._actual_chunk_count = len([
            chunk for chunk in subtask.chunk_graph
            if not isinstance(chunk.op, (Fetch, FetchShuffle))
        ])
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

    async def _load_input_data(self):
        keys, gets, accept_nones = [], [], []
        for chunk in self._chunk_graph.iter_indep():
            if isinstance(chunk.op, Fetch):
                keys.append(chunk.key)
                gets.append(self._storage_api.get.delay(chunk.key))
                accept_nones.append(True)
            elif isinstance(chunk.op, FetchShuffle):
                for key in self._chunk_key_to_data_keys[chunk.key]:
                    keys.append(key)
                    gets.append(self._storage_api.get.delay(key, error='ignore'))
                    accept_nones.append(False)
        if keys:
            logger.debug('Start getting input data, keys: %s, '
                         'subtask id: %s', keys, self.subtask.subtask_id)
            inputs = await self._storage_api.get.batch(*gets)
            self._datastore.update(
                {key: get for key, get, accept_none in zip(keys, inputs, accept_nones)
                 if accept_none or get is not None}
            )
            logger.debug('Finish getting input data keys: %s, '
                         'subtask id: %s', keys, self.subtask.subtask_id)
        return keys

    @staticmethod
    async def notify_task_manager_result(supervisor_address: str,
                                         result: SubtaskResult):
        task_api = await TaskAPI.create(
            result.session_id, supervisor_address)
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
                                     ctx: Dict[str, Any],
                                     op: OperandType):
        if not isinstance(op, (Fetch, FetchShuffle)):
            self._op_progress[op.key] = 0.0
        get_context().set_running_operand_key(self._session_id, op.key)
        return asyncio.to_thread(self._execute_operand, ctx, op)

    def set_op_progress(self, op_key: str, progress: float):
        if op_key in self._op_progress:  # pragma: no branch
            self._op_progress[op_key] = progress

    @enter_mode(build=False, kernel=True)
    def _execute_operand(self,
                         ctx: Dict[str, Any],
                         op: OperandType):  # noqa: R0201  # pylint: disable=no-self-use
        return execute(ctx, op)

    async def _execute_graph(self, chunk_graph: ChunkGraph):
        loop = asyncio.get_running_loop()
        ref_counts = self._init_ref_counts()

        # from data_key to results
        for chunk in chunk_graph.topological_iter():
            if chunk.key not in self._datastore:
                # since `op.execute` may be a time-consuming operation,
                # we make it run in a thread pool to not block current thread.
                logger.debug('Start executing operand: %s,'
                             'chunk: %s, subtask id: %s', chunk.op, chunk,
                             self.subtask.subtask_id)
                future = asyncio.create_task(
                    await self._async_execute_operand(self._datastore, chunk.op))
                to_wait = loop.create_future()

                def cb(fut):
                    if not to_wait.done():
                        if fut.exception():
                            to_wait.set_exception(fut.exception())
                        else:
                            to_wait.set_result(fut.result())
                future.add_done_callback(cb)

                try:
                    await to_wait
                    logger.debug('Finish executing operand: %s,'
                                 'chunk: %s, subtask id: %s', chunk.op, chunk,
                                 self.subtask.subtask_id)
                except asyncio.CancelledError:
                    logger.debug('Receive cancel instruction for operand: %s,'
                                 'chunk: %s, subtask id: %s', chunk.op, chunk,
                                 self.subtask.subtask_id)
                    # wait for this computation to finish
                    await future
                    # if cancelled, stop next computation
                    logger.debug('Cancelled operand: %s, chunk: %s, '
                                 'subtask id: %s', chunk.op, chunk,
                                 self.subtask.subtask_id)
                    self.result.status = SubtaskStatus.cancelled
                    raise

            self.set_op_progress(chunk.op.key, 1.0)

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

                put = self._storage_api.put.delay(data_key, result_data)
                puts.append(put)
            else:
                assert isinstance(result_chunk.op, MapReduceOperand)
                keys = [store_key for store_key in self._datastore
                        if isinstance(store_key, tuple) and store_key[0] == data_key]
                for key in keys:
                    stored_keys.append(key)
                    result_data = self._datastore[key]
                    put = self._storage_api.put.delay(key, result_data)
                    puts.append(put)
        logger.debug('Start putting data keys: %s, '
                     'subtask id: %s', stored_keys, self.subtask.subtask_id)
        data_key_to_store_size = dict()
        data_key_to_memory_size = dict()
        if puts:
            put_infos = asyncio.create_task(self._storage_api.put.batch(*puts))
            try:
                store_infos = await put_infos
                for store_key, store_info in zip(stored_keys, store_infos):
                    data_key_to_store_size[store_key] = store_info.store_size
                    data_key_to_memory_size[store_key] = store_info.memory_size
                logger.debug('Finish putting data keys: %s, '
                             'subtask id: %s', stored_keys, self.subtask.subtask_id)
            except asyncio.CancelledError:
                logger.debug('Cancelling put data keys: %s, '
                             'subtask id: %s', stored_keys, self.subtask.subtask_id)
                put_infos.cancel()

                logger.debug('Cancelled put data keys: %s, '
                             'subtask id: %s', stored_keys, self.subtask.subtask_id)
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
        key_to_result_chunk = {c.key: c for c in chunk_graph.result_chunks}
        # store meta
        set_chunk_metas = []
        result_data_size = 0
        for chunk_key in stored_keys:
            if isinstance(chunk_key, tuple):
                result_chunk = key_to_result_chunk[chunk_key[0]]
            else:
                result_chunk = key_to_result_chunk[chunk_key]
            store_size = data_key_to_store_size[chunk_key]
            memory_size = data_key_to_memory_size[chunk_key]
            result_data_size += memory_size
            set_chunk_metas.append(
                self._meta_api.set_chunk_meta.delay(
                    result_chunk, memory_size=memory_size,
                    store_size=store_size, bands=[self._band],
                    chunk_key=chunk_key))
        for chunk in chunk_graph.result_chunks:
            if chunk.key not in data_key_to_store_size:
                # mapper, set meta, so that storage can make sure
                # this operand is executed, some sub key is absent
                # due to it's empty actually
                set_chunk_metas.append(self._meta_api.set_chunk_meta.delay(
                    chunk, memory_size=0, store_size=0, bands=[self._band]))
        logger.debug('Start storing chunk metas for data keys: %s, '
                     'subtask id: %s', stored_keys, self.subtask.subtask_id)
        if set_chunk_metas:
            set_chunks_meta = asyncio.create_task(
                self._meta_api.set_chunk_meta.batch(*set_chunk_metas))
            try:
                await set_chunks_meta
                logger.debug('Finish store chunk metas for data keys: %s, '
                             'subtask id: %s', stored_keys, self.subtask.subtask_id)
            except asyncio.CancelledError:
                logger.debug('Cancelling store chunk metas for data keys: %s, '
                             'subtask id: %s', stored_keys, self.subtask.subtask_id)
                set_chunks_meta.cancel()

                # remote stored data
                deletes = []
                for data_key in stored_keys:
                    deletes.append(self._storage_api.delete.delay(data_key))
                await self._storage_api.delete.batch(*deletes)

                self.result.status = SubtaskStatus.cancelled
                logger.debug('Cancelled store chunk metas for data keys: %s, '
                             'subtask id: %s', stored_keys, self.subtask.subtask_id)
                raise
        # set result data size
        self.result.data_size = result_data_size

    async def done(self):
        if self.result.status == SubtaskStatus.running:
            self.result.status = SubtaskStatus.succeeded
        self.result.progress = 1.0
        self.is_done.set()

    async def run(self):
        self.result.status = SubtaskStatus.running
        input_keys = None
        unpinned = False
        try:
            chunk_graph = optimize(self._chunk_graph, self._engines)
            self._chunk_key_to_data_keys = get_chunk_key_to_data_keys(chunk_graph)
            report_progress = asyncio.create_task(
                self.report_progress_periodically())

            # load inputs data
            input_keys = await self._load_input_data()
            try:
                # execute chunk graph
                await self._execute_graph(chunk_graph)
            finally:
                # unpin inputs data
                unpinned = True
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
            if input_keys is not None and not unpinned:
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
            size = self._actual_chunk_count
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
            StorageAPI.create(self._session_id, self.address, self._band[1]),
            MetaAPI.create(self._session_id, self._supervisor_address)]
        coros = [asyncio.ensure_future(coro) for coro in coros]
        await asyncio.gather(*coros)
        self._session_api, self._storage_api, self._meta_api = \
            [coro.result() for coro in coros]

    async def _init_context(self, session_id: str):
        loop = asyncio.get_running_loop()
        context = ThreadedServiceContext(
            session_id, self._supervisor_address,
            self.address, loop, band=self._band)
        await context.init()
        set_context(context)

    async def run(self, subtask: Subtask):
        logger.debug('Start to run subtask: %s', subtask.subtask_id)

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
        logger.debug('Cancelling subtask: %s', self._processor.subtask_id)

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

    def set_running_op_progress(self, op_key: str, progress: float):
        self._processor.set_op_progress(op_key, progress)
