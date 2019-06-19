# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import logging
import time
import uuid
from functools import partial

from .spill import read_spill_file, write_spill_file, spill_exists
from .utils import WorkerActor, concat_operand_keys
from .. import promise
from ..config import options
from ..errors import *
from ..utils import deserialize_graph, log_unhandled, calc_data_size
from ..compat import six, OrderedDict3
from ..tensor.execution.core import Executor

logger = logging.getLogger(__name__)

_calc_result_cache = OrderedDict3()


class InProcessCacheActor(WorkerActor):
    """
    Actor managing calculation result in rss memory
    """
    def __init__(self):
        super(InProcessCacheActor, self).__init__()
        self._chunk_holder_ref = None
        self._mem_quota_ref = None

        self._spill_dump_pool = None

    def post_create(self):
        from .chunkholder import ChunkHolderActor
        from .quota import MemQuotaActor

        super(InProcessCacheActor, self).post_create()
        self._chunk_holder_ref = self.promise_ref(ChunkHolderActor.default_uid())
        self._mem_quota_ref = self.promise_ref(MemQuotaActor.default_uid())
        if options.worker.spill_directory:
            self._spill_dump_pool = self.ctx.threadpool(len(options.worker.spill_directory))

    @promise.reject_on_exception
    @log_unhandled
    def dump_cache(self, keys, callback):
        """
        Dump data in rss memory into shared cache
        """
        @log_unhandled
        def _try_put_chunk(session_id, chunk_key, data_size, data_shape):
            logger.debug('Try putting %s into shared cache.', chunk_key)
            try:
                if chunk_key not in _calc_result_cache:
                    if not self._chunk_store.contains(session_id, chunk_key):
                        raise KeyError('Data key %s not found in inproc cache', chunk_key)
                    return

                ref = None
                try:
                    ref = self._chunk_store.put(session_id, chunk_key, _calc_result_cache[chunk_key][1])
                    del _calc_result_cache[chunk_key]
                    self._mem_quota_ref.release_quota(chunk_key, _tell=True)

                    self._chunk_holder_ref.register_chunk(session_id, chunk_key)
                    data_size = self._chunk_store.get_actual_size(session_id, chunk_key)
                    self.get_meta_client().set_chunk_meta(
                        session_id, chunk_key, size=data_size, shape=data_shape, workers=(self.address,))
                finally:
                    del ref

            except StoreFull:
                # if we cannot put data into shared cache, we store it into spill directly
                self._chunk_holder_ref.spill_size(data_size, _tell=True)
                _put_spill_directly(session_id, chunk_key, data_size, data_shape)

        @log_unhandled
        def _put_spill_directly(session_id, chunk_key, data_size, data_shape, *_):
            if self._spill_dump_pool is None:
                raise SpillNotConfigured

            logger.debug('Writing data %s directly into spill.', chunk_key)
            self._spill_dump_pool.submit(write_spill_file, chunk_key, _calc_result_cache[chunk_key][1]).result()

            del _calc_result_cache[chunk_key]
            self._mem_quota_ref.release_quota(chunk_key, _tell=True)

            self.get_meta_client().set_chunk_meta(
                session_id, chunk_key, size=data_size, shape=data_shape, workers=(self.address,))

        promises = []
        for k in keys:
            session_id, value = _calc_result_cache[k]
            data_size = calc_data_size(value)
            # for some special operands(argmax, argmean, mean, ..), intermediate chunk data has multiple parts, choose
            # first part's shape as chunk's shape.
            data_shape = value[0].shape if isinstance(value, tuple) else value.shape
            del value
            promises.append(
                promise.Promise(done=True).then(partial(_try_put_chunk, session_id, k, data_size, data_shape))
            )
        promise.all_(promises).then(lambda *_: self.tell_promise(callback)) \
            .catch(lambda *exc: self.tell_promise(callback, *exc, **dict(_accept=False)))

    @log_unhandled
    def remove_cache(self, keys):
        """
        Remove data from cache
        """
        for k in keys:
            del _calc_result_cache[k]


class CpuCalcActor(WorkerActor):
    def __init__(self):
        super(CpuCalcActor, self).__init__()
        self._mem_quota_ref = None
        self._inproc_cache_ref = None
        self._dispatch_ref = None
        self._status_ref = None

        self._execution_pool = None
        self._spill_load_pool = None

    def post_create(self):
        from .quota import MemQuotaActor
        from .dispatcher import DispatchActor
        from .status import StatusActor
        from .daemon import WorkerDaemonActor

        super(CpuCalcActor, self).post_create()
        if isinstance(self.uid, six.string_types) and ':' in self.uid:
            uid_parts = self.uid.split(':')
            inproc_uid = 'w:' + uid_parts[1] + ':inproc-cache-' + str(uuid.uuid4())
        else:
            inproc_uid = None

        raw_ref = self.ctx.create_actor(InProcessCacheActor, uid=inproc_uid)
        self._inproc_cache_ref = self.promise_ref(raw_ref)
        daemon_ref = self.ctx.actor_ref(WorkerDaemonActor.default_uid())
        if self.ctx.has_actor(daemon_ref):
            daemon_ref.register_child_actor(raw_ref, _tell=True)

        self._mem_quota_ref = self.promise_ref(MemQuotaActor.default_uid())
        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())
        self._dispatch_ref.register_free_slot(self.uid, 'cpu')

        self._status_ref = self.ctx.actor_ref(StatusActor.default_uid())
        if not self.ctx.has_actor(self._status_ref):
            self._status_ref = None

        self._execution_pool = self.ctx.threadpool(1)
        if options.worker.spill_directory:
            self._spill_load_pool = self.ctx.threadpool(len(options.worker.spill_directory))

    @staticmethod
    def _build_load_key(graph_key, chunk_key):
        return '%s_load_memory_%s' % (graph_key, chunk_key)

    @promise.reject_on_exception
    @log_unhandled
    def calc(self, session_id, ser_graph, targets, callback):
        """
        Do actual calculation. This method should be called when all data
        is available (i.e., either in shared cache or in memory)
        :param session_id: session id
        :param ser_graph: serialized executable graph
        :param targets: keys of target chunks
        :param callback: promise callback, returns the uid of InProcessCacheActor
        """
        from ..tensor.expressions.datasource import TensorFetchChunk
        graph = deserialize_graph(ser_graph)
        op_key, op_name = concat_operand_keys(graph, '_')

        try:
            context_dict = dict()
            comp_nodes = []
            absent_keys = []
            spill_load_futures = dict()
            for chunk in graph.iter_nodes():
                try:
                    # try load chunk from shared cache
                    if isinstance(chunk.op, TensorFetchChunk):
                        context_dict[chunk.key] = self._chunk_store.get(session_id, chunk.key)
                        self._mem_quota_ref.release_quota(self._build_load_key(op_key, chunk.key))
                    else:
                        comp_nodes.append(chunk.op.key)
                except KeyError:
                    # chunk not in shared cache, we load it from spill directly
                    if self._spill_load_pool is not None and spill_exists(chunk.key):
                        logger.debug('Load chunk %s directly from spill', chunk.key)
                        self._mem_quota_ref.process_quota(self._build_load_key(op_key, chunk.key))
                        spill_load_futures[chunk.key] = self._spill_load_pool.submit(read_spill_file, chunk.key)
                    else:
                        absent_keys.append(chunk.key)
            if absent_keys:
                raise ObjectNotInPlasma(absent_keys)

            # collect results from greenlets
            if spill_load_futures:
                for k, future in spill_load_futures.items():
                    context_dict[k] = future.result()
                    self._mem_quota_ref.hold_quota(self._build_load_key(op_key, k))
                spill_load_futures.clear()

            logger.debug('Start calculating operand %r.', comp_nodes)

            start_time = time.time()

            # mark targets as processing
            target_keys = [k for k in targets if not self._chunk_store.contains(session_id, k)]
            [self._mem_quota_ref.process_quota(k) for k in target_keys]

            # start actual execution
            executor = Executor(storage=context_dict)
            results = self._execution_pool.submit(executor.execute_graph, graph, targets).result()

            for k in list(context_dict.keys()):
                del context_dict[k]
                self._mem_quota_ref.release_quota(self._build_load_key(op_key, k))

            end_time = time.time()

            [self._mem_quota_ref.hold_quota(k) for k in target_keys]

            # adjust sizes in allocation
            save_sizes = dict()
            for k, v in zip(targets, results):
                if not self._chunk_store.contains(session_id, k):
                    _calc_result_cache[k] = (session_id, v)
                    save_sizes[k] = calc_data_size(v)
                    self._mem_quota_ref.apply_allocation(k, save_sizes[k])

            if self._status_ref:
                self._status_ref.update_mean_stats(
                    'calc_speed.' + op_name, sum(save_sizes.values()) * 1.0 / (end_time - start_time),
                    _tell=True, _wait=False)

            logger.debug('Finish calculating operand %r.', comp_nodes)
            self.tell_promise(callback, self._inproc_cache_ref.uid, save_sizes)
            self._dispatch_ref.register_free_slot(self.uid, 'cpu', _tell=True)
        except:  # noqa: E722
            self._dispatch_ref.register_free_slot(self.uid, 'cpu', _tell=True)
            raise
