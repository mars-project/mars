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
from collections import defaultdict
from functools import partial

from .. import promise
from ..compat import six, OrderedDict3
from ..config import options
from ..errors import *
from ..executor import Executor
from ..utils import to_str, deserialize_graph, log_unhandled, calc_data_size
from .spill import read_spill_file, write_spill_file, spill_exists
from .utils import WorkerActor, concat_operand_keys, build_load_key, get_chunk_key

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
    def dump_cache(self, session_id, keys, callback):
        """
        Dump data in rss memory into shared cache
        """
        from ..scheduler.chunkmeta import WorkerMeta
        meta_dict = dict()

        @log_unhandled
        def _try_put_chunk(chunk_key, data_size, data_shape):
            logger.debug('Try putting %s into shared cache.', chunk_key)
            session_chunk_key = (session_id, chunk_key)
            try:
                if session_chunk_key not in _calc_result_cache:
                    if not self._chunk_store.contains(session_id, chunk_key):
                        raise KeyError('Data key %s not found in inproc cache', chunk_key)
                    return

                buf = None
                try:
                    buf = self._chunk_store.put(session_id, chunk_key,
                                                _calc_result_cache[session_chunk_key])
                    del _calc_result_cache[session_chunk_key]
                    self._mem_quota_ref.release_quota(chunk_key, _tell=True)

                    self._chunk_holder_ref.register_chunk(session_id, chunk_key)
                    data_size = self._chunk_store.get_actual_size(session_id, chunk_key)
                    meta_dict[chunk_key] = WorkerMeta(
                        chunk_size=data_size, chunk_shape=data_shape, workers=(self.address,))
                finally:
                    del buf

            except StoreFull:
                # if we cannot put data into shared cache, we store it into spill directly
                if not isinstance(chunk_key, tuple):
                    self._chunk_holder_ref.spill_size(data_size, _tell=True)
                _put_spill_directly(chunk_key, data_size, data_shape)

        @log_unhandled
        def _put_spill_directly(chunk_key, data_size, data_shape, *_):
            if self._spill_dump_pool is None:
                raise SpillNotConfigured

            session_chunk_key = (session_id, chunk_key)
            logger.debug('Writing data %s directly into spill.', chunk_key)
            self._spill_dump_pool.submit(write_spill_file, chunk_key,
                                         _calc_result_cache[session_chunk_key]).result()

            del _calc_result_cache[session_chunk_key]
            self._mem_quota_ref.release_quota(chunk_key, _tell=True)

            meta_dict[chunk_key] = WorkerMeta(
                chunk_size=data_size, chunk_shape=data_shape, workers=(self.address,))

        @log_unhandled
        def _finish_store(*_):
            data_sizes = dict()
            meta_client = self.get_meta_client()

            keys, metas = [], []
            for k, v in meta_dict.items():
                keys.append(k)
                metas.append(v)
                data_sizes[k] = v.chunk_size

            meta_client.batch_set_chunk_meta(session_id, keys, metas)
            self.tell_promise(callback, data_sizes)

        promises = []
        for k in keys:
            value = _calc_result_cache[(session_id, k)]
            data_size = calc_data_size(value)
            # for some special operands(argmax, argmean, mean, ..), intermediate chunk data has multiple parts, choose
            # first part's shape as chunk's shape.
            data_shape = value[0].shape if isinstance(value, tuple) else value.shape
            del value
            promises.append(
                promise.finished().then(partial(_try_put_chunk, k, data_size, data_shape))
            )
        promise.all_(promises).then(_finish_store) \
            .catch(lambda *exc: self.tell_promise(callback, *exc, **dict(_accept=False)))

    @staticmethod
    @log_unhandled
    def remove_cache(session_id, keys):
        """
        Remove data from cache
        """
        for k in keys:
            del _calc_result_cache[(session_id, k)]


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

    def _collect_input_data(self, session_id, graph, op_key):
        from ..operands import Fetch, FetchShuffle

        comp_nodes = set()
        fetch_keys = set()
        for chunk in graph:
            if isinstance(chunk.op, Fetch):
                fetch_keys.add(chunk.op.to_fetch_key or chunk.key)
            elif isinstance(chunk.op, FetchShuffle):
                shuffle_key = graph.successors(chunk)[0].op.shuffle_key
                for k in chunk.op.to_fetch_keys:
                    fetch_keys.add((k, shuffle_key))
            else:
                comp_nodes.add(chunk.key)

        context_dict = dict()
        absent_keys = []
        spill_load_futures = dict()
        for key in fetch_keys:
            try:
                # try load chunk from shared cache
                context_dict[key] = self._chunk_store.get(session_id, key)
                self._mem_quota_ref.release_quota(build_load_key(op_key, key))
            except KeyError:
                # chunk not in shared cache, we load it from spill directly
                if self._spill_load_pool is not None and spill_exists(key):
                    logger.debug('Load chunk %s directly from spill', key)
                    self._mem_quota_ref.process_quota(build_load_key(op_key, key))
                    spill_load_futures[key] = self._spill_load_pool.submit(read_spill_file, key)
                else:
                    absent_keys.append(key)
        if absent_keys:
            logger.error('Chunk requirements %r unmet.')
            raise ObjectNotInPlasma(absent_keys)

        # collect results from futures
        direct_load_keys = []
        if spill_load_futures:
            for k, future in spill_load_futures.items():
                context_dict[k] = future.result()
                load_key = build_load_key(op_key, k)
                direct_load_keys.append(load_key)
                self._mem_quota_ref.hold_quota(load_key)
            spill_load_futures.clear()

        return context_dict, direct_load_keys

    def _calc_results(self, graph, context_dict, chunk_targets):
        # mark targets as processing
        for k in chunk_targets:
            self._mem_quota_ref.process_quota(k)

        # start actual execution
        executor = Executor(storage=context_dict)
        self._execution_pool.submit(executor.execute_graph, graph,
                                    chunk_targets, retval=False).result()

        # collect results
        result_pairs = []
        collected_chunk_keys = set()
        for k, v in context_dict.items():
            if isinstance(k, tuple):
                k = tuple(to_str(i) for i in k)
            else:
                k = to_str(k)

            chunk_key = get_chunk_key(k)
            if chunk_key in chunk_targets:
                result_pairs.append((k, v))
                collected_chunk_keys.add(chunk_key)

        for k in chunk_targets:
            if k not in collected_chunk_keys:  # pragma: no cover
                raise KeyError(k)
            self._mem_quota_ref.hold_quota(k)
        return result_pairs

    @promise.reject_on_exception
    @log_unhandled
    def calc(self, session_id, ser_graph, chunk_targets, callback):
        """
        Do actual calculation. This method should be called when all data
        is available (i.e., either in shared cache or in memory)
        :param session_id: session id
        :param ser_graph: serialized executable graph
        :param chunk_targets: keys of target chunks
        :param callback: promise callback, returns the uid of InProcessCacheActor
        """
        graph = deserialize_graph(ser_graph)
        op_key, op_name = concat_operand_keys(graph, '_')
        chunk_targets = set(chunk_targets)

        try:
            context_dict, direct_load_keys = self._collect_input_data(session_id, graph, op_key)

            logger.debug('Start calculating operand %s.', op_key)
            start_time = time.time()

            try:
                result_pairs = self._calc_results(graph, context_dict, chunk_targets)
            finally:
                # release memory alloc for load keys
                for k in direct_load_keys:
                    self._mem_quota_ref.release_quota(build_load_key(op_key, k))

            end_time = time.time()

            # adjust sizes in allocation
            save_sizes = dict()
            apply_alloc_sizes = defaultdict(lambda: 0)
            for k, v in result_pairs:
                if not self._chunk_store.contains(session_id, k):
                    _calc_result_cache[(session_id, k)] = v
                    data_size = save_sizes[k] = calc_data_size(v)
                    apply_alloc_sizes[get_chunk_key(k)] += data_size

            for k, v in apply_alloc_sizes.items():
                self._mem_quota_ref.alter_allocation(k, v)

            if self._status_ref:
                self._status_ref.update_mean_stats(
                    'calc_speed.' + op_name, sum(save_sizes.values()) * 1.0 / (end_time - start_time),
                    _tell=True, _wait=False)

            logger.debug('Finish calculating operand %s.', op_key)
            self.tell_promise(callback, self._inproc_cache_ref.uid, save_sizes)
            self._dispatch_ref.register_free_slot(self.uid, 'cpu', _tell=True)
        except:  # noqa: E722
            self._dispatch_ref.register_free_slot(self.uid, 'cpu', _tell=True)
            raise
