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

import functools
import logging
import random
import sys
import time
from collections import defaultdict, OrderedDict

from .. import promise
from ..config import options
from ..compat import six
from ..errors import *
from ..utils import deserialize_graph, log_unhandled, to_str
from .chunkholder import ensure_chunk
from .spill import spill_exists
from .utils import WorkerActor

logger = logging.getLogger(__name__)

_WORKER_RETRY_ERRORS = (PinChunkFailed, NoDataToSpill, ObjectNotInPlasma)


class ExecutionActor(WorkerActor):
    """
    Actor for execution control
    """
    _graph_stages = dict()
    _mem_requests = dict()
    _pin_requests = dict()
    _est_finish_times = dict()
    _retry_delays = defaultdict(lambda: 0)
    _last_dump_time = time.time()
    _stop_requests = set()

    def __init__(self):
        super(ExecutionActor, self).__init__()
        self._chunk_holder_ref = None
        self._dispatch_ref = None
        self._mem_quota_ref = None
        self._status_ref = None

        self._scheduler_resource_ref = None

        self._callbacks = defaultdict(list)
        self._callback_cache = OrderedDict()
        self._size_cache = dict()

    def post_create(self):
        from ..scheduler.resource import ResourceActor
        from .chunkholder import ChunkHolderActor
        from .dispatcher import DispatchActor
        from .quota import MemQuotaActor
        from .status import StatusActor

        super(ExecutionActor, self).post_create()
        self._chunk_holder_ref = self.promise_ref(ChunkHolderActor.default_uid())

        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())
        self._dispatch_ref.register_free_slot(self.uid, 'execution')

        self._mem_quota_ref = self.promise_ref(MemQuotaActor.default_uid())

        scheduler_addr = self.get_scheduler(self.uid)
        self._scheduler_resource_ref = self.ctx.actor_ref(ResourceActor.default_uid(),
                                                          address=scheduler_addr)

        self.register_process_down_handler()

        self._status_ref = self.ctx.actor_ref(StatusActor.default_uid())
        if not self.ctx.has_actor(self._status_ref):
            self._status_ref = None

        self.periodical_dump()

    def periodical_dump(self):
        """
        Periodically dump debug information
        """
        if logger.getEffectiveLevel() > logging.DEBUG:
            return
        cls = type(self)
        if cls._last_dump_time < time.time() - 10:
            cls._last_dump_time = time.time()
            if self._graph_stages:
                self._dump_execution_stages()
        self.ref().periodical_dump(_tell=True, _delay=10)

    @staticmethod
    def _estimate_calc_memory(graph, data_sizes, targets):
        from ..tensor.execution.core import Executor
        size_ctx = dict((k, (v, v)) for k, v in data_sizes.items())
        executor = Executor(storage=size_ctx)
        res = executor.execute_graph(graph, targets, mock=True)
        target_sizes = dict(zip(targets, res))

        total_mem = sum(target_sizes[key][1] for key in targets)
        if total_mem:
            for key in targets:
                r = target_sizes[key]
                target_sizes[key] = (r[0], max(r[1], r[1] * executor.mock_max_memory // total_mem))
        return target_sizes

    @staticmethod
    def _build_load_key(graph_key, chunk_key):
        return '%s_load_memory_%s' % (graph_key, chunk_key)

    @promise.reject_on_exception
    @log_unhandled
    def _fetch_remote_data(self, session_id, graph_key, chunk_key, remote_addr, *_, **kwargs):
        """
        Asynchronously send data receiving command to a remote address
        :param session_id: session id
        :param graph_key: graph key
        :param chunk_key: chunk key
        :param remote_addr: remote server containing provided chunk key
        :return: promise object
        """
        from .dispatcher import DispatchActor

        remote_disp_ref = self.promise_ref(uid=DispatchActor.default_uid(),
                                           address=remote_addr)
        ensure_cached = kwargs.pop('ensure_cached', True)

        @log_unhandled
        def _finish_fetch(*_):
            self._chunk_holder_ref.pin_chunks(graph_key, chunk_key)
            if self._chunk_holder_ref.is_stored(chunk_key):
                self._mem_quota_ref.release_quota(self._build_load_key(graph_key, chunk_key))

        @log_unhandled
        def _fetch_step(sender_uid):
            sender_ref = self.promise_ref(sender_uid, address=remote_addr)
            logger.debug('Request for chunk %s transferring from %s', chunk_key, remote_addr)
            return sender_ref.send_data(
                session_id, chunk_key, self.address, ensure_cached=ensure_cached,
                timeout=options.worker.prepare_data_timeout, _promise=True
            ).then(_finish_fetch)

        return remote_disp_ref.get_free_slot('sender', _promise=True).then(_fetch_step)

    def estimate_graph_finish_time(self, graph_key, graph, calc_fetch=True, base_time=None):
        """
        Calc predictions for given chunk graph
        """
        if graph_key not in self._graph_stages:
            return

        from ..tensor.expressions.datasource import TensorFetchChunk
        ops = set(type(c.op).__name__ for c in graph if not isinstance(c.op, TensorFetchChunk))
        op_calc_key = ('calc_speed.' + list(ops)[0]) if len(ops) == 1 else None

        stats = defaultdict(lambda: dict(count=0))
        if self._status_ref:
            stats.update(self._status_ref.get_stats(['disk_read_speed', 'disk_write_speed',
                                                     'net_transfer_speed', op_calc_key]))

        if op_calc_key not in stats:
            return None
        if stats[op_calc_key]['count'] < options.optimize.min_stats_count:
            return None
        if abs(stats[op_calc_key]['count']) < 1e-6:
            return None

        input_size = 0
        net_size = 0
        disk_size = 0
        base_time = base_time or time.time()

        if calc_fetch:
            for c in graph:
                if not isinstance(c.op, TensorFetchChunk):
                    break
                input_size += c.nbytes
                if self._chunk_holder_ref.is_stored(c.key):
                    continue
                if spill_exists(c.key):
                    disk_size += c.nbytes
                else:
                    net_size += c.nbytes

            if stats['net_transfer_speed']['count'] >= options.optimize.min_stats_count:
                base_time += net_size * 1.0 / stats['net_transfer_speed']['mean']
            if stats['disk_read_speed']['count'] >= options.optimize.min_stats_count:
                base_time += disk_size * 1.0 / stats['disk_read_speed']['mean']
            else:
                base_time += disk_size * 1.0 / options.optimize.default_disk_io_speed

        est_finish_time = base_time + input_size * 1.0 / stats[op_calc_key]['mean']

        self._est_finish_times[graph_key] = est_finish_time
        self._status_ref.update_stats(dict(
            min_est_finish_time=min(self._est_finish_times.values()),
            max_est_finish_time=max(self._est_finish_times.values()),
        ), _tell=True, _wait=False)

        self.ref().estimate_graph_finish_time(graph_key, graph, _tell=True, _delay=1)

    def _update_stage_info(self, session_id, key, ops, stage):
        self._graph_stages[key] = (stage, time.time())
        if self._status_ref:
            self._status_ref.update_progress(session_id, key, ops, stage, _tell=True, _wait=False)

    @promise.reject_on_exception
    @log_unhandled
    def execute_graph(self, session_id, graph_key, graph_ser, io_meta, data_sizes, send_targets=None,
                      callback=None):
        """
        Submit graph to the worker and control the execution
        :param session_id: session id
        :param graph_key: graph key
        :param graph_ser: serialized executable graph
        :param io_meta: io meta of the chunk
        :param data_sizes: data size of each input chunk, as a dict
        :param send_targets: targets to send results after execution
        :param callback: promise callback
        """
        from ..tensor.expressions.datasource import TensorFetchChunk
        data_sizes = data_sizes or dict()
        graph = deserialize_graph(graph_ser)
        targets = io_meta['chunks']
        chunks_use_once = set(io_meta.get('input_chunks', [])) - set(io_meta.get('shared_input_chunks', []))

        graph_ops = ','.join(type(c.op).__name__ for c in graph if not isinstance(c.op, TensorFetchChunk))
        logger.debug('Worker graph %s(%s) targeting at %r accepted.', graph_key, graph_ops, targets)

        self._update_stage_info(session_id, graph_key, graph_ops, 'allocate_resource')

        # add callbacks to callback store
        if callback is None:
            callback = []
        elif not isinstance(callback, list):
            callback = [callback]
        self._callbacks[graph_key].extend(callback)
        if graph_key in self._callback_cache:
            del self._callback_cache[graph_key]

        unspill_keys = []
        transfer_keys = []
        calc_keys = set()

        alloc_mem_batch = dict()
        alloc_cache_batch = dict()
        input_chunk_keys = dict()

        if self._status_ref:
            self.estimate_graph_finish_time(graph_key, graph)

        memory_estimations = self._estimate_calc_memory(graph, data_sizes, targets)

        # collect potential allocation sizes
        for chunk in graph:
            if not isinstance(chunk.op, TensorFetchChunk) and chunk.key in targets:
                # use estimated size as potential allocation size
                calc_keys.add(chunk.key)
                alloc_cache_batch[chunk.key], alloc_mem_batch[chunk.key] = memory_estimations[chunk.key]
            else:
                # use actual size as potential allocation size
                input_chunk_keys[chunk.key] = data_sizes.get(chunk.key, chunk.nbytes)

        calc_keys = [to_str(k) for k in calc_keys]

        keys_to_pin = list(input_chunk_keys.keys())
        try:
            self._pin_requests[graph_key] = set(self._chunk_holder_ref.pin_chunks(graph_key, keys_to_pin))
        except PinChunkFailed:
            # cannot pin input chunks: retry later
            callback = self._callbacks[graph_key]
            self._cleanup_graph(session_id, graph_key)

            retry_delay = self._retry_delays[graph_key] + 0.5 + random.random()
            self._retry_delays[graph_key] = min(1 + self._retry_delays[graph_key], 30)
            self.ref().execute_graph(session_id, graph_key, graph_ser, io_meta, data_sizes, send_targets, callback,
                                     _tell=True, _delay=retry_delay)
            return

        load_chunk_sizes = dict((k, v) for k, v in input_chunk_keys.items()
                                if k not in self._pin_requests[graph_key])
        alloc_mem_batch.update((self._build_load_key(graph_key, k), v)
                               for k, v in load_chunk_sizes.items() if k in chunks_use_once)
        self._chunk_holder_ref.spill_size(sum(alloc_cache_batch.values()), _tell=True)

        # build allocation promises
        batch_alloc_promises = []
        if alloc_mem_batch:
            self._mem_requests[graph_key] = list(alloc_mem_batch.keys())
            batch_alloc_promises.append(self._mem_quota_ref.request_batch_quota(alloc_mem_batch, _promise=True))

        @log_unhandled
        def _prepare_inputs(*_):
            if graph_key in self._stop_requests:
                raise ExecutionInterrupted

            logger.debug('Start preparing input data for graph %s', graph_key)
            self._update_stage_info(session_id, graph_key, graph_ops, 'prepare_inputs')
            prepare_promises = []

            handled_keys = set()
            for chunk in graph:
                if chunk.key in handled_keys:
                    continue
                if not isinstance(chunk.op, TensorFetchChunk):
                    continue
                handled_keys.add(chunk.key)

                if self._chunk_holder_ref.is_stored(chunk.key):
                    # data already in plasma: we just pin it
                    pinned_keys = self._chunk_holder_ref.pin_chunks(graph_key, chunk.key)
                    if chunk.key in pinned_keys:
                        self._mem_quota_ref.release_quota(self._build_load_key(graph_key, chunk.key))
                        continue

                if spill_exists(chunk.key):
                    if chunk.key in chunks_use_once:
                        # input only use in current operand, we only need to load it into process memory
                        continue
                    self._mem_quota_ref.release_quota(self._build_load_key(graph_key, chunk.key))
                    load_fun = functools.partial(lambda gk, ck, *_: self._chunk_holder_ref.pin_chunks(gk, ck),
                                       graph_key, chunk.key)
                    unspill_keys.append(chunk.key)
                    prepare_promises.append(ensure_chunk(self, session_id, chunk.key, move_to_end=True) \
                                            .then(load_fun))
                    continue

                # load data from another worker
                chunk_key = to_str(chunk.key)
                chunk_meta = self.get_meta_client().get_chunk_meta(session_id, chunk_key)
                if chunk_meta is None:
                    raise DependencyMissing('Dependency %s not met on sending.' % chunk_key)

                worker_priorities = []
                for w in chunk_meta.workers:
                    # todo sort workers by speed of network and other possible factors
                    worker_priorities.append((w, (0, )))

                transfer_keys.append(chunk_key)

                # fetch data from other workers, if one fails, try another
                sorted_workers = sorted(worker_priorities, key=lambda pr: pr[1])
                p = self._fetch_remote_data(session_id, graph_key, chunk_key, sorted_workers[0][0],
                                            ensure_cached=chunk_key not in chunks_use_once)
                for wp in sorted_workers[1:]:
                    p = p.catch(functools.partial(self._fetch_remote_data, session_id, graph_key, chunk_key, wp[0],
                                                  ensure_cached=chunk_key not in chunks_use_once))
                prepare_promises.append(p)

            logger.debug('Graph key %s: Targets %r, unspill keys %r, transfer keys %r',
                         graph_key, targets, unspill_keys, transfer_keys)
            return promise.all_(prepare_promises)

        @log_unhandled
        def _wait_free_slot(*_):
            logger.debug('Waiting for free CPU slot for graph %s', graph_key)
            self._update_stage_info(session_id, graph_key, graph_ops, 'fetch_free_slot')
            return self._dispatch_ref.get_free_slot('cpu', _promise=True)

        @log_unhandled
        def _send_calc_request(calc_uid):
            if graph_key in self._stop_requests:
                raise ExecutionInterrupted

            # get allocation for calc, in case that memory exhausts
            target_allocs = dict()
            for chunk in graph:
                if isinstance(chunk.op, TensorFetchChunk):
                    if not self._chunk_holder_ref.is_stored(chunk.key):
                        alloc_key = self._build_load_key(graph_key, chunk.key)
                        if alloc_key in alloc_mem_batch:
                            target_allocs[alloc_key] = alloc_mem_batch[alloc_key]
                elif chunk.key in targets:
                    target_allocs[chunk.key] = alloc_mem_batch[chunk.key]

            logger.debug('Start calculation for graph %s', graph_key)

            self._update_stage_info(session_id, graph_key, graph_ops, 'calculate')
            calc_ref = self.promise_ref(calc_uid)

            self.estimate_graph_finish_time(graph_key, graph, calc_fetch=False)
            # make sure that memory suffices before actually run execution
            return self._mem_quota_ref.request_batch_quota(target_allocs, _promise=True) \
                .then(lambda *_: self._deallocate_scheduler_resource(session_id, graph_key, delay=2)) \
                .then(lambda *_: calc_ref.calc(session_id, graph_ser, targets, _promise=True))

        @log_unhandled
        def _dump_cache(inproc_uid, save_sizes):
            # do some clean up
            self._deallocate_scheduler_resource(session_id, graph_key)
            inproc_ref = self.promise_ref(inproc_uid)

            if graph_key in self._stop_requests:
                inproc_ref.remove_cache(calc_keys, _tell=True)
                raise ExecutionInterrupted

            self._update_stage_info(session_id, graph_key, graph_ops, 'dump_cache')

            logger.debug('Graph %s: Start putting %r into shared cache. Target actor uid %s.',
                         graph_key, calc_keys, inproc_uid)

            self._chunk_holder_ref.unpin_chunks(graph_key, list(set(c.key for c in graph)), _tell=True)
            if logger.getEffectiveLevel() <= logging.DEBUG:
                self._dump_execution_stages()
                # self._cache_ref.dump_cache_status(_tell=True)

            self._size_cache[graph_key] = save_sizes

            if not send_targets:
                # no endpoints to send, dump keys into shared memory and return
                logger.debug('Worker graph %s(%s) finished execution. Dumping %r into plasma...',
                             graph_key, graph_ops, calc_keys)
                return inproc_ref.dump_cache(calc_keys, _promise=True)
            else:
                # dump keys into shared memory and send
                logger.debug('Worker graph %s(%s) finished execution. Dumping %r into plasma '
                             'while actively transferring %r...',
                             graph_key, graph_ops, calc_keys, send_targets)

                return inproc_ref.dump_cache(calc_keys, _promise=True) \
                        .then(_do_active_transfer)

        @log_unhandled
        def _do_active_transfer(*_):
            # transfer the result chunk to expected endpoints
            @log_unhandled
            def _send_chunk(sender_uid, chunk_key, target_addrs):
                sender_ref = self.promise_ref(sender_uid)
                logger.debug('Request for chunk %s sent to %s', chunk_key, target_addrs)
                return sender_ref.send_data(session_id, chunk_key, target_addrs, ensure_cached=False,
                                            timeout=options.worker.prepare_data_timeout, _promise=True)

            if graph_key in self._mem_requests:
                self._mem_quota_ref.release_quotas(self._mem_requests[graph_key], _tell=True)
                del self._mem_requests[graph_key]

            promises = []
            for key, targets in send_targets.items():
                promises.append(self._dispatch_ref.get_free_slot('sender', _promise=True) \
                                .then(functools.partial(_send_chunk, chunk_key=key, target_addrs=targets)) \
                                .catch(lambda *_: None))
            return promise.all_(promises)

        @log_unhandled
        def _handle_rejection(*exc):
            # some error occurred...
            logger.debug('Entering _handle_rejection() for graph %s', graph_key)
            if logger.getEffectiveLevel() <= logging.DEBUG:
                self._dump_execution_stages()
                # self._cache_ref.dump_cache_status(_tell=True)

            if graph_key in self._stop_requests:
                self._stop_requests.remove(graph_key)

            self._mem_quota_ref.cancel_requests(list(alloc_mem_batch.keys()), _tell=True)

            if not issubclass(exc[0], _WORKER_RETRY_ERRORS):
                # exception not retryable: call back to scheduler
                if isinstance(exc[0], ExecutionInterrupted):
                    logger.warning('Execution of graph %s interrupted.', graph_key)
                else:
                    try:
                        six.reraise(*exc)
                    except:
                        logger.exception('Unexpected error occurred in executing %s', graph_key)
                self._invoke_finish_callbacks(session_id, graph_key, *exc, **dict(_accept=False))
                return

            logger.debug('Graph %s rejected from execution because of %s', graph_key, exc[0].__name__)

            cb = self._callbacks[graph_key]
            self._cleanup_graph(session_id, graph_key)

            if issubclass(exc[0], ObjectNotInPlasma):
                retry_delay = 0
            else:
                retry_delay = self._retry_delays[graph_key] + 0.5 + random.random()
                self._retry_delays[graph_key] = min(1 + self._retry_delays[graph_key], 30)

            self.ref().execute_graph(session_id, graph_key, graph_ser, io_meta, data_sizes, send_targets, cb,
                                     _tell=True, _delay=retry_delay)

        promise.all_(batch_alloc_promises).then(_prepare_inputs) \
            .then(_wait_free_slot).then(_send_calc_request) \
            .then(_dump_cache).then(lambda *_: self._invoke_finish_callbacks(session_id, graph_key,
                                                                             self._size_cache.get(graph_key))) \
            .catch(_handle_rejection)

    def _deallocate_scheduler_resource(self, session_id, graph_key, delay=0):
        try:
            self._scheduler_resource_ref.deallocate_resource(
                session_id, graph_key, self.address, _delay=delay, _tell=True, _wait=False)
        except:
            pass

    def _cleanup_graph(self, session_id, graph_key):
        """
        Do clean up after graph is executed
        :param session_id: session id
        :param graph_key: graph key
        """
        logger.debug('Cleaning callbacks for graph %s', graph_key)
        if graph_key in self._callbacks:
            del self._callbacks[graph_key]
        if graph_key in self._graph_stages:
            del self._graph_stages[graph_key]
        if graph_key in self._est_finish_times:
            del self._est_finish_times[graph_key]

        if graph_key in self._mem_requests:
            self._mem_quota_ref.release_quotas(self._mem_requests[graph_key], _tell=True)
            del self._mem_requests[graph_key]

        if graph_key in self._pin_requests:
            self._chunk_holder_ref.unpin_chunks(graph_key, self._pin_requests[graph_key], _tell=True)
            del self._pin_requests[graph_key]

        if self._status_ref:
            self._status_ref.remove_progress(session_id, graph_key, _tell=True, _wait=False)

    @promise.reject_on_exception
    @log_unhandled
    def add_finish_callback(self, session_id, graph_key, callback):
        """
        Register a callback to callback store
        :param session_id: session id
        :param graph_key: graph key
        :param callback: promise call
        """
        logger.debug('Adding callback %r for graph %s', callback, graph_key)
        self._callbacks[graph_key].append(callback)
        if graph_key in self._callback_cache:
            _, args, kwargs = self._callback_cache[graph_key]
            self._invoke_finish_callbacks(session_id, graph_key, *args, **kwargs)

    @log_unhandled
    def stop_execution(self, graph_key):
        """
        Mark graph for stopping
        :param graph_key: graph key
        """
        logger.debug('Receive stop for graph %s', graph_key)
        if graph_key not in self._graph_stages:
            return
        if self._graph_stages[graph_key][0] == 'allocate_resource':
            try:
                raise ExecutionInterrupted
            except:  # noqa: E722
                exc_info = sys.exc_info()
            if graph_key in self._mem_requests:
                self._mem_quota_ref.cancel_requests(self._mem_requests[graph_key], exc_info, _tell=True)
        self._stop_requests.add(graph_key)

    @log_unhandled
    def _invoke_finish_callbacks(self, session_id, graph_key, *args, **kwargs):
        """
        Call finish callback when execution is done
        :param session_id: session id
        :param graph_key: graph key
        """
        logger.debug('Send finish callback for graph %s into %d targets', graph_key,
                     len(self._callbacks[graph_key]))
        for cb in self._callbacks[graph_key]:
            self.tell_promise(cb, *args, **kwargs)
        self._cleanup_graph(session_id, graph_key)
        if graph_key in self._retry_delays:
            del self._retry_delays[graph_key]

        if graph_key not in self._callback_cache:
            # preserve callback result for several time to allow add_finish_callback()
            # after execution done
            clean_keys = []
            cur_time = time.time()
            last_finish_time = cur_time - options.worker.callback_preserve_time
            self._callback_cache[graph_key] = (cur_time, args, kwargs)
            for k, tp in self._callback_cache.items():
                if tp[0] < last_finish_time:
                    clean_keys.append(k)
                else:
                    break
            for k in clean_keys:
                del self._callback_cache[k]
                if k in self._size_cache:
                    del self._size_cache[k]

    def _dump_execution_stages(self):
        if logger.getEffectiveLevel() <= logging.DEBUG:
            cur_time = time.time()
            stages = dict((k, (cur_time - v[-1], v[0])) for k, v in self._graph_stages.items())
            logger.debug('Executing stages: %r', stages)
