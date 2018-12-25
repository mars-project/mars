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
from functools import partial
from collections import defaultdict

from .. import promise
from ..compat import Enum
from ..config import options
from ..errors import PinChunkFailed, WorkerProcessStopped, ExecutionInterrupted, DependencyMissing
from ..tensor.expressions.datasource import TensorFetch
from ..utils import deserialize_graph, log_unhandled
from .chunkholder import ensure_chunk
from .spill import spill_exists
from .utils import WorkerActor, ExpiringCache, concat_operand_keys

logger = logging.getLogger(__name__)


class ExecutionState(Enum):
    PRE_PUSHED = 'pre_pushed'
    ALLOCATING = 'allocating'
    PREPARING_INPUTS = 'preparing_inputs'
    CALCULATING = 'calculating'
    STORING = 'storing'


class GraphExecutionRecord(object):
    """
    Execution records of the graph
    """
    __slots__ = ('graph', 'graph_serialized', '_state', 'op_string', 'targets', 'io_meta',
                 'priority_data', 'data_sizes', 'chunks_use_once', 'state_time',
                 'mem_request', 'pin_request', 'est_finish_time', 'calc_actor_uid',
                 'send_addresses', 'retry_delay', 'enqueue_callback', 'finish_callbacks',
                 'stop_requested', 'succ_keys', 'undone_pred_keys')

    def __init__(self, graph_serialized, state, targets=None, io_meta=None, priority_data=None,
                 data_sizes=None, chunks_use_once=None, mem_request=None, pin_request=None,
                 est_finish_time=None, calc_actor_uid=None, send_addresses=None,
                 retry_delay=None, enqueue_callback=None, finish_callbacks=None,
                 stop_requested=False):
        self.graph_serialized = graph_serialized
        graph = self.graph = deserialize_graph(graph_serialized)

        self._state = state
        self.state_time = time.time()
        self.targets = targets or []
        self.io_meta = io_meta or dict()
        self.data_sizes = data_sizes or dict()
        self.priority_data = priority_data or dict()
        self.chunks_use_once = chunks_use_once or set()
        self.mem_request = mem_request or dict()
        self.pin_request = pin_request or set()
        self.est_finish_time = est_finish_time or time.time()
        self.calc_actor_uid = calc_actor_uid
        self.send_addresses = send_addresses
        self.retry_delay = retry_delay or 0
        self.enqueue_callback = enqueue_callback
        self.finish_callbacks = finish_callbacks or []
        self.stop_requested = stop_requested or False

        self.succ_keys = set()
        self.undone_pred_keys = set()

        _, self.op_string = concat_operand_keys(graph)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self.state_time = time.time()


class GraphResultRecord(object):
    """
    Execution result of a graph
    """
    __slots__ = 'data_sizes', 'exc', 'accept'

    def __init__(self, *args, **kwargs):
        accept = self.accept = kwargs.pop('_accept', True)
        if accept:
            self.data_sizes = args[0]
        else:
            self.exc = args

    def build_args(self):
        if self.accept:
            return (self.data_sizes,), {}
        else:
            return self.exc, dict(_accept=False)


class ExecutionActor(WorkerActor):
    """
    Actor for execution control
    """
    _last_dump_time = time.time()

    def __init__(self):
        super(ExecutionActor, self).__init__()
        self._chunk_holder_ref = None
        self._dispatch_ref = None
        self._task_queue_ref = None
        self._mem_quota_ref = None
        self._status_ref = None
        self._daemon_ref = None

        self._graph_records = dict()  # type: dict[tuple, GraphExecutionRecord]
        self._result_cache = ExpiringCache()  # type: dict[tuple, GraphResultRecord]

    def post_create(self):
        from .chunkholder import ChunkHolderActor
        from .daemon import WorkerDaemonActor
        from .dispatcher import DispatchActor
        from .quota import MemQuotaActor
        from .status import StatusActor
        from .taskqueue import TaskQueueActor

        super(ExecutionActor, self).post_create()
        self._chunk_holder_ref = self.promise_ref(ChunkHolderActor.default_name())
        self._dispatch_ref = self.promise_ref(DispatchActor.default_name())
        self._task_queue_ref = self.promise_ref(TaskQueueActor.default_name())
        self._mem_quota_ref = self.promise_ref(MemQuotaActor.default_name())

        self._daemon_ref = self.ctx.actor_ref(WorkerDaemonActor.default_name())
        if self.ctx.has_actor(self._daemon_ref):
            self._daemon_ref.register_callback(self.ref(), self.handle_process_down.__name__, _tell=True)
        else:
            self._daemon_ref = None

        self._status_ref = self.ctx.actor_ref(StatusActor.default_name())
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
            if self._graph_records:
                self._dump_execution_states()
        self.ref().periodical_dump(_tell=True, _delay=10)

    @promise.reject_on_exception
    @log_unhandled
    def enqueue_graph(self, session_id, graph_key, graph_ser, io_meta, data_sizes,
                      priority_data=None, send_addresses=None, undone_pred_keys=None, callback=None):
        """
        Submit graph to the worker and control the execution
        :param session_id: session id
        :param graph_key: graph key
        :param graph_ser: serialized executable graph
        :param io_meta: io meta of the chunk
        :param data_sizes: data size of each input chunk, as a dict
        :param priority_data: data priority
        :param send_addresses: targets to send results after execution
        :param undone_pred_keys: predecessor keys, available when the submitted graph require predecessors to finish
        :param callback: promise callback
        """
        priority_data = priority_data or dict()

        graph_record = self._graph_records[(session_id, graph_key)] = GraphExecutionRecord(
            graph_ser, ExecutionState.ALLOCATING,
            io_meta=io_meta,
            data_sizes=data_sizes,
            enqueue_callback=callback,
            priority_data=priority_data,
            targets=io_meta['chunks'],
            chunks_use_once=set(io_meta.get('input_chunks', [])) - set(io_meta.get('shared_input_chunks', [])),
            send_addresses=send_addresses,
        )

        if undone_pred_keys:
            for k in undone_pred_keys:
                try:
                    self._graph_records[(session_id, k)].succ_keys.add(graph_key)
                except KeyError:
                    pass
                if (session_id, k) not in self._result_cache or \
                        not self._result_cache[(session_id, k)].accept:
                    graph_record.undone_pred_keys.add(k)

        if not graph_record.undone_pred_keys:
            logger.debug('Worker graph %s(%s) targeting at %r accepted.', graph_key,
                         graph_record.op_string, graph_record.targets)
            self._update_state(session_id, graph_key, ExecutionState.ALLOCATING)
            self._task_queue_ref.enqueue_task(session_id, graph_key, priority_data, _promise=True) \
                .then(lambda *_: self.tell_promise(callback) if callback else None)
        else:
            logger.debug('Worker graph %s(%s) targeting at %r pre-pushed.', graph_key,
                         graph_record.op_string, graph_record.targets)
            self._update_state(session_id, graph_key, ExecutionState.PRE_PUSHED)
            logger.debug('Worker graph %s(%s) now has unfinished predecessors %r.',
                         graph_key, graph_record.op_string, graph_record.undone_pred_keys)

    def _notify_successors(self, session_id, graph_key):
        graph_rec = self._graph_records[(session_id, graph_key)]
        for succ_key in graph_rec.succ_keys:
            succ_query_key = (session_id, succ_key)
            try:
                succ_rec = self._graph_records[succ_query_key]
            except KeyError:
                continue

            try:
                succ_rec.data_sizes.update(self._result_cache[succ_query_key].data_sizes)
            except (KeyError, AttributeError):
                pass
            succ_rec.undone_pred_keys.difference_update((graph_key,))
            if succ_rec.undone_pred_keys:
                logger.debug('Worker graph %s(%s) now has unfinished predecessors %r.',
                             succ_key, succ_rec.op_string, succ_rec.undone_pred_keys)
                continue

            missing_keys = [c.key for c in succ_rec.graph if c.key not in succ_rec.data_sizes
                            and isinstance(c.op, TensorFetchChunk)]
            if missing_keys:
                sizes = self.get_meta_ref(session_id, graph_key, local=False) \
                    .batch_get_chunk_size(session_id, missing_keys)
                succ_rec.data_sizes.update(zip(missing_keys, sizes))
            logger.debug('Worker graph %s(%s) targeting at %r from PRE_PUSHED into ALLOCATING.',
                         succ_key, succ_rec.op_string, succ_rec.targets)
            self._update_state(session_id, succ_key, ExecutionState.ALLOCATING)

            enqueue_callback = succ_rec.enqueue_callback
            p = self._task_queue_ref.enqueue_task(
                session_id, succ_key, succ_rec.priority_data, _promise=True)
            if enqueue_callback:
                p.then(partial(self.tell_promise, enqueue_callback))

    @log_unhandled
    def prepare_quota_request(self, session_id, graph_key):
        """
        Calculate quota request for an execution graph
        :param session_id: session id
        :param graph_key: key of the execution graph
        :return: allocation dict
        """
        try:
            graph_record = self._graph_records[(session_id, graph_key)]
        except KeyError:
            return None

        graph = graph_record.graph
        alloc_mem_batch = dict()
        alloc_cache_batch = dict()
        input_chunk_keys = dict()

        if self._status_ref:
            self.estimate_graph_finish_time(session_id, graph_key)

        # collect potential allocation sizes
        for chunk in graph:
            if not isinstance(chunk.op, TensorFetch) and chunk.key in graph_record.targets:
                # use estimated size as potential allocation size
                alloc_mem_batch[chunk.key] = chunk.rough_nbytes * 2
                alloc_cache_batch[chunk.key] = chunk.rough_nbytes
            else:
                # use actual size as potential allocation size
                input_chunk_keys[chunk.key] = graph_record.data_sizes.get(chunk.key, chunk.nbytes)

        keys_to_pin = list(input_chunk_keys.keys())
        try:
            graph_record.pin_request = set(self._chunk_holder_ref.pin_chunks(graph_key, keys_to_pin))
        except PinChunkFailed:
            # cannot pin input chunks: retry later
            self.dequeue_graph(session_id, graph_key)

            retry_delay = graph_record.retry_delay + 0.5 + random.random()
            graph_record.retry_delay = min(1 + graph_record.retry_delay, 30)
            self.ref().enqueue_graph(
                session_id, graph_key, graph_record.graph_serialized, graph_record.io_meta,
                graph_record.data_sizes, priority_data=graph_record.priority_data,
                send_addresses=graph_record.send_addresses, callback=graph_record.enqueue_callback,
                _tell=True, _delay=retry_delay)
            return None

        load_chunk_sizes = dict((k, v) for k, v in input_chunk_keys.items()
                                if k not in graph_record.pin_request)
        alloc_mem_batch.update((self._build_load_key(graph_key, k), v)
                               for k, v in load_chunk_sizes.items() if k in graph_record.chunks_use_once)
        self._chunk_holder_ref.spill_size(sum(alloc_cache_batch.values()), _tell=True)

        if alloc_mem_batch:
            graph_record.mem_request = alloc_mem_batch
        return alloc_mem_batch

    @log_unhandled
    def dequeue_graph(self, session_id, graph_key):
        """
        Remove execution graph task from queue
        :param session_id: session id
        :param graph_key: key of the execution graph
        """
        self._cleanup_graph(session_id, graph_key)

    @log_unhandled
    def update_priority(self, session_id, graph_key, priority_data):
        """
        Update priority data for given execution graph
        :param session_id: session id
        :param graph_key: key of the execution graph
        :param priority_data: priority data
        """
        query_key = (session_id, graph_key)
        if query_key not in self._graph_records:
            return
        self._graph_records[query_key].priority_data = priority_data
        self._task_queue_ref.update_priority(session_id, graph_key, priority_data)

    @staticmethod
    def _build_load_key(graph_key, chunk_key):
        return '%s_load_memory_%s' % (graph_key, chunk_key)

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

        remote_disp_ref = self.promise_ref(uid=DispatchActor.default_name(),
                                           address=remote_addr)
        ensure_cached = kwargs.pop('ensure_cached', True)

        @log_unhandled
        def _finish_fetch(*_):
            self._chunk_holder_ref.pin_chunks(graph_key, chunk_key)
            if self._chunk_holder_ref.is_stored(chunk_key):
                self._mem_quota_ref.release_quota(self._build_load_key(graph_key, chunk_key))

        @log_unhandled
        def _fetch_step(sender_uid):
            if self._graph_records[(session_id, graph_key)].stop_requested:
                self._dispatch_ref.register_free_slot(sender_uid, 'sender')
                raise ExecutionInterrupted

            sender_ref = self.promise_ref(sender_uid, address=remote_addr)
            logger.debug('Request for chunk %s transferring from %s', chunk_key, remote_addr)
            return sender_ref.send_data(
                session_id, chunk_key, self.address, ensure_cached=ensure_cached,
                timeout=options.worker.prepare_data_timeout, _promise=True
            ).then(_finish_fetch)

        return promise.Promise(done=True) \
            .then(lambda *_: remote_disp_ref.get_free_slot('sender', _promise=True)) \
            .then(_fetch_step)

    def estimate_graph_finish_time(self, session_id, graph_key, calc_fetch=True, base_time=None):
        """
        Calc predictions for given chunk graph
        """
        session_graph_key = (session_id, graph_key)
        if session_graph_key not in self._graph_records:
            return
        graph_record = self._graph_records[session_graph_key]
        graph = graph_record.graph

        ops = set(type(c.op).__name__ for c in graph if not isinstance(c.op, TensorFetch))
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
                if not isinstance(c.op, TensorFetch):
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

        graph_record.est_finish_time = est_finish_time
        self._status_ref.update_stats(dict(
            min_est_finish_time=min(rec.est_finish_time for rec in self._graph_records.values()),
            max_est_finish_time=max(rec.est_finish_time for rec in self._graph_records.values()),
        ), _tell=True, _wait=False)

        self.ref().estimate_graph_finish_time(session_id, graph_key, _tell=True, _delay=1)

    def _update_state(self, session_id, key, state):
        logger.debug('Operand %s switched to %s', key, getattr(state, 'name'))
        record = self._graph_records[(session_id, key)]
        record.state = state
        if self._status_ref:
            self._status_ref.update_progress(session_id, key, record.op_string, state.name,
                                             _tell=True, _wait=False)

    @promise.reject_on_exception
    @log_unhandled
    def start_execution(self, session_id, graph_key, send_addresses=None, callback=None):
        """
        Submit graph to the worker and control the execution
        :param session_id: session id
        :param graph_key: key of the execution graph
        :param send_addresses: targets to send results after execution
        :param callback: promise callback
        """
        graph_record = self._graph_records[(session_id, graph_key)]
        if send_addresses:
            graph_record.send_addresses = send_addresses

        # add callbacks to callback store
        if callback is None:
            callback = []
        elif not isinstance(callback, list):
            callback = [callback]
        graph_record.finish_callbacks.extend(callback)
        try:
            del self._result_cache[(session_id, graph_key)]
        except KeyError:
            pass

        @log_unhandled
        def _wait_free_slot(*_):
            return self._dispatch_ref.get_free_slot('cpu', _promise=True)

        @log_unhandled
        def _handle_success(*_):
            self._notify_successors(session_id, graph_key)
            self._invoke_finish_callbacks(session_id, graph_key)

        @log_unhandled
        def _handle_rejection(*exc):
            # some error occurred...
            logger.debug('Entering _handle_rejection() for graph %s', graph_key)
            self._dump_execution_states()

            if graph_record.stop_requested:
                graph_record.stop_requested = False
                if not isinstance(exc[1], ExecutionInterrupted):
                    try:
                        raise ExecutionInterrupted
                    except ExecutionInterrupted:
                        exc = sys.exc_info()

            if isinstance(exc[1], ExecutionInterrupted):
                logger.warning('Execution of graph %s interrupted.', graph_key)
            else:
                logger.exception('Unexpected error occurred in executing %s', graph_key, exc_info=exc)

            self._result_cache[(session_id, graph_key)] = GraphResultRecord(*exc, **dict(_accept=False))
            self._invoke_finish_callbacks(session_id, graph_key)

        self._prepare_graph_inputs(session_id, graph_key) \
            .then(_wait_free_slot) \
            .then(lambda uid: self._send_calc_request(session_id, graph_key, uid)) \
            .then(lambda uid, sizes: self._dump_cache(session_id, graph_key, uid, sizes)) \
            .then(_handle_success, _handle_rejection)

    @log_unhandled
    def _prepare_graph_inputs(self, session_id, graph_key):
        """
        Load input data from spilled storage and other workers
        :param session_id: session id
        :param graph_key: key of the execution graph
        """
        graph_record = self._graph_records[(session_id, graph_key)]
        if graph_record.stop_requested:
            raise ExecutionInterrupted

        unspill_keys = []
        transfer_keys = []

        logger.debug('Start preparing input data for graph %s', graph_key)
        self._update_state(session_id, graph_key, ExecutionState.PREPARING_INPUTS)
        prepare_promises = []
        chunks_use_once = graph_record.chunks_use_once

        handled_keys = set()
        for chunk in graph_record.graph:
            if not isinstance(chunk.op, TensorFetch):
                continue
            if chunk.key in handled_keys:
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
                load_fun = partial(lambda gk, ck, *_: self._chunk_holder_ref.pin_chunks(gk, ck),
                                   graph_key, chunk.key)
                unspill_keys.append(chunk.key)
                prepare_promises.append(ensure_chunk(self, session_id, chunk.key, move_to_end=True) \
                                        .then(load_fun))
                continue

            # load data from another worker
            chunk_meta = self.get_meta_ref(session_id, chunk.key) \
                .get_chunk_meta(session_id, chunk.key)
            if chunk_meta is None:
                raise DependencyMissing('Dependency %s not met on sending.' % chunk.key)
            worker_results = chunk_meta.workers

            worker_priorities = []
            for worker_ip in worker_results:
                # todo sort workers by speed of network and other possible factors
                worker_priorities.append((worker_ip, (0, )))

            transfer_keys.append(chunk.key)

            # fetch data from other workers, if one fails, try another
            sorted_workers = sorted(worker_priorities, key=lambda pr: pr[1])
            p = self._fetch_remote_data(session_id, graph_key, chunk.key, sorted_workers[0][0],
                                        ensure_cached=chunk.key not in chunks_use_once)
            for wp in sorted_workers[1:]:
                p = p.catch(functools.partial(self._fetch_remote_data, session_id, graph_key, chunk.key, wp[0],
                                              ensure_cached=chunk.key not in chunks_use_once))
            prepare_promises.append(p)

        logger.debug('Graph key %s: Targets %r, unspill keys %r, transfer keys %r',
                     graph_key, graph_record.targets, unspill_keys, transfer_keys)
        return promise.all_(prepare_promises)

    @log_unhandled
    def _send_calc_request(self, session_id, graph_key, calc_uid):
        """
        Start actual calculation in CpuCalcActor
        :param session_id: session id
        :param graph_key: key of the execution graph
        :param calc_uid: uid of the allocated CpuCalcActor
        """
        graph_record = self._graph_records[(session_id, graph_key)]
        try:
            if graph_record.stop_requested:
                raise ExecutionInterrupted

            graph_record.calc_actor_uid = calc_uid

            # get allocation for calc, in case that memory exhausts
            target_allocs = dict()
            for chunk in graph_record.graph:
                if isinstance(chunk.op, TensorFetch):
                    if not self._chunk_holder_ref.is_stored(chunk.key):
                        alloc_key = self._build_load_key(graph_key, chunk.key)
                        if alloc_key in graph_record.mem_request:
                            target_allocs[alloc_key] = graph_record.mem_request[alloc_key]
                elif chunk.key in graph_record.targets:
                    try:
                        target_allocs[chunk.key] = graph_record.mem_request[chunk.key]
                    except KeyError:
                        raise

            logger.debug('Start calculation for graph %s in actor %s', graph_key, calc_uid)

            self._update_state(session_id, graph_key, ExecutionState.CALCULATING)
            raw_calc_ref = self.ctx.actor_ref(calc_uid)
            calc_ref = self.promise_ref(raw_calc_ref)

            def _start_calc(*_):
                if self._daemon_ref is None or self._daemon_ref.is_actor_process_alive(raw_calc_ref):
                    return calc_ref.calc(session_id, graph_record.graph_serialized,
                                         graph_record.targets, _promise=True)
                else:
                    raise WorkerProcessStopped

            self.estimate_graph_finish_time(session_id, graph_key, calc_fetch=False)
        except:
            self._dispatch_ref.register_free_slot(calc_uid, 'cpu')
            raise

        # make sure that memory suffices before actually run execution
        return self._mem_quota_ref.request_batch_quota(target_allocs, _promise=True) \
            .then(_start_calc)

    @log_unhandled
    def _dump_cache(self, session_id, graph_key, inproc_uid, save_sizes):
        """
        Dump calc results into shared cache or spill
        :param session_id: session id
        :param graph_key: key of the execution graph
        :param inproc_uid: uid of the InProcessCacheActor
        :param save_sizes: sizes of data
        """
        graph_record = self._graph_records[session_id, graph_key]
        calc_keys = graph_record.targets
        send_addresses = graph_record.send_addresses

        @log_unhandled
        def _do_active_transfer(*_):
            # transfer the result chunk to expected endpoints
            @log_unhandled
            def _send_chunk(sender_uid, chunk_key, target_addrs):
                if graph_record.stop_requested:
                    self._dispatch_ref.register_free_slot(sender_uid, 'sender')
                    raise ExecutionInterrupted

                sender_ref = self.promise_ref(sender_uid)
                logger.debug('Request for chunk %s sent to %s', chunk_key, target_addrs)
                return sender_ref.send_data(session_id, chunk_key, target_addrs, ensure_cached=False,
                                            timeout=options.worker.prepare_data_timeout, _promise=True)

            if graph_record.mem_request:
                self._mem_quota_ref.release_quotas(tuple(graph_record.mem_request.keys()), _tell=True)
                graph_record.mem_request = dict()

            promises = []
            for key, targets in send_addresses.items():
                promises.append(self._dispatch_ref.get_free_slot('sender', _promise=True)
                                .then(partial(_send_chunk, chunk_key=key, target_addrs=targets))
                                .catch(lambda *_: None))
            return promise.all_(promises)

        logger.debug('Graph %s: Start putting %r into shared cache. Target actor uid %s.',
                     graph_key, calc_keys, inproc_uid)
        self._update_state(session_id, graph_key, ExecutionState.STORING)

        raw_inproc_ref = self.ctx.actor_ref(inproc_uid)
        inproc_ref = self.promise_ref(raw_inproc_ref)

        if graph_record.stop_requested:
            logger.debug('Graph %s already marked for stop, quit.', graph_key)
            if (self._daemon_ref is None or self._daemon_ref.is_actor_process_alive(raw_inproc_ref)) \
                    and self.ctx.has_actor(raw_inproc_ref):
                logger.debug('Try remove keys for graph %s.', graph_key)
                raw_inproc_ref.remove_cache(list(calc_keys), _tell=True)
            logger.debug('Graph %s already marked for stop, quit.', graph_key)
            raise ExecutionInterrupted

        self._chunk_holder_ref.unpin_chunks(
            graph_key, list(set(c.key for c in graph_record.graph)), _tell=True)
        self._dump_execution_states()

        if self._daemon_ref is not None and not self._daemon_ref.is_actor_process_alive(raw_inproc_ref):
            raise WorkerProcessStopped

        def _cache_result(*_):
            self._result_cache[(session_id, graph_key)] = GraphResultRecord(save_sizes)

        if not send_addresses:
            # no endpoints to send, dump keys into shared memory and return
            logger.debug('Worker graph %s(%s) finished execution. Dumping %r into plasma...',
                         graph_key, graph_record.op_string, calc_keys)
            return inproc_ref.dump_cache(calc_keys, _promise=True) \
                .then(_cache_result)
        else:
            # dump keys into shared memory and send
            logger.debug('Worker graph %s(%s) finished execution. Dumping %r into plasma '
                         'while actively transferring %r...',
                         graph_key, graph_record.op_string, calc_keys, send_addresses)

            return inproc_ref.dump_cache(calc_keys, _promise=True) \
                .then(_do_active_transfer) \
                .then(_cache_result)

    def _cleanup_graph(self, session_id, graph_key):
        """
        Do clean up after graph is executed
        :param session_id: session id
        :param graph_key: graph key
        """
        logger.debug('Cleaning callbacks for graph %s', graph_key)
        self._task_queue_ref.release_task(session_id, graph_key, _tell=True)

        try:
            graph_record = self._graph_records[(session_id, graph_key)]
        except KeyError:
            return

        self._mem_quota_ref.cancel_requests(tuple(graph_record.mem_request.keys()), _tell=True)
        if graph_record.mem_request:
            self._mem_quota_ref.release_quotas(tuple(graph_record.mem_request.keys()), _tell=True)
        if graph_record.pin_request:
            self._chunk_holder_ref.unpin_chunks(graph_key, graph_record.pin_request, _tell=True)

        if self._status_ref:
            self._status_ref.remove_progress(session_id, graph_key, _tell=True, _wait=False)
        del self._graph_records[(session_id, graph_key)]

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
        try:
            args, kwargs = self._result_cache[(session_id, graph_key)].build_args()
            self.tell_promise(callback, *args, **kwargs)
        except KeyError:
            self._graph_records[(session_id, graph_key)].finish_callbacks.append(callback)

    @log_unhandled
    def stop_execution(self, session_id, graph_key):
        """
        Mark graph for stopping
        :param graph_key: graph key
        """
        logger.debug('Receive stop for graph %s', graph_key)
        try:
            graph_record = self._graph_records[(session_id, graph_key)]
        except KeyError:
            return

        graph_record.stop_requested = True
        if graph_record.state == ExecutionState.ALLOCATING:
            try:
                raise ExecutionInterrupted
            except:
                exc_info = sys.exc_info()
            if graph_record.mem_request:
                self._mem_quota_ref.cancel_requests(tuple(graph_record.mem_request.keys()), exc_info, _tell=True)
        elif graph_record.state == ExecutionState.CALCULATING:
            if self._daemon_ref is not None and graph_record.calc_actor_uid is not None:
                self._daemon_ref.kill_actor_process(self.ctx.actor_ref(graph_record.calc_actor_uid), _tell=True)

    @log_unhandled
    def _invoke_finish_callbacks(self, session_id, graph_key):
        """
        Call finish callback when execution is done
        :param session_id: session id
        :param graph_key: graph key
        """
        query_key = (session_id, graph_key)
        callbacks = self._graph_records[query_key].finish_callbacks
        args, kwargs = self._result_cache[query_key].build_args()
        logger.debug('Send finish callback for graph %s into %d targets', graph_key, len(callbacks))
        for cb in callbacks:
            self.tell_promise(cb, *args, **kwargs)
        self._cleanup_graph(session_id, graph_key)

    def _dump_execution_states(self):
        if logger.getEffectiveLevel() <= logging.DEBUG:
            cur_time = time.time()
            states = dict((k[1], (cur_time - v.state_time, v.state.name))
                          for k, v in self._graph_records.items()
                          if v.state not in (ExecutionState.PRE_PUSHED, ExecutionState.ALLOCATING))
            logger.debug('Executing states: %r', states)

    def handle_process_down(self, halt_refs):
        """
        Handle process down event
        :param halt_refs: actor refs in halt processes
        """
        logger.debug('Process halt detected. Trying to reject affected promises %r.',
                     [ref.uid for ref in halt_refs])
        try:
            raise WorkerProcessStopped
        except WorkerProcessStopped:
            exc_info = sys.exc_info()

        for ref in halt_refs:
            self.reject_promise_ref(ref, *exc_info)
