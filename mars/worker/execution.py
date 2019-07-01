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
from collections import defaultdict

from .. import promise
from ..compat import reduce, six, Enum, BrokenPipeError, \
    ConnectionRefusedError, TimeoutError  # pylint: disable=W0622
from ..config import options
from ..errors import PinChunkFailed, WorkerProcessStopped, WorkerDead, \
    ExecutionInterrupted, DependencyMissing
from ..executor import Executor
from ..operands import Fetch, FetchShuffle
from ..utils import deserialize_graph, log_unhandled, build_exc_info, calc_data_size, BlacklistSet
from .chunkholder import ensure_chunk
from .spill import spill_exists, get_spill_data_size
from .utils import WorkerActor, ExpiringCache, concat_operand_keys, build_load_key

logger = logging.getLogger(__name__)


class ExecutionState(Enum):
    ALLOCATING = 'allocating'
    PREPARING_INPUTS = 'preparing_inputs'
    CALCULATING = 'calculating'
    STORING = 'storing'


class GraphExecutionRecord(object):
    """
    Execution records of the graph
    """
    __slots__ = ('graph', 'graph_serialized', '_state', 'op_string', 'data_targets',
                 'chunk_targets', 'io_meta', 'data_sizes', 'shared_input_chunks',
                 'state_time', 'mem_request', 'pin_request', 'est_finish_time',
                 'calc_actor_uid', 'send_addresses', 'retry_delay', 'finish_callbacks',
                 'stop_requested')

    def __init__(self, graph_serialized, state, chunk_targets=None, data_targets=None,
                 io_meta=None, data_sizes=None, mem_request=None,
                 shared_input_chunks=None, pin_request=None, est_finish_time=None,
                 calc_actor_uid=None, send_addresses=None, retry_delay=None,
                 finish_callbacks=None, stop_requested=False):

        self.graph_serialized = graph_serialized
        graph = self.graph = deserialize_graph(graph_serialized)

        self._state = state
        self.state_time = time.time()
        self.data_targets = data_targets or []
        self.chunk_targets = chunk_targets or []
        self.io_meta = io_meta or dict()
        self.data_sizes = data_sizes or dict()
        self.shared_input_chunks = shared_input_chunks or set()
        self.mem_request = mem_request or dict()
        self.pin_request = pin_request or set()
        self.est_finish_time = est_finish_time or time.time()
        self.calc_actor_uid = calc_actor_uid
        self.send_addresses = send_addresses
        self.retry_delay = retry_delay or 0
        self.finish_callbacks = finish_callbacks or []
        self.stop_requested = stop_requested or False

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
    __slots__ = 'data_sizes', 'exc', 'succeeded'

    def __init__(self, *args, **kwargs):
        succeeded = self.succeeded = kwargs.pop('succeeded', True)
        if succeeded:
            self.data_sizes = args[0]
        else:
            self.exc = args

    def build_args(self):
        if self.succeeded:
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
        self._mem_quota_ref = None
        self._status_ref = None
        self._daemon_ref = None

        self._resource_ref = None

        self._graph_records = dict()  # type: dict[tuple, GraphExecutionRecord]
        self._result_cache = ExpiringCache()  # type: dict[tuple, GraphResultRecord]

        self._peer_blacklist = BlacklistSet(options.worker.peer_blacklist_time)

    def post_create(self):
        from .chunkholder import ChunkHolderActor
        from .daemon import WorkerDaemonActor
        from .dispatcher import DispatchActor
        from .quota import MemQuotaActor
        from .status import StatusActor

        super(ExecutionActor, self).post_create()
        self.set_cluster_info_ref()

        self._chunk_holder_ref = self.promise_ref(ChunkHolderActor.default_uid())
        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())
        self._mem_quota_ref = self.promise_ref(MemQuotaActor.default_uid())

        self._daemon_ref = self.ctx.actor_ref(WorkerDaemonActor.default_uid())
        if not self.ctx.has_actor(self._daemon_ref):
            self._daemon_ref = None
        else:
            self.register_actors_down_handler()

        self._status_ref = self.ctx.actor_ref(StatusActor.default_uid())
        if not self.ctx.has_actor(self._status_ref):
            self._status_ref = None

        from ..scheduler import ResourceActor
        self._resource_ref = self.get_actor_ref(ResourceActor.default_uid())
        if not self.ctx.has_actor(self._resource_ref):
            self._resource_ref = None

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

    def _estimate_calc_memory(self, session_id, graph_key):
        graph_record = self._graph_records[(session_id, graph_key)]
        size_ctx = dict((k, (v, v)) for k, v in graph_record.data_sizes.items())
        executor = Executor(storage=size_ctx, sync_provider_type=Executor.SyncProviderType.MOCK)
        res = executor.execute_graph(graph_record.graph, graph_record.chunk_targets, mock=True)
        targets = graph_record.chunk_targets
        target_sizes = dict(zip(targets, res))

        total_mem = sum(target_sizes[key][1] for key in targets)
        if total_mem:
            for key in targets:
                r = target_sizes[key]
                target_sizes[key] = (r[0], max(r[1], r[1] * executor.mock_max_memory // total_mem))
        return target_sizes

    def _prepare_quota_request(self, session_id, graph_key):
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
        input_data_sizes = graph_record.data_sizes
        alloc_mem_batch = dict()
        alloc_cache_batch = dict()
        input_chunk_keys = dict()

        if self._status_ref:
            self.estimate_graph_finish_time(session_id, graph_key)

        memory_estimations = self._estimate_calc_memory(session_id, graph_key)

        # collect potential allocation sizes
        for chunk in graph:
            op = chunk.op
            if isinstance(op, Fetch):
                # use actual size as potential allocation size
                input_chunk_keys[chunk.key] = input_data_sizes.get(chunk.key, calc_data_size(chunk))
            elif isinstance(op, FetchShuffle):
                shuffle_key = graph.successors(chunk)[0].op.shuffle_key
                for k in op.to_fetch_keys:
                    part_key = (k, shuffle_key)
                    try:
                        input_chunk_keys[part_key] = input_data_sizes[part_key]
                    except KeyError:
                        pass
            elif chunk.key in graph_record.chunk_targets:
                # use estimated size as potential allocation size
                cache_batch, alloc_mem_batch[chunk.key] = memory_estimations[chunk.key]
                if not isinstance(chunk.key, tuple):
                    alloc_cache_batch[chunk.key] = cache_batch

        keys_to_pin = list(input_chunk_keys.keys())
        graph_record.pin_request = set(self._chunk_holder_ref.pin_chunks(graph_key, keys_to_pin))

        load_chunk_sizes = dict((k, v) for k, v in input_chunk_keys.items()
                                if k not in graph_record.pin_request)
        alloc_mem_batch.update((build_load_key(graph_key, k), v) for k, v in load_chunk_sizes.items()
                               if k not in graph_record.shared_input_chunks)
        if alloc_cache_batch:
            self._chunk_holder_ref.spill_size(sum(alloc_cache_batch.values()), _tell=True)

        if alloc_mem_batch:
            graph_record.mem_request = alloc_mem_batch
        return alloc_mem_batch

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

        if remote_addr in self._peer_blacklist:
            raise DependencyMissing

        remote_disp_ref = self.promise_ref(uid=DispatchActor.default_uid(),
                                           address=remote_addr)
        ensure_cached = kwargs.pop('ensure_cached', True)
        timeout = options.worker.prepare_data_timeout

        @log_unhandled
        def _finish_fetch(*_):
            self._chunk_holder_ref.pin_chunks(graph_key, chunk_key)
            if self._chunk_holder_ref.is_stored(chunk_key):
                self._mem_quota_ref.release_quota(build_load_key(graph_key, chunk_key))

        @log_unhandled
        def _handle_network_error(*exc):
            try:
                logger.warning('Communicating to %s encountered %s when fetching %s',
                               remote_addr, exc[0].__name__, chunk_key)

                if not self._chunk_holder_ref.is_stored(chunk_key):
                    logger.debug('Deleting chunk %s from worker because of failure of transfer', chunk_key)
                    self._chunk_store.delete(session_id, chunk_key)
                else:
                    # as data already transferred, we can skip the error
                    logger.debug('Chunk %s already transferred', chunk_key)
                    _finish_fetch()
                    return

                six.reraise(*exc)
            except (BrokenPipeError, ConnectionRefusedError, TimeoutError,
                    WorkerDead, promise.PromiseTimeout):
                if self._resource_ref:
                    self._resource_ref.detach_dead_workers([remote_addr], _tell=True)
                raise DependencyMissing((session_id, chunk_key))

        @log_unhandled
        def _fetch_step(sender_uid):
            if remote_addr in self._peer_blacklist:
                raise DependencyMissing

            if self._graph_records[(session_id, graph_key)].stop_requested:
                remote_disp_ref.register_free_slot(sender_uid, 'sender', _tell=True)
                raise ExecutionInterrupted

            sender_ref = self.promise_ref(sender_uid, address=remote_addr)
            logger.debug('Fetching chunk %s in %s to %s with slot %s',
                         chunk_key, remote_addr, self.address, sender_uid)
            try:
                return sender_ref.send_data(
                    session_id, chunk_key, self.address, ensure_cached=ensure_cached,
                    timeout=timeout, _timeout=timeout, _promise=True
                ).then(_finish_fetch)
            except:  # noqa: E722
                _handle_network_error(*sys.exc_info())

        return promise.finished() \
            .then(lambda *_: remote_disp_ref.get_free_slot('sender', _promise=True, _timeout=timeout)) \
            .then(_fetch_step) \
            .catch(_handle_network_error)

    def estimate_graph_finish_time(self, session_id, graph_key, calc_fetch=True, base_time=None):
        """
        Calc predictions for given chunk graph
        """
        session_graph_key = (session_id, graph_key)
        if session_graph_key not in self._graph_records:
            return
        graph_record = self._graph_records[session_graph_key]
        graph = graph_record.graph

        ops = set(type(c.op).__name__ for c in graph if not isinstance(c.op, Fetch))
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
                if not isinstance(c.op, Fetch):
                    break
                data_size = calc_data_size(c)
                input_size += data_size
                if self._chunk_holder_ref.is_stored(c.key):
                    continue
                if spill_exists(c.key):
                    disk_size += data_size
                else:
                    net_size += data_size

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
    def execute_graph(self, session_id, graph_key, graph_ser, io_meta, data_sizes,
                      send_addresses=None, callback=None):
        """
        Submit graph to the worker and control the execution
        :param session_id: session id
        :param graph_key: graph key
        :param graph_ser: serialized executable graph
        :param io_meta: io meta of the chunk
        :param data_sizes: data size of each input chunk, as a dict
        :param send_addresses: targets to send results after execution
        :param callback: promise callback
        """
        session_graph_key = (session_id, graph_key)
        try:
            old_callbacks = self._graph_records[session_graph_key].finish_callbacks or []
        except KeyError:
            old_callbacks = []

        graph_record = self._graph_records[(session_id, graph_key)] = GraphExecutionRecord(
            graph_ser, ExecutionState.ALLOCATING,
            io_meta=io_meta,
            data_sizes=data_sizes,
            chunk_targets=io_meta['chunks'],
            data_targets=list(io_meta.get('data_targets') or io_meta['chunks']),
            shared_input_chunks=set(io_meta.get('shared_input_chunks', [])),
            send_addresses=send_addresses,
            finish_callbacks=old_callbacks,
        )

        logger.debug('Worker graph %s(%s) targeting at %r accepted.', graph_key,
                     graph_record.op_string, graph_record.chunk_targets)
        self._update_state(session_id, graph_key, ExecutionState.ALLOCATING)

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
            self._invoke_finish_callbacks(session_id, graph_key)

        @log_unhandled
        def _handle_rejection(*exc):
            # some error occurred...
            logger.debug('Entering _handle_rejection() for graph %s', graph_key)
            self._dump_execution_states()

            if graph_record.stop_requested:
                graph_record.stop_requested = False
                if not isinstance(exc[1], ExecutionInterrupted):
                    exc = build_exc_info(ExecutionInterrupted)

            if isinstance(exc[1], ExecutionInterrupted):
                logger.warning('Execution of graph %s interrupted.', graph_key)
            else:
                logger.exception('Unexpected error occurred in executing graph %s', graph_key, exc_info=exc)

            self._result_cache[(session_id, graph_key)] = GraphResultRecord(*exc, **dict(succeeded=False))
            self._invoke_finish_callbacks(session_id, graph_key)

        # collect target data already computed
        save_sizes = dict()
        for target_key in graph_record.data_targets:
            if self._chunk_store.contains(session_id, target_key):
                save_sizes[target_key] = self._chunk_store.get_actual_size(session_id, target_key)
            elif spill_exists(target_key):
                save_sizes[target_key] = get_spill_data_size(target_key)

        # when all target data are computed, report success directly
        if all(k in save_sizes for k in graph_record.data_targets):
            logger.debug('All predecessors of graph %s already computed, call finish directly.', graph_key)
            self._result_cache[(session_id, graph_key)] = GraphResultRecord(save_sizes)
            _handle_success()
        else:
            try:
                quota_request = self._prepare_quota_request(session_id, graph_key)
            except PinChunkFailed:
                logger.debug('Failed to pin chunk for graph %s', graph_key)

                # cannot pin input chunks: retry later
                retry_delay = graph_record.retry_delay + 0.5 + random.random()
                graph_record.retry_delay = min(1 + graph_record.retry_delay, 30)
                self.ref().execute_graph(
                    session_id, graph_key, graph_record.graph_serialized, graph_record.io_meta,
                    graph_record.data_sizes, _tell=True, _delay=retry_delay)
                return

            promise.finished() \
                .then(lambda *_: self._prepare_graph_inputs(session_id, graph_key)) \
                .then(lambda *_: self._mem_quota_ref.request_batch_quota(quota_request, _promise=True)) \
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
        graph = graph_record.graph
        input_metas = graph_record.io_meta.get('input_data_metas', dict())

        if graph_record.stop_requested:
            raise ExecutionInterrupted

        unspill_keys = []
        transfer_keys = []

        logger.debug('Start preparing input data for graph %s', graph_key)
        self._update_state(session_id, graph_key, ExecutionState.PREPARING_INPUTS)
        prepare_promises = []
        shared_input_chunks = graph_record.shared_input_chunks

        input_keys = set()
        shuffle_keys = set()
        for chunk in graph:
            op = chunk.op
            if isinstance(op, Fetch):
                input_keys.add(op.to_fetch_key or chunk.key)
            elif isinstance(op, FetchShuffle):
                shuffle_key = graph.successors(chunk)[0].op.shuffle_key
                for k in op.to_fetch_keys:
                    part_key = (k, shuffle_key)
                    input_keys.add(part_key)
                    shuffle_keys.add(part_key)

        for input_key in input_keys:
            if self._chunk_holder_ref.is_stored(input_key):
                # data already in plasma: we just pin it
                pinned_keys = self._chunk_holder_ref.pin_chunks(graph_key, (input_key,))
                if input_key in pinned_keys:
                    self._mem_quota_ref.release_quota(build_load_key(graph_key, input_key))
                    continue

            if spill_exists(input_key):
                if input_key not in shared_input_chunks or input_key in shuffle_keys:
                    # input only use in current operand, we only need to load it into process memory
                    continue
                self._mem_quota_ref.release_quota(build_load_key(graph_key, input_key))
                load_fun = functools.partial(lambda gk, ck, *_: self._chunk_holder_ref.pin_chunks(gk, ck),
                                             graph_key, input_key)
                unspill_keys.append(input_key)
                prepare_promises.append(ensure_chunk(self, session_id, input_key, move_to_end=True) \
                                        .then(load_fun))
                continue

            # load data from another worker
            try:
                chunk_meta = input_metas[input_key]
                chunk_meta.workers = tuple(w for w in chunk_meta.workers if w != self.address)
            except KeyError:
                chunk_meta = self.get_meta_client().get_chunk_meta(session_id, input_key)

            if chunk_meta is None or not chunk_meta.workers:
                raise DependencyMissing('Dependency %r not met on sending.' % (input_key,))
            worker_results = chunk_meta.workers

            worker_priorities = []
            for worker_ip in worker_results:
                # todo sort workers by speed of network and other possible factors
                worker_priorities.append((worker_ip, (0, )))

            transfer_keys.append(input_key)

            # fetch data from other workers, if one fails, try another
            sorted_workers = sorted(worker_priorities, key=lambda pr: pr[1])
            p = promise.finished(_accept=False)
            for wp in sorted_workers:
                p = p.catch(functools.partial(self._fetch_remote_data, session_id, graph_key, input_key, wp[0],
                                              ensure_cached=input_key in shared_input_chunks))
            prepare_promises.append(p)

        logger.debug('Graph key %s: Targets %r, unspill keys %r, transfer keys %r',
                     graph_key, graph_record.chunk_targets, unspill_keys, transfer_keys)
        p = promise.all_(prepare_promises) \
            .then(lambda *_: logger.debug('Data preparation for graph %s finished', graph_key))
        return p

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
                if isinstance(chunk.op, Fetch):
                    if not self._chunk_holder_ref.is_stored(chunk.key):
                        alloc_key = build_load_key(graph_key, chunk.key)
                        if alloc_key in graph_record.mem_request:
                            target_allocs[alloc_key] = graph_record.mem_request[alloc_key]
                elif chunk.key in graph_record.chunk_targets:
                    target_allocs[chunk.key] = graph_record.mem_request[chunk.key]

            logger.debug('Start calculation for graph %s in actor %s', graph_key, calc_uid)

            self._update_state(session_id, graph_key, ExecutionState.CALCULATING)
            raw_calc_ref = self.ctx.actor_ref(calc_uid)
            calc_ref = self.promise_ref(raw_calc_ref)

            def _start_calc(*_):
                if self._daemon_ref is None or self._daemon_ref.is_actor_process_alive(raw_calc_ref):
                    return calc_ref.calc(session_id, graph_record.graph_serialized,
                                         graph_record.chunk_targets, _promise=True)
                else:
                    raise WorkerProcessStopped

            self.estimate_graph_finish_time(session_id, graph_key, calc_fetch=False)
        except:  # noqa: E722
            self._dispatch_ref.register_free_slot(calc_uid, 'cpu', _tell=True)
            raise

        # make sure that memory suffices before actually run execution
        return self._mem_quota_ref.request_batch_quota(target_allocs, _promise=True) \
            .then(lambda *_: self._deallocate_scheduler_resource(session_id, graph_key, delay=2)) \
            .then(_start_calc)

    @log_unhandled
    def _do_active_transfer(self, session_id, graph_key, data_to_addresses):
        if graph_key:
            logger.debug('Start active transfer for graph %s', graph_key)
            graph_record = self._graph_records[session_id, graph_key]
        else:
            graph_record = None

        # transfer the result chunk to expected endpoints
        @log_unhandled
        def _send_chunk(sender_uid, data_key, target_addrs):
            if graph_record and graph_record.stop_requested:
                self._dispatch_ref.register_free_slot(sender_uid, 'sender', _tell=True)
                raise ExecutionInterrupted

            sender_ref = self.promise_ref(sender_uid)
            timeout = options.worker.prepare_data_timeout
            logger.debug('Actively sending chunk %s to %s with slot %s',
                         data_key, target_addrs, sender_uid)
            return sender_ref.send_data(session_id, data_key, target_addrs, ensure_cached=False,
                                        timeout=timeout, _timeout=timeout, _promise=True)

        if graph_record:
            if graph_record.mem_request:
                self._mem_quota_ref.release_quotas(tuple(graph_record.mem_request.keys()), _tell=True)
                graph_record.mem_request = dict()

        promises = []
        for key, targets in data_to_addresses.items():
            promises.append(promise.finished()
                            .then(lambda: self._dispatch_ref.get_free_slot('sender', _promise=True))
                            .then(functools.partial(_send_chunk, data_key=key, target_addrs=targets))
                            .catch(lambda *_: None))
        return promise.all_(promises)

    @log_unhandled
    def _dump_cache(self, session_id, graph_key, inproc_uid, save_sizes):
        """
        Dump calc results into shared cache or spill
        :param session_id: session id
        :param graph_key: key of the execution graph
        :param inproc_uid: uid of the InProcessCacheActor
        :param save_sizes: sizes of data
        """
        self._deallocate_scheduler_resource(session_id, graph_key)

        graph_record = self._graph_records[session_id, graph_key]
        chunk_keys = graph_record.chunk_targets
        calc_keys = list(save_sizes.keys())
        send_addresses = graph_record.send_addresses

        logger.debug('Graph %s: Start putting %r into shared cache. Target actor uid %s.',
                     graph_key, chunk_keys, inproc_uid)
        self._update_state(session_id, graph_key, ExecutionState.STORING)

        raw_inproc_ref = self.ctx.actor_ref(inproc_uid)
        inproc_ref = self.promise_ref(raw_inproc_ref)

        if graph_record.stop_requested:
            logger.debug('Graph %s already marked for stop, quit.', graph_key)
            if (self._daemon_ref is None or self._daemon_ref.is_actor_process_alive(raw_inproc_ref)) \
                    and self.ctx.has_actor(raw_inproc_ref):
                logger.debug('Try remove keys for graph %s.', graph_key)
                raw_inproc_ref.remove_cache(session_id, list(calc_keys), _tell=True)
            logger.debug('Graph %s already marked for stop, quit.', graph_key)
            raise ExecutionInterrupted

        self._chunk_holder_ref.unpin_chunks(
            graph_key, list(set(c.key for c in graph_record.graph)), _tell=True)
        self._dump_execution_states()

        if self._daemon_ref is not None and not self._daemon_ref.is_actor_process_alive(raw_inproc_ref):
            raise WorkerProcessStopped

        def _cache_result(result_sizes):
            save_sizes.update(result_sizes)
            self._result_cache[(session_id, graph_key)] = GraphResultRecord(save_sizes)

        if not send_addresses:
            # no endpoints to send, dump keys into shared memory and return
            logger.debug('Worker graph %s(%s) finished execution. Dumping results into plasma...',
                         graph_key, graph_record.op_string)
            return inproc_ref.dump_cache(session_id, calc_keys, _promise=True) \
                .then(_cache_result)
        else:
            # dump keys into shared memory and send
            all_addresses = [{v} if isinstance(v, six.string_types) else set(v)
                             for v in send_addresses.values()]
            all_addresses = list(reduce(lambda a, b: a | b, all_addresses, set()))
            logger.debug('Worker graph %s(%s) finished execution. Dumping results into plasma '
                         'while actively transferring into %r...',
                         graph_key, graph_record.op_string, all_addresses)

            data_to_addresses = dict((k, v) for k, v in send_addresses.items()
                                     if k in save_sizes)

            return inproc_ref.dump_cache(session_id, calc_keys, _promise=True) \
                .then(_cache_result) \
                .then(lambda *_: functools.partial(self._do_active_transfer,
                                                   session_id, graph_key, data_to_addresses))

    def _deallocate_scheduler_resource(self, session_id, graph_key, delay=0):
        try:
            self._resource_ref.deallocate_resource(
                session_id, graph_key, self.address, _delay=delay, _tell=True, _wait=False)
        except:  # noqa: E722
            pass

    def _cleanup_graph(self, session_id, graph_key):
        """
        Do clean up after graph is executed
        :param session_id: session id
        :param graph_key: graph key
        """
        logger.debug('Cleaning callbacks for graph %s', graph_key)
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

    @promise.reject_on_exception
    @log_unhandled
    def send_data_to_workers(self, session_id, data_to_addresses, callback=None):
        """
        Send data stored in worker to other addresses
        :param session_id: session id
        :param data_to_addresses: dict mapping data to a list of addresses
        :param callback: promise call
        :return:
        """
        self._do_active_transfer(session_id, None, data_to_addresses) \
            .then(lambda *_: self.tell_promise(callback) if callback else None) \
            .catch(lambda *exc: self.tell_promise(*exc, **dict(_accept=False)) if callback else None)

    @log_unhandled
    def stop_execution(self, session_id, graph_key):
        """
        Mark graph for stopping
        :param session_id: session id
        :param graph_key: graph key
        """
        logger.debug('Receive stop for graph %s', graph_key)
        try:
            graph_record = self._graph_records[(session_id, graph_key)]
        except KeyError:
            return

        graph_record.stop_requested = True
        if graph_record.state == ExecutionState.ALLOCATING:
            if graph_record.mem_request:
                self._mem_quota_ref.cancel_requests(
                    tuple(graph_record.mem_request.keys()), build_exc_info(ExecutionInterrupted), _tell=True)
        elif graph_record.state == ExecutionState.CALCULATING:
            if self._daemon_ref is not None and graph_record.calc_actor_uid is not None:
                self._daemon_ref.kill_actor_process(self.ctx.actor_ref(graph_record.calc_actor_uid), _tell=True)

    @log_unhandled
    def handle_worker_change(self, _adds, removes):
        """
        Handle worker dead event
        :param removes: list of dead workers
        """
        if not removes:
            return
        self._peer_blacklist.update(removes)

        handled_refs = self.reject_dead_endpoints(removes, *build_exc_info(DependencyMissing))
        logger.debug('Peer worker halt received. Affected promises %r rejected.',
                     [ref.uid for ref in handled_refs])

        for sender_slot in self._dispatch_ref.get_slots('sender'):
            self.ctx.actor_ref(sender_slot).reject_dead_endpoints(removes, _tell=True)
        for receiver_slot in self._dispatch_ref.get_slots('receiver'):
            self.ctx.actor_ref(receiver_slot).notify_dead_senders(removes, _tell=True)

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
                          if v.state != ExecutionState.ALLOCATING)
            logger.debug('Executing states: %r', states)
