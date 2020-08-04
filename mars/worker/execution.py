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

import functools
import logging
import random
import time
from collections import defaultdict

from .. import promise
from ..config import options
from ..errors import PinDataKeyFailed, WorkerProcessStopped, WorkerDead, \
    ExecutionInterrupted, DependencyMissing
from ..executor import Executor
from ..operands import Fetch, FetchShuffle
from ..utils import BlacklistSet, deserialize_graph, log_unhandled, build_exc_info, \
    calc_data_size, get_chunk_shuffle_key, calc_object_overhead
from .storage import DataStorageDevice
from .transfer import ReceiverManagerActor
from .utils import WorkerActor, ExpiringCache, ExecutionState, concat_operand_keys, \
    build_quota_key, change_quota_key_owner

logger = logging.getLogger(__name__)


class GraphExecutionRecord(object):
    """
    Execution records of the graph
    """
    __slots__ = ('graph', 'graph_serialized', '_state', 'op_string', 'data_targets',
                 'chunk_targets', 'io_meta', 'data_metas', 'shared_input_chunks',
                 'state_time', 'mem_request', 'pinned_keys', 'mem_overhead_keys',
                 'est_finish_time', 'calc_actor_uid', 'send_addresses', 'retry_delay',
                 'retry_pending', 'finish_callbacks', 'stop_requested', 'calc_device',
                 'preferred_data_device', 'resource_released', 'no_prepare_chunk_keys')

    def __init__(self, graph_serialized, state, chunk_targets=None, data_targets=None,
                 io_meta=None, data_metas=None, mem_request=None, shared_input_chunks=None,
                 pinned_keys=None, mem_overhead_keys=None, est_finish_time=None,
                 calc_actor_uid=None, send_addresses=None, retry_delay=None,
                 finish_callbacks=None, stop_requested=False, calc_device=None,
                 preferred_data_device=None, resource_released=False,
                 no_prepare_chunk_keys=None):

        self.graph_serialized = graph_serialized
        graph = self.graph = deserialize_graph(graph_serialized)

        self._state = state
        self.state_time = time.time()
        self.data_targets = data_targets or []
        self.chunk_targets = chunk_targets or []
        self.io_meta = io_meta or dict()
        self.data_metas = data_metas or dict()
        self.shared_input_chunks = shared_input_chunks or set()
        self.mem_request = mem_request or dict()
        self.pinned_keys = set(pinned_keys or [])
        self.mem_overhead_keys = set(mem_overhead_keys or [])
        self.est_finish_time = est_finish_time or time.time()
        self.calc_actor_uid = calc_actor_uid
        self.send_addresses = send_addresses
        self.retry_delay = retry_delay or 0
        self.retry_pending = False
        self.finish_callbacks = finish_callbacks or []
        self.stop_requested = stop_requested or False
        self.calc_device = calc_device
        self.preferred_data_device = preferred_data_device
        self.resource_released = resource_released
        self.no_prepare_chunk_keys = no_prepare_chunk_keys or set()

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
    __slots__ = 'data_sizes', 'data_shapes', 'exc', 'succeeded'

    def __init__(self, *args, **kwargs):
        succeeded = self.succeeded = kwargs.pop('succeeded', True)
        if succeeded:
            self.data_sizes = args[0]
            self.data_shapes = args[1]
        else:
            self.exc = args

    def build_args(self):
        if self.succeeded:
            return (self.data_sizes, self.data_shapes), {}
        else:
            return self.exc, dict(_accept=False)


class ExecutionActor(WorkerActor):
    """
    Actor for execution control
    """
    _last_dump_time = time.time()

    def __init__(self):
        super().__init__()
        self._dispatch_ref = None
        self._mem_quota_ref = None
        self._status_ref = None
        self._daemon_ref = None
        self._receiver_manager_ref = None

        self._resource_ref = None

        self._graph_records = dict()  # type: dict[tuple, GraphExecutionRecord]
        self._result_cache = ExpiringCache()  # type: dict[tuple, GraphResultRecord]

        self._peer_blacklist = BlacklistSet(options.worker.peer_blacklist_time)

    def post_create(self):
        from .daemon import WorkerDaemonActor
        from .dispatcher import DispatchActor
        from .quota import MemQuotaActor
        from .status import StatusActor

        super().post_create()
        self.set_cluster_info_ref()

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

        self._receiver_manager_ref = self.ctx.actor_ref(ReceiverManagerActor.default_uid())
        if not self.ctx.has_actor(self._receiver_manager_ref):
            self._receiver_manager_ref = None
        else:
            self._receiver_manager_ref = self.promise_ref(self._receiver_manager_ref)

        from ..scheduler import ResourceActor
        self._resource_ref = self.get_actor_ref(ResourceActor.default_uid())

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

    def _pin_shared_data_keys(self, session_id, graph_key, data_keys):
        if not data_keys:
            return []
        try:
            graph_record = self._graph_records[(session_id, graph_key)]
        except KeyError:
            return []
        pinned_keys = self.storage_client.pin_data_keys(
            session_id, data_keys, graph_key, [graph_record.preferred_data_device])
        graph_record.pinned_keys.update(pinned_keys)
        return pinned_keys

    def _estimate_calc_memory(self, session_id, graph_key):
        graph_record = self._graph_records[(session_id, graph_key)]
        data_metas = graph_record.data_metas
        size_ctx = dict((k, (v.chunk_size, v.chunk_size)) for k, v in data_metas.items())

        # update shapes
        for n in graph_record.graph:
            if isinstance(n.op, Fetch):
                try:
                    meta = data_metas[n.key]
                    if hasattr(n, '_shape') and meta.chunk_shape is not None:
                        n._shape = meta.chunk_shape
                except KeyError:
                    pass
            elif isinstance(n.op, FetchShuffle):
                shuffle_key = get_chunk_shuffle_key(graph_record.graph.successors(n)[0])
                shapes = [data_metas.get((k, shuffle_key)).chunk_shape for k in n.op.to_fetch_keys]
                n.extra_params['_shapes'] = \
                    dict(((k, shuffle_key), v) for k, v in zip(n.op.to_fetch_keys, shapes))

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
        storage_client = self.storage_client
        input_data_sizes = dict((k, v.chunk_size) for k, v in graph_record.data_metas.items())
        alloc_mem_batch = dict()
        alloc_cache_batch = dict()
        input_chunk_keys = dict()

        if self._status_ref:
            self.estimate_graph_finish_time(session_id, graph_key)

        if graph_record.preferred_data_device == DataStorageDevice.SHARED_MEMORY or \
                graph_record.preferred_data_device == DataStorageDevice.VINEYARD:  # pragma: no cover
            memory_estimations = self._estimate_calc_memory(session_id, graph_key)
        else:
            memory_estimations = dict()

        graph_record.mem_overhead_keys = set()
        # collect potential allocation sizes
        for chunk in graph:
            op = chunk.op
            overhead_keys_and_shapes = []

            if isinstance(op, Fetch):
                if chunk.key in graph_record.no_prepare_chunk_keys:
                    continue
                # use actual size as potential allocation size
                input_chunk_keys[chunk.key] = input_data_sizes.get(chunk.key) or calc_data_size(chunk)
                overhead_keys_and_shapes = [(chunk.key, getattr(chunk, 'shape', None))]
            elif isinstance(op, FetchShuffle):
                shuffle_key = get_chunk_shuffle_key(graph.successors(chunk)[0])
                overhead_keys_and_shapes = chunk.extra_params.get('_shapes', dict()).items()
                for k in op.to_fetch_keys:
                    part_key = (k, shuffle_key)
                    try:
                        input_chunk_keys[part_key] = input_data_sizes[part_key]
                    except KeyError:
                        pass
            elif chunk.key in graph_record.chunk_targets:
                # use estimated size as potential allocation size
                estimation_data = memory_estimations.get(chunk.key)
                if not estimation_data:
                    continue
                quota_key = build_quota_key(session_id, chunk.key, owner=graph_key)
                cache_batch, alloc_mem_batch[quota_key] = estimation_data
                if not isinstance(chunk.key, tuple):
                    alloc_cache_batch[chunk.key] = cache_batch

            for key, shape in overhead_keys_and_shapes:
                overhead = calc_object_overhead(chunk, shape)
                if overhead:
                    graph_record.mem_overhead_keys.add(key)
                    quota_key = build_quota_key(session_id, key, owner=graph_key)
                    alloc_mem_batch[quota_key] = overhead

        keys_to_pin = list(input_chunk_keys.keys())
        graph_record.pinned_keys = set()
        self._pin_shared_data_keys(session_id, graph_key, keys_to_pin)

        for k, v in input_chunk_keys.items():
            quota_key = build_quota_key(session_id, k, owner=graph_key)
            if quota_key not in alloc_mem_batch:
                if k in graph_record.pinned_keys or k in graph_record.shared_input_chunks:
                    continue
            alloc_mem_batch[quota_key] = alloc_mem_batch.get(quota_key, 0) + v

        if alloc_cache_batch:
            storage_client.spill_size(sum(alloc_cache_batch.values()), [graph_record.preferred_data_device])

        graph_record.mem_request = alloc_mem_batch or dict()
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
            raise DependencyMissing((session_id, graph_key))

        remote_disp_ref = self.promise_ref(uid=DispatchActor.default_uid(),
                                           address=remote_addr)
        ensure_cached = kwargs.pop('ensure_cached', True)
        timeout = options.worker.prepare_data_timeout
        storage_client = self.storage_client
        graph_record = self._graph_records[(session_id, graph_key)]

        @log_unhandled
        def _finish_fetch(*_):
            locations = set(locs[1] for locs in storage_client.get_data_locations(session_id, [chunk_key])[0])
            if DataStorageDevice.PROC_MEMORY not in locations:
                self._pin_shared_data_keys(session_id, graph_key, [chunk_key])
                if chunk_key not in graph_record.mem_overhead_keys:
                    self._mem_quota_ref.release_quotas(
                        [build_quota_key(session_id, chunk_key, owner=graph_key)], _tell=True, _wait=False)
                if graph_record.preferred_data_device not in locations:
                    return storage_client.copy_to(session_id, [chunk_key], [graph_record.preferred_data_device])

        @log_unhandled
        def _handle_network_error(*exc):
            try:
                logger.warning('Communicating to %s encountered %s when fetching %s',
                               remote_addr, exc[0].__name__, chunk_key)

                if not storage_client.get_data_locations(session_id, [chunk_key])[0]:
                    logger.debug('Deleting chunk %s from worker because of failure of transfer', chunk_key)
                    storage_client.delete(session_id, [chunk_key])
                else:
                    # as data already transferred, we can skip the error
                    logger.debug('Chunk %s already transferred', chunk_key)
                    _finish_fetch()
                    return

                raise exc[1].with_traceback(exc[2])
            except (BrokenPipeError, ConnectionRefusedError, TimeoutError,
                    WorkerDead, promise.PromiseTimeout):
                self._resource_ref.detach_dead_workers([remote_addr], _tell=True, _wait=False)
                raise DependencyMissing((session_id, chunk_key))

        @log_unhandled
        def _fetch_step(sender_uid):
            if remote_addr in self._peer_blacklist:
                raise DependencyMissing((session_id, chunk_key))

            if graph_record.stop_requested:
                remote_disp_ref.register_free_slot(sender_uid, 'sender', _tell=True)
                raise ExecutionInterrupted

            sender_ref = self.promise_ref(sender_uid, address=remote_addr)
            logger.debug('Fetching chunk %s in %s to %s with slot %s',
                         chunk_key, remote_addr, self.address, sender_uid)

            return sender_ref.send_data(
                session_id, [chunk_key], [self.address], ensure_cached=ensure_cached,
                pin_token=graph_key, timeout=timeout, _timeout=timeout, _promise=True
            ).then(_finish_fetch)

        return promise.finished() \
            .then(lambda *_: remote_disp_ref.acquire_free_slot('sender', _promise=True, _timeout=timeout)) \
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
                try:
                    data_size = calc_data_size(c)
                except (AttributeError, TypeError, ValueError):
                    data_size = 0
                input_size += data_size
                data_locations = self.storage_client.get_data_locations(session_id, [c.key])[0]
                if (0, DataStorageDevice.VINEYARD) in data_locations or \
                        (0, DataStorageDevice.SHARED_MEMORY) in data_locations:  # pragma: no cover
                    continue
                elif (0, DataStorageDevice.DISK) in data_locations:
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
            self._status_ref.update_progress(
                session_id, key, record.op_string, state, _tell=True, _wait=False)

    @promise.reject_on_exception
    @log_unhandled
    def execute_graph(self, session_id, graph_key, graph_ser, io_meta, data_metas,
                      calc_device=None, send_addresses=None, callback=None):
        """
        Submit graph to the worker and control the execution
        :param session_id: session id
        :param graph_key: graph key
        :param graph_ser: serialized executable graph
        :param io_meta: io meta of the chunk
        :param data_metas: data meta of each input chunk, as a dict
        :param calc_device: device for calculation, can be 'gpu' or 'cpu'
        :param send_addresses: targets to send results after execution
        :param callback: promise callback
        """
        session_graph_key = (session_id, graph_key)
        callback = callback or []
        if not isinstance(callback, list):
            callback = [callback]
        try:
            all_callbacks = self._graph_records[session_graph_key].finish_callbacks or []
            self._graph_records[session_graph_key].finish_callbacks.extend(callback)
            if not self._graph_records[session_graph_key].retry_pending:
                self._graph_records[session_graph_key].finish_callbacks = all_callbacks + callback
                return
        except KeyError:
            all_callbacks = []
        all_callbacks.extend(callback)

        calc_device = calc_device or 'cpu'
        if calc_device == 'cpu':  # pragma: no cover
            if options.vineyard.socket:
                preferred_data_device = DataStorageDevice.VINEYARD
            else:
                preferred_data_device = DataStorageDevice.SHARED_MEMORY
        else:
            preferred_data_device = DataStorageDevice.CUDA

        # todo change this when handling multiple devices
        if preferred_data_device == DataStorageDevice.CUDA:
            slot = self._dispatch_ref.get_slots(calc_device)[0]
            proc_id = self.ctx.distributor.distribute(slot)
            preferred_data_device = (proc_id, preferred_data_device)

        graph_record = self._graph_records[(session_id, graph_key)] = GraphExecutionRecord(
            graph_ser, ExecutionState.ALLOCATING,
            io_meta=io_meta,
            data_metas=data_metas,
            chunk_targets=io_meta['chunks'],
            data_targets=list(io_meta.get('data_targets') or io_meta['chunks']),
            shared_input_chunks=set(io_meta.get('shared_input_chunks', [])),
            send_addresses=send_addresses,
            finish_callbacks=all_callbacks,
            calc_device=calc_device,
            preferred_data_device=preferred_data_device,
            no_prepare_chunk_keys=io_meta.get('no_prepare_chunk_keys') or set(),
        )

        logger.debug('Worker graph %s(%s) targeting at %r accepted.', graph_key,
                     graph_record.op_string, graph_record.chunk_targets)
        self._update_state(session_id, graph_key, ExecutionState.ALLOCATING)

        try:
            del self._result_cache[session_graph_key]
        except KeyError:
            pass

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

            self._result_cache[(session_id, graph_key)] = GraphResultRecord(*exc, succeeded=False)
            self._invoke_finish_callbacks(session_id, graph_key)

        # collect target data already computed
        attrs = self.storage_client.get_data_attrs(session_id, graph_record.data_targets)
        save_attrs = dict((k, v) for k, v in zip(graph_record.data_targets, attrs) if v)

        # when all target data are computed, report success directly
        if all(k in save_attrs for k in graph_record.data_targets):
            logger.debug('All predecessors of graph %s already computed, call finish directly.', graph_key)
            sizes = dict((k, v.size) for k, v in save_attrs.items())
            shapes = dict((k, v.shape) for k, v in save_attrs.items())
            self._result_cache[(session_id, graph_key)] = GraphResultRecord(sizes, shapes)
            _handle_success()
        else:
            try:
                quota_request = self._prepare_quota_request(session_id, graph_key)
            except PinDataKeyFailed:
                logger.debug('Failed to pin chunk for graph %s', graph_key)

                # cannot pin input chunks: retry later
                retry_delay = graph_record.retry_delay + 0.5 + random.random()
                graph_record.retry_delay = min(1 + graph_record.retry_delay, 30)
                graph_record.retry_pending = True

                self.ref().execute_graph(
                    session_id, graph_key, graph_record.graph_serialized, graph_record.io_meta,
                    graph_record.data_metas, calc_device=calc_device, send_addresses=send_addresses,
                    _tell=True, _delay=retry_delay)
                return

            promise.finished() \
                .then(lambda *_: self._mem_quota_ref.request_batch_quota(
                    quota_request, _promise=True) if quota_request else None) \
                .then(lambda *_: self._prepare_graph_inputs(session_id, graph_key)) \
                .then(lambda *_: self._dispatch_ref.acquire_free_slot(calc_device, _promise=True)) \
                .then(lambda uid: self._send_calc_request(session_id, graph_key, uid)) \
                .then(lambda saved_keys: self._store_results(session_id, graph_key, saved_keys)) \
                .then(_handle_success, _handle_rejection)

    def _release_shared_store_quotas(self, session_id, graph_key, keys):
        graph_record = self._graph_records[(session_id, graph_key)]
        quota_keys_to_release = [
            build_quota_key(session_id, k, owner=graph_key) for k in keys
            if k not in graph_record.mem_overhead_keys
        ]
        for quota_key in quota_keys_to_release:
            graph_record.mem_request.pop(quota_key, None)
        self._mem_quota_ref.release_quotas(quota_keys_to_release, _tell=True)

    @log_unhandled
    def _prepare_graph_inputs(self, session_id, graph_key):
        """
        Load input data from spilled storage and other workers
        :param session_id: session id
        :param graph_key: key of the execution graph
        """
        storage_client = self.storage_client
        graph_record = self._graph_records[(session_id, graph_key)]
        graph = graph_record.graph
        input_metas = graph_record.io_meta.get('input_data_metas', dict())

        if graph_record.stop_requested:
            raise ExecutionInterrupted

        logger.debug('Start preparing input data for graph %s', graph_key)
        self._update_state(session_id, graph_key, ExecutionState.PREPARING_INPUTS)
        prepare_promises = []

        input_keys = set()
        shuffle_keys = set()
        for chunk in graph:
            op = chunk.op
            if isinstance(op, Fetch):
                if chunk.key in graph_record.no_prepare_chunk_keys:
                    continue
                input_keys.add(op.to_fetch_key or chunk.key)
            elif isinstance(op, FetchShuffle):
                shuffle_key = get_chunk_shuffle_key(graph.successors(chunk)[0])
                for input_key in op.to_fetch_keys:
                    part_key = (input_key, shuffle_key)
                    input_keys.add(part_key)
                    shuffle_keys.add(part_key)

        local_keys = graph_record.pinned_keys & set(input_keys)
        non_local_keys = [k for k in input_keys if k not in local_keys]
        non_local_locations = storage_client.get_data_locations(session_id, non_local_keys)
        copy_keys = set(k for k, loc in zip(non_local_keys, non_local_locations) if loc)
        remote_keys = [k for k in non_local_keys if k not in copy_keys]

        # handle local keys
        self._release_shared_store_quotas(session_id, graph_key, local_keys)
        # handle move keys
        prepare_promises.extend(
            self._prepare_copy_keys(session_id, graph_key, copy_keys))
        # handle remote keys
        prepare_promises.extend(
            self._prepare_remote_keys(session_id, graph_key, remote_keys, input_metas))

        logger.debug('Graph key %s: Targets %r, loaded keys %r, copy keys %s, remote keys %r',
                     graph_key, graph_record.chunk_targets, local_keys, copy_keys, remote_keys)
        p = promise.all_(prepare_promises) \
            .then(lambda *_: logger.debug('Data preparation for graph %s finished', graph_key))
        return p

    @log_unhandled
    def _prepare_copy_keys(self, session_id, graph_key, copy_keys):
        promises = []
        graph_record = self._graph_records[(session_id, graph_key)]
        ensure_shared_keys = [k for k in copy_keys if k in graph_record.shared_input_chunks]
        better_shared_keys = [k for k in copy_keys if k not in graph_record.shared_input_chunks]

        def _release_copied_keys(keys):
            actual_moved_keys = self._pin_shared_data_keys(session_id, graph_key, keys)
            self._release_shared_store_quotas(session_id, graph_key, actual_moved_keys)

        if ensure_shared_keys:
            promises.append(
                self.storage_client.copy_to(
                    session_id, ensure_shared_keys, [graph_record.preferred_data_device],
                    ensure=True, pin_token=graph_key)
                .then(lambda *_: _release_copied_keys(ensure_shared_keys),
                      lambda *_: _release_copied_keys(ensure_shared_keys))
            )
        if better_shared_keys:
            promises.append(
                self.storage_client.copy_to(
                    session_id, better_shared_keys, [graph_record.preferred_data_device],
                    ensure=False, pin_token=graph_key)
                .then(lambda *_: _release_copied_keys(better_shared_keys),
                      lambda *_: _release_copied_keys(better_shared_keys))
            )
        return promises

    @log_unhandled
    def _prepare_remote_keys(self, session_id, graph_key, remote_keys, input_metas):
        promises = []
        graph_record = self._graph_records[(session_id, graph_key)]

        filtered_remote_keys = remote_keys
        if self._receiver_manager_ref:
            transferring_keys = set(self._receiver_manager_ref.filter_receiving_keys(
                session_id, remote_keys))
            if transferring_keys:
                logger.debug('Data %s already scheduled for fetch, just wait.', transferring_keys)
                filtered_remote_keys = [k for k in remote_keys if k not in transferring_keys]
                promises.append(self._receiver_manager_ref.add_keys_callback(
                    session_id, transferring_keys, _promise=True,
                    _timeout=options.worker.prepare_data_timeout))

        for input_key in filtered_remote_keys:
            # load data from another worker
            chunk_meta = input_metas.get(input_key) or graph_record.data_metas.get(input_key)
            if chunk_meta:
                chunk_meta.workers = tuple(w for w in chunk_meta.workers if w != self.address)

            if chunk_meta is None or not chunk_meta.workers:
                raise DependencyMissing('Dependency %r not met on sending.' % (input_key,))
            worker_results = list(chunk_meta.workers)
            random.shuffle(worker_results)

            # fetch data from other workers, if one fails, try another
            p = promise.finished(_accept=False)
            for worker in worker_results:
                p = p.catch(functools.partial(
                    self._fetch_remote_data, session_id, graph_key, input_key, worker,
                    ensure_cached=input_key in graph_record.shared_input_chunks))
            promises.append(p)
        return promises

    @log_unhandled
    def _send_calc_request(self, session_id, graph_key, calc_uid):
        """
        Start actual calculation in CpuCalcActor
        :param session_id: session id
        :param graph_key: key of the execution graph
        :param calc_uid: uid of the allocated CpuCalcActor
        """
        try:
            graph_record = self._graph_records[(session_id, graph_key)]

            if graph_record.stop_requested:
                raise ExecutionInterrupted

            graph_record.calc_actor_uid = calc_uid

            self._update_state(session_id, graph_key, ExecutionState.CALCULATING)
            calc_ref = self.promise_ref(calc_uid)

            def _start_calc(*_):
                logger.debug('Submit calculation for graph %s in actor %s', graph_key, calc_uid)
                if self._daemon_ref is None or self._daemon_ref.is_actor_process_alive(calc_ref):
                    return calc_ref.calc(
                        session_id, graph_key, graph_record.graph_serialized, graph_record.chunk_targets,
                        _promise=True
                    )
                else:
                    raise WorkerProcessStopped

            self.estimate_graph_finish_time(session_id, graph_key, calc_fetch=False)
        except:  # noqa: E722
            self._dispatch_ref.register_free_slot(calc_uid, 'cpu', _tell=True)
            raise

        # make sure that memory suffices before actually run execution
        mem_request = graph_record.mem_request
        if mem_request:
            logger.debug('Ensuring resource %r for graph %s', mem_request, graph_key)
            return self._mem_quota_ref.request_batch_quota(mem_request, process_quota=True, _promise=True) \
                .then(lambda *_: self._deallocate_scheduler_resource(session_id, graph_key, delay=2)) \
                .then(_start_calc)
        else:
            self._deallocate_scheduler_resource(session_id, graph_key, delay=2)
            return _start_calc()

    @log_unhandled
    def _do_active_transfer(self, session_id, graph_key, data_to_addresses):
        if graph_key:
            logger.debug('Start active transfer for graph %s', graph_key)
            graph_record = self._graph_records[session_id, graph_key]
        else:
            graph_record = None

        # transfer the result chunk to expected endpoints
        @log_unhandled
        def _send_chunk(sender_uid, data_keys, target_addr):
            if graph_record and graph_record.stop_requested:
                self._dispatch_ref.register_free_slot(sender_uid, 'sender', _tell=True)
                raise ExecutionInterrupted

            sender_ref = self.promise_ref(sender_uid)
            timeout = options.worker.prepare_data_timeout
            logger.debug('Actively sending chunks %r to %s with slot %s',
                         data_keys, target_addr, sender_uid)
            return sender_ref.send_data(session_id, data_keys, [target_addr], ensure_cached=False,
                                        timeout=timeout, _timeout=timeout, _promise=True)

        if graph_record:
            if graph_record.mem_request:
                self._mem_quota_ref.release_quotas(tuple(graph_record.mem_request.keys()), _tell=True)
                graph_record.mem_request = dict()

        promises = []
        address_to_data_keys = defaultdict(set)
        for key, targets in data_to_addresses.items():
            for target in targets:
                address_to_data_keys[target].add(key)
        for target, keys in address_to_data_keys.items():
            if target == self.address:
                continue
            promises.append(promise.finished()
                            .then(lambda: self._dispatch_ref.acquire_free_slot('sender', _promise=True))
                            .then(functools.partial(_send_chunk, data_keys=keys, target_addr=target))
                            .catch(lambda *_: None))

        for target, keys in address_to_data_keys.items():
            self.ctx.actor_ref(ReceiverManagerActor.default_uid(), address=target) \
                .register_pending_keys(session_id, keys, _tell=True, _wait=False)

        return promise.all_(promises)

    @log_unhandled
    def _store_results(self, session_id, graph_key, saved_keys):
        """
        Store calc results into shared cache or spill
        :param session_id: session id
        :param graph_key: key of the execution graph
        """
        storage_client = self.storage_client

        self._deallocate_scheduler_resource(session_id, graph_key)

        graph_record = self._graph_records[session_id, graph_key]
        chunk_keys = graph_record.chunk_targets
        send_addresses = graph_record.send_addresses

        logger.debug('Graph %s: Start putting %r into shared cache using actor uid %s.',
                     graph_key, chunk_keys, graph_record.calc_actor_uid)
        self._update_state(session_id, graph_key, ExecutionState.STORING)

        raw_calc_ref = self.ctx.actor_ref(graph_record.calc_actor_uid)
        calc_ref = self.promise_ref(raw_calc_ref)

        if graph_record.stop_requested:
            logger.debug('Graph %s already marked for stop, quit.', graph_key)
            if (self._daemon_ref is None or self._daemon_ref.is_actor_process_alive(raw_calc_ref)) \
                    and self.ctx.has_actor(raw_calc_ref):
                logger.debug('Try remove keys for graph %s.', graph_key)
                raw_calc_ref.remove_cache(session_id, list(saved_keys), _tell=True)
            logger.debug('Graph %s already marked for stop, quit.', graph_key)
            raise ExecutionInterrupted

        storage_client.unpin_data_keys(session_id, graph_record.pinned_keys, graph_key)
        self._dump_execution_states()

        data_attrs = storage_client.get_data_attrs(session_id, saved_keys)

        if self._daemon_ref is not None and not self._daemon_ref.is_actor_process_alive(raw_calc_ref):
            raise WorkerProcessStopped

        def _cache_result(*_):
            save_sizes = dict((k, v.size) for k, v in zip(saved_keys, data_attrs) if v)
            save_shapes = dict((k, v.shape) for k, v in zip(saved_keys, data_attrs) if v)
            self._result_cache[(session_id, graph_key)] = GraphResultRecord(save_sizes, save_shapes)

        if not send_addresses:
            # no endpoints to send, dump keys into shared memory and return
            logger.debug('Worker graph %s(%s) finished execution. Dumping results...',
                         graph_key, graph_record.op_string)
            return calc_ref.store_results(session_id, graph_key, saved_keys, data_attrs, _promise=True) \
                .then(_cache_result)
        else:
            # dump keys into shared memory and send
            all_addresses = [{v} if isinstance(v, str) else set(v)
                             for v in send_addresses.values()]
            all_addresses = list(functools.reduce(lambda a, b: a | b, all_addresses, set()))
            logger.debug('Worker graph %s(%s) finished execution. Dumping results '
                         'while actively transferring into %r...',
                         graph_key, graph_record.op_string, all_addresses)

            data_to_addresses = dict((k, v) for k, v in send_addresses.items()
                                     if k in saved_keys)

            return calc_ref.store_results(session_id, graph_key, saved_keys, data_attrs, _promise=True) \
                .then(_cache_result) \
                .then(lambda *_: functools.partial(self._do_active_transfer,
                                                   session_id, graph_key, data_to_addresses))

    def _deallocate_scheduler_resource(self, session_id, graph_key, delay=0):
        try:
            graph_record = self._graph_records[(session_id, graph_key)]
            if not graph_record.resource_released:
                graph_record.resource_released = True
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

        mem_quota_keys = tuple(graph_record.mem_request.keys())
        self._mem_quota_ref.cancel_requests(mem_quota_keys, _tell=True)
        if graph_record.mem_request:
            self._mem_quota_ref.release_quotas(mem_quota_keys, _tell=True)
            if graph_record.calc_actor_uid:
                target_proc_id = self.ctx.distributor.distribute(graph_record.calc_actor_uid)
                owned_quota_keys = tuple(change_quota_key_owner(k, target_proc_id)
                                         for k in mem_quota_keys)
                self._mem_quota_ref.release_quotas(owned_quota_keys, _tell=True)

        if graph_record.pinned_keys:
            self.storage_client.unpin_data_keys(session_id, graph_record.pinned_keys, graph_key)

        if self._status_ref:
            self._status_ref.remove_progress(session_id, graph_key, _tell=True, _wait=False)
        del self._graph_records[(session_id, graph_key)]
        self.ref().delete_result_cache(session_id, graph_key, _tell=True, _delay=20)

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
            .catch(lambda *exc: self.tell_promise(*exc, _accept=False) if callback else None)

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
    def delete_data_by_keys(self, session_id, keys):
        self.storage_client.delete(session_id, keys, _tell=True)

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
        kwargs['_wait'] = False
        for cb in callbacks:
            self.tell_promise(cb, *args, **kwargs)
        self._cleanup_graph(session_id, graph_key)

    def delete_result_cache(self, session_id, graph_key):
        del self._result_cache[(session_id, graph_key)]

    def _dump_execution_states(self):
        if logger.getEffectiveLevel() <= logging.DEBUG:
            cur_time = time.time()
            states = dict((k[1], (cur_time - v.state_time, v.state.name))
                          for k, v in self._graph_records.items()
                          if v.state != ExecutionState.ALLOCATING)
            logger.debug('Executing states: %r', states)
