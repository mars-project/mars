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

import logging
import time
from collections import defaultdict

from .. import promise
from ..config import options
from ..executor import Executor
from ..utils import to_str, deserialize_graph, log_unhandled, calc_data_size, \
    get_chunk_shuffle_key
from ..context import DistributedDictContext
from .events import EventContext, EventCategory, EventLevel, ProcedureEventType
from .storage import DataStorageDevice
from .utils import WorkerActor, concat_operand_keys, get_chunk_key, build_quota_key

logger = logging.getLogger(__name__)


class BaseCalcActor(WorkerActor):
    _slot_name = None
    _calc_event_type = None
    _calc_source_devices = None
    _calc_intermediate_device = None
    _calc_dest_devices = None

    def __init__(self):
        super().__init__()
        self._remove_intermediate = self._calc_intermediate_device not in self._calc_dest_devices

        self._dispatch_ref = None
        self._mem_quota_ref = None
        self._events_ref = None
        self._status_ref = None
        self._resource_ref = None

        self._executing_set = set()
        self._marked_as_destroy = False

        self._execution_pool = None
        self._n_cpu = None

    def post_create(self):
        super().post_create()

        from .quota import MemQuotaActor
        from .dispatcher import DispatchActor
        from .events import EventsActor
        from .status import StatusActor
        from ..scheduler import ResourceActor

        self._mem_quota_ref = self.promise_ref(MemQuotaActor.default_uid())
        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())
        self._dispatch_ref.register_free_slot(self.uid, self._slot_name)

        status_ref = self.ctx.actor_ref(StatusActor.default_uid())
        self._status_ref = status_ref if self.ctx.has_actor(status_ref) else None

        self._resource_ref = self.get_actor_ref(ResourceActor.default_uid())

        self._events_ref = self.ctx.actor_ref(EventsActor.default_uid())
        if not self.ctx.has_actor(self._events_ref):
            self._events_ref = None

        self._execution_pool = self.ctx.threadpool(1)

    def pre_destroy(self):
        logger.debug('Destroying actor %s', self.uid)
        self._dispatch_ref.remove_slot(self.uid, self._slot_name, _tell=True, _wait=False)
        super().pre_destroy()

    @staticmethod
    def _get_keys_to_fetch(graph):
        from ..operands import Fetch, FetchShuffle
        fetch_keys = set()
        exclude_fetch_keys = set()
        for chunk in graph:
            if isinstance(chunk.op, Fetch):
                fetch_keys.add(chunk.op.to_fetch_key or chunk.key)
            elif isinstance(chunk.op, FetchShuffle):
                shuffle_key = get_chunk_shuffle_key(graph.successors(chunk)[0])
                for k in chunk.op.to_fetch_keys:
                    fetch_keys.add((k, shuffle_key))
            else:
                for inp, prepare_inp in zip(chunk.inputs, chunk.op.prepare_inputs):
                    if not prepare_inp:
                        exclude_fetch_keys.add(inp.key)
        return list(fetch_keys - exclude_fetch_keys)

    def _make_quotas_local(self, session_id, graph_key, data_keys, process_quota=False):
        old_keys, new_keys = [], []
        for k in data_keys:
            old_keys.append(build_quota_key(session_id, k, owner=graph_key))
            new_keys.append(build_quota_key(session_id, k, owner=self.proc_id))
        self._mem_quota_ref.alter_allocations(
            old_keys, new_keys=new_keys, process_quota=process_quota)
        return new_keys

    def _release_local_quotas(self, session_id, data_keys):
        self._mem_quota_ref.release_quotas(
            [build_quota_key(session_id, k, owner=self.proc_id) for k in data_keys],
            _tell=True, _wait=False)

    def _fetch_keys_to_process(self, session_id, keys_to_fetch):
        context_dict = dict()
        storage_client = self.storage_client

        if not keys_to_fetch:
            return promise.finished(context_dict)

        def _handle_loaded(objs):
            try:
                data_locations = storage_client.get_data_locations(session_id, keys_to_fetch)
                shared_quota_keys = []
                inproc_keys = []
                inproc_quota_keys = []

                context_dict.update(zip(keys_to_fetch, objs))
                for k, locations in zip(keys_to_fetch, data_locations):
                    quota_key = build_quota_key(session_id, k, owner=self.proc_id)
                    if (self.proc_id, DataStorageDevice.PROC_MEMORY) not in locations:
                        shared_quota_keys.append(quota_key)
                    else:
                        inproc_keys.append(k)
                        inproc_quota_keys.append(quota_key)

                if shared_quota_keys:
                    self._mem_quota_ref.hold_quotas(shared_quota_keys, _tell=True)
                if inproc_keys:
                    self._mem_quota_ref.hold_quotas(inproc_quota_keys, _tell=True)
                    if self._remove_intermediate:
                        storage_client.delete(session_id, inproc_keys, [self._calc_intermediate_device])
            finally:
                objs[:] = []

        def _handle_load_fail(*exc):
            if self._remove_intermediate:
                storage_client.delete(session_id, keys_to_fetch, [self._calc_intermediate_device])
            self._release_local_quotas(session_id, keys_to_fetch)

            context_dict.clear()
            raise exc[1].with_traceback(exc[2])

        return storage_client.get_objects(session_id, keys_to_fetch, self._calc_source_devices) \
            .then(_handle_loaded, _handle_load_fail).then(lambda *_: context_dict)

    def _get_n_cpu(self):
        if self._n_cpu is None:
            self._n_cpu = len(self._dispatch_ref.get_slots('cpu'))
        return self._n_cpu

    def _calc_results(self, session_id, graph_key, graph, context_dict, chunk_targets):
        _, op_name = concat_operand_keys(graph, '_')

        logger.debug('Start calculating operand %s in %s.', graph_key, self.uid)
        start_time = time.time()

        for chunk in graph:
            for inp, prepare_inp in zip(chunk.inputs, chunk.op.prepare_inputs):
                if not prepare_inp:
                    context_dict[inp.key] = None

        local_context_dict = DistributedDictContext(
            self.get_scheduler(self.default_uid()), session_id, actor_ctx=self.ctx,
            address=self.address, n_cpu=self._get_n_cpu())
        local_context_dict['_actor_cls'] = type(self)
        local_context_dict['_actor_uid'] = self.uid
        local_context_dict['_op_key'] = graph_key
        local_context_dict.update(context_dict)
        context_dict.clear()

        # start actual execution
        executor = Executor(storage=local_context_dict)
        with EventContext(self._events_ref, EventCategory.PROCEDURE, EventLevel.NORMAL,
                          self._calc_event_type, self.uid):
            self._execution_pool.submit(executor.execute_graph, graph,
                                        chunk_targets, retval=False).result()

        end_time = time.time()

        # collect results
        result_keys = []
        result_values = []
        result_sizes = []
        collected_chunk_keys = set()
        for k, v in local_context_dict.items():
            if isinstance(k, tuple):
                k = tuple(to_str(i) for i in k)
            else:
                k = to_str(k)

            chunk_key = get_chunk_key(k)
            if chunk_key in chunk_targets:
                result_keys.append(k)
                result_values.append(v)
                result_sizes.append(calc_data_size(v))
                collected_chunk_keys.add(chunk_key)

        local_context_dict.clear()

        # check if all targets generated
        if any(k not in collected_chunk_keys for k in chunk_targets):
            raise KeyError([k for k in chunk_targets if k not in collected_chunk_keys])

        # adjust sizes in allocation
        apply_allocs = defaultdict(lambda: 0)
        for k, size in zip(result_keys, result_sizes):
            apply_allocs[get_chunk_key(k)] += size

        apply_alloc_quota_keys, apply_alloc_sizes = [], []
        for k, v in apply_allocs.items():
            apply_alloc_quota_keys.append(build_quota_key(session_id, k, owner=self.proc_id))
            apply_alloc_sizes.append(v)
        self._mem_quota_ref.alter_allocations(apply_alloc_quota_keys, apply_alloc_sizes, _tell=True, _wait=False)
        self._mem_quota_ref.hold_quotas(apply_alloc_quota_keys, _tell=True)

        if self._status_ref:
            self._status_ref.update_mean_stats(
                'calc_speed.' + op_name, sum(apply_alloc_sizes) * 1.0 / (end_time - start_time),
                _tell=True, _wait=False)

        logger.debug('Finish calculating operand %s.', graph_key)

        return self.storage_client.put_objects(
            session_id, result_keys, result_values, [self._calc_intermediate_device], sizes=result_sizes) \
            .then(lambda *_: result_keys)

    @promise.reject_on_exception
    @log_unhandled
    def calc(self, session_id, graph_key, ser_graph, chunk_targets, callback):
        """
        Do actual calculation. This method should be called when all data
        is available (i.e., either in shared cache or in memory)
        :param session_id: session id
        :param graph_key: key of executable graph
        :param ser_graph: serialized executable graph
        :param chunk_targets: keys of target chunks
        :param callback: promise callback, returns the uid of InProcessCacheActor
        """
        self._executing_set.add(graph_key)
        graph = deserialize_graph(ser_graph)
        chunk_targets = set(chunk_targets)
        keys_to_fetch = self._get_keys_to_fetch(graph)

        self._make_quotas_local(session_id, graph_key, keys_to_fetch + list(chunk_targets),
                                process_quota=True)

        def _start_calc(context_dict):
            return self._calc_results(session_id, graph_key, graph, context_dict, chunk_targets)

        def _finalize(keys, exc_info):
            if not self._marked_as_destroy:
                self._dispatch_ref.register_free_slot(self.uid, self._slot_name, _tell=True, _wait=False)

            keys_to_delete = []
            keys_to_release = []

            for k in keys_to_fetch:
                if get_chunk_key(k) not in chunk_targets:
                    if self._remove_intermediate:
                        keys_to_delete.append(k)
                    keys_to_release.append(k)

            if not exc_info:
                self.tell_promise(callback, keys)
            else:
                for k in chunk_targets:
                    if self._remove_intermediate:
                        keys_to_delete.append(k)
                    keys_to_release.append(k)
                self.tell_promise(callback, *exc_info, _accept=False)

            if keys_to_delete:
                self.storage_client.delete(session_id, keys_to_delete, [self._calc_intermediate_device])
            self._release_local_quotas(session_id, keys_to_release)

        return self._fetch_keys_to_process(session_id, keys_to_fetch) \
            .then(lambda context_dict: _start_calc(context_dict)) \
            .then(lambda keys: _finalize(keys, None), lambda *exc_info: _finalize(None, exc_info))

    @promise.reject_on_exception
    @log_unhandled
    def store_results(self, session_id, graph_key, keys_to_store, data_attrs, callback):
        logger.debug('Start dumping keys for graph %s', graph_key)
        from ..scheduler.chunkmeta import WorkerMeta

        storage_client = self.storage_client
        store_keys, store_metas = [], []

        if data_attrs is None:
            data_attrs = storage_client.get_data_attrs(session_id, keys_to_store)

        for k, attr in zip(keys_to_store, data_attrs):
            if attr is None or isinstance(k, tuple):
                continue
            store_keys.append(k)
            store_metas.append(WorkerMeta(attr.size, attr.shape, (self.address,)))
        meta_future = self.get_meta_client().batch_set_chunk_meta(
            session_id, store_keys, store_metas, _wait=False)

        def _delete_keys(*_):
            logger.debug('Finish dumping keys for graph %s', graph_key)
            if self._remove_intermediate:
                storage_client.delete(
                    session_id, keys_to_store, [self._calc_intermediate_device], _tell=True)
            self._release_local_quotas(session_id, keys_to_store)

        def _finalize(*_):
            self._executing_set.remove(graph_key)
            if self._marked_as_destroy and not self._executing_set:
                # delay destroying to allow requests to enter
                self.ref().destroy_self(_tell=True, _delay=0.5)

        return storage_client.copy_to(session_id, keys_to_store, self._calc_dest_devices) \
            .then(_delete_keys) \
            .then(lambda *_: meta_future.result()) \
            .then(lambda *_: self.tell_promise(callback),
                  lambda *exc: self.tell_promise(callback, *exc, _accept=False)) \
            .then(_finalize)

    def destroy_self(self):
        if self._executing_set:
            return

        from .daemon import WorkerDaemonActor
        daemon_ref = self.ctx.actor_ref(WorkerDaemonActor.default_uid())
        if self.ctx.has_actor(daemon_ref):
            daemon_ref.destroy_actor(self.ref(), _tell=True)
        else:
            self.ctx.destroy_actor(self.ref())

    @log_unhandled
    def mark_destroy(self):
        self._marked_as_destroy = True
        self._dispatch_ref.remove_slot(self.uid, self._slot_name)
        if not self._executing_set:
            # delay destroying to allow requests to enter
            self.ref().destroy_self(_tell=True, _delay=0.5)


class CpuCalcActor(BaseCalcActor):
    _slot_name = 'cpu'
    _calc_event_type = ProcedureEventType.CPU_CALC
    if options.vineyard.socket:
        shared_memory_device = DataStorageDevice.VINEYARD  # pragma: no cover
    else:
        shared_memory_device = DataStorageDevice.SHARED_MEMORY

    # PROC_MEMORY must come first to avoid loading to shared storage
    # which will easily cause errors
    _calc_source_devices = (DataStorageDevice.PROC_MEMORY, shared_memory_device)
    _calc_intermediate_device = DataStorageDevice.PROC_MEMORY
    _calc_dest_devices = (shared_memory_device, DataStorageDevice.DISK)


class CudaCalcActor(BaseCalcActor):
    _slot_name = 'cuda'
    _calc_event_type = ProcedureEventType.GPU_CALC
    _calc_source_devices = (DataStorageDevice.CUDA, )
    _calc_intermediate_device = DataStorageDevice.CUDA
    _calc_dest_devices = (DataStorageDevice.CUDA, )
