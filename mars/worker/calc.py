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
import functools
from collections import defaultdict

from .. import promise
from ..compat import six
from ..executor import Executor
from ..utils import to_str, deserialize_graph, log_unhandled, calc_data_size, \
    get_chunk_shuffle_key
from .events import EventContext, EventCategory, EventLevel, ProcedureEventType
from .storage import DataStorageDevice
from .utils import WorkerActor, concat_operand_keys, get_chunk_key, build_quota_key

logger = logging.getLogger(__name__)


class CpuCalcActor(WorkerActor):
    def __init__(self):
        super(CpuCalcActor, self).__init__()
        self._mem_quota_ref = None
        self._dispatch_ref = None
        self._events_ref = None
        self._status_ref = None

        self._execution_pool = None

    def post_create(self):
        super(CpuCalcActor, self).post_create()

        from .quota import MemQuotaActor
        from .dispatcher import DispatchActor
        from .events import EventsActor
        from .status import StatusActor

        self._mem_quota_ref = self.promise_ref(MemQuotaActor.default_uid())
        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())
        self._dispatch_ref.register_free_slot(self.uid, 'cpu')

        status_ref = self.ctx.actor_ref(StatusActor.default_uid())
        self._status_ref = status_ref if self.ctx.has_actor(status_ref) else None

        self._events_ref = self.ctx.actor_ref(EventsActor.default_uid())
        if not self.ctx.has_actor(self._events_ref):
            self._events_ref = None

        self._execution_pool = self.ctx.threadpool(1)

    @staticmethod
    def _get_keys_to_fetch(graph):
        from ..operands import Fetch, FetchShuffle
        fetch_keys = set()
        for chunk in graph:
            if isinstance(chunk.op, Fetch):
                fetch_keys.add(chunk.op.to_fetch_key or chunk.key)
            elif isinstance(chunk.op, FetchShuffle):
                shuffle_key = get_chunk_shuffle_key(graph.successors(chunk)[0])
                for k in chunk.op.to_fetch_keys:
                    fetch_keys.add((k, shuffle_key))
        return list(fetch_keys)

    def _make_quotas_local(self, session_id, graph_key, data_keys, process_quota=False):
        old_keys, new_keys = [], []
        for k in data_keys:
            old_keys.append(build_quota_key(session_id, k, owner=graph_key))
            new_keys.append(build_quota_key(session_id, k, owner=self.proc_id))
        self._mem_quota_ref.alter_allocations(
            old_keys, new_keys=new_keys, process_quota=process_quota, _tell=True, _wait=False)
        return new_keys

    def _release_local_quota(self, session_id, data_key):
        self._mem_quota_ref.release_quota(
            build_quota_key(session_id, data_key, owner=self.proc_id), _tell=True, _wait=False)

    def _fetch_keys_to_process(self, session_id, keys_to_fetch):
        context_dict = dict()
        failed = [False]
        promises = []
        device_order = [DataStorageDevice.SHARED_MEMORY, DataStorageDevice.PROC_MEMORY]
        storage_client = self.storage_client

        def _handle_single_loaded(k, obj):
            locations = storage_client.get_data_locations(session_id, k)
            quota_key = build_quota_key(session_id, k, owner=self.proc_id)
            if (self.proc_id, DataStorageDevice.PROC_MEMORY) not in locations:
                self._mem_quota_ref.release_quota(quota_key, _tell=True, _wait=False)
            else:
                self._mem_quota_ref.hold_quota(quota_key, _tell=True)
                storage_client.delete(session_id, k, [DataStorageDevice.PROC_MEMORY])
            if not failed[0]:
                context_dict[k] = obj

        def _handle_single_load_fail(*exc, **kwargs):
            data_key = kwargs.pop('key')
            quota_key = build_quota_key(session_id, data_key, owner=self.proc_id)
            storage_client.delete(session_id, data_key, [DataStorageDevice.PROC_MEMORY])
            self._mem_quota_ref.release_quota(quota_key, _tell=True, _wait=False)

            failed[0] = True
            context_dict.clear()
            six.reraise(*exc)

        for key in keys_to_fetch:
            promises.append(storage_client.get_object(session_id, key, device_order)
                            .then(functools.partial(_handle_single_loaded, key),
                                  functools.partial(_handle_single_load_fail, key=key)))

        return promise.all_(promises).then(lambda *_: context_dict)

    def _calc_results(self, session_id, graph_key, graph, context_dict, chunk_targets):
        _, op_name = concat_operand_keys(graph, '_')

        logger.debug('Start calculating operand %s in %s.', graph_key, self.uid)
        start_time = time.time()

        local_context_dict = context_dict.copy()
        context_dict.clear()

        # start actual execution
        executor = Executor(storage=local_context_dict)
        with EventContext(self._events_ref, EventCategory.PROCEDURE, EventLevel.NORMAL,
                          ProcedureEventType.CALCULATION, self.uid):
            self._execution_pool.submit(executor.execute_graph, graph,
                                        chunk_targets, retval=False).result()

        end_time = time.time()

        # collect results
        result_pairs = []
        collected_chunk_keys = set()
        for k, v in local_context_dict.items():
            if isinstance(k, tuple):
                k = tuple(to_str(i) for i in k)
            else:
                k = to_str(k)

            chunk_key = get_chunk_key(k)
            if chunk_key in chunk_targets:
                result_pairs.append((k, v))
                collected_chunk_keys.add(chunk_key)

        local_context_dict.clear()

        # check if all targets generated
        if any(k not in collected_chunk_keys for k in chunk_targets):
            raise KeyError([k for k in chunk_targets if k not in collected_chunk_keys])

        # adjust sizes in allocation
        apply_alloc_sizes = defaultdict(lambda: 0)
        for k, v in result_pairs:
            apply_alloc_sizes[get_chunk_key(k)] += calc_data_size(v)

        for k, v in apply_alloc_sizes.items():
            quota_key = build_quota_key(session_id, k, owner=self.proc_id)
            self._mem_quota_ref.alter_allocation(quota_key, v, _tell=True)
            self._mem_quota_ref.hold_quota(quota_key, _tell=True)

        if self._status_ref:
            self._status_ref.update_mean_stats(
                'calc_speed.' + op_name, sum(apply_alloc_sizes.values()) * 1.0 / (end_time - start_time),
                _tell=True, _wait=False)

        logger.debug('Finish calculating operand %s.', graph_key)

        result_keys = [p[0] for p in result_pairs]

        return promise.all_([
            self.storage_client.put_object(session_id, k, v, [DataStorageDevice.PROC_MEMORY])
            for k, v in result_pairs
        ]).then(lambda *_: result_keys)

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
        graph = deserialize_graph(ser_graph)
        chunk_targets = set(chunk_targets)
        keys_to_fetch = self._get_keys_to_fetch(graph)

        self._make_quotas_local(session_id, graph_key, keys_to_fetch, process_quota=True)
        target_quotas = self._make_quotas_local(session_id, graph_key, chunk_targets)

        def _start_calc(context_dict):
            for k in target_quotas:
                self._mem_quota_ref.process_quota(k, _tell=True, _wait=False)
            return self._calc_results(session_id, graph_key, graph, context_dict, chunk_targets)

        def _finalize(keys, exc_info):
            self._dispatch_ref.register_free_slot(self.uid, 'cpu', _tell=True, _wait=False)

            for k in keys_to_fetch:
                if get_chunk_key(k) not in chunk_targets:
                    self.storage_client.delete(session_id, k, [DataStorageDevice.PROC_MEMORY])
                    self._release_local_quota(session_id, k)

            if not exc_info:
                self.tell_promise(callback, keys)
            else:
                for k in chunk_targets:
                    self.storage_client.delete(session_id, k, [DataStorageDevice.PROC_MEMORY])
                    self._release_local_quota(session_id, k)
                self.tell_promise(callback, *exc_info, **dict(_accept=False))

        return self._fetch_keys_to_process(session_id, keys_to_fetch) \
            .then(lambda context_dict: _start_calc(context_dict)) \
            .then(lambda keys: _finalize(keys, None), lambda *exc_info: _finalize(None, exc_info))

    @promise.reject_on_exception
    @log_unhandled
    def store_results(self, session_id, keys_to_store, callback):
        from ..scheduler.chunkmeta import WorkerMeta

        storage_client = self.storage_client

        sizes = storage_client.get_data_sizes(session_id, keys_to_store)
        shapes = storage_client.get_data_shapes(session_id, keys_to_store)

        store_keys, store_metas = [], []

        for k, size in sizes.items():
            if isinstance(k, tuple):
                continue
            store_keys.append(k)
            store_metas.append(WorkerMeta(size, shapes.get(k), (self.address,)))
        meta_future = self.get_meta_client().batch_set_chunk_meta(
            session_id, store_keys, store_metas, _wait=False)

        def _delete_key(k, *_):
            storage_client.delete(session_id, k, [DataStorageDevice.PROC_MEMORY], _tell=True)
            self._mem_quota_ref.release_quota(
                build_quota_key(session_id, k, owner=self.proc_id), _tell=True, _wait=False)

        copy_targets = [DataStorageDevice.SHARED_MEMORY, DataStorageDevice.DISK]
        promise.all_([
            storage_client.copy_to(session_id, k, copy_targets)
                .then(functools.partial(_delete_key, k))
            for k in keys_to_store]) \
            .then(lambda *_: meta_future.result()) \
            .then(lambda *_: self.tell_promise(callback),
                  lambda *exc: self.tell_promise(callback, *exc, **dict(_accept=False)))
