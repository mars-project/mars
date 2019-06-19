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
from functools import partial

from .. import promise
from ..compat import OrderedDict3, six, functools32
from ..config import options
from ..utils import parse_memory_limit, log_unhandled, readable_size
from ..errors import *
from .utils import WorkerActor

logger = logging.getLogger(__name__)


@log_unhandled
def ensure_chunk(promise_actor, session_id, chunk_key, move_to_end=False):
    from .dispatcher import DispatchActor

    chunk_holder_ref = promise_actor.promise_ref(ChunkHolderActor.default_uid())
    dispatch_ref = promise_actor.promise_ref(DispatchActor.default_uid())
    if chunk_holder_ref.is_stored(chunk_key):
        logger.debug('No need to load key %s from spill.', chunk_key)
        if move_to_end:
            chunk_holder_ref.move_to_end(chunk_key)
        return promise.Promise(done=True)

    logger.debug('Try starting loading data %s from spill.', chunk_key)

    @log_unhandled
    def _release_mem(*args, **kwargs):
        accepts = kwargs.pop('accepts', True)
        if accepts:
            logger.debug('Chunk %s loaded into plasma.', chunk_key)
        else:
            logger.error('Chunk %s failed to load in plasma.', chunk_key)
            six.reraise(*args)

    uid = dispatch_ref.get_hash_slot('spill', chunk_key)
    spill_ref = promise_actor.promise_ref(uid)

    logger.debug('Requesting data load for chunk %s in %s.', chunk_key, uid)
    return spill_ref.load(session_id, chunk_key, _timeout=options.worker.prepare_data_timeout, _promise=True) \
        .then(_release_mem, partial(_release_mem, accepts=False))


class ChunkHolderActor(WorkerActor):
    def __init__(self, plasma_limit=0):
        super(ChunkHolderActor, self).__init__()
        self._plasma_limit = plasma_limit
        self._cache_chunk_sessions = dict()

        self._chunk_holder = OrderedDict3()
        self._total_size = 0
        self._total_hold = 0
        self._pinned_counter = dict()
        self._spill_pending_keys = set()

        self._total_spill = 0
        self._min_spill_size = 0
        self._max_spill_size = 0

        self._dispatch_ref = None
        self._status_ref = None

    def post_create(self):
        from .dispatcher import DispatchActor
        from .status import StatusActor

        super(ChunkHolderActor, self).post_create()
        self.register_process_down_handler()
        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())

        self._plasma_limit = self._chunk_store.get_actual_capacity(self._plasma_limit)
        logger.info('Detected actual plasma store size: %s', readable_size(self._plasma_limit))
        self._total_size = self._plasma_limit
        parse_num, is_percent = parse_memory_limit(options.worker.min_spill_size)
        self._min_spill_size = int(self._plasma_limit * parse_num if is_percent else parse_num)
        parse_num, is_percent = parse_memory_limit(options.worker.max_spill_size)
        self._max_spill_size = int(self._plasma_limit * parse_num if is_percent else parse_num)

        self._status_ref = self.ctx.actor_ref(StatusActor.default_uid())
        if self.ctx.has_actor(self._status_ref):
            self._status_ref.set_cache_allocations(
                dict(hold=self._total_hold, total=self._total_size), _tell=True, _wait=False)
        else:
            self._status_ref = None

    def pre_destroy(self):
        for k in list(self._chunk_holder):
            del self._chunk_holder[k]

    @functools32.lru_cache(100)
    def _get_spill_actor_ref(self, chunk_key):
        if not options.worker.spill_directory:
            return None
        try:
            uid = self._dispatch_ref.get_hash_slot('spill', chunk_key)
            return self.promise_ref(uid)
        except KeyError:
            return None

    @promise.reject_on_exception
    @log_unhandled
    def spill_size(self, size, multiplier=1, callback=None):
        request_size = size

        if request_size < self._min_spill_size:
            request_size = self._min_spill_size
        request_size *= multiplier
        if request_size > self._plasma_limit:
            raise NoDataToSpill
        if request_size > self._max_spill_size:
            request_size = self._max_spill_size

        logger.debug('Start spilling %d(x%d) bytes from shared cache.', request_size, multiplier)

        if request_size + self._total_hold > self._plasma_limit:
            acc_free = 0
            free_keys = []
            for k in self._chunk_holder:
                if k in self._pinned_counter or k in self._spill_pending_keys:
                    continue
                acc_free += len(self._chunk_holder[k])
                free_keys.append(k)
                self._spill_pending_keys.add(k)
                if request_size + self._total_hold - acc_free <= self._plasma_limit:
                    break

            if not free_keys:
                logger.warning('Cannot spill further. Rejected. request=%d', request_size)
                raise NoDataToSpill

            logger.debug('Decide to spill %d chunks. request=%d', len(free_keys), request_size)

            @log_unhandled
            def _release_spill_allocations(key):
                if key in self._spill_pending_keys:
                    self._spill_pending_keys.remove(key)
                if key in self._pinned_counter:
                    return
                if key not in self._chunk_holder:
                    return

                logger.debug('Removing reference of chunk %s from %s when spilling', key, self.uid)
                if key in self._chunk_holder:
                    self._total_hold -= len(self._chunk_holder[key])
                    del self._chunk_holder[key]
                    session_id = self._cache_chunk_sessions[key]
                    self._chunk_store.delete(session_id, key)

            @log_unhandled
            def _handle_spill_reject(*exc, **kwargs):
                key = kwargs['chunk_key']
                if key in self._spill_pending_keys:
                    self._spill_pending_keys.remove(key)
                six.reraise(*exc)

            @log_unhandled
            def _spill_key(key):
                if key in self._pinned_counter:
                    return
                if key not in self._cache_chunk_sessions:
                    return
                spill_ref = self._get_spill_actor_ref(key)
                logger.debug('Send spill request for key %s to spill actor %s', key, spill_ref.uid)
                return spill_ref.spill(self._cache_chunk_sessions[key], key, _promise=True) \
                    .then(lambda *_: _release_spill_allocations(key)) \
                    .catch(functools32.partial(_handle_spill_reject, chunk_key=key))

            @log_unhandled
            def _finalize_spill(*_):
                logger.debug('Finish spilling %d chunks.', len(free_keys))
                self._plasma_client.evict(request_size)
                if callback:
                    self.tell_promise(callback)

                if self._status_ref:
                    self._status_ref.set_cache_allocations(
                        dict(hold=self._total_hold, total=self._total_size), _tell=True, _wait=False)

            promise.all_(_spill_key(k) for k in free_keys).then(_finalize_spill) \
                .catch(lambda *exc: self.tell_promise(callback, *exc, **dict(_accept=False)))
        else:
            logger.debug('No need to spill. request=%d', request_size)

            self._plasma_client.evict(request_size)
            if callback:
                self.tell_promise(callback)

    @log_unhandled
    def register_chunk(self, session_id, chunk_key):
        if chunk_key in self._chunk_holder:
            self._total_hold -= len(self._chunk_holder[chunk_key])
            del self._chunk_holder[chunk_key]

        self._chunk_holder[chunk_key] = self._chunk_store.get_buffer(session_id, chunk_key)
        self._chunk_holder.move_to_end(chunk_key)

        self._total_hold += len(self._chunk_holder[chunk_key])
        self._cache_chunk_sessions[chunk_key] = session_id
        logger.debug('Chunk %s registered in %s. total_hold=%d', chunk_key, self.uid, self._total_hold)

        if self._status_ref:
            self._status_ref.set_cache_allocations(
                dict(hold=self._total_hold, total=self._total_size), _tell=True, _wait=False)

    @log_unhandled
    def unregister_chunk(self, session_id, chunk_key):
        spill_ref = self._get_spill_actor_ref(chunk_key)
        if spill_ref:
            spill_ref.delete(chunk_key, _tell=True)

        if chunk_key in self._cache_chunk_sessions:
            del self._cache_chunk_sessions[chunk_key]

        if chunk_key in self._chunk_holder:
            self._total_hold -= self._chunk_holder[chunk_key].size
            del self._chunk_holder[chunk_key]
            self._chunk_store.delete(session_id, chunk_key)

        logger.debug('Chunk %s unregistered in %s. total_hold=%d', chunk_key, self.uid, self._total_hold)

        if self._status_ref:
            self._status_ref.set_cache_allocations(
                dict(hold=self._total_hold, total=self._total_size), _tell=True, _wait=False)

    def is_stored(self, chunk_key):
        return chunk_key in self._chunk_holder

    def move_to_end(self, chunk_key):
        self._chunk_holder.move_to_end(chunk_key)

    @log_unhandled
    def pin_chunks(self, graph_key, chunk_keys):
        if isinstance(chunk_keys, six.string_types):
            chunk_keys = (chunk_keys,)
        spilling_keys = list(k for k in chunk_keys if k in self._spill_pending_keys)
        if spilling_keys:
            logger.warning('Cannot pin chunks %r', spilling_keys)
            raise PinChunkFailed
        pinned = []
        for k in chunk_keys:
            if k not in self._chunk_holder:
                continue
            if k not in self._pinned_counter:
                self._pinned_counter[k] = set()
            self._pinned_counter[k].add(graph_key)
            pinned.append(k)
        return pinned

    @log_unhandled
    def unpin_chunks(self, graph_key, chunk_keys):
        if isinstance(chunk_keys, six.string_types):
            chunk_keys = (chunk_keys,)
        for k in chunk_keys:
            if k not in self._pinned_counter:
                continue
            self._pinned_counter[k].difference_update((graph_key,))
            if not self._pinned_counter[k]:
                del self._pinned_counter[k]

    def dump_keys(self):
        return list(self._chunk_holder.keys())
