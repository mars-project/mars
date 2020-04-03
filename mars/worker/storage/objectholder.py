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
import os
import time
from collections import OrderedDict

from ... import promise
from ...config import options
from ...utils import parse_readable_size, log_unhandled, readable_size, tokenize
from ...errors import SpillNotConfigured, SpillSizeExceeded, NoDataToSpill, PinDataKeyFailed
from ..utils import WorkerActor
from .core import DataStorageDevice
from .manager import StorageManagerActor

logger = logging.getLogger(__name__)


class ObjectHolderActor(WorkerActor):
    _storage_device = None
    _spill_devices = None

    def __init__(self, size_limit=0):
        super().__init__()
        self._size_limit = size_limit

        self._data_holder = OrderedDict()
        self._data_sizes = dict()

        self._total_hold = 0
        self._pinned_counter = dict()
        self._spill_pending_keys = set()

        self._total_spill = 0
        self._min_spill_size = 0
        self._max_spill_size = 0

        self._dispatch_ref = None
        self._status_ref = None
        self._storage_handler = None

    def post_create(self):
        from ..dispatcher import DispatchActor
        from ..status import StatusActor

        super().post_create()
        self.register_actors_down_handler()
        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())

        parse_num, is_percent = parse_readable_size(options.worker.min_spill_size)
        self._min_spill_size = int(self._size_limit * parse_num if is_percent else parse_num)
        parse_num, is_percent = parse_readable_size(options.worker.max_spill_size)
        self._max_spill_size = int(self._size_limit * parse_num if is_percent else parse_num)

        status_ref = self.ctx.actor_ref(StatusActor.default_uid())
        self._status_ref = status_ref if self.ctx.has_actor(status_ref) else None

        self._storage_handler = self.storage_client.get_storage_handler(
            self._storage_device.build_location(self.proc_id))

    def pre_destroy(self):
        for k in self._data_holder:
            self._data_holder[k] = None

    def update_cache_status(self):
        raise NotImplementedError

    def post_delete(self, session_id, data_keys):
        raise NotImplementedError

    def get_size_limit(self):
        return self._size_limit

    @promise.reject_on_exception
    @log_unhandled
    def spill_size(self, size, multiplier=1, callback=None):
        if not self._spill_devices:  # pragma: no cover
            raise SpillNotConfigured

        request_size = int(size * multiplier)
        request_size = max(request_size, self._min_spill_size)
        if request_size > self._size_limit:
            raise SpillSizeExceeded
        request_size = min(request_size, self._max_spill_size)

        spill_ref_key = tokenize((time.time(), size, multiplier))

        logger.debug('Start spilling %d(x%d) bytes in %s. ref_key==%s.',
                     request_size, multiplier, self.uid, spill_ref_key)

        if request_size + self._total_hold > self._size_limit:
            acc_free = 0
            free_keys = []
            for k in self._data_holder.keys():
                if k in self._pinned_counter or k in self._spill_pending_keys:
                    continue
                acc_free += self._data_sizes[k]
                free_keys.append(k)
                self._spill_pending_keys.add(k)
                if request_size + self._total_hold - acc_free <= self._size_limit:
                    break

            if not free_keys:
                logger.warning('Cannot spill further in %s. Rejected. request=%d ref_key=%s',
                               self.uid, request_size, spill_ref_key)
                raise NoDataToSpill

            logger.debug('Decide to spill %d data keys %r in %s. request=%d ref_key=%s',
                         len(free_keys), free_keys, self.uid, request_size, spill_ref_key)

            @log_unhandled
            def _release_spill_allocations(key):
                logger.debug('Removing reference of data %s from %s when spilling. ref_key=%s',
                             key, self.uid, spill_ref_key)
                self.delete_objects(key[0], [key[1]])

            @log_unhandled
            def _handle_spill_reject(*exc, **kwargs):
                key = kwargs['session_data_key']
                self._remove_spill_pending(*key)
                raise exc[1].with_traceback(exc[2])

            @log_unhandled
            def _spill_key(key):
                if key in self._pinned_counter or key not in self._data_holder:
                    self._remove_spill_pending(*key)
                    return
                logger.debug('Spilling key %s in %s. ref_key=%s', key, self.uid, spill_ref_key)
                return self.storage_client.copy_to(key[0], [key[1]], self._spill_devices) \
                    .then(lambda *_: _release_spill_allocations(key),
                          functools.partial(_handle_spill_reject, session_data_key=key))

            @log_unhandled
            def _finalize_spill(*_):
                logger.debug('Finish spilling %d data keys in %s. ref_key=%s',
                             len(free_keys), self.uid, spill_ref_key)
                self._plasma_client.evict(request_size)
                if callback:
                    self.tell_promise(callback)
                self.update_cache_status()

            promise.all_(_spill_key(k) for k in free_keys).then(_finalize_spill) \
                .catch(lambda *exc: self.tell_promise(callback, *exc, _accept=False))
        else:
            logger.debug('No need to spill in %s. request=%d ref_key=%s',
                         self.uid, request_size, spill_ref_key)

            self._plasma_client.evict(request_size)
            if callback:
                self.tell_promise(callback)

    @log_unhandled
    def _internal_put_object(self, session_id, data_key, obj, size):
        try:
            session_data_key = (session_id, data_key)
            if session_data_key in self._data_holder:
                self._total_hold -= self._data_sizes[session_data_key]
                del self._data_holder[session_data_key]

            self._data_holder[session_data_key] = obj
            self._data_sizes[session_data_key] = size
            self._total_hold += size
        finally:
            del obj

    def _finish_put_objects(self, _session_id, data_keys):
        if logger.getEffectiveLevel() <= logging.DEBUG:  # pragma: no cover
            simplified_keys = sorted(set(k[0] if isinstance(k, tuple) else k for k in data_keys))
            logger.debug('Data %r registered in %s. total_hold=%d', simplified_keys,
                         self.uid, self._total_hold)
        self.update_cache_status()

    def _remove_spill_pending(self, session_id, data_key):
        try:
            self._spill_pending_keys.remove((session_id, data_key))
            logger.debug('Spill-pending key (%s, %s) removed in %s', session_id, data_key, self.uid)
        except KeyError:
            pass

    @log_unhandled
    def delete_objects(self, session_id, data_keys):
        actual_removed = []
        for data_key in data_keys:
            session_data_key = (session_id, data_key)

            self._remove_spill_pending(session_id, data_key)

            try:
                del self._pinned_counter[session_data_key]
            except KeyError:
                pass

            if session_data_key in self._data_holder:
                actual_removed.append(data_key)

                data_size = self._data_sizes[session_data_key]
                self._total_hold -= data_size
                del self._data_holder[session_data_key]
                del self._data_sizes[session_data_key]

        self.post_delete(session_id, actual_removed)
        if actual_removed:
            logger.debug('Data %s unregistered in %s. total_hold=%d', actual_removed, self.uid, self._total_hold)
            self.update_cache_status()

    def lift_data_keys(self, session_id, data_keys, last=True):
        for k in data_keys:
            self._data_holder.move_to_end((session_id, k), last)

    @log_unhandled
    def pin_data_keys(self, session_id, data_keys, token):
        spilling_keys = list(k for k in data_keys if (session_id, k) in self._spill_pending_keys)
        if spilling_keys:
            logger.warning('Cannot pin data key %r: under spilling', spilling_keys)
            raise PinDataKeyFailed
        pinned = []
        for k in data_keys:
            session_k = (session_id, k)
            if session_k not in self._data_holder:
                continue
            if session_k not in self._pinned_counter:
                self._pinned_counter[session_k] = set()
            self._pinned_counter[session_k].add(token)
            pinned.append(k)
        logger.debug('Data keys %r pinned in %s', pinned, self.uid)
        return pinned

    @log_unhandled
    def unpin_data_keys(self, session_id, data_keys, token):
        unpinned = []
        for k in data_keys:
            session_k = (session_id, k)
            try:
                self._pinned_counter[session_k].remove(token)
                if not self._pinned_counter[session_k]:
                    del self._pinned_counter[session_k]
                unpinned.append(k)
            except KeyError:
                continue
        if unpinned:
            logger.debug('Data keys %r unpinned in %s', unpinned, self.uid)
        return unpinned

    def dump_keys(self):  # pragma: no cover
        return list(self._data_holder.keys())


class SimpleObjectHolderActor(ObjectHolderActor):
    def post_create(self):
        super().post_create()
        manager_ref = self.ctx.actor_ref(StorageManagerActor.default_uid())
        manager_ref.register_process_holder(
            self.proc_id, self._storage_device, self.ref())

    def put_objects(self, session_id, data_keys, data_objs, data_sizes, pin_token=None):
        try:
            for data_key, obj, size in zip(data_keys, data_objs, data_sizes):
                self._internal_put_object(session_id, data_key, obj, size)
            if pin_token:
                self.pin_data_keys(session_id, data_keys, pin_token)
            self._finish_put_objects(session_id, data_keys)
        finally:
            del data_objs

    def get_object(self, session_id, data_key):
        return self._data_holder[(session_id, data_key)]

    def get_objects(self, session_id, data_keys):
        return [self._data_holder[(session_id, key)] for key in data_keys]

    def update_cache_status(self):
        pass

    def post_delete(self, session_id, data_keys):
        pass


class SharedHolderActor(ObjectHolderActor):
    if options.vineyard.socket:
        _storage_device = DataStorageDevice.VINEYARD  # pragma: no cover
    else:
        _storage_device = DataStorageDevice.SHARED_MEMORY
    _spill_devices = (DataStorageDevice.DISK,)

    def post_create(self):
        super().post_create()
        self._size_limit = self._shared_store.get_actual_capacity(self._size_limit)
        logger.info('Detected actual plasma store size: %s', readable_size(self._size_limit))

    def update_cache_status(self):
        if self._status_ref:
            self._status_ref.set_cache_allocations(
                dict(hold=self._total_hold, total=self._size_limit), _tell=True, _wait=False)

    def post_delete(self, session_id, data_keys):
        self._shared_store.batch_delete(session_id, data_keys)
        self._storage_handler.unregister_data(session_id, data_keys)

    def put_objects_by_keys(self, session_id, data_keys, shapes=None, pin_token=None):
        sizes = []
        for data_key in data_keys:
            buf = None
            try:
                buf = self._shared_store.get_buffer(session_id, data_key)
                size = len(buf)
                self._internal_put_object(session_id, data_key, buf, size)
            finally:
                del buf
            sizes.append(size)
        if pin_token:
            self.pin_data_keys(session_id, data_keys, pin_token)

        self.storage_client.register_data(
            session_id, data_keys, (0, self._storage_device), sizes, shapes=shapes)


class InProcHolderActor(SimpleObjectHolderActor):
    _storage_device = DataStorageDevice.PROC_MEMORY


class CudaHolderActor(SimpleObjectHolderActor):
    _storage_device = DataStorageDevice.CUDA
    if options.vineyard.socket:
        shared_memory_device = DataStorageDevice.VINEYARD  # pragma: no cover
    else:
        shared_memory_device = DataStorageDevice.SHARED_MEMORY

    _spill_devices = [shared_memory_device, DataStorageDevice.DISK]

    def __init__(self, size_limit=0, device_id=None):
        super().__init__(size_limit=size_limit)
        if device_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

        # warm up cupy
        try:
            import cupy
            cupy.zeros((10, 10)).sum()
        except ImportError:
            pass
        # warm up cudf
        try:
            import cudf
            import numpy as np
            import pandas as pd
            cudf.from_pandas(pd.DataFrame(np.zeros((10, 10))))
        except ImportError:
            pass
