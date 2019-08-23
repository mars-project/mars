# Copyright 1999-2019 Alibaba Group Holding Ltd.
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

from ... import promise
from ...compat import six
from ...errors import StorageFull
from ...utils import calc_data_size, build_exc_info
from .core import DataStorageDevice, get_storage_handler_cls
from .manager import StorageManagerActor

logger = logging.getLogger(__name__)


class StorageClient(object):

    def __init__(self, host_actor):
        self._storage_handlers = {}

        self._host_actor = host_actor
        self._actor_ctx = host_actor.ctx
        self._shared_store = host_actor.shared_store
        self._proc_id = host_actor.proc_id
        self._manager_ref = self._actor_ctx.actor_ref(StorageManagerActor.default_uid())

    @property
    def host_actor(self):
        return self._host_actor

    @property
    def actor_ctx(self):
        return self._actor_ctx

    @property
    def proc_id(self):
        return self._proc_id

    @property
    def shared_store(self):
        return self._shared_store

    @property
    def manager_ref(self):
        return self._manager_ref

    def actor_ref(self, *args, **kw):
        return self._actor_ctx.actor_ref(*args, **kw)

    def has_actor(self, *args, **kw):
        return self._actor_ctx.has_actor(*args, **kw)

    def promise_ref(self, *args, **kw):
        return self._host_actor.promise_ref(*args, **kw)

    def get_storage_handler(self, storage_type):
        try:
            return self._storage_handlers[storage_type]
        except KeyError:
            handler = self._storage_handlers[storage_type] = \
                get_storage_handler_cls(storage_type)(self)
            return handler

    def __getattr__(self, item):
        return getattr(self._manager_ref, item)

    def _get_stored_devices(self, session_id, data_key):
        return [loc[1] for loc in (self._manager_ref.get_data_locations(session_id, data_key) or ())
                if loc[0] == self._proc_id or loc[1] in DataStorageDevice.GLOBAL_DEVICES]

    def _do_with_spill(self, action, total_bytes, device_order,
                       device_pos=0, spill_multiplier=1.0, ensure=True):
        def _handle_err(*exc_info):
            if issubclass(exc_info[0], StorageFull):
                req_bytes = max(total_bytes, exc_info[1].request_size)
                if device_pos < len(device_order) - 1:
                    return self._do_with_spill(
                        action, req_bytes, device_order,
                        device_pos=device_pos + 1, spill_multiplier=1.0, ensure=ensure,
                    )
                elif ensure:
                    new_multiplier = min(spill_multiplier + 0.1, 10)
                    return handler.spill_size(req_bytes, spill_multiplier) \
                        .then(lambda *_: self._do_with_spill(
                            action, req_bytes, device_order, device_pos=device_pos,
                            spill_multiplier=new_multiplier, ensure=ensure,
                        ))
            six.reraise(*exc_info)

        cur_device_type = device_order[device_pos]
        handler = self.get_storage_handler(cur_device_type)
        return action(handler).catch(_handle_err)

    def create_reader(self, session_id, data_key, source_devices, packed=False,
                      packed_compression=None, _promise=True):
        """
        Create a data reader from existing data and return in a Promise.
        If no readers can be created, will try copying the data into a
        readable storage.

        :param session_id: session id
        :param data_key: data key
        :param source_devices: devices to read from
        :param packed: create a reader to read packed data format
        :param packed_compression: compression format to use when reading as packed
        :param _promise: return a promise
        """
        stored_devs = set(self._get_stored_devices(session_id, data_key))
        for src_dev in source_devices:
            if src_dev not in stored_devs:
                continue
            handler = self.get_storage_handler(src_dev)
            try:
                logger.debug('Creating %s reader for (%s, %s) on %s', 'packed' if packed else 'bytes',
                             session_id, data_key, handler.storage_type)
                return handler.create_bytes_reader(
                    session_id, data_key, packed=packed, packed_compression=packed_compression,
                    _promise=_promise)
            except AttributeError:  # pragma: no cover
                raise IOError('Device %r does not support direct reading.' % src_dev)

        if _promise:
            return self.copy_to(session_id, data_key, source_devices) \
                .then(lambda *_: self.create_reader(session_id, data_key, source_devices))
        else:
            raise IOError('Cannot return a non-promise result')

    def create_writer(self, session_id, data_key, total_bytes, device_order, packed=False,
                      packed_compression=None, _promise=True):
        def _action(handler, _promise=True):
            logger.debug('Creating %s writer for (%s, %s) on %s', 'packed' if packed else 'bytes',
                         session_id, data_key, handler.storage_type)
            return handler.create_bytes_writer(
                session_id, data_key, total_bytes, packed=packed,
                packed_compression=packed_compression, _promise=_promise)

        if _promise:
            return self._do_with_spill(_action, total_bytes, device_order)
        else:
            for device_type in device_order:
                try:
                    handler = self.get_storage_handler(device_type)
                    return _action(handler, _promise=False)
                except StorageFull:
                    pass
            raise StorageFull

    def get_object(self, session_id, data_key, source_devices, serialized=False, _promise=True):
        stored_devs = set(self._get_stored_devices(session_id, data_key))
        for src_dev in source_devices:
            if src_dev not in stored_devs:
                continue
            handler = self.get_storage_handler(src_dev)
            try:
                return handler.get_object(session_id, data_key, serialized=serialized, _promise=_promise)
            except AttributeError:  # pragma: no cover
                raise IOError('Device %r does not support direct reading.' % src_dev)

        if _promise:
            return self.copy_to(session_id, data_key, source_devices) \
                .then(lambda *_: self.get_object(session_id, data_key, source_devices, serialized=serialized))
        else:
            raise IOError('Getting object without promise not supported')

    def put_object(self, session_id, data_key, obj, device_order, serialized=False):
        data_size = self._manager_ref.get_data_size(session_id, data_key) or calc_data_size(obj)

        def _action(h):
            return h.put_object(session_id, data_key, obj, serialized=serialized, _promise=True)

        return self._do_with_spill(_action, data_size, device_order)

    def copy_to(self, session_id, data_key, device_order, ensure=True):
        existing_devs = set(self._get_stored_devices(session_id, data_key))
        data_size = self._manager_ref.get_data_size(session_id, data_key)

        if not existing_devs or not data_size:
            return promise.finished(
                *build_exc_info(KeyError, 'Data key (%s, %s) does not exist.' % (session_id, data_key)),
                **dict(_accept=False)
            )

        target = next((d for d in device_order if d in existing_devs), None)
        if target is not None:
            handler = self.get_storage_handler(target)
            if getattr(handler, '_spillable', False):
                handler.lift_data_key(session_id, data_key)
            return promise.finished(target)

        source_handler = self.get_storage_handler(max(existing_devs))

        def _action(h):
            return h.load_from(session_id, data_key, source_handler)

        def _handle_exc(*exc):
            existing = set(self._get_stored_devices(session_id, data_key))
            if not any(d for d in device_order if d in existing):
                six.reraise(*exc)

        return self._do_with_spill(_action, data_size, device_order, ensure=ensure) \
            .catch(_handle_exc)

    def delete(self, session_id, data_key, devices=None, _tell=False):
        devices = devices or self._get_stored_devices(session_id, data_key) or ()
        for dev_type in devices:
            handler = self.get_storage_handler(dev_type)
            handler.delete(session_id, data_key, _tell=_tell)

    def filter_exist_keys(self, session_id, data_keys, devices=None):
        devices = [d.build_location(self.proc_id)
                   for d in (devices or DataStorageDevice.__members__.values())]
        return self.manager_ref.filter_exist_keys(session_id, data_keys, devices)

    def pin_data_keys(self, session_id, data_keys, token, devices=None):
        devices = devices or DataStorageDevice.__members__.values()
        pinned = set()
        for dev in devices:
            handler = self.get_storage_handler(dev)
            if not getattr(handler, '_spillable', False):
                continue
            keys = handler.pin_data_keys(session_id, data_keys, token)
            pinned.update(keys)
        return list(pinned)

    def unpin_data_keys(self, session_id, data_keys, token, devices=None):
        devices = devices or DataStorageDevice.__members__.values()
        for dev in devices:
            handler = self.get_storage_handler(dev)
            if not getattr(handler, '_spillable', False):
                continue
            handler.unpin_data_keys(session_id, data_keys, token)

    def spill_size(self, data_size, devices):
        promises = []
        for dev in devices:
            handler = self.get_storage_handler(dev)
            if not getattr(handler, '_spillable', False):
                continue
            promises.append(handler.spill_size(data_size))
        return promise.all_(promises)
