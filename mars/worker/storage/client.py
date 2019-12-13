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
import operator
from collections import defaultdict

from ... import promise
from ...compat import six, reduce
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

    def get_storage_handler(self, location):
        try:
            return self._storage_handlers[location]
        except KeyError:
            handler = self._storage_handlers[location] = \
                get_storage_handler_cls(location[1])(self, proc_id=location[0])
            return handler

    def __getattr__(self, item):
        return getattr(self._manager_ref, item)

    def _normalize_devices(self, devices):
        return [d if isinstance(d, tuple) else d.build_location(self.proc_id)
                for d in devices] if devices is not None else None

    def _do_with_spill(self, action, data_keys, total_bytes, device_order,
                       device_pos=0, spill_multiplier=1.0, ensure=True):
        def _handle_err(*exc_info):
            if issubclass(exc_info[0], StorageFull):
                req_bytes = max(total_bytes, exc_info[1].request_size)
                if device_pos < len(device_order) - 1:
                    return self._do_with_spill(
                        action, exc_info[1].affected_keys, req_bytes, device_order,
                        device_pos=device_pos + 1, spill_multiplier=1.0, ensure=ensure,
                    )
                elif ensure:
                    new_multiplier = min(spill_multiplier + 0.1, 10)
                    return handler.spill_size(req_bytes, spill_multiplier) \
                        .then(lambda *_: self._do_with_spill(
                            action, exc_info[1].affected_keys, req_bytes, device_order,
                            device_pos=device_pos, spill_multiplier=new_multiplier, ensure=ensure,
                        ))
            six.reraise(*exc_info)

        cur_device = device_order[device_pos]
        handler = self.get_storage_handler(cur_device)
        return action(handler, data_keys).catch(_handle_err)

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
        source_devices = self._normalize_devices(source_devices)
        stored_devs = set(self._manager_ref.get_data_locations(session_id, [data_key])[0])
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
            return self.copy_to(session_id, [data_key], source_devices) \
                .then(lambda *_: self.create_reader(session_id, data_key, source_devices, packed=packed))
        else:
            raise IOError('Cannot return a non-promise result')

    def create_writer(self, session_id, data_key, total_bytes, device_order, packed=False,
                      packed_compression=None, _promise=True):
        device_order = self._normalize_devices(device_order)

        def _action(handler, _keys, _promise=True):
            logger.debug('Creating %s writer for (%s, %s) on %s', 'packed' if packed else 'bytes',
                         session_id, data_key, handler.storage_type)
            return handler.create_bytes_writer(
                session_id, data_key, total_bytes, packed=packed,
                packed_compression=packed_compression, _promise=_promise)

        if _promise:
            return self._do_with_spill(_action, [data_key], total_bytes, device_order)
        else:
            exc = ValueError('Missing device type')
            for device_type in device_order:
                try:
                    handler = self.get_storage_handler(device_type)
                    return _action(handler, [data_key], _promise=False)
                except StorageFull as ex:
                    exc = ex
            raise exc

    def get_object(self, session_id, data_key, source_devices, serialize=False, _promise=True):
        if _promise:
            return self.get_objects(session_id, [data_key], source_devices, serialize=serialize,
                                    _promise=True).then(lambda objs: objs[0])
        else:
            return self.get_objects(session_id, [data_key], source_devices, serialize=serialize,
                                    _promise=False)[0]

    def get_objects(self, session_id, data_keys, source_devices, serialize=False, _promise=True):
        source_devices = self._normalize_devices(source_devices)
        stored_dev_lists = self._manager_ref.get_data_locations(session_id, data_keys)
        dev_to_keys = defaultdict(list)
        for key, devs in zip(data_keys, stored_dev_lists):
            first_dev = next((stored_dev for stored_dev in source_devices if stored_dev in devs),
                             None) or sorted(devs)[0]
            dev_to_keys[first_dev].append(key)

        data_dict = dict()
        if not _promise:
            if any(dev not in source_devices for dev in dev_to_keys.keys()):
                raise IOError('Getting objects without promise not supported')
            for stored_dev, keys in dev_to_keys.items():
                handler = self.get_storage_handler(stored_dev)
                data_dict.update(zip(keys, handler.get_objects(
                    session_id, data_keys, serialize=serialize, _promise=False)))
            return [data_dict[k] for k in data_keys]
        else:
            promises = []
            for stored_dev, keys in dev_to_keys.items():
                handler = self.get_storage_handler(stored_dev)
                loc_getter = functools.partial(
                    lambda keys, *_: self.get_objects(
                        session_id, keys, source_devices, serialize=serialize, _promise=True), keys)
                updater = functools.partial(lambda keys, objs: data_dict.update(zip(keys, objs)),
                                            keys)
                if stored_dev in source_devices:
                    promises.append(handler.get_objects(
                        session_id, keys, serialize=serialize, _promise=True).then(updater))
                else:
                    promises.append(self.copy_to(session_id, keys, source_devices)
                        .then(loc_getter).then(updater))
            return promise.all_(promises).then(lambda *_: [data_dict[k] for k in data_keys])

    def put_objects(self, session_id, data_keys, objs, device_order, sizes=None, serialize=False):
        device_order = self._normalize_devices(device_order)
        if sizes:
            sizes_dict = dict(zip(data_keys, sizes))
        else:
            sizes_dict = dict((k, calc_data_size(obj)) for k, obj in zip(data_keys, objs))

        data_dict = dict(zip(data_keys, objs))

        def _action(h, keys):
            objects = [data_dict[k] for k in keys]
            data_sizes = [sizes_dict[k] for k in keys]
            try:
                return h.put_objects(session_id, keys, objects, sizes=data_sizes, serialize=serialize,
                                     _promise=True)
            finally:
                objects[:] = []

        return self._do_with_spill(_action, data_keys, sum(sizes_dict.values()), device_order)

    def copy_to(self, session_id, data_keys, device_order, ensure=True, pin_token=None):
        device_order = self._normalize_devices(device_order)
        existing_devs = self._manager_ref.get_data_locations(session_id, data_keys)
        data_sizes = self._manager_ref.get_data_sizes(session_id, data_keys)

        device_to_keys = defaultdict(list)
        device_total_size = defaultdict(lambda: 0)
        lift_reqs = defaultdict(list)
        for k, devices, size in zip(data_keys, existing_devs, data_sizes):
            if not devices or not size:
                return promise.finished(
                    *build_exc_info(KeyError, 'Data key (%s, %s) does not exist.' % (session_id, k)),
                    **dict(_accept=False)
                )

            target = next((d for d in device_order if d in devices), None)
            if target is not None:
                lift_reqs[target].append(k)
            else:
                max_device = max(devices)
                device_to_keys[max_device].append(k)
                device_total_size[max_device] += size

        for target, data_keys in lift_reqs.items():
            handler = self.get_storage_handler(target)
            if getattr(handler, '_spillable', False):
                handler.lift_data_keys(session_id, data_keys)
        if not device_to_keys:
            return promise.finished()

        def _action(src_handler, h, keys):
            return h.load_from(session_id, keys, src_handler, pin_token=pin_token)

        def _handle_exc(keys, *exc):
            existing = self._manager_ref.get_data_locations(session_id, keys)
            for devices in existing:
                if not any(d for d in device_order if d in devices):
                    six.reraise(*exc)

        promises = []
        for d in device_to_keys.keys():
            action = functools.partial(_action, self.get_storage_handler(d))
            keys = device_to_keys[d]
            total_size = device_total_size[d]
            promises.append(
                self._do_with_spill(action, keys, total_size, device_order, ensure=ensure)
                    .catch(functools.partial(_handle_exc, keys))
            )
        return promise.all_(promises)

    def delete(self, session_id, data_keys, devices=None, _tell=False):
        if not devices:
            devices = reduce(operator.ior,
                             self._manager_ref.get_data_locations(session_id, data_keys), set())
        else:
            devices = self._normalize_devices(devices)

        devices = self._normalize_devices(devices)
        for dev_type in devices:
            handler = self.get_storage_handler(dev_type)
            handler.delete(session_id, data_keys, _tell=_tell)

    def filter_exist_keys(self, session_id, data_keys, devices=None):
        devices = self._normalize_devices(devices or DataStorageDevice.__members__.values())
        return self.manager_ref.filter_exist_keys(session_id, data_keys, devices)

    def pin_data_keys(self, session_id, data_keys, token, devices=None):
        if not devices:
            devices = reduce(operator.ior,
                             self._manager_ref.get_data_locations(session_id, data_keys), set())
        else:
            devices = self._normalize_devices(devices)

        pinned = set()
        for dev in devices:
            handler = self.get_storage_handler(dev)
            if not getattr(handler, '_spillable', False):
                continue
            keys = handler.pin_data_keys(session_id, data_keys, token)
            pinned.update(keys)
        return list(pinned)

    def unpin_data_keys(self, session_id, data_keys, token, devices=None):
        if not devices:
            devices = reduce(operator.ior,
                             self._manager_ref.get_data_locations(session_id, data_keys), set())
        else:
            devices = self._normalize_devices(devices)

        for dev in devices:
            handler = self.get_storage_handler(dev)
            if not getattr(handler, '_spillable', False):
                continue
            handler.unpin_data_keys(session_id, data_keys, token)

    def spill_size(self, data_size, devices):
        promises = []
        devices = self._normalize_devices(devices)
        for dev in devices:
            handler = self.get_storage_handler(dev)
            if not getattr(handler, '_spillable', False):
                continue
            promises.append(handler.spill_size(data_size))
        return promise.all_(promises)
