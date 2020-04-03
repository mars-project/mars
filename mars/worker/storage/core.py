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
import sys
from collections import defaultdict
from enum import Enum

from ... import promise
from ...config import options
from ...utils import classproperty, log_unhandled
from ...serialize import dataserializer
from .iorunner import IORunnerActor

logger = logging.getLogger(__name__)


class DataStorageDevice(Enum):
    PROC_MEMORY = 0
    CUDA = 1
    SHARED_MEMORY = 2
    DISK = 3
    VINEYARD = 4

    def __lt__(self, other):
        return self.value < other.value

    @classproperty
    def GLOBAL_DEVICES(cls):
        if options.vineyard.socket:
            return cls.VINEYARD, cls.DISK  # pragma: no cover
        else:
            return cls.SHARED_MEMORY, cls.DISK

    def build_location(self, proc_id):
        if self in self.GLOBAL_DEVICES:
            return 0, self
        else:
            return proc_id, self


class BytesStorageIO(object):
    storage_type = None

    def __init__(self, session_id, data_key, mode='w', handler=None, **kwargs):
        self._session_id = session_id
        self._data_key = data_key
        self._mode = mode
        self._buf = None
        self._handler = handler
        self._storage_ctx = handler.storage_ctx

        self._is_readable = 'r' in mode
        self._is_writable = 'w' in mode
        self._closed = False

    @property
    def is_readable(self):
        return self._is_readable and not self._closed

    @property
    def is_writable(self):
        return self._is_writable and not self._closed

    @property
    def closed(self):
        return self._closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(finished=exc_val is None)

    def __del__(self):
        self.close()

    def get_io_pool(self, pool_name=None):
        return self._handler.get_io_pool(pool_name)

    def register(self, size, shape=None):
        location = self.storage_type.build_location(self._storage_ctx.proc_id)
        self._storage_ctx.manager_ref \
            .register_data(self._session_id, [self._data_key], location, [size], shapes=[shape])

    def close(self, finished=True):
        self._closed = True

    @property
    def nbytes(self):
        raise NotImplementedError

    def read(self, size=-1):
        raise NotImplementedError

    def write(self, d):
        raise NotImplementedError


class StorageHandler(object):
    storage_type = None

    def __init__(self, storage_ctx, proc_id=None):
        self._storage_ctx = storage_ctx
        self._actor_ctx = storage_ctx.actor_ctx
        self._proc_id = proc_id if proc_id is not None else storage_ctx.host_actor.proc_id
        if self.is_device_global():
            self._proc_id = 0

        from ..dispatcher import DispatchActor
        self._dispatch_ref = self.actor_ref(DispatchActor.default_uid())

    def get_io_pool(self, pool_name=None):
        actor_obj = self.host_actor
        pool_var = '_pool_%s_attr_%s' % (pool_name or '', self.storage_type.value)
        if getattr(actor_obj, pool_var, None) is None:
            setattr(actor_obj, pool_var, self._actor_ctx.threadpool(1))
        return getattr(actor_obj, pool_var)

    @classmethod
    def is_device_global(cls):
        return cls.storage_type in DataStorageDevice.GLOBAL_DEVICES

    def is_other_process(self):
        return not self.is_device_global() and self.proc_id != self._storage_ctx.proc_id

    @staticmethod
    def pass_on_exc(func, exc):
        func()
        raise exc[1].with_traceback(exc[2]) from None

    @property
    def storage_ctx(self):
        return self._storage_ctx

    @property
    def host_actor(self):
        return self._storage_ctx.host_actor

    @property
    def proc_id(self):
        return self._proc_id

    @property
    def location(self):
        return self.storage_type.build_location(self._proc_id)

    def is_io_runner(self):
        return getattr(self.host_actor, '_io_runner', False)

    def actor_ref(self, *args, **kwargs):
        return self._actor_ctx.actor_ref(*args, **kwargs)

    def promise_ref(self, *args, **kwargs):
        return self._storage_ctx.host_actor.promise_ref(self.actor_ref(*args, **kwargs))

    def delete(self, session_id, data_keys, _tell=False):
        raise NotImplementedError

    def load_from(self, session_id, data_keys, src_handler, pin_token=None):
        """
        :param session_id: session id
        :param data_keys: keys of data to load
        :param src_handler: source data handler containing source data
        :type src_handler: StorageHandler
        :param pin_token: if not None, this token is used to pin loaded data.
                          pass this to unpin_data_keys() if unpin is needed
        """
        logger.debug('Try loading data %s from device %s into %s',
                     data_keys, src_handler.storage_type, self.storage_type)
        left_has_obj_io = getattr(self, '_has_object_io', False)
        right_has_obj_io = getattr(src_handler, '_has_object_io', False)
        right_has_bytes_io = getattr(src_handler, '_has_bytes_io', False)

        if left_has_obj_io and right_has_obj_io:
            return self.load_from_object_io(session_id, data_keys, src_handler, pin_token=pin_token)
        elif right_has_bytes_io:
            return self.load_from_bytes_io(session_id, data_keys, src_handler, pin_token=pin_token)
        else:
            return self.load_from_object_io(session_id, data_keys, src_handler, pin_token=pin_token)

    def load_from_bytes_io(self, session_id, data_keys, src_handler, pin_token=None):
        raise NotImplementedError

    def load_from_object_io(self, session_id, data_keys, src_handler, pin_token=None):
        raise NotImplementedError

    def register_data(self, session_id, data_keys, sizes, shapes=None):
        self._storage_ctx.manager_ref \
            .register_data(session_id, data_keys, self.location, sizes, shapes=shapes)

    def _dispatch_keys_to_uids(self, session_id, data_keys):
        uid_to_keys = defaultdict(list)
        runner_uids = self._dispatch_ref.get_hash_slots(
            'iorunner', [(session_id, k) for k in data_keys])
        for k, uid in zip(data_keys, runner_uids):
            uid_to_keys[uid].append(k)
        return uid_to_keys.items()

    def transfer_in_runner(self, session_id, data_keys, src_handler, fallback=None):
        if self.is_io_runner():
            return fallback() if fallback is not None else promise.finished()

        if self.is_device_global() and src_handler.is_device_global():
            return promise.all_(
                self.promise_ref(uid).load_from(
                    self.location, session_id, keys, src_handler.location, _promise=True)
                for uid, keys in self._dispatch_keys_to_uids(session_id, data_keys))
        elif self.is_device_global() or src_handler.is_device_global():
            if self.is_other_process() or src_handler.is_other_process():
                runner_proc_id = self.proc_id or src_handler.proc_id
                runner_ref = self.promise_ref(IORunnerActor.gen_uid(runner_proc_id))
                return runner_ref.load_from(
                    self.location, session_id, data_keys, src_handler.location, _promise=True)
            elif fallback is not None:
                if src_handler.storage_type != DataStorageDevice.DISK and \
                        self.storage_type != DataStorageDevice.DISK:
                    return fallback()

                uid_to_work_item_ids = dict()

                def _fallback_runner(uid, work_item_id):
                    uid_to_work_item_ids[uid] = work_item_id
                    return fallback()

                def _unlocker(uid, *exc, **kwargs):
                    self.promise_ref(uid).unlock(uid_to_work_item_ids[uid])
                    if not kwargs.get('accept', True):
                        raise exc[1].with_traceback(exc[2])

                return promise.all_(
                    self.promise_ref(uid).lock(session_id, keys, _promise=True)
                        .then(functools.partial(_fallback_runner, uid))
                        .then(functools.partial(_unlocker, uid), functools.partial(_unlocker, uid, accept=False))
                    for uid, keys in self._dispatch_keys_to_uids(session_id, data_keys)
                )
            else:
                return promise.finished()
        else:
            return fallback() if fallback is not None else promise.finished()

    def unregister_data(self, session_id, data_keys, _tell=False):
        self._storage_ctx.manager_ref \
            .unregister_data(session_id, data_keys, self.location, _tell=_tell)


class BytesStorageMixin(object):
    _has_bytes_io = True

    def create_bytes_reader(self, session_id, data_key, packed=False, packed_compression=None,
                            _promise=False):
        raise NotImplementedError

    def create_bytes_writer(self, session_id, data_key, total_bytes, packed=False,
                            packed_compression=None, auto_register=True, pin_token=None,
                            _promise=False):
        raise NotImplementedError

    def _copy_bytes_data(self, reader, writer, on_close=None):
        copy_block_size = options.worker.copy_block_size
        async_read_pool = reader.get_io_pool()
        async_write_pool = writer.get_io_pool()

        @log_unhandled
        def _copy(_reader, _writer):
            with_exc = False
            block = None
            write_future = None
            try:
                while True:
                    block = async_read_pool.submit(_reader.read, copy_block_size).result()
                    if write_future:
                        write_future.result()
                    if not block:
                        break
                    write_future = async_write_pool.submit(_writer.write, block)
            except:  # noqa: E722
                with_exc = True
                raise
            finally:
                if on_close:
                    on_close(_reader, _writer, not with_exc)
                _reader.close()
                _writer.close(finished=not with_exc)
                del _reader, _writer, block

        try:
            return self.host_actor.spawn_promised(_copy, reader, writer)
        finally:
            del reader, writer

    def _copy_object_data(self, serialized_obj, writer):
        def _copy(ser):
            try:
                with writer:
                    async_write_pool = writer.get_io_pool()
                    if hasattr(ser, 'write_to'):
                        async_write_pool.submit(ser.write_to, writer).result()
                    else:
                        async_write_pool.submit(writer.write, ser).result()
            finally:
                del ser

        try:
            return self.host_actor.spawn_promised(_copy, serialized_obj)
        finally:
            del serialized_obj


class ObjectStorageMixin(object):
    _has_object_io = True

    @staticmethod
    def _deserial(obj):
        if hasattr(obj, 'deserialize'):
            return obj.deserialize()
        else:
            return dataserializer.deserialize(obj)

    def _batch_load_objects(self, session_id, data_keys, key_loader,
                            batch_get=False, serialize=False, pin_token=None):
        is_success = True
        data_dict = dict()
        sizes = self._storage_ctx.manager_ref.get_data_sizes(session_id, data_keys)

        def _record_data(k, o):
            try:
                if is_success:
                    data_dict[k] = o
            finally:
                del o

        def _put_objects(*_):
            keys, objs = zip(*data_dict.items())
            data_dict.clear()
            key_list, obj_list = list(keys), list(objs)
            try:
                return self.put_objects(session_id, key_list, obj_list, sizes, serialize=serialize,
                                        pin_token=pin_token, _promise=True)
            finally:
                del objs
                obj_list[:] = []

        def _handle_err(*exc_info):
            nonlocal is_success
            is_success = False
            data_dict.clear()
            raise exc_info[1].with_traceback(exc_info[2]) from None

        def _batch_put_objects(objs):
            try:
                return self.put_objects(session_id, data_keys, objs, sizes,
                                        serialize=serialize, pin_token=pin_token, _promise=True)
            finally:
                del objs

        if batch_get:
            return key_loader(data_keys).then(_batch_put_objects)
        else:
            return promise.all_(
                key_loader(k).then(functools.partial(_record_data, k), _handle_err)
                for k in data_keys).then(_put_objects, _handle_err)

    def get_objects(self, session_id, data_keys, serialize=False, _promise=False):
        raise NotImplementedError

    def put_objects(self, session_id, data_keys, objs, sizes=None, serialize=False, pin_token=False,
                    _promise=False):
        raise NotImplementedError


class SpillableStorageMixin(object):
    _spillable = True

    def spill_size(self, size, multiplier=1):
        raise NotImplementedError

    def lift_data_keys(self, session_id, data_keys):
        raise NotImplementedError

    def pin_data_keys(self, session_id, data_keys, token):
        raise NotImplementedError

    def unpin_data_keys(self, session_id, data_keys, token, _tell=False):
        raise NotImplementedError


def wrap_promised(func):
    @functools.wraps(func)
    def _wrapped(*args, **kwargs):
        try:
            promised = kwargs.get('_promise')
            if not promised:
                return func(*args, **kwargs)
            else:
                try:
                    val = func(*args, **kwargs)
                    if isinstance(val, promise.Promise):
                        return val
                    else:
                        return promise.finished(val)
                except:  # noqa: E722
                    return promise.finished(*sys.exc_info(), _accept=False)
        finally:
            del args, kwargs

    return _wrapped


_storage_handler_cls = {}


def register_storage_handler_cls(storage_type, handler_cls):
    _storage_handler_cls[storage_type] = handler_cls


def get_storage_handler_cls(storage_type):
    try:
        return _storage_handler_cls[storage_type]
    except KeyError:  # pragma: no cover
        raise NotImplementedError('Storage type %r not supported' % (storage_type, ))
