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

import functools
import logging
import sys

from ... import promise
from ...compat import Enum, six
from ...config import options
from ...utils import classproperty
from ...serialize import dataserializer

logger = logging.getLogger(__name__)


class DataStorageDevice(Enum):
    PROC_MEMORY = 0
    SHARED_MEMORY = 1
    DISK = 2

    def __lt__(self, other):
        return self.value < other.value

    @classproperty
    def GLOBAL_DEVICES(cls):
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
            .register_data(self._session_id, self._data_key, location, size, shape=shape)

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

    def __init__(self, storage_ctx):
        self._storage_ctx = storage_ctx
        self._actor_ctx = storage_ctx.actor_ctx

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

    @staticmethod
    def pass_on_exc(func, exc):
        func()
        six.reraise(*exc)

    @property
    def storage_ctx(self):
        return self._storage_ctx

    @property
    def host_actor(self):
        return self._storage_ctx.host_actor

    @property
    def location(self):
        return self.storage_type.build_location(self._storage_ctx.proc_id)

    def is_io_runner(self):
        return getattr(self.host_actor, '_io_runner', False)

    def actor_ref(self, *args, **kwargs):
        return self._actor_ctx.actor_ref(*args, **kwargs)

    def promise_ref(self, *args, **kwargs):
        return self._storage_ctx.host_actor.promise_ref(self.actor_ref(*args, **kwargs))

    def delete(self, session_id, data_key, _tell=False):
        raise NotImplementedError

    def load_from(self, session_id, data_key, src_handler):
        """
        :param session_id:
        :param data_key:
        :param src_handler:
        :type src_handler: StorageHandler
        """
        logger.debug('Try loading data (%s, %s) from device %s into %s',
                     session_id, data_key, src_handler.storage_type, self.storage_type)
        left_has_obj_io = getattr(self, '_has_object_io', False)
        right_has_obj_io = getattr(src_handler, '_has_object_io', False)
        right_has_bytes_io = getattr(src_handler, '_has_bytes_io', False)

        if left_has_obj_io and right_has_obj_io:
            return self.load_from_object_io(session_id, data_key, src_handler)
        elif right_has_bytes_io:
            return self.load_from_bytes_io(session_id, data_key, src_handler)
        else:
            return self.load_from_object_io(session_id, data_key, src_handler)

    def load_from_bytes_io(self, session_id, data_key, src_handler):
        raise NotImplementedError

    def load_from_object_io(self, session_id, data_key, src_handler):
        raise NotImplementedError

    def register_data(self, session_id, data_key, size, shape=None):
        self._storage_ctx.manager_ref \
            .register_data(session_id, data_key, self.location, size, shape=shape)

    def transfer_in_global_runner(self, session_id, data_key, src_handler, fallback=None):
        if fallback:
            if self.is_io_runner():
                return fallback()
            elif src_handler.storage_type != DataStorageDevice.DISK and \
                    self.storage_type != DataStorageDevice.DISK:
                return fallback()

        runner_ref = self.promise_ref(self._dispatch_ref.get_hash_slot(
            'iorunner', (session_id, data_key)))

        if src_handler.is_device_global():
            return runner_ref.load_from(
                self.storage_type, session_id, data_key, src_handler.storage_type, _promise=True)
        elif fallback is not None:
            def _unlocker(*exc):
                runner_ref.unlock(session_id, data_key)
                if exc:
                    six.reraise(*exc)

            return runner_ref.lock(session_id, data_key, _promise=True) \
                .then(fallback) \
                .then(lambda *_: _unlocker(), _unlocker)
        return promise.finished()

    def unregister_data(self, session_id, data_key, _tell=False):
        self._storage_ctx.manager_ref \
            .unregister_data(session_id, data_key, self.location, _tell=_tell)


class BytesStorageMixin(object):
    _has_bytes_io = True

    def create_bytes_reader(self, session_id, data_key, packed=False, packed_compression=None,
                            _promise=False):
        raise NotImplementedError

    def create_bytes_writer(self, session_id, data_key, total_bytes, packed=False,
                            packed_compression=None, _promise=False):
        raise NotImplementedError

    def _copy_bytes_data(self, reader, writer):
        copy_block_size = options.worker.copy_block_size
        async_read_pool = reader.get_io_pool()
        async_write_pool = writer.get_io_pool()

        def _copy():
            with reader:
                with_exc = False
                try:
                    write_future = None
                    while True:
                        block = async_read_pool.submit(reader.read, copy_block_size).result()
                        if write_future:
                            write_future.result()
                        if not block:
                            break
                        write_future = async_write_pool.submit(writer.write, block)
                except:  # noqa: E722
                    with_exc = True
                    raise
                finally:
                    writer.close(finished=not with_exc)
        return self.host_actor.spawn_promised(_copy)

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

    def get_object(self, session_id, data_key, serialized=False, _promise=False):
        raise NotImplementedError

    def put_object(self, session_id, data_key, obj, serialized=False, _promise=False):
        raise NotImplementedError


class SpillableStorageMixin(object):
    _spillable = True

    def spill_size(self, size, multiplier=1):
        raise NotImplementedError

    def lift_data_key(self, session_id, data_key):
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
                    return promise.finished(*sys.exc_info(), **dict(_accept=False))
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
        raise NotImplementedError('Storage type %r not supported' % storage_type)
