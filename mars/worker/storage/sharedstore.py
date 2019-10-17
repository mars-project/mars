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

from ...actors import FunctionActor
from ...errors import StorageFull, StorageDataExists
from ...utils import calc_data_size

try:
    import pyarrow
    from pyarrow import plasma
    try:
        from pyarrow.plasma import PlasmaObjectNonexistent, PlasmaStoreFull
    except ImportError:
        from pyarrow.lib import PlasmaObjectNonexistent, PlasmaStoreFull
except ImportError:  # pragma: no cover
    pyarrow, plasma, PlasmaObjectNonexistent, PlasmaStoreFull = None, None, None, None

logger = logging.getLogger(__name__)


class PlasmaKeyMapActor(FunctionActor):
    @classmethod
    def default_uid(cls):
        return 'w:0:' + cls.__name__

    def __init__(self):
        super(PlasmaKeyMapActor, self).__init__()
        self._mapping = dict()

    def put(self, session_id, chunk_key, obj_id):
        session_chunk_key = (session_id, chunk_key)
        if session_chunk_key in self._mapping:
            raise StorageDataExists(session_chunk_key)
        self._mapping[session_chunk_key] = obj_id

    def get(self, session_id, chunk_key):
        return self._mapping.get((session_id, chunk_key))

    def delete(self, session_id, chunk_key):
        try:
            del self._mapping[(session_id, chunk_key)]
        except KeyError:
            pass


class PlasmaSharedStore(object):
    """
    Wrapper of plasma client for Mars objects
    """
    def __init__(self, plasma_client, mapper_ref):
        from ...serialize.dataserializer import mars_serialize_context

        self._plasma_client = plasma_client
        self._actual_size = None
        self._serialize_context = mars_serialize_context()

        self._mapper_ref = mapper_ref

    def get_actual_capacity(self, store_limit):
        """
        Get actual capacity of plasma store
        :return: actual storage size in bytes
        """
        if self._actual_size is None:
            bufs = []
            left_size = store_limit
            total_size = 0
            alloc_fraction = 0.9
            while left_size:
                allocate_size = int(left_size * alloc_fraction)
                if allocate_size < 1 * 1024 ** 2:
                    break

                try:
                    obj_id = plasma.ObjectID.from_random()
                    bufs.append(self._plasma_client.create(obj_id, allocate_size))
                    self._plasma_client.seal(obj_id)
                    total_size += allocate_size
                    left_size -= allocate_size
                    alloc_fraction = 0.9
                except PlasmaStoreFull:
                    alloc_fraction -= 0.1
                    if alloc_fraction < 1e-6:
                        break
            del bufs
            self._plasma_client.evict(total_size)
            self._actual_size = total_size
        return self._actual_size

    def _new_object_id(self, session_id, data_key):
        """
        Calc unique object id for chunks
        """
        while True:
            new_id = plasma.ObjectID.from_random()
            if not self._plasma_client.contains(new_id):
                break
        self._mapper_ref.put(session_id, data_key, new_id)
        return new_id

    def _get_object_id(self, session_id, data_key):
        obj_id = self._mapper_ref.get(session_id, data_key)
        if obj_id is None:
            raise KeyError((session_id, data_key))
        return obj_id

    def create(self, session_id, data_key, size):
        obj_id = self._new_object_id(session_id, data_key)

        try:
            self._plasma_client.evict(size)
            buffer = self._plasma_client.create(obj_id, size)
            return buffer
        except PlasmaStoreFull:
            exc_type = PlasmaStoreFull
            self._mapper_ref.delete(session_id, data_key)
            logger.warning('Data %s(%d) failed to store to plasma due to StorageFull',
                           data_key, size)
        except:  # noqa: E722
            self._mapper_ref.delete(session_id, data_key)
            raise

        if exc_type is PlasmaStoreFull:
            raise StorageFull(request_size=size, total_size=self._actual_size)

    def seal(self, session_id, data_key):
        obj_id = self._get_object_id(session_id, data_key)
        try:
            self._plasma_client.seal(obj_id)
        except PlasmaObjectNonexistent:
            self._mapper_ref.delete(session_id, data_key)
            raise KeyError((session_id, data_key))

    def get(self, session_id, data_key):
        """
        Get deserialized Mars object from plasma store
        """
        obj_id = self._get_object_id(session_id, data_key)
        obj = self._plasma_client.get(obj_id, serialization_context=self._serialize_context, timeout_ms=10)
        if obj is plasma.ObjectNotAvailable:
            self._mapper_ref.delete(session_id, data_key)
            raise KeyError((session_id, data_key))
        return obj

    def get_buffer(self, session_id, data_key):
        """
        Get raw buffer from plasma store
        """
        obj_id = self._get_object_id(session_id, data_key)
        [buf] = self._plasma_client.get_buffers([obj_id], timeout_ms=10)
        if buf is None:
            self._mapper_ref.delete(session_id, data_key)
            raise KeyError((session_id, data_key))
        return buf

    def get_actual_size(self, session_id, data_key):
        """
        Get actual size of Mars object from plasma store
        """
        buf = None
        try:
            obj_id = self._get_object_id(session_id, data_key)
            [buf] = self._plasma_client.get_buffers([obj_id], timeout_ms=10)
            if buf is None:
                self._mapper_ref.delete(session_id, data_key)
                raise KeyError((session_id, data_key))
            return buf.size
        finally:
            del buf

    def put(self, session_id, data_key, value):
        """
        Put a Mars object into plasma store
        :param session_id: session id
        :param data_key: chunk key
        :param value: Mars object to be put
        """
        data_size = calc_data_size(value)

        try:
            obj_id = self._new_object_id(session_id, data_key)
        except StorageDataExists:
            obj_id = self._get_object_id(session_id, data_key)
            if self._plasma_client.contains(obj_id):
                logger.debug('Data %s already exists, returning existing', data_key)
                [buffer] = self._plasma_client.get_buffers([obj_id], timeout_ms=10)
                del value
                return buffer
            else:
                logger.warning('Data %s registered but no data found, reconstructed', data_key)
                self._mapper_ref.delete(session_id, data_key)
                obj_id = self._new_object_id(session_id, data_key)

        try:
            serialized = pyarrow.serialize(value, self._serialize_context)
            del value
            data_size = serialized.total_bytes
            try:
                buffer = self._plasma_client.create(obj_id, serialized.total_bytes)
                stream = pyarrow.FixedSizeBufferWriter(buffer)
                stream.set_memcopy_threads(6)
                serialized.write_to(stream)
                self._plasma_client.seal(obj_id)
            finally:
                del serialized
            return buffer
        except PlasmaStoreFull:
            self._mapper_ref.delete(session_id, data_key)
            logger.warning('Data %s(%d) failed to store to plasma due to StorageFull',
                           data_key, data_size)
            exc = PlasmaStoreFull
        except:  # noqa: E722
            self._mapper_ref.delete(session_id, data_key)
            raise

        if exc is PlasmaStoreFull:
            raise StorageFull(request_size=data_size, total_size=self._actual_size)

    def contains(self, session_id, data_key):
        """
        Check if given chunk key exists in current plasma store
        """
        try:
            obj_id = self._get_object_id(session_id, data_key)
            if self._plasma_client.contains(obj_id):
                return True
            else:
                self._mapper_ref.delete(session_id, data_key)
                return False
        except KeyError:
            return False

    def delete(self, session_id, data_key):
        self._mapper_ref.delete(session_id, data_key)
