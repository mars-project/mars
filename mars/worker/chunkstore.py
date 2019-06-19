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

from ..actors import FunctionActor
from ..errors import StoreFull, StoreKeyExists
from ..utils import calc_data_size

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
            raise StoreKeyExists(session_chunk_key)
        self._mapping[session_chunk_key] = obj_id

    def get(self, session_id, chunk_key):
        return self._mapping.get((session_id, chunk_key))

    def delete(self, session_id, chunk_key):
        try:
            del self._mapping[(session_id, chunk_key)]
        except KeyError:
            pass


class PlasmaChunkStore(object):
    """
    Wrapper of plasma client for Mars objects
    """
    def __init__(self, plasma_client, mapper_ref):
        from ..serialize.dataserializer import mars_serialize_context

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
            from pyarrow import plasma, lib

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
                except lib.PlasmaStoreFull:
                    alloc_fraction -= 0.1
                    if alloc_fraction < 1e-6:
                        break
            del bufs
            self._plasma_client.evict(total_size)
            self._actual_size = total_size
        return self._actual_size

    def _new_object_id(self, session_id, chunk_key):
        """
        Calc unique object id for chunks
        """
        from pyarrow.plasma import ObjectID
        while True:
            new_id = ObjectID.from_random()
            if not self._plasma_client.contains(new_id):
                break
        self._mapper_ref.put(session_id, chunk_key, new_id)
        return new_id

    def _get_object_id(self, session_id, chunk_key):
        obj_id = self._mapper_ref.get(session_id, chunk_key)
        if obj_id is None:
            raise KeyError((session_id, chunk_key))
        return obj_id

    def create(self, session_id, chunk_key, size):
        from pyarrow.lib import PlasmaStoreFull
        obj_id = self._new_object_id(session_id, chunk_key)

        try:
            self._plasma_client.evict(size)
            buffer = self._plasma_client.create(obj_id, size)
            return buffer
        except PlasmaStoreFull:
            exc_type = PlasmaStoreFull
            self._mapper_ref.delete(session_id, chunk_key)
            logger.warning('Chunk %s(%d) failed to store to plasma due to StoreFullError',
                           chunk_key, size)
        except:  # noqa: E722  # pragma: no cover
            self._mapper_ref.delete(session_id, chunk_key)
            raise

        if exc_type is PlasmaStoreFull:
            raise StoreFull

    def seal(self, session_id, chunk_key):
        from pyarrow.lib import PlasmaObjectNonexistent
        obj_id = self._get_object_id(session_id, chunk_key)
        try:
            self._plasma_client.seal(obj_id)
        except PlasmaObjectNonexistent:
            raise KeyError((session_id, chunk_key))

    def get(self, session_id, chunk_key):
        """
        Get deserialized Mars object from plasma store
        """
        from pyarrow.plasma import ObjectNotAvailable

        obj_id = self._get_object_id(session_id, chunk_key)
        obj = self._plasma_client.get(obj_id, serialization_context=self._serialize_context, timeout_ms=10)
        if obj is ObjectNotAvailable:
            raise KeyError((session_id, chunk_key))
        return obj

    def get_buffer(self, session_id, chunk_key):
        """
        Get raw buffer from plasma store
        """
        obj_id = self._get_object_id(session_id, chunk_key)
        [buf] = self._plasma_client.get_buffers([obj_id], timeout_ms=10)
        if buf is None:
            raise KeyError((session_id, chunk_key))
        return buf

    def get_actual_size(self, session_id, chunk_key):
        """
        Get actual size of Mars object from plasma store
        """
        buf = None
        try:
            obj_id = self._get_object_id(session_id, chunk_key)
            [buf] = self._plasma_client.get_buffers([obj_id], timeout_ms=10)
            if buf is None:
                raise KeyError((session_id, chunk_key))
            size = buf.size
        finally:
            del buf
        return size

    def put(self, session_id, chunk_key, value):
        """
        Put a Mars object into plasma store
        :param session_id: session id
        :param chunk_key: chunk key
        :param value: Mars object to be put
        """
        import pyarrow
        from pyarrow.lib import PlasmaStoreFull

        data_size = calc_data_size(value)

        try:
            obj_id = self._new_object_id(session_id, chunk_key)
        except StoreKeyExists:
            obj_id = self._get_object_id(session_id, chunk_key)
            if self._plasma_client.contains(obj_id):
                logger.debug('Chunk %s already exists, returning existing', chunk_key)
                [buffer] = self._plasma_client.get_buffers([obj_id], timeout_ms=10)
                return buffer
            else:
                logger.warning('Chunk %s registered but no data found, reconstructed', chunk_key)
                self.delete(session_id, chunk_key)
                obj_id = self._new_object_id(session_id, chunk_key)

        try:
            serialized = pyarrow.serialize(value, self._serialize_context)
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
            self._mapper_ref.delete(session_id, chunk_key)
            logger.warning('Chunk %s(%d) failed to store to plasma due to StoreFullError',
                           chunk_key, data_size)
            exc = PlasmaStoreFull
        except:  # noqa: E722  # pragma: no cover
            self._mapper_ref.delete(session_id, chunk_key)
            raise

        if exc is PlasmaStoreFull:
            raise StoreFull

    def contains(self, session_id, chunk_key):
        """
        Check if given chunk key exists in current plasma store
        """
        try:
            obj_id = self._get_object_id(session_id, chunk_key)
            return self._plasma_client.contains(obj_id)
        except KeyError:
            return False

    def delete(self, session_id, chunk_key):
        self._mapper_ref.delete(session_id, chunk_key)
