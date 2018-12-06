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
import hashlib

from ..utils import to_binary, calc_data_size
from ..compat import functools32
from ..errors import StoreFull, StoreKeyExists


logger = logging.getLogger(__name__)


class PlasmaChunkStore(object):
    """
    Wrapper of plasma client for Mars objects
    """
    def __init__(self, plasma_client, kv_store_ref):
        from ..serialize.dataserializer import mars_serialize_context

        self._plasma_client = plasma_client
        self._kv_store_ref = kv_store_ref
        self._actual_size = None
        self._serialize_context = mars_serialize_context()

    def get_actual_capacity(self):
        """
        Get actual capacity of plasma store
        :return: actual storage size in bytes
        """
        if self._actual_size is None:
            from pyarrow import plasma, lib

            bufs = []
            total_size = 0
            allocate_unit = 4 * 1024 * 1024
            try:
                while True:
                    obj_id = plasma.ObjectID.from_random()
                    bufs.append(self._plasma_client.create(obj_id, allocate_unit))
                    self._plasma_client.seal(obj_id)
                    total_size += allocate_unit
            except lib.PlasmaStoreFull:
                pass
            del bufs
            self._plasma_client.evict(total_size)
            self._actual_size = total_size
        return self._actual_size

    @staticmethod
    @functools32.lru_cache(100)
    def _calc_object_id(session_id, chunk_key):
        """
        Calc unique object id for chunks
        """
        from pyarrow.plasma import ObjectID
        key = '%s#%s' % (session_id, chunk_key)
        digest = hashlib.md5(to_binary(key)).digest()
        return ObjectID(digest + digest[:4])

    def create(self, session_id, chunk_key, size):
        from pyarrow.lib import PlasmaStoreFull, PlasmaObjectExists

        obj_id = self._calc_object_id(session_id, chunk_key)
        try:
            buffer = self._plasma_client.create(obj_id, size)
            return buffer
        except PlasmaStoreFull:
            exc_type = PlasmaStoreFull
            logger.warning('Chunk %s(%d) failed to store to plasma due to StoreFullError',
                           chunk_key, size)
        except PlasmaObjectExists:
            exc_type = PlasmaObjectExists
            logger.warning('Chunk %s(%d) already exists in plasma store', chunk_key, size)

        if exc_type is PlasmaStoreFull:
            raise StoreFull
        elif exc_type is PlasmaObjectExists:
            raise StoreKeyExists

    def seal(self, session_id, chunk_key):
        from pyarrow.lib import PlasmaObjectNonexistent
        obj_id = self._calc_object_id(session_id, chunk_key)
        try:
            self._plasma_client.seal(obj_id)
        except PlasmaObjectNonexistent:
            raise KeyError('(%r, %r)' % (session_id, chunk_key))

    def get(self, session_id, chunk_key):
        """
        Get deserialized Mars object from plasma store
        """
        from pyarrow.plasma import ObjectNotAvailable

        obj_id = self._calc_object_id(session_id, chunk_key)
        obj = self._plasma_client.get(obj_id, serialization_context=self._serialize_context, timeout_ms=10)
        if obj is ObjectNotAvailable:
            raise KeyError('(%r, %r)' % (session_id, chunk_key))
        return obj

    def get_buffer(self, session_id, chunk_key):
        """
        Get raw buffer from plasma store
        """
        obj_id = self._calc_object_id(session_id, chunk_key)
        [buf] = self._plasma_client.get_buffers([obj_id], timeout_ms=10)
        if buf is None:
            raise KeyError('(%r, %r)' % (session_id, chunk_key))
        return buf

    def get_actual_size(self, session_id, chunk_key):
        """
        Get actual size of Mars object from plasma store
        """
        buf = None
        try:
            obj_id = self._calc_object_id(session_id, chunk_key)
            [buf] = self._plasma_client.get_buffers([obj_id], timeout_ms=10)
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
        from pyarrow.lib import PlasmaStoreFull, PlasmaObjectExists
        from ..serialize.dataserializer import DataTuple

        data_size = calc_data_size(value)
        if isinstance(value, tuple):
            value = DataTuple(*value)

        obj_id = self._calc_object_id(session_id, chunk_key)

        try:
            serialized = pyarrow.serialize(value, self._serialize_context)
            try:
                buffer = self._plasma_client.create(obj_id, serialized.total_bytes)
                stream = pyarrow.FixedSizeBufferWriter(buffer)
                stream.set_memcopy_threads(6)
                serialized.write_to(stream)
                self._plasma_client.seal(obj_id)
            except PlasmaObjectExists:
                [buffer] = self._plasma_client.get_buffers([obj_id])
            finally:
                del serialized

            self._kv_store_ref.write('/sessions/%s/chunks/%s/data_size' % (session_id, chunk_key),
                                     data_size)
            return buffer
        except PlasmaStoreFull:
            logger.warning('Chunk %s(%d) failed to store to plasma due to StoreFullError',
                           chunk_key, data_size)
            exc = PlasmaStoreFull
        if exc is PlasmaStoreFull:
            raise StoreFull

    def contains(self, session_id, chunk_key):
        """
        Check if given chunk key exists in current plasma store
        """
        try:
            obj_id = self._calc_object_id(session_id, chunk_key)
            return self._plasma_client.contains(obj_id)
        except KeyError:
            return False
