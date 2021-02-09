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

import sys
from typing import Dict, List

from ..utils import TypeDispatcher

if sys.version_info[:2] < (3, 8):
    try:
        import pickle5 as pickle  # nosec  # pylint: disable=import_pickle
    except ImportError:
        import pickle  # nosec  # pylint: disable=import_pickle
else:
    import pickle  # nosec  # pylint: disable=import_pickle

HAS_PICKLE_BUFFER = pickle.HIGHEST_PROTOCOL >= 5
BUFFER_PICKLE_PROTOCOL = max(pickle.DEFAULT_PROTOCOL, 5)


_serial_dispatcher = TypeDispatcher()
_deserializers = dict()


class Serializer:
    serializer_name = None

    def serialize(self, obj):
        raise NotImplementedError

    def deserialize(self, header: Dict, buffers: List):
        raise NotImplementedError

    @classmethod
    def register(cls, obj_type):
        inst = cls()
        _serial_dispatcher.register(obj_type, inst)
        _deserializers[cls.serializer_name] = inst


def pickle_buffers(obj):
    buffers = [None]
    if HAS_PICKLE_BUFFER:
        def buffer_cb(x):
            x = x.raw()
            if x.ndim > 1:
                # ravel n-d memoryview
                x = x.cast(x.format)
            buffers.append(memoryview(x))

        buffers[0] = pickle.dumps(
            obj,
            buffer_callback=buffer_cb,
            protocol=BUFFER_PICKLE_PROTOCOL,
        )
    else:  # pragma: no cover
        buffers[0] = pickle.dumps(obj)
    return buffers


def unpickle_buffers(buffers):
    return pickle.loads(buffers[0], buffers=buffers[1:])


class ScalarSerializer(Serializer):
    serializer_name = 'scalar'

    def serialize(self, obj):
        header = {'val': obj}
        return header, []

    def deserialize(self, header: Dict, buffers: List):
        return header['val']


class StrSerializer(Serializer):
    serializer_name = 'str'

    def serialize(self, obj):
        header = {}
        if isinstance(obj, str):
            header['unicode'] = True
            bytes_data = obj.encode()
        else:
            bytes_data = obj
        return header, [bytes_data]

    def deserialize(self, header: Dict, buffers: List):
        if header.get('unicode'):
            return buffers[0].decode()
        return buffers[0]


class PickleSerializer(Serializer):
    serializer_name = 'pickle'

    def serialize(self, obj):
        return {}, pickle_buffers(obj)

    def deserialize(self, header: Dict, buffers: List):
        return unpickle_buffers(buffers)


class CollectionSerializer(Serializer):
    obj_type = None

    @staticmethod
    def _serialize(c):
        headers = []
        buffers_list = []
        for obj in c:
            header, buffers = serialize(obj)
            headers.append(header)
            buffers_list.append(buffers)
        return headers, buffers_list

    def serialize(self, obj):
        buffers = []
        headers_list, buffers_list = self._serialize(obj)
        for b in buffers_list:
            buffers.extend(b)
        headers = {'headers': headers_list}
        if type(obj) is not self.obj_type:
            headers['obj_type'] = pickle.dumps(type(obj))
        return headers, buffers

    def _iter_deserial(self, headers: Dict, buffers: List):
        pos = 0
        for sub_header in headers:
            buf_num = sub_header['buf_num']
            sub_buffers = buffers[pos:pos + buf_num]
            yield deserialize(sub_header, sub_buffers)
            pos += buf_num

    def deserialize(self, header: Dict, buffers: List):
        obj_type = self.obj_type
        if 'obj_type' in header:
            obj_type = pickle.loads(header['obj_type'])
        return obj_type(self._iter_deserial(header['headers'], buffers))


class ListSerializer(CollectionSerializer):
    serializer_name = 'list'
    obj_type = list


class TupleSerializer(CollectionSerializer):
    serializer_name = 'tuple'
    obj_type = tuple


class DictSerializer(CollectionSerializer):
    serializer_name = 'dict'

    def serialize(self, obj: Dict):
        key_headers, key_buffers_list = self._serialize(obj.keys())
        value_headers, value_buffers_list = self._serialize(obj.values())

        buffers = []
        for b in key_buffers_list:
            buffers.extend(b)
        key_buf_num = len(buffers)

        for b in value_buffers_list:
            buffers.extend(b)

        header = {'key_headers': key_headers, 'key_buf_num': key_buf_num,
                  'value_headers': value_headers}
        if type(obj) is not dict:
            header['obj_type'] = pickle.dumps(type(obj))

        return header, buffers

    def deserialize(self, header: Dict, buffers: List):
        key_buffers = buffers[:header['key_buf_num']]
        value_buffers = buffers[header['key_buf_num']:]

        obj_type = dict
        if 'obj_type' in header:
            obj_type = pickle.loads(header['obj_type'])

        return obj_type(zip(
            self._iter_deserial(header['key_headers'], key_buffers),
            self._iter_deserial(header['value_headers'], value_buffers),
        ))


PickleSerializer.register(object)
ScalarSerializer.register(bool)
ScalarSerializer.register(int)
ScalarSerializer.register(float)
ScalarSerializer.register(complex)
StrSerializer.register(bytes)
StrSerializer.register(str)
ListSerializer.register(list)
TupleSerializer.register(tuple)
DictSerializer.register(dict)


def serialize(obj):
    serializer = _serial_dispatcher.get_handler(type(obj))
    header, buffers = serializer.serialize(obj)
    header['serializer'] = serializer.serializer_name
    header['buf_num'] = len(buffers)
    return header, buffers


def deserialize(header: Dict, buffers: List):
    serializer = _deserializers[header.pop('serializer')]
    return serializer.deserialize(header, buffers)
