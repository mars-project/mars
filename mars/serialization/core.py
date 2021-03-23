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
from functools import partial
from typing import Any, Dict, List

from ..utils import TypeDispatcher

import cloudpickle
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

    def serialize(self, obj: Any, context: Dict):
        raise NotImplementedError

    def deserialize(self, header: Dict, buffers: List, context: Dict):
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

        buffers[0] = cloudpickle.dumps(
            obj,
            buffer_callback=buffer_cb,
            protocol=BUFFER_PICKLE_PROTOCOL,
        )
    else:  # pragma: no cover
        buffers[0] = cloudpickle.dumps(obj)
    return buffers


def unpickle_buffers(buffers):
    return cloudpickle.loads(buffers[0], buffers=buffers[1:])


class ScalarSerializer(Serializer):
    serializer_name = 'scalar'

    def serialize(self, obj: Any, context: Dict):
        header = {'val': obj}
        return header, []

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        return header['val']


class StrSerializer(Serializer):
    serializer_name = 'str'

    def serialize(self, obj, context: Dict):
        header = {}
        if isinstance(obj, str):
            header['unicode'] = True
            bytes_data = obj.encode()
        else:
            bytes_data = obj
        return header, [bytes_data]

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        if header.get('unicode'):
            return buffers[0].decode()
        return buffers[0]


class PickleSerializer(Serializer):
    serializer_name = 'pickle'

    def serialize(self, obj, context: Dict):
        return {}, pickle_buffers(obj)

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        return unpickle_buffers(buffers)


class CollectionSerializer(Serializer):
    obj_type = None

    @staticmethod
    def _serialize(c, context: Dict):
        headers = []
        buffers_list = []
        for obj in c:
            header, buffers = serialize(obj, context)
            headers.append(header)
            buffers_list.append(buffers)
        return headers, buffers_list

    def serialize(self, obj: Any, context: Dict):
        buffers = []
        headers_list, buffers_list = self._serialize(obj, context)
        for b in buffers_list:
            buffers.extend(b)
        headers = {'headers': headers_list}
        if type(obj) is not self.obj_type:
            headers['obj_type'] = pickle.dumps(type(obj))
        return headers, buffers

    def _iter_deserial(self, headers: Dict, buffers: List, context: Dict):
        pos = 0
        for sub_header in headers:
            buf_num = sub_header['buf_num']
            sub_buffers = buffers[pos:pos + buf_num]
            yield deserialize(sub_header, sub_buffers, context)
            pos += buf_num

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        obj_type = self.obj_type
        if 'obj_type' in header:
            obj_type = pickle.loads(header['obj_type'])
        if hasattr(obj_type, '_fields'):
            # namedtuple
            return obj_type(*self._iter_deserial(header['headers'], buffers, context))
        else:
            return obj_type(self._iter_deserial(header['headers'], buffers, context))


class ListSerializer(CollectionSerializer):
    serializer_name = 'list'
    obj_type = list

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        ret = super().deserialize(header, buffers, context)
        for idx, v in enumerate(ret):
            if isinstance(v, Placeholder):
                v.callbacks.append(partial(ret.__setitem__, idx))
        return ret


class TupleSerializer(CollectionSerializer):
    serializer_name = 'tuple'
    obj_type = tuple

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        ret = super().deserialize(header, buffers, context)
        assert not any(isinstance(v, Placeholder) for v in ret)
        return ret


class DictSerializer(CollectionSerializer):
    serializer_name = 'dict'

    def serialize(self, obj: Dict, context: Dict):
        key_headers, key_buffers_list = self._serialize(obj.keys(), context)
        value_headers, value_buffers_list = self._serialize(obj.values(), context)

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

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        key_buffers = buffers[:header['key_buf_num']]
        value_buffers = buffers[header['key_buf_num']:]

        obj_type = dict
        if 'obj_type' in header:
            obj_type = pickle.loads(header['obj_type'])

        keys = list(self._iter_deserial(
            header['key_headers'], key_buffers, context))
        values = list(self._iter_deserial(
            header['value_headers'], value_buffers, context))

        def _key_replacer(key, real_key):
            ret[real_key] = ret.pop(key)

        def _value_replacer(key, real_value):
            if isinstance(key, Placeholder):
                key = context[key.id]
            ret[key] = real_value

        ret = obj_type(zip(keys, values))
        for k, v in zip(keys, values):
            if isinstance(k, Placeholder):
                k.callbacks.append(partial(_key_replacer, k))
            if isinstance(v, Placeholder):
                v.callbacks.append(partial(_value_replacer, k))
        return ret


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


class Placeholder:
    id: int
    callbacks: List

    def __init__(self, id_: int):
        self.id = id_
        self.callbacks = []

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):  # pragma: no cover
        if not isinstance(other, Placeholder):
            return False
        return self.id == other.id


def serialize(obj, context: Dict = None):
    serializer = _serial_dispatcher.get_handler(type(obj))
    context = context if context is not None else dict()

    if id(obj) in context:
        return {
            'id': id(obj),
            'serializer': 'ref',
            'buf_num': 0,
        }, []
    else:
        context[id(obj)] = obj

    header, buffers = serializer.serialize(obj, context)
    header['serializer'] = serializer.serializer_name
    header['buf_num'] = len(buffers)
    header['id'] = id(obj)
    return header, buffers


def deserialize(header: Dict, buffers: List, context: Dict = None):
    context = context if context is not None else dict()

    serializer_name = header['serializer']
    obj_id = header['id']
    if serializer_name == 'ref':
        if obj_id not in context:
            context[obj_id] = Placeholder(obj_id)
    else:
        serializer = _deserializers[serializer_name]
        deserialized = serializer.deserialize(header, buffers, context)
        context_val, context[obj_id] = context.get(obj_id), deserialized
        if isinstance(context_val, Placeholder):
            for cb in context_val.callbacks:
                cb(deserialized)
    return context[obj_id]
