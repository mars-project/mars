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

import inspect
import sys
import types
from functools import partial, wraps
from typing import Any, Dict, List

import pandas as pd

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
_PANDAS_HAS_MGR = hasattr(pd.Series([0]), '_mgr')


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

    @staticmethod
    def unregister(obj_type):
        handler = _serial_dispatcher.get_handler(obj_type)
        _serial_dispatcher.unregister(obj_type)
        _deserializers.pop(handler.__class__.serializer_name, None)

    @classmethod
    def get_registered_types(cls):
        return _serial_dispatcher.get_registered_types()


def buffered(func):
    @wraps(func)
    def wrapped(self, obj: Any, context: Dict):
        if id(obj) in context:
            return {
                       'id': id(obj),
                       'serializer': 'ref',
                       'buf_num': 0,
                   }, []
        else:
            context[id(obj)] = obj
            return func(self, obj, context)
    return wrapped


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
    result = cloudpickle.loads(buffers[0], buffers=buffers[1:])

    # as pandas prior to 1.1.0 use _data instead of _mgr to hold BlockManager,
    # deserializing from high versions may produce mal-functioned pandas objects,
    # thus the patch is needed
    if _PANDAS_HAS_MGR:
        return result
    else:  # pragma: no cover
        if hasattr(result, '_mgr') and isinstance(result, (pd.DataFrame, pd.Series)):
            result._data = getattr(result, '_mgr')
            delattr(result, '_mgr')
        return result


class ScalarSerializer(Serializer):
    serializer_name = 'scalar'

    def serialize(self, obj: Any, context: Dict):
        header = {'val': obj}
        return header, []

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        return header['val']


class StrSerializer(Serializer):
    serializer_name = 'str'

    @buffered
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

    @buffered
    def serialize(self, obj, context: Dict):
        return {}, pickle_buffers(obj)

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        return unpickle_buffers(buffers)


class CollectionSerializer(Serializer):
    obj_type = None

    @staticmethod
    def _serialize(c):
        headers = [None] * len(c)
        buffers_list = [None] * len(c)
        for idx, obj in enumerate(c):
            header, buffers = yield obj
            headers[idx] = header
            buffers_list[idx] = buffers
        return headers, buffers_list

    @buffered
    def serialize(self, obj: Any, context: Dict):
        buffers = []
        headers_list, buffers_list = yield from self._serialize(obj)
        for b in buffers_list:
            buffers.extend(b)
        headers = {'headers': headers_list}
        if type(obj) is not self.obj_type:
            headers['obj_type'] = pickle.dumps(type(obj))
        return headers, buffers

    @staticmethod
    def _list_deserial(headers: Dict, buffers: List):
        pos = 0
        ret = [None] * len(headers)
        for idx, sub_header in enumerate(headers):
            buf_num = sub_header['buf_num']
            sub_buffers = buffers[pos:pos + buf_num]
            ret[idx] = yield sub_header, sub_buffers
            pos += buf_num
        return ret

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        obj_type = self.obj_type
        if 'obj_type' in header:
            obj_type = pickle.loads(header['obj_type'])
        if hasattr(obj_type, '_fields'):
            # namedtuple
            return obj_type(*(yield from self._list_deserial(header['headers'], buffers)))
        else:
            return obj_type((yield from self._list_deserial(header['headers'], buffers)))


class ListSerializer(CollectionSerializer):
    serializer_name = 'list'
    obj_type = list

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        ret = yield from super().deserialize(header, buffers, context)
        for idx, v in enumerate(ret):
            if isinstance(v, Placeholder):
                v.callbacks.append(partial(ret.__setitem__, idx))
        return ret


class TupleSerializer(CollectionSerializer):
    serializer_name = 'tuple'
    obj_type = tuple

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        ret = yield from super().deserialize(header, buffers, context)
        assert not any(isinstance(v, Placeholder) for v in ret)
        return ret


class DictSerializer(CollectionSerializer):
    serializer_name = 'dict'
    _inspected_inherits = set()

    @buffered
    def serialize(self, obj: Dict, context: Dict):
        obj_type = type(obj)
        if obj_type is not dict and obj_type not in self._inspected_inherits:
            inspect_init = inspect.getfullargspec(obj_type.__init__)
            if inspect_init.args == ['self'] and not inspect_init.varargs \
                    and not inspect_init.varkw:
                # dict inheritance
                # remove context to generate real serialized result
                context.pop(id(obj))
                PickleSerializer.register(obj_type)
                return (yield obj)
            else:
                self._inspected_inherits.add(obj_type)

        key_headers, key_buffers_list = yield from self._serialize(obj.keys())
        value_headers, value_buffers_list = yield from self._serialize(obj.values())

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

        keys = yield from self._list_deserial(header['key_headers'], key_buffers)
        values = yield from self._list_deserial(header['value_headers'], value_buffers)

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
ScalarSerializer.register(type(None))
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
    # todo remove this when gevent dependency removed
    # workaround for traceback pickling error
    from ..lib.tblib import pickling_support
    pickling_support.install()

    def _wrap_headers(_obj, _serializer_name, _header, _buffers):
        if _header.get('serializer') == 'ref':
            return _header, _buffers
        # if serializer already defined, do not change
        _header['serializer'] = _header.get('serializer', _serializer_name)
        _header['buf_num'] = len(_buffers)
        _header['id'] = id(_obj)
        return _header, _buffers

    context = context if context is not None else dict()

    serializer = _serial_dispatcher.get_handler(type(obj))
    result = serializer.serialize(obj, context)


    if not isinstance(result, types.GeneratorType):
        # result is not a generator, return directly
        header, buffers = result
        return _wrap_headers(obj, serializer.serializer_name, header, buffers)
    else:
        # result is a generator, iter it till final result
        gen_stack = [(result, obj, serializer.serializer_name)]
        last_serial = None
        while gen_stack:
            gen, call_obj, ser_name = gen_stack[-1]
            try:
                gen_to_serial = gen.send(last_serial)
                gen_serializer = _serial_dispatcher.get_handler(type(gen_to_serial))
                gen_result = gen_serializer.serialize(gen_to_serial, context)
                if isinstance(gen_result, types.GeneratorType):
                    # when intermediate result still generator, push its contexts
                    # into stack and handle it first
                    gen_stack.append(
                        (gen_result, gen_to_serial, gen_serializer.serializer_name)
                    )
                    # result need to be emptied to run the generator
                    last_serial = None
                else:
                    # when intermediate result is not generator, pass it
                    # to the generator again
                    last_serial = _wrap_headers(
                        gen_to_serial, gen_serializer.serializer_name, *gen_result)
            except StopIteration as si:
                # when current generator finishes, jump to the previous one
                # and pass final result to it
                gen_stack.pop()
                last_serial = _wrap_headers(call_obj, ser_name, *si.value)

        return last_serial


def deserialize(header: Dict, buffers: List, context: Dict = None):
    def _deserialize(_header, _buffers):
        serializer_name = _header['serializer']
        obj_id = _header['id']
        if serializer_name == 'ref':
            try:
                result = context[obj_id]
            except KeyError:
                result = context[obj_id] = Placeholder(obj_id)
        else:
            serializer = _deserializers[serializer_name]
            result = serializer.deserialize(_header, _buffers, context)
            if not isinstance(result, types.GeneratorType):
                _fill_context(obj_id, result)
        return result

    def _fill_context(obj_id, result):
        context_val, context[obj_id] = context.get(obj_id), result
        if isinstance(context_val, Placeholder):
            for cb in context_val.callbacks:
                cb(result)

    context = context if context is not None else dict()

    deserialized = _deserialize(header, buffers)
    if not isinstance(deserialized, types.GeneratorType):
        # result is not a generator, return directly
        return deserialized
    else:
        # result is a generator, iter it till final result
        gen_stack = [(deserialized, (header, buffers))]
        last_deserial = None
        while gen_stack:
            gen, to_deserial = gen_stack[-1]
            try:
                gen_to_deserial = gen.send(last_deserial)
                gen_deserialized = _deserialize(*gen_to_deserial)
                if isinstance(gen_deserialized, types.GeneratorType):
                    # when intermediate result still generator, push its contexts
                    # into stack and handle it first
                    gen_stack.append(
                        (gen_deserialized, gen_to_deserial)
                    )
                    # result need to be emptied to run the generator
                    last_deserial = None
                else:
                    # when intermediate result is not generator, pass it
                    # to the generator again
                    last_deserial = gen_deserialized
            except StopIteration as si:
                # when current generator finishes, jump to the previous one
                # and pass final result to it
                gen_stack.pop()
                last_deserial = si.value
                # remember to fill Placeholders when some result is generated
                _fill_context(to_deserial[0]['id'], last_deserial)
        return last_deserial
