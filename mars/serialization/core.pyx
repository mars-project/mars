# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

import datetime
import inspect
import sys
from cpython cimport PyObject
from functools import partial, wraps
from libc.stdint cimport uint32_t, int64_t, uintptr_t
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .._utils cimport TypeDispatcher
from ..utils import tokenize_int

import cloudpickle

if sys.version_info[:2] < (3, 8):
    try:
        import pickle5 as pickle  # nosec  # pylint: disable=import_pickle
    except ImportError:
        import pickle  # nosec  # pylint: disable=import_pickle
else:
    import pickle  # nosec  # pylint: disable=import_pickle

BUFFER_PICKLE_PROTOCOL = max(pickle.DEFAULT_PROTOCOL, 5)
cdef bint HAS_PICKLE_BUFFER = pickle.HIGHEST_PROTOCOL >= 5
cdef bint _PANDAS_HAS_MGR = hasattr(pd.Series([0]), "_mgr")


cdef TypeDispatcher _serial_dispatcher = TypeDispatcher()
cdef dict _deserializers = dict()

cdef uint32_t _MAX_STR_PRIMITIVE_LEN = 1024


cdef class Serializer:
    serializer_id = None

    cpdef serial(self, obj: Any, dict context):
        raise NotImplementedError

    cpdef deserial(self, tuple serialized, dict context, list subs):
        raise NotImplementedError

    @classmethod
    def calc_default_serializer_id(cls):
        return tokenize_int(f"{cls.__module__}.{cls.__qualname__}") & 0x7fffffff

    @classmethod
    def register(cls, obj_type):
        inst = cls()
        if (
            cls.serializer_id is None
            or cls.serializer_id == getattr(super(cls, cls), "serializer_id")
        ):
            # a class should have its own serializer_id without inheritance
            cls.serializer_id = cls.calc_default_serializer_id()
        _serial_dispatcher.register(obj_type, inst)
        _deserializers[cls.serializer_id] = inst

    @classmethod
    def unregister(cls, obj_type):
        _serial_dispatcher.unregister(obj_type)
        _deserializers.pop(cls.serializer_id, None)


cdef inline uint32_t _short_id(object obj) nogil:
    cdef void* ptr = <PyObject*>obj
    return (<uintptr_t>ptr) & <uint32_t>0xffffffff


def short_id(obj):
    return _short_id(obj)


def buffered(func):
    @wraps(func)
    def wrapped(self, obj: Any, dict context):
        cdef uint32_t short_id = _short_id(obj)
        if short_id in context:
            return Placeholder(_short_id(obj))
        else:
            context[short_id] = obj
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
        if hasattr(result, "_mgr") and isinstance(result, (pd.DataFrame, pd.Series)):
            result._data = getattr(result, "_mgr")
            delattr(result, "_mgr")
        return result


cdef class PickleSerializer(Serializer):
    serializer_id = 0

    cpdef serial(self, obj: Any, dict context):
        cdef uint32_t obj_id
        obj_id = _short_id(obj)
        if obj_id in context:
            return Placeholder(obj_id)
        context[obj_id] = obj

        return (), pickle_buffers(obj), True

    cpdef deserial(self, tuple serialized, dict context, list subs):
        return unpickle_buffers(subs)


cdef set _primitive_types = {
    type(None),
    bool,
    int,
    float,
    complex,
    datetime.datetime,
    datetime.date,
    datetime.timedelta,
    type(max),  # builtin functions
    np.dtype,
    np.number,
}


class PrimitiveSerializer(Serializer):
    serializer_id = 1

    @buffered
    def serial(self, obj: Any, context: Dict):
        return (obj,), [], True

    def deserial(self, tuple obj, context: Dict, subs: List[Any]):
        return obj[0]


cdef class BytesSerializer(Serializer):
    serializer_id = 2

    cpdef serial(self, obj: Any, dict context):
        cdef uint32_t obj_id
        obj_id = _short_id(obj)
        if obj_id in context:
            return Placeholder(obj_id)
        context[obj_id] = obj

        return (), [obj], True

    cpdef deserial(self, tuple serialized, dict context, list subs):
        return subs[0]


cdef class StrSerializer(Serializer):
    serializer_id = 3

    cpdef serial(self, obj: Any, dict context):
        cdef uint32_t obj_id
        obj_id = _short_id(obj)
        if obj_id in context:
            return Placeholder(obj_id)
        context[obj_id] = obj

        return (), [(<str>obj).encode()], True

    cpdef deserial(self, tuple serialized, dict context, list subs):
        buffer = subs[0]
        if type(buffer) is memoryview:
            buffer = buffer.tobytes()
        return buffer.decode()


cdef class CollectionSerializer(Serializer):
    obj_type = None

    cpdef tuple _serial_iterable(self, obj: Any):
        cdef list idx_to_propagate = []
        cdef list obj_to_propagate = []
        cdef list obj_list = list(obj)
        cdef int64_t idx

        for idx in range(len(obj_list)):
            item = obj_list[idx]
            item_type = type(item)

            if (
                (item_type is bytes or item_type is str)
                and len(<str>item) < _MAX_STR_PRIMITIVE_LEN
            ):
                # treat short strings as primitives
                continue
            elif item_type in _primitive_types:
                continue

            obj_list[idx] = None
            idx_to_propagate.append(idx)
            obj_to_propagate.append(item)

        if self.obj_type is not None and type(obj) is not self.obj_type:
            obj_type = type(obj)
        else:
            obj_type = None
        return (obj_list, idx_to_propagate, obj_type), obj_to_propagate, False

    cpdef serial(self, obj: Any, dict context):
        cdef uint32_t obj_id
        obj_id = _short_id(obj)
        if obj_id in context:
            return Placeholder(obj_id)
        context[obj_id] = obj

        return self._serial_iterable(obj)

    cpdef list _deserial_iterable(self, tuple serialized, list subs):
        cdef list res_list, idx_to_propagate
        cdef int64_t i

        res_list, idx_to_propagate, _ = serialized

        for i in range(len(idx_to_propagate)):
            res_list[idx_to_propagate[i]] = subs[i]
        return res_list


cdef class TupleSerializer(CollectionSerializer):
    serializer_id = 4
    obj_type = tuple

    cpdef deserial(self, tuple serialized, dict context, list subs):
        cdef list res = self._deserial_iterable(serialized, subs)
        for v in res:
            assert type(v) is not Placeholder

        obj_type = serialized[-1] or tuple
        if hasattr(obj_type, "_fields"):
            # namedtuple
            return obj_type(*res)
        else:
            return obj_type(res)


cdef class ListSerializer(CollectionSerializer):
    serializer_id = 5
    obj_type = list

    cpdef deserial(self, tuple serialized, dict context, list subs):
        cdef int64_t idx
        cdef list res = self._deserial_iterable(serialized, subs)

        obj_type = serialized[-1]
        if obj_type is None:
            result = res
        else:
            result = obj_type(res)

        for idx, v in enumerate(res):
            if type(v) is Placeholder:
                (<Placeholder>v).callbacks.append(
                    partial(result.__setitem__, idx)
                )
        return result


def _dict_key_replacer(ret, key, real_key):
    ret[real_key] = ret.pop(key)


def _dict_value_replacer(context, ret, key, real_value):
    if type(key) is Placeholder:
        key = context[(<Placeholder>key).id]
    ret[key] = real_value


cdef class DictSerializer(CollectionSerializer):
    serializer_id = 6
    _inspected_inherits = set()

    cpdef serial(self, obj: Any, dict context):
        cdef uint32_t obj_id
        cdef tuple key_obj, value_obj
        cdef list key_bufs, value_bufs

        obj_id = _short_id(obj)
        if obj_id in context:
            return Placeholder(obj_id)
        context[obj_id] = obj

        obj_type = type(obj)
        if obj_type is not dict and obj_type not in self._inspected_inherits:
            inspect_init = inspect.getfullargspec(obj_type.__init__)
            if (
                inspect_init.args == ["self"]
                and not inspect_init.varargs
                and not inspect_init.varkw
            ):
                # dict inheritance
                # remove context to generate real serialized result
                context.pop(obj_id)
                return (obj,), [], True
            else:
                self._inspected_inherits.add(obj_type)

        key_obj, key_bufs, _ = self._serial_iterable(obj.keys())
        value_obj, value_bufs, _ = self._serial_iterable(obj.values())
        if type(obj) is not dict:
            obj_type = type(obj)
        else:
            obj_type = None
        ser_obj = (key_obj[:-1], value_obj[:-1], len(key_bufs), obj_type)
        return ser_obj, key_bufs + value_bufs, False

    cpdef deserial(self, tuple serialized, dict context, list subs):
        cdef int64_t i, num_key_bufs
        cdef list key_subs, value_subs, keys, values

        if len(serialized) == 1:
            # serialized directly
            return serialized[0]

        key_serialized, value_serialized, num_key_bufs, obj_type = serialized
        key_subs = subs[:num_key_bufs]
        value_subs = subs[num_key_bufs:]

        keys = self._deserial_iterable(<tuple>key_serialized + (None,), key_subs)
        values = self._deserial_iterable(<tuple>value_serialized + (None,), value_subs)

        if obj_type is None:
            ret = dict(zip(keys, values))
        else:
            try:
                ret = obj_type(zip(keys, values))
            except TypeError:
                # defaultdict
                ret = obj_type()
                ret.update(zip(keys, values))

        for i in range(len(keys)):
            k, v = keys[i], values[i]
            if type(k) is Placeholder:
                (<Placeholder>k).callbacks.append(
                    partial(_dict_key_replacer, ret, k)
                )
            if type(v) is Placeholder:
                (<Placeholder>v).callbacks.append(
                    partial(_dict_value_replacer, context, ret, k)
                )
        return ret


cdef class Placeholder:
    cpdef public uint32_t id
    cpdef public list callbacks

    def __init__(self, uint32_t id_):
        self.id = id_
        self.callbacks = []

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):  # pragma: no cover
        if type(other) is not Placeholder:
            return False
        return self.id == other.id


cdef class PlaceholderSerializer(Serializer):
    serializer_id = 7

    cpdef serial(self, obj: Any, dict context):
        return ((<Placeholder>obj).id,), [], True

    cpdef deserial(self, tuple serialized, dict context, list subs):
        obj_id = serialized[0]
        try:
            return context[obj_id]
        except KeyError:
            return Placeholder(obj_id)


PickleSerializer.register(object)
for _primitive in _primitive_types:
    PrimitiveSerializer.register(_primitive)
BytesSerializer.register(bytes)
StrSerializer.register(str)
ListSerializer.register(list)
TupleSerializer.register(tuple)
DictSerializer.register(dict)
PlaceholderSerializer.register(Placeholder)


cdef class _SerialStackItem:
    cdef public tuple serialized
    cdef public list subs
    cdef public list subs_serialized

    def __cinit__(self, tuple serialized, list subs):
        self.serialized = serialized
        self.subs = subs
        self.subs_serialized = []


cdef tuple _serial_single(obj, dict context):
    cdef uint32_t obj_id
    cdef Serializer serializer
    cdef tuple common_header, serialized

    while True:
        serializer = _serial_dispatcher.get_handler(type(obj))
        ret_serial = serializer.serial(obj, context)
        if type(ret_serial) is tuple:
            serialized, subs, final = <tuple>ret_serial

            if type(obj) is Placeholder:
                obj_id = (<Placeholder>obj).id
            else:
                obj_id = _short_id(obj)

            common_header = (
                serializer.serializer_id, obj_id, len(subs), final
            )
            break
        else:
            obj = ret_serial
    return common_header + serialized, subs, final


def serialize(obj, dict context = None):
    cdef list serial_stack = []
    cdef _SerialStackItem stack_item
    cdef list result_bufs_list = []
    cdef tuple serialized
    cdef list subs
    cdef bint final
    cdef int64_t num_serialized

    context = context if context is not None else dict()
    serialized, subs, final = _serial_single(obj, context)
    if final or not subs:
        return ({}, serialized), subs

    serial_stack.append(_SerialStackItem(serialized, subs))
    serialized = None

    while serial_stack:
        stack_item = serial_stack[-1]
        if serialized is not None:
            stack_item.subs_serialized.append(serialized)
        num_serialized = len(stack_item.subs_serialized)
        if len(stack_item.subs) == num_serialized:
            serialized = stack_item.serialized + tuple(stack_item.subs_serialized)
            serial_stack.pop()
        else:
            serialized, subs, final = _serial_single(
                stack_item.subs[num_serialized], context
            )
            if final or not subs:
                if subs:
                    result_bufs_list.extend(subs)
            else:
                stack_item = _SerialStackItem(serialized, subs)
                serial_stack.append(stack_item)
                serialized = None
    return ({}, serialized), result_bufs_list


cdef class _DeserialStackItem:
    cdef public tuple serialized
    cdef public tuple subs
    cdef public list subs_deserialized

    def __cinit__(self, tuple serialized, tuple subs):
        self.serialized = serialized
        self.subs = subs
        self.subs_deserialized = []


cdef void _fill_placeholders(dict context, obj_id, result):
    context_val, context[obj_id] = context.get(obj_id), result
    if type(context_val) is Placeholder:
        for cb in (<Placeholder>context_val).callbacks:
            cb(result)


cdef _deserial_single(tuple serialized, dict context, list subs):
    cdef Serializer serializer
    cdef int64_t num_subs

    serializer_id, obj_id, num_subs, final = serialized[:4]
    serializer = _deserializers[serializer_id]
    res = serializer.deserial(serialized[4:], context, subs)

    _fill_placeholders(context, obj_id, res)
    return res


def deserialize(tuple serialized, list buffers, dict context = None):
    cdef list deserial_stack = []
    cdef _DeserialStackItem stack_item
    cdef int64_t num_subs, num_deserialized, buf_pos = 0
    cdef bint final
    cdef Serializer serializer
    cdef object deserialized = None

    context = context if context is not None else dict()
    # drop extra prop field
    serialized = serialized[-1]
    serializer_id, obj_id, num_subs, final = serialized[:4]
    if final or num_subs == 0:
        return _deserial_single(serialized, context, buffers)

    deserial_stack.append(
        _DeserialStackItem(
            serialized[:-num_subs], serialized[-num_subs:]
        )
    )

    while deserial_stack:
        stack_item = deserial_stack[-1]
        if deserialized is not None:
            stack_item.subs_deserialized.append(deserialized)
        num_deserialized = len(stack_item.subs_deserialized)
        if len(stack_item.subs) == num_deserialized:
            deserialized = _deserial_single(
                stack_item.serialized, context, stack_item.subs_deserialized
            )
            deserial_stack.pop()
        else:
            serialized = stack_item.subs[num_deserialized]
            serializer_id, obj_id, num_subs, final = serialized[:4]
            if final or num_subs == 0:
                deserialized = _deserial_single(
                    serialized, context, buffers[buf_pos : buf_pos + num_subs]
                )
                buf_pos += num_subs
            else:
                stack_item = _DeserialStackItem(
                    serialized[:-num_subs], serialized[-num_subs:]
                )
                deserial_stack.append(stack_item)
                deserialized = None
    return deserialized
