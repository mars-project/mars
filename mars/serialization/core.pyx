# distutils: language = c++
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

import asyncio
import datetime
import enum
import hashlib
import inspect
import sys
from functools import partial, wraps
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from cpython cimport PyObject
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t, uintptr_t
from libcpp.unordered_map cimport unordered_map

from .._utils cimport TypeDispatcher

import cloudpickle

if sys.version_info[:2] < (3, 8):  # pragma: no cover
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
# prime modulus for serializer ids
# use the largest prime number smaller than 32767
cdef int32_t _SERIALIZER_ID_PRIME = 32749


cdef class Serializer:
    serializer_id = None

    cpdef serial(self, object obj, dict context):
        """
        Returns intermediate serialization result of certain object.
        The returned value can be a Placeholder or a tuple comprising
        of three parts: a header, a group of subcomponents and
        a finalizing flag.
        
        * Header is a pickle-serializable tuple
        * Subcomponents are parts or buffers for iterative
          serialization.
        * Flag is a boolean value. If true, subcomponents should be 
          buffers (for instance, bytes, memory views, GPU buffers,
          etc.) that can be read and written directly. If false, 
          subcomponents will be serialized iteratively.
        
        Parameters
        ----------
        obj: Any
            Object to serialize
        context: Dict
            Serialization context to help creating Placeholder objects
            for reducing duplicated serialization

        Returns
        -------
        result: Placeholder | Tuple[Tuple, List, bool]
            Intermediate result of serialization
        """
        raise NotImplementedError

    cpdef deserial(self, tuple serialized, dict context, list subs):
        """
        Returns deserialized object given serialized headers and
        deserialized subcomponents.
        
        Parameters
        ----------
        serialized: Tuple
            Serialized object header as a tuple
        context
            Serialization context for instantiation of Placeholder
            objects
        subs: List
            Deserialized subcomponents

        Returns
        -------
        result: Any
            Deserialized objects
        """
        raise NotImplementedError

    cpdef on_deserial_error(
        self,
        tuple serialized,
        dict context,
        list subs_serialized,
        int error_index,
        object exc,
    ):
        """
        Returns rewritten exception when subcomponent deserialization fails
        
        Parameters
        ----------
        serialized: Tuple
            Serialized object header as a tuple
        context
            Serialization context for instantiation of Placeholder
            objects
        subs_serialized: List
            Serialized subcomponents
        error_index: int
            Index of subcomponent causing error
        exc: BaseException
            Exception raised
            
        Returns
        -------
        exc: BaseException | None
            Rewritten exception. If None, original exception is kept.
        """
        return None

    @classmethod
    def calc_default_serializer_id(cls):
        s = f"{cls.__module__}.{cls.__qualname__}"
        h = hashlib.md5(s.encode())
        return int(h.hexdigest(), 16) % _SERIALIZER_ID_PRIME

    @classmethod
    def register(cls, obj_type):
        inst = cls()
        if (
            cls.serializer_id is None
            or cls.serializer_id == getattr(super(cls, cls), "serializer_id", None)
        ):
            # a class should have its own serializer_id
            # inherited serializer_id not acceptable
            cls.serializer_id = cls.calc_default_serializer_id()
        _serial_dispatcher.register(obj_type, inst)
        if _deserializers.get(cls.serializer_id) is not None:
            assert type(_deserializers[cls.serializer_id]) is cls
        else:
            _deserializers[cls.serializer_id] = inst

    @classmethod
    def unregister(cls, obj_type):
        _serial_dispatcher.unregister(obj_type)
        _deserializers.pop(cls.serializer_id, None)


cdef inline uint64_t _fast_id(object obj) nogil:
    return <uintptr_t><PyObject*>obj


def fast_id(obj):
    """C version of id() used for serialization"""
    return _fast_id(obj)


def buffered(func):
    """
    Wrapper for serial() method to reduce duplicated serialization
    """
    @wraps(func)
    def wrapped(self, obj: Any, dict context):
        cdef uint64_t obj_id = _fast_id(obj)
        if obj_id in context:
            return Placeholder(_fast_id(obj))
        else:
            context[obj_id] = obj
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
        cdef uint64_t obj_id
        obj_id = _fast_id(obj)
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
    enum.Enum,
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
        cdef uint64_t obj_id
        obj_id = _fast_id(obj)
        if obj_id in context:
            return Placeholder(obj_id)
        context[obj_id] = obj

        return (), [obj], True

    cpdef deserial(self, tuple serialized, dict context, list subs):
        return subs[0]


cdef class StrSerializer(Serializer):
    serializer_id = 3

    cpdef serial(self, obj: Any, dict context):
        cdef uint64_t obj_id
        obj_id = _fast_id(obj)
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
        cdef uint64_t obj_id
        obj_id = _fast_id(obj)
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
                cb = partial(result.__setitem__, idx)
                (<Placeholder>v).callbacks.append(cb)
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
        cdef uint64_t obj_id
        cdef tuple key_obj, value_obj
        cdef list key_bufs, value_bufs

        obj_id = _fast_id(obj)
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
                # inherited dicts may not have proper initializers
                # for deserialization
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
                # first arg of defaultdict is a callable
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
    """
    Placeholder object to reduce duplicated serialization

    The object records object identifier and keeps callbacks
    to replace itself in parent objects.
    """
    cdef public uint64_t id
    cdef public list callbacks

    def __init__(self, uint64_t id_):
        self.id = id_
        self.callbacks = []

    def __hash__(self):
        return self.id

    def __eq__(self, other):  # pragma: no cover
        if type(other) is not Placeholder:
            return False
        return self.id == other.id

    def __repr__(self):
        return (
            f"Placeholder(id={self.id}, "
            f"callbacks=[list of {len(self.callbacks)}])"
        )


cdef class PlaceholderSerializer(Serializer):
    serializer_id = 7

    cpdef serial(self, obj: Any, dict context):
        return (), [], True

    cpdef deserial(self, tuple serialized, dict context, list subs):
        return Placeholder(0)


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


cdef class _IdContextHolder:
    cdef unordered_map[uint64_t, uint64_t] d


cdef int _COMMON_HEADER_LEN = 4


cdef tuple _serial_single(
    obj, dict context, _IdContextHolder id_context_holder
):
    """Serialize single object and return serialized tuples"""
    cdef uint64_t obj_id, ordered_id
    cdef Serializer serializer
    cdef tuple common_header, serialized

    while True:
        serializer = _serial_dispatcher.get_handler(type(obj))
        ret_serial = serializer.serial(obj, context)
        if type(ret_serial) is tuple:
            # object is serialized, form a common header and return
            serialized, subs, final = <tuple>ret_serial

            if type(obj) is Placeholder:
                obj_id = (<Placeholder>obj).id
                ordered_id = id_context_holder.d[obj_id]
            else:
                obj_id = _fast_id(obj)
                ordered_id = id_context_holder.d.size()
                id_context_holder.d[obj_id] = ordered_id

            # REMEMBER to change _COMMON_HEADER_LEN when content of
            # this header changed
            common_header = (
                serializer.serializer_id, ordered_id, len(subs), final
            )
            break
        else:
            # object is converted into another (usually a Placeholder)
            obj = ret_serial
    return common_header + serialized, subs, final


class _SerializeObjectOverflow(Exception):
    def __init__(self, tuple cur_serialized, int num_total_serialized):
        super(_SerializeObjectOverflow, self).__init__(cur_serialized)
        self.cur_serialized = cur_serialized
        self.num_total_serialized = num_total_serialized


cpdef object _serialize_with_stack(
    list serial_stack,
    tuple serialized,
    dict context,
    _IdContextHolder id_context_holder,
    list result_bufs_list,
    int64_t num_overflow = 0,
    int64_t num_total_serialized = 0,
):
    cdef _SerialStackItem stack_item
    cdef list subs
    cdef bint final
    cdef int64_t num_sub_serialized
    cdef bint is_resume = num_total_serialized > 0

    while serial_stack:
        stack_item = serial_stack[-1]
        if serialized is not None:
            # have previously-serialized results, record first
            stack_item.subs_serialized.append(serialized)

        num_sub_serialized = len(stack_item.subs_serialized)
        if len(stack_item.subs) == num_sub_serialized:
            # all subcomponents serialized, serialization of current is done
            # and we can move to the parent object
            serialized = stack_item.serialized + tuple(stack_item.subs_serialized)
            num_total_serialized += 1
            serial_stack.pop()
        else:
            # serialize next subcomponent at stack top
            serialized, subs, final = _serial_single(
                stack_item.subs[num_sub_serialized], context, id_context_holder
            )
            num_total_serialized += 1
            if final or not subs:
                # the subcomponent is a leaf
                if subs:
                    result_bufs_list.extend(subs)
            else:
                # the subcomponent has its own subcomponents, we push itself
                # into stack and process its children
                stack_item = _SerialStackItem(serialized, subs)
                serial_stack.append(stack_item)
                # note that the serialized header should not be recorded
                # as we are now processing the subcomponent itself
                serialized = None
        if 0 < num_overflow < num_total_serialized:
            raise _SerializeObjectOverflow(serialized, num_total_serialized)

    # we keep an empty dict for extra metas required for other modules
    if is_resume:
        # returns num of deserialized objects when resumed
        extra_meta = {"_N": num_total_serialized}
    else:
        # otherwise does not record the number to reduce result size
        extra_meta = {}
    return (extra_meta, serialized), result_bufs_list


def serialize(obj, dict context = None):
    """
    Serialize an object and return a header and buffers.
    Buffers are intended for zero-copy data manipulation.

    Parameters
    ----------
    obj: Any
        Object to serialize
    context:
        Serialization context for instantiation of Placeholder
        objects

    Returns
    -------
    result: Tuple[Tuple, List]
        Picklable header and buffers
    """
    cdef list serial_stack = []
    cdef list result_bufs_list = []
    cdef tuple serialized
    cdef list subs
    cdef bint final
    cdef _IdContextHolder id_context_holder = _IdContextHolder()

    context = context if context is not None else dict()
    serialized, subs, final = _serial_single(obj, context, id_context_holder)
    if final or not subs:
        # marked as a leaf node, return directly
        return ({}, serialized), subs

    serial_stack.append(_SerialStackItem(serialized, subs))
    return _serialize_with_stack(
        serial_stack, None, context, id_context_holder, result_bufs_list
    )


async def serialize_with_spawn(
    obj, dict context = None, int spawn_threshold = 100, object executor = None
):
    """
    Serialize an object and return a header and buffers.
    Buffers are intended for zero-copy data manipulation.

    Parameters
    ----------
    obj: Any
        Object to serialize
    context: Dict
        Serialization context for instantiation of Placeholder
        objects
    spawn_threshold: int
        Threshold to spawn into a ThreadPoolExecutor
    executor: ThreadPoolExecutor
        ThreadPoolExecutor to spawn rest serialization into

    Returns
    -------
    result: Tuple[Tuple, List]
        Picklable header and buffers
    """
    cdef list serial_stack = []
    cdef list result_bufs_list = []
    cdef tuple serialized
    cdef list subs
    cdef bint final
    cdef _IdContextHolder id_context_holder = _IdContextHolder()

    context = context if context is not None else dict()
    serialized, subs, final = _serial_single(obj, context, id_context_holder)
    if final or not subs:
        # marked as a leaf node, return directly
        return ({}, serialized), subs

    serial_stack.append(_SerialStackItem(serialized, subs))

    try:
        result = _serialize_with_stack(
            serial_stack, None, context, id_context_holder, result_bufs_list, spawn_threshold
        )
    except _SerializeObjectOverflow as ex:
        result = await asyncio.get_running_loop().run_in_executor(
            executor,
            _serialize_with_stack,
            serial_stack,
            ex.cur_serialized,
            context,
            id_context_holder,
            result_bufs_list,
            0,
            ex.num_total_serialized,
        )
    return result


cdef class _DeserialStackItem:
    cdef public tuple serialized
    cdef public tuple subs
    cdef public list subs_deserialized

    def __cinit__(self, tuple serialized, tuple subs):
        self.serialized = serialized
        self.subs = subs
        self.subs_deserialized = []


cdef _deserial_single(tuple serialized, dict context, list subs):
    """Deserialize a single object"""
    cdef Serializer serializer
    cdef int64_t num_subs

    serializer_id, obj_id, num_subs, final = serialized[:_COMMON_HEADER_LEN]
    serializer = _deserializers[serializer_id]
    res = serializer.deserial(serialized[_COMMON_HEADER_LEN:], context, subs)

    if type(res) is Placeholder:
        try:
            res = context[obj_id]
        except KeyError:
            (<Placeholder>res).id = obj_id

    # get previously-recorded context values
    context_val, context[obj_id] = context.get(obj_id), res
    # if previously recorded object is a Placeholder,
    # replace it with callbacks
    if type(context_val) is Placeholder:
        for cb in (<Placeholder>context_val).callbacks:
            cb(res)
    return res


def deserialize(tuple serialized, list buffers, dict context = None):
    """
    Deserialize an object with serialized headers and buffers

    Parameters
    ----------
    serialized: Tuple
        Serialized object header
    buffers: List
        List of buffers extracted from serialize() calls
    context: Dict
        Serialization context for replacing Placeholder
        objects

    Returns
    -------
    result: Any
        Deserialized object
    """
    cdef list deserial_stack = []
    cdef _DeserialStackItem stack_item
    cdef int64_t num_subs, num_deserialized, buf_pos = 0
    cdef bint final
    cdef Serializer serializer
    cdef object deserialized = None, exc_value = None
    cdef bint has_deserialized = False

    context = context if context is not None else dict()
    # drop extra meta field
    serialized = serialized[-1]
    serializer_id, obj_id, num_subs, final = serialized[:4]
    if final or num_subs == 0:
        # marked as a leaf node, return directly
        return _deserial_single(serialized, context, buffers)

    deserial_stack.append(
        _DeserialStackItem(
            serialized[:-num_subs], serialized[-num_subs:]
        )
    )

    while deserial_stack:
        stack_item = deserial_stack[-1]
        # the deserialized result can be None, hence we cannot
        # simply judge from the value deserialized
        if has_deserialized:
            # have previously-deserialized results, record first
            stack_item.subs_deserialized.append(deserialized)
        elif exc_value is not None:
            # have exception in successor components, try rewrite
            # and pass to predecessors
            serializer_id = stack_item.serialized[0]
            serializer = _deserializers[serializer_id]
            new_exc_value = serializer.on_deserial_error(
                stack_item.serialized[_COMMON_HEADER_LEN:],
                context,
                list(stack_item.subs),
                len(stack_item.subs_deserialized),
                exc_value,
            )
            exc_value = new_exc_value if new_exc_value is not None else exc_value
            deserial_stack.pop()
            continue

        num_deserialized = len(stack_item.subs_deserialized)
        if len(stack_item.subs) == num_deserialized:
            try:
                # all subcomponents deserialized, we can deserialize the object itself
                deserialized = _deserial_single(
                    stack_item.serialized, context, stack_item.subs_deserialized
                )
                has_deserialized = True
                deserial_stack.pop()
            except BaseException as ex:
                has_deserialized = False
                exc_value = ex
                deserial_stack.pop()
        else:
            # select next subcomponent to process
            serialized = stack_item.subs[num_deserialized]
            serializer_id, obj_id, num_subs, final = serialized[:4]
            if final or num_subs == 0:
                try:
                    # next subcomponent is a leaf, just deserialize
                    deserialized = _deserial_single(
                        serialized, context, buffers[buf_pos : buf_pos + num_subs]
                    )
                    has_deserialized = True
                    buf_pos += num_subs
                except BaseException as ex:
                    has_deserialized = False
                    exc_value = ex
            else:
                # next subcomponent has its own subcomponents, we push it
                # into stack and start handling its children
                stack_item = _DeserialStackItem(
                    serialized[:-num_subs], serialized[-num_subs:]
                )
                deserial_stack.append(stack_item)
                # note that the deserialized object should be cleaned
                # as we are just starting to handle the subcomponent itself
                has_deserialized = False

    if exc_value is not None:
        raise exc_value
    return deserialized
