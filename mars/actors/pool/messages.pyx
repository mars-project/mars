#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import errno
import socket
from pickle import dumps, loads

import numpy as np
cimport numpy as np
cimport cython
from libc.string cimport memcpy
from cpython.bytearray cimport PyByteArray_AS_STRING, PyByteArray_GET_SIZE, PyByteArray_Resize
from cpython.version cimport PY_MAJOR_VERSION

from ..core cimport ActorRef
from ...compat import BrokenPipeError, ConnectionResetError
from ...lib.tblib import pickling_support

pickling_support.install()

# Internal message types includes:
# 1) create actor
# 2) destroy actor
# 3) has actor
# 4) result that return back by the actor in a different process or machine
# 5) error that throws back by the actor in a different process or machine
# 6) send all, the send actor message
# 7) tell all, the tell actor message
cpdef BYTE_t CREATE_ACTOR = MessageType.create_actor
cpdef BYTE_t DESTROY_ACTOR = MessageType.destroy_actor
cpdef BYTE_t HAS_ACTOR = MessageType.has_actor
cpdef BYTE_t RESULT = MessageType.result
cpdef BYTE_t ERROR = MessageType.error
cpdef BYTE_t SEND_ALL = MessageType.send_all
cpdef BYTE_t TELL_ALL = MessageType.tell_all


cpdef enum MessageSerialType:
    # Internal message serialized type
    null = 0
    raw_bytes = 1
    pickle = 2


cdef BYTE_t NONE = MessageSerialType.null
cdef BYTE_t RAW_BYTES = MessageSerialType.raw_bytes
cdef BYTE_t PICKLE = MessageSerialType.pickle


cdef INT32_t DEFAULT_PROTOCOL = 0


cdef class _BASE_ACTOR_MESSAGE:
    cdef public INT32_t message_type
    cdef public bytes message_id
    cdef public INT32_t from_index
    cdef public INT32_t to_index


cdef class CREATE_ACTOR_MESSAGE(_BASE_ACTOR_MESSAGE):
    cdef public ActorRef actor_ref
    cdef public object actor_cls
    cdef public tuple args
    cdef public dict kwargs

    def __init__(self, INT32_t message_type=-1, bytes message_id=None,
                 INT32_t from_index=0, INT32_t to_index=0, ActorRef actor_ref=None,
                 object actor_cls=None, tuple args=None, dict kwargs=None):
        self.message_type = message_type
        self.message_id = message_id
        self.from_index = from_index
        self.to_index = to_index
        self.actor_ref = actor_ref
        self.actor_cls = actor_cls
        self.args = args
        self.kwargs = kwargs


cdef class DESTROY_ACTOR_MESSAGE(_BASE_ACTOR_MESSAGE):
    cdef public ActorRef actor_ref

    def __init__(self, INT32_t message_type=-1, bytes message_id=None,
                 INT32_t from_index=0, INT32_t to_index=0, object actor_ref=None):
        self.message_type = message_type
        self.message_id = message_id
        self.from_index = from_index
        self.to_index = to_index
        self.actor_ref = actor_ref


cdef class HAS_ACTOR_MESSAGE(_BASE_ACTOR_MESSAGE):
    cdef public ActorRef actor_ref

    def __init__(self, INT32_t message_type=-1, bytes message_id=None,
                 INT32_t from_index=0, INT32_t to_index=0, ActorRef actor_ref=None):
        self.message_type = message_type
        self.message_id = message_id
        self.from_index = from_index
        self.to_index = to_index
        self.actor_ref = actor_ref


cdef class RESULT_MESSAGE(_BASE_ACTOR_MESSAGE):
    cdef public object result

    def __init__(self, INT32_t message_type=-1, bytes message_id=None,
                 INT32_t from_index=0, INT32_t to_index=0, object result=None):
        self.message_type = message_type
        self.message_id = message_id
        self.from_index = from_index
        self.to_index = to_index
        self.result = result


cdef class ERROR_MESSAGE(_BASE_ACTOR_MESSAGE):
    cdef public object error_type
    cdef public object error
    cdef public object traceback

    def __init__(self, INT32_t message_type=-1, bytes message_id=None,
                 INT32_t from_index=0, INT32_t to_index=0, object error_type=None,
                 object error=None, object traceback=None):
        self.message_type = message_type
        self.message_id = message_id
        self.from_index = from_index
        self.to_index = to_index
        self.error_type = error_type
        self.error = error
        self.traceback = traceback


cdef class SEND_MESSAGE(_BASE_ACTOR_MESSAGE):
    cdef public ActorRef actor_ref
    cdef public object message

    def __init__(self, object message_type=-1, bytes message_id=None,
                 INT32_t from_index=0, INT32_t to_index=0, ActorRef actor_ref=None,
                 object message=None):
        self.message_type = message_type
        self.message_id = message_id
        self.from_index = from_index
        self.to_index = to_index
        self.actor_ref = actor_ref
        self.message = message


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _pack_byte(BYTE_t val, bytearray arr):
    cdef char *ptr
    cdef Py_ssize_t size = PyByteArray_GET_SIZE(arr)
    PyByteArray_Resize(arr, size + 1)
    ptr = PyByteArray_AS_STRING(arr) + size
    ptr[0] = val


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline BYTE_t _unpack_byte(const char *s) nogil:
    return s[0]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _pack_short(INT16_t val, bytearray arr):
    cdef char *ptr
    cdef Py_ssize_t size = PyByteArray_GET_SIZE(arr)
    PyByteArray_Resize(arr, size + sizeof(INT16_t))
    ptr = PyByteArray_AS_STRING(arr) + size
    memcpy(ptr, &val, sizeof(INT16_t))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline INT16_t _unpack_short(const char *s) nogil:
    cdef INT16_t ret
    memcpy(&ret, s, sizeof(INT16_t))
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _pack_int(INT32_t val, bytearray arr):
    cdef char *ptr
    cdef Py_ssize_t size = PyByteArray_GET_SIZE(arr)
    PyByteArray_Resize(arr, size + sizeof(INT32_t))
    ptr = PyByteArray_AS_STRING(arr) + size
    memcpy(ptr, &val, sizeof(INT32_t))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline INT32_t _unpack_int(const char *s) nogil:
    cdef INT32_t ret
    memcpy(&ret, s, sizeof(INT32_t))
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _pack_long(INT64_t val, bytearray arr):
    cdef char *ptr
    cdef Py_ssize_t size = PyByteArray_GET_SIZE(arr)
    PyByteArray_Resize(arr, size + sizeof(INT64_t))
    ptr = PyByteArray_AS_STRING(arr) + size
    memcpy(ptr, &val, sizeof(INT64_t))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline INT64_t _unpack_long(const char *s) nogil:
    cdef INT64_t ret
    memcpy(&ret, s, sizeof(INT64_t))
    return ret


cdef object np_random_bytes = np.random.bytes


cdef inline bytes new_message_id():
    return np_random_bytes(32)


cdef inline void _pack_message_type(INT32_t message_type, bytearray buf, INT32_t protocol=DEFAULT_PROTOCOL):
    cdef INT32_t value

    # use 1 byte to represent protocol and message type, from left to right, 0-2, protocol, 3-7 message_type
    value = (protocol << 5) | message_type
    _pack_byte(value, buf)


cdef inline INT32_t _unpack_message_type_value(const char *binary, size_t* pos) nogil:
    cdef INT32_t value
    cdef INT32_t protocol

    value = binary[pos[0]]
    pos[0] += 1
    protocol = value >> 5
    if protocol != 0:
        raise NotImplementedError('Unsupported protocol')

    return value & ((1 << 5) - 1)


cdef inline object _unpack_message_type(const char *binary, size_t* pos):
    return MessageType(_unpack_message_type_value(binary, pos))


cpdef INT32_t unpack_message_type_value(bytes binary):
    cdef size_t pos = 0
    return _unpack_message_type_value(binary, &pos)


cpdef object unpack_message_type(bytes binary):
    return MessageType(unpack_message_type_value(binary))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void _skip_message_type(const char *binary, size_t* pos) nogil:
    pos[0] += 1


cpdef bytes unpack_message_id(bytes binary):
    cdef size_t pos = 1

    return _unpack_message_id(binary, &pos)


cdef inline void _pack_message_id(bytes message_id, bytearray buf):
    _pack_object(message_id, buf)


cdef inline bytes _unpack_message_id(const char *buf, size_t* pos):
    return _unpack_object(buf, pos)


cdef inline void _skip_message_id(const char *buf, size_t* pos) nogil:
    _skip_object(buf, pos)


cdef inline void _pack_object(object obj, bytearray buf) except *:
    cdef BYTE_t st
    cdef bytes m

    if obj is None:
        _pack_byte(NONE, buf)
        return

    if isinstance(obj, bytes):
        st = RAW_BYTES
        m = obj
    else:
        st = PICKLE
        m = dumps(obj)

    _pack_byte(st, buf)
    _pack_long(len(m), buf)
    buf.extend(m)


cdef inline object _unpack_object(const char *binary, size_t* pos):
    cdef BYTE_t st
    cdef size_t size
    cdef bytes ret_bytes

    st = binary[pos[0]]
    pos[0] += 1

    if st == NONE:
        return None

    size = _unpack_long(binary + pos[0])
    pos[0] += 8

    ret_bytes = binary[pos[0]:pos[0] + size]
    pos[0] += size
    if st == RAW_BYTES:
        return ret_bytes
    else:
        return loads(ret_bytes)


cdef inline void _skip_object(const char *binary, size_t* pos) nogil:
    cdef BYTE_t st
    cdef size_t size

    st = binary[pos[0]]
    pos[0] += 1

    if st == NONE:
        return

    size = _unpack_long(binary + pos[0])
    pos[0] += size + sizeof(INT64_t)


cdef inline void _pack_index(INT32_t index, bytearray buf):
    # 2 bytes
    _pack_short(index, buf)


cdef inline INT32_t _unpack_index(const char *binary, size_t* pos) nogil:
    cdef INT32_t ret = _unpack_short(binary + pos[0])
    pos[0] += 2
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void _skip_index(const char *binary, size_t* pos) nogil:
    pos[0] += 2


cdef inline void _pack_actor_ref(ActorRef actor_ref, bytearray buf):
    # this line circumvents a bug in Cython which led to application crash
    assert actor_ref is not None
    _pack_object(actor_ref.address, buf)
    _pack_object(actor_ref.uid, buf)


cdef inline ActorRef _unpack_actor_ref(const char *binary, size_t* pos):
    cdef object address
    cdef object uid

    address = _unpack_object(binary, pos)
    uid = _unpack_object(binary, pos)
    return ActorRef(address, uid)


cpdef bytes pack_actor_ref(ActorRef actor_ref):
    cdef bytearray buf = bytearray()

    _pack_actor_ref(actor_ref, buf)
    return bytes(buf)


cpdef void unpack_actor_ref(bytes binary, ActorRef actor_ref):
    cdef size_t pos = 0
    cdef object address
    cdef object uid

    address = _unpack_object(binary, &pos)
    uid = _unpack_object(binary, &pos)

    actor_ref.address = address
    actor_ref.uid = uid


cdef inline void _skip_address(const char *binary, size_t* pos) nogil:
    _skip_object(binary, pos)


cdef inline object _unpack_uid(const char *binary, size_t* pos):
    return _unpack_object(binary, pos)


cdef inline void _pack_message_size(size_t size, bytearray buf):
    _pack_long(size, buf)


cdef inline INT32_t _unpack_message_size(const char *binary, size_t* pos) nogil:
    cdef INT32_t ret = _unpack_long(binary + pos[0])
    pos[0] += 8
    return ret


@cython.nonecheck(False)
cdef inline list _pack_tuple_message(tuple messages):
    cdef list ret
    cdef bytearray bio
    cdef BYTE_t st
    cdef bytes m

    ret = []
    for message in messages:
        bio = bytearray()

        # use 1 byte to record the serialize type
        if isinstance(message, bytes):
            st = RAW_BYTES
            m = message
        else:
            st = PICKLE
            m = dumps(message)

        _pack_byte(st, bio)
        _pack_long(len(m), bio)
        ret.append(<bytes>bio)
        ret.append(m)

    return ret


cdef inline bytes _pack_sole_message(object message, bytearray buf):
    cdef BYTE_t st
    cdef bytes m

    # use 1 byte to record the serialize type
    if isinstance(message, bytes):
        st = RAW_BYTES
        m = message
    elif message is None:
        st = NONE
        m = bytes()
    else:
        st = PICKLE
        m = dumps(message)

    _pack_byte(st, buf)
    _pack_long(len(m), buf)
    return m


cdef inline list _pack_message(object message, bytearray buf):
    if isinstance(message, tuple):
        _pack_message_size(len(message), buf)
        return _pack_tuple_message(message)
    else:
        _pack_message_size(0, buf)
        return [_pack_sole_message(message, buf)]


cdef inline tuple _unpack_tuple_message(const char *buf, size_t size, size_t* pos):
    return tuple(_unpack_sole_message(buf, pos) for _ in range(size))


cdef inline object _unpack_sole_message(const char *binary, size_t* pos):
    cdef BYTE_t st
    cdef size_t size
    cdef bytes ret_bytes

    st = binary[pos[0]]
    pos[0] += 1
    size = _unpack_long(binary + pos[0])
    pos[0] += 8

    ret_bytes = binary[pos[0]:pos[0] + size]
    pos[0] += size
    if st == RAW_BYTES:
        return ret_bytes
    elif st == NONE:
        return None
    else:
        return loads(ret_bytes)


cdef inline object _unpack_message(const char *buf, size_t* pos):
    cdef size_t size

    size = _unpack_message_size(buf, pos)
    if size > 0:
        return _unpack_tuple_message(buf, size, pos)
    else:
        return _unpack_sole_message(buf, pos)


cdef inline object _pack_send_message(INT32_t from_index, INT32_t to_index, ActorRef actor_ref, object message,
                                      bint send, object write=None, INT32_t protocol=DEFAULT_PROTOCOL):
    cdef bytes message_id
    cdef INT32_t message_type
    cdef bytearray buf
    cdef object m

    # from_index -1 means main process, -2 means remote, other is the subprocess id
    message_id = new_message_id()
    message_type = SEND_ALL if send else TELL_ALL

    buf = bytearray()
    _pack_message_type(message_type, buf, protocol=protocol)
    _pack_message_id(message_id, buf)
    _pack_index(from_index, buf)
    _pack_index(to_index, buf)
    _pack_actor_ref(actor_ref, buf)
    m = _pack_message(message, buf)

    if write is not None:
        write(buf, *m)
        return message_id

    return [message_id, buf] + m


cdef inline object _unpack_send_message(bytes binary, send=True):
    cdef object message_type
    cdef bytes message_id
    cdef INT32_t from_index
    cdef INT32_t to_index
    cdef ActorRef actor_ref
    cdef object message
    cdef size_t pos = 0
    cdef const char *binary_ptr = binary

    _unpack_message_type_value(binary_ptr, &pos)
    message_type = MessageType.send_all if send else MessageType.tell_all
    message_id = _unpack_message_id(binary_ptr, &pos)
    from_index = _unpack_index(binary_ptr, &pos)
    to_index = _unpack_index(binary_ptr, &pos)
    actor_ref = _unpack_actor_ref(binary_ptr, &pos)
    message = _unpack_message(binary_ptr, &pos)

    return SEND_MESSAGE(message_type=message_type, message_id=message_id,
                        from_index=from_index, to_index=to_index,
                        actor_ref=actor_ref, message=message)


cpdef object pack_send_message(INT32_t from_index, INT32_t to_index, ActorRef actor_ref, object message,
                               object write=None, INT32_t protocol=DEFAULT_PROTOCOL):
    return _pack_send_message(from_index, to_index, actor_ref, message, True,
                              write=write, protocol=protocol)


cpdef object unpack_send_message(bytes message):
    return _unpack_send_message(message)


cpdef object pack_tell_message(INT32_t from_index, INT32_t to_index, ActorRef actor_ref, object message,
                               object write=None, INT32_t protocol=DEFAULT_PROTOCOL):
    return _pack_send_message(from_index, to_index, actor_ref, message, False,
                              write=write, protocol=protocol)


cpdef object unpack_tell_message(bytes message):
    return _unpack_send_message(message, send=False)


cpdef tuple get_index(bytes binary, object calc_from_uid):
    cdef object buf
    cdef INT32_t from_index
    cdef INT32_t to_index
    cdef object uid
    cdef size_t pos = 0
    cdef const char *binary_ptr = binary

    _skip_message_type(binary_ptr, &pos)
    _skip_message_id(binary_ptr, &pos)

    from_index = _unpack_index(binary_ptr, &pos)
    to_index = _unpack_index(binary_ptr, &pos)
    if to_index != -1 or calc_from_uid is None:
        return from_index, to_index

    _skip_address(binary_ptr, &pos)

    uid = _unpack_uid(binary_ptr, &pos)
    return from_index, calc_from_uid(uid)


cpdef object pack_create_actor_message(INT32_t from_index, INT32_t to_index, ActorRef actor_ref, object actor_cls,
                                       tuple args, dict kw, object write=None, INT32_t protocol=DEFAULT_PROTOCOL):
    cdef bytes message_id
    cdef bytearray buf

    # from_index -1 means main process, -2 means remote, other is the subprocess id
    message_id = new_message_id()

    buf = bytearray()
    _pack_message_type(CREATE_ACTOR, buf, protocol=protocol)
    _pack_message_id(message_id, buf)
    _pack_index(from_index, buf)
    _pack_index(to_index, buf)
    _pack_actor_ref(actor_ref, buf)
    _pack_object(actor_cls, buf)
    _pack_object(args if args else None, buf)
    _pack_object(kw if kw else None, buf)

    if write is not None:
        write(buf)
        return message_id

    return message_id, buf


cpdef object unpack_create_actor_message(bytes binary):
    cdef object buf
    cdef object message_type
    cdef bytes message_id
    cdef INT32_t from_index
    cdef INT32_t to_index
    cdef ActorRef actor_ref
    cdef object actor_cls
    cdef tuple args
    cdef dict kw
    cdef size_t pos = 0
    cdef const char *binary_ptr = binary

    _unpack_message_type_value(binary_ptr, &pos)
    message_type = MessageType.create_actor
    message_id = _unpack_message_id(binary_ptr, &pos)
    from_index = _unpack_index(binary_ptr, &pos)
    to_index = _unpack_index(binary_ptr, &pos)
    actor_ref = _unpack_actor_ref(binary_ptr, &pos)
    actor_cls = _unpack_object(binary_ptr, &pos)
    args = _unpack_object(binary_ptr, &pos) or tuple()
    kw = _unpack_object(binary_ptr, &pos) or dict()

    return CREATE_ACTOR_MESSAGE(message_type=message_type, message_id=message_id,
                                from_index=from_index, to_index=to_index,
                                actor_ref=actor_ref, actor_cls=actor_cls,
                                args=args, kwargs=kw)


cpdef object pack_destroy_actor_message(INT32_t from_index, INT32_t to_index, ActorRef actor_ref,
                                        object write=None, INT32_t protocol=DEFAULT_PROTOCOL):
    cdef bytes message_id
    cdef bytearray buf

    # from_index -1 means main process, -2 means remote, other is the subprocess id
    message_id = new_message_id()

    buf = bytearray()
    _pack_message_type(DESTROY_ACTOR, buf, protocol=protocol)
    _pack_message_id(message_id, buf)
    _pack_index(from_index, buf)
    _pack_index(to_index, buf)
    _pack_actor_ref(actor_ref, buf)

    if write is not None:
        write(buf)
        return message_id

    return message_id, buf


cpdef object unpack_destroy_actor_message(bytes binary):
    cdef object buf
    cdef object message_type
    cdef bytes message_id
    cdef INT32_t from_index
    cdef INT32_t to_index
    cdef ActorRef actor_ref
    cdef size_t pos = 0
    cdef const char *binary_ptr = binary

    message_type = _unpack_message_type(binary_ptr, &pos)
    message_id = _unpack_message_id(binary_ptr, &pos)
    from_index = _unpack_index(binary_ptr, &pos)
    to_index = _unpack_index(binary_ptr, &pos)
    actor_ref = _unpack_actor_ref(binary_ptr, &pos)

    return DESTROY_ACTOR_MESSAGE(message_type=message_type, message_id=message_id,
                                 from_index=from_index, to_index=to_index,
                                 actor_ref=actor_ref)


cpdef object pack_has_actor_message(INT32_t from_index, INT32_t to_index, ActorRef actor_ref,
                                    object write=None, INT32_t protocol=DEFAULT_PROTOCOL):
    cdef bytearray buf

    # from_index -1 means main process, -2 means remote, other is the subprocess id
    message_id = new_message_id()

    buf = bytearray()
    _pack_message_type(HAS_ACTOR, buf, protocol=protocol)
    _pack_message_id(message_id, buf)
    _pack_index(from_index, buf)
    _pack_index(to_index, buf)
    _pack_actor_ref(actor_ref, buf)

    if write is not None:
        write(buf)
        return message_id

    return message_id, buf


cpdef object unpack_has_actor_message(bytes binary):
    cdef object message_type
    cdef bytes message_id
    cdef INT32_t from_index
    cdef INT32_t to_index
    cdef ActorRef actor_ref
    cdef size_t pos = 0
    cdef const char *binary_ptr = binary

    message_type = _unpack_message_type(binary_ptr, &pos)
    message_id = _unpack_message_id(binary_ptr, &pos)
    from_index = _unpack_index(binary_ptr, &pos)
    to_index = _unpack_index(binary_ptr, &pos)
    actor_ref = _unpack_actor_ref(binary_ptr, &pos)

    return HAS_ACTOR_MESSAGE(message_type=message_type, message_id=message_id,
                             from_index=from_index, to_index=to_index,
                             actor_ref=actor_ref)


cpdef object pack_result_message(bytes message_id, INT32_t from_index, INT32_t to_index, object result,
                                 object write=None, INT32_t protocol=DEFAULT_PROTOCOL):
    cdef bytearray buf

    buf = bytearray()
    _pack_message_type(RESULT, buf, protocol=protocol)
    _pack_message_id(message_id, buf)
    _pack_index(from_index, buf)
    _pack_index(to_index, buf)
    _pack_object(result, buf)

    if write is not None:
        write(buf)
        return message_id

    return message_id, buf


cpdef object unpack_result_message(bytes binary, object from_index=None, object to_index=None):
    cdef object message_type
    cdef bytes message_id
    cdef object result
    cdef size_t pos = 0
    cdef const char *binary_ptr = binary

    _unpack_message_type_value(binary_ptr, &pos)
    message_type = MessageType.result
    message_id = _unpack_message_id(binary_ptr, &pos)
    if from_index is not None:
        _skip_index(binary_ptr, &pos)
    else:
        from_index = _unpack_index(binary_ptr, &pos)
    if to_index is not None:
        _skip_index(binary_ptr, &pos)
    else:
        to_index = _unpack_index(binary_ptr, &pos)
    result = _unpack_object(binary_ptr, &pos)

    return RESULT_MESSAGE(message_type=message_type, message_id=message_id,
                          from_index=from_index, to_index=to_index,
                          result=result)


cpdef object pack_error_message(bytes message_id, INT32_t from_index, INT32_t to_index,
                                object error_type, object error, object tb,
                                object write=None, INT32_t protocol=DEFAULT_PROTOCOL):
    cdef bytearray buf

    buf = bytearray()
    _pack_message_type(ERROR, buf, protocol=protocol)
    _pack_message_id(message_id, buf)
    _pack_index(from_index, buf)
    _pack_index(to_index, buf)
    _pack_object(error_type, buf)
    _pack_object(error, buf)
    _pack_object(tb, buf)

    if write is not None:
        write(buf)
        return message_id

    return message_id, buf


cpdef object unpack_error_message(bytes binary):
    cdef object message_type
    cdef bytes message_id
    cdef INT32_t from_index
    cdef INT32_t to_index
    cdef object error_type
    cdef object error
    cdef object tb
    cdef size_t pos = 0
    cdef const char *binary_ptr = binary

    message_type = _unpack_message_type(binary_ptr, &pos)
    message_id = _unpack_message_id(binary_ptr, &pos)
    from_index = _unpack_index(binary_ptr, &pos)
    to_index = _unpack_index(binary_ptr, &pos)
    error_type = _unpack_object(binary_ptr, &pos)
    error = _unpack_object(binary_ptr, &pos)
    tb = _unpack_object(binary_ptr, &pos)

    return ERROR_MESSAGE(message_type=message_type, message_id=message_id,
                         from_index=from_index, to_index=to_index,
                         error_type=error_type, error=error, traceback=tb)


cdef inline bytes _wrap_read_func(object read_func, size_t size):
    cdef bytes read_bytes

    try:
         read_bytes = read_func(size)
    except ConnectionResetError:
        raise BrokenPipeError('The remote server is closed')

    if len(read_bytes) == 0:
        raise BrokenPipeError('The remote server is closed')

    return read_bytes


cpdef bytes read_remote_message(object read_func):
    cdef size_t size
    cdef bytearray buf
    cdef INT32_t received_size
    cdef INT32_t left
    cdef bytes read_bytes

    read_bytes = _wrap_read_func(read_func, 8)
    size = _unpack_long(read_bytes)
    buf = bytearray()
    received_size = 0
    left = size

    while True:
        read_bytes = _wrap_read_func(read_func, left)
        buf.extend(read_bytes)
        received_size += len(read_bytes)
        if received_size >= size:
            break
        left = size - received_size

    return bytes(buf)


def write_remote_message(write_func, *binary):
    cdef bytes size_bytes
    cdef INT64_t size = 0

    for b in binary:
        size += len(b)
    size_bytes = (<char *>&size)[:sizeof(INT64_t)]
    write_func(size_bytes)

    for b in binary:
        write_func(b)


if PY_MAJOR_VERSION < 3:  # pragma: no cover
    _write_remote_message = write_remote_message

    def write_remote_message(*args, **kwargs):
        try:
            return _write_remote_message(*args, **kwargs)
        except socket.error as ex:
            if ex.errno == errno.EPIPE:
                raise BrokenPipeError
            else:
                raise
