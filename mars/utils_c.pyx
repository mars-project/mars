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

import os
from hashlib import md5
from datetime import date, datetime, timedelta
from collections import deque

import numpy as np

from cpython.version cimport PY_MAJOR_VERSION


cpdef str to_str(s, encoding='utf-8'):
    if type(s) is str:
        return <str>s
    elif PY_MAJOR_VERSION >= 3 and isinstance(s, bytes):
        return (<bytes>s).decode(encoding)
    elif PY_MAJOR_VERSION < 3 and isinstance(s, unicode):
        return (<unicode>s).encode(encoding)
    elif isinstance(s, str):
        return str(s)
    elif s is None:
        return s
    else:
        raise TypeError("Could not convert from unicode.")


cpdef bytes to_binary(s, encoding='utf-8'):
    if type(s) is bytes:
        return <bytes>s
    elif isinstance(s, unicode):
        return (<unicode>s).encode(encoding)
    elif isinstance(s, bytes):
        return bytes(s)
    else:
        raise TypeError("Could not convert to bytes.")


cpdef unicode to_text(s, encoding='utf-8'):
    if type(s) is unicode:
        return <unicode>s
    elif isinstance(s, bytes):
        return (<bytes>s).decode('utf-8')
    elif isinstance(s, unicode):
        return unicode(s)
    else:
        raise TypeError("Could not convert to unicode.")


def tokenize(*args, **kwargs):
    if kwargs:
        args = args + (kwargs,)
    return md5(str([h(arg) for arg in args]).encode('utf-8')).hexdigest()


cdef object h(object ob):
    if isinstance(ob, dict):
        return h_iterative(sorted(list(ob.items()), key=str))
    if isinstance(ob, set):
        return h_iterative(sorted(ob, key=str))
    if isinstance(ob, (tuple, list)):
        return h_iterative(ob)
    return h_non_iterative(ob)


cdef list h_iterative(object ob):
    nested = deque(ob)
    h_list = []
    dq = deque()
    while nested or dq:
        x = dq.pop() if dq else nested.popleft()
        if isinstance(x, (list, tuple)):
            dq.extend(reversed(x))
        else:
            h_list.append(h_non_iterative(x))
    return h_list


cdef inline object h_non_iterative(object ob):
    if isinstance(ob, (int, float, str, unicode, bytes,
                       type(None), type, slice, date, datetime, timedelta)):
        return ob
    if isinstance(ob, (int, long, complex)):
        return ob
    if hasattr(ob, 'key'):
        return ob.key
    # numpy relative
    if isinstance(ob, np.ndarray):
        return h_numpy(ob)
    if isinstance(ob, (np.dtype, np.generic)):
        return repr(ob)

    raise TypeError('Cannot generate token for %s, type: %s' % (ob, type(ob)))


cdef h_numpy(ob):
    cdef int offset
    cdef str data

    if not ob.shape:
        return str(ob), ob.dtype
    if hasattr(ob, 'mode') and getattr(ob, 'filename', None):
        if hasattr(ob.base, 'ctypes'):
            offset = (ob.ctypes.get_as_parameter().value -
                      ob.base.ctypes.get_as_parameter().value)
        else:
            offset = 0  # root memmap's have mmap object as base
        return (ob.filename, os.path.getmtime(ob.filename), ob.dtype,
                ob.shape, ob.strides, offset)
    if ob.dtype.hasobject:
        try:
            data = md5('-'.join(ob.flat).encode('utf-8')).hexdigest()
        except TypeError:
            data = md5(b'-'.join([unicode(item).encode('utf-8') for item in
                                  ob.flat])).hexdigest()
    else:
        try:
            data = md5(ob.ravel().view('i1').data).hexdigest()
        except (BufferError, AttributeError, ValueError):
            data = md5(ob.copy().ravel().view('i1').data).hexdigest()
    return data, ob.dtype, ob.shape, ob.strides


__all__ = ['to_str', 'to_binary', 'to_text', 'tokenize']
