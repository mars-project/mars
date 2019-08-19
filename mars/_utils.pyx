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
import pickle
import uuid
from binascii import hexlify
from datetime import date, datetime, timedelta
from collections import deque

from .lib.mmh3 import hash as mmh_hash, hash_bytes as mmh_hash_bytes
from .compat import Enum

import numpy as np
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

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
        raise TypeError("Could not convert from %r to str." % (s,))


cpdef bytes to_binary(s, encoding='utf-8'):
    if type(s) is bytes:
        return <bytes>s
    elif isinstance(s, unicode):
        return (<unicode>s).encode(encoding)
    elif isinstance(s, bytes):
        return bytes(s)
    elif s is None:
        return None
    else:
        raise TypeError("Could not convert from %r to bytes." % (s,))


cpdef unicode to_text(s, encoding='utf-8'):
    if type(s) is unicode:
        return <unicode>s
    elif isinstance(s, bytes):
        return (<bytes>s).decode('utf-8')
    elif isinstance(s, unicode):
        return unicode(s)
    elif s is None:
        return None
    else:
        raise TypeError("Could not convert from %r to unicode." % (s,))


cdef inline build_canonical_bytes(tuple args, kwargs):
    if kwargs:
        args = args + (kwargs,)
    return str([tokenize_handler.tokenize(arg) for arg in args]).encode('utf-8')


def tokenize(*args, **kwargs):
    return to_hex(mmh_hash_bytes(build_canonical_bytes(args, kwargs)))


def tokenize_int(*args, **kwargs):
    return mmh_hash(build_canonical_bytes(args, kwargs))


cdef inline to_hex(bytes s):
    if PY_MAJOR_VERSION >= 3:
        return s.hex()
    else:
        return hexlify(s)


cdef class Tokenizer:
    cdef dict _handlers
    def __init__(self):
        self._handlers = dict()

    def register(self, cls, handler):
        self._handlers[cls] = handler

    cdef inline tokenize(self, object obj):
        object_type = type(obj)
        try:
            handler = self._handlers[object_type]
            return handler(obj)
        except KeyError:
            if hasattr(obj, '_key'):
                return obj._key
            if hasattr(obj, '__mars_tokenize__'):
                return self.tokenize(obj.__mars_tokenize__())

            for clz in object_type.__mro__:
                if clz in self._handlers:
                    self._handlers[object_type] = self._handlers[clz]
                    return self._handlers[clz](obj)
            raise TypeError('Cannot generate token for %s, type: %s' % (obj, object_type))


cdef inline list iterative_tokenize(object ob):
    dq = deque(ob)
    h_list = []
    while dq:
        x = dq.pop()
        if isinstance(x, (list, tuple)):
            dq.extend(x)
        elif isinstance(x, set):
            dq.extend(sorted(x))
        elif isinstance(x, dict):
            dq.extend(sorted(list(x.items())))
        else:
            h_list.append(tokenize_handler.tokenize(x))
    return h_list


cdef inline tuple tokenize_numpy(ob):
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
            data = to_hex(mmh_hash_bytes('-'.join(ob.flat).encode('utf-8', errors='surrogatepass')))
        except UnicodeDecodeError:
            data = to_hex(mmh_hash_bytes(b'-'.join([to_binary(x) for x in ob.flat])))
        except TypeError:
            try:
                data = to_hex(mmh_hash_bytes(pickle.dumps(ob, pickle.HIGHEST_PROTOCOL)))
            except:
                # nothing can do, generate uuid
                data = uuid.uuid4().hex
    else:
        try:
            data = to_hex(mmh_hash_bytes(ob.ravel().view('i1').data))
        except (BufferError, AttributeError, ValueError):
            data = to_hex(mmh_hash_bytes(ob.copy().ravel().view('i1').data))
    return data, ob.dtype, ob.shape, ob.strides


cdef inline _extract_range_index_attr(object range_index, str attr):
    try:
        return getattr(range_index, attr)
    except AttributeError:  # pragma: no cover
        return getattr(range_index, '_' + attr)


cdef list tokenize_pandas_index(ob):
    cdef int start
    cdef int stop
    cdef int end
    if isinstance(ob, pd.RangeIndex):
        start = _extract_range_index_attr(ob, 'start')
        stop = _extract_range_index_attr(ob, 'stop')
        step = _extract_range_index_attr(ob, 'step')
        # for range index, there is no need to get the values
        return iterative_tokenize([ob.name, getattr(ob, 'names', None), slice(start, stop, step)])
    else:
        return iterative_tokenize([ob.name, getattr(ob, 'names', None), ob.values])


cdef list tokenize_pandas_series(ob):
    return iterative_tokenize([ob.name, ob.dtype, ob.values, ob.index])


cdef list tokenize_pandas_dataframe(ob):
    l = [block.values for block in ob._data.blocks]
    l.extend([ob.columns, ob.index])
    return iterative_tokenize(l)


cdef Tokenizer tokenize_handler = Tokenizer()

base_types = (int, long, float, str, unicode, bytes, complex,
              type(None), type, slice, date, datetime, timedelta)
for t in base_types:
    tokenize_handler.register(t, lambda ob: ob)

for t in (np.dtype, np.generic):
    tokenize_handler.register(t, lambda ob: repr(ob))

for t in (list, tuple, dict, set):
    tokenize_handler.register(t, iterative_tokenize)

tokenize_handler.register(np.ndarray, tokenize_numpy)
tokenize_handler.register(dict, lambda ob: iterative_tokenize(sorted(list(ob.items()))))
tokenize_handler.register(set, lambda ob: iterative_tokenize(sorted(ob)))
tokenize_handler.register(np.random.RandomState, lambda ob: iterative_tokenize(ob.get_state()))
tokenize_handler.register(Enum, lambda ob: iterative_tokenize((type(ob), ob.name)))
tokenize_handler.register(pd.Index, tokenize_pandas_index)
tokenize_handler.register(pd.Series, tokenize_pandas_series)
tokenize_handler.register(pd.DataFrame, tokenize_pandas_dataframe)

__all__ = ['to_str', 'to_binary', 'to_text', 'tokenize', 'tokenize_int']
