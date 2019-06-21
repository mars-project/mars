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
    return str([h(arg) for arg in args]).encode('utf-8')


def tokenize(*args, **kwargs):
    return to_hex(mmh_hash_bytes(build_canonical_bytes(args, kwargs)))


def tokenize_int(*args, **kwargs):
    return mmh_hash(build_canonical_bytes(args, kwargs))


cdef inline to_hex(bytes s):
    if PY_MAJOR_VERSION >= 3:
        return s.hex()
    else:
        return hexlify(s)


cdef inline object h(object ob):
    if isinstance(ob, dict):
        return h_iterative(sorted(list(ob.items()), key=str))
    elif isinstance(ob, set):
        return h_iterative(sorted(ob, key=str))
    elif isinstance(ob, (tuple, list)):
        return h_iterative(ob)
    return h_non_iterative(ob)


cdef inline list h_iterative(object ob):
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
    if isinstance(ob, (int, long, float, str, unicode, bytes, complex,
                       type(None), type, slice, date, datetime, timedelta)):
        return ob
    if hasattr(ob, 'key'):
        return ob.key
    # numpy relative
    if isinstance(ob, np.ndarray):
        return h_numpy(ob)
    elif isinstance(ob, (np.dtype, np.generic)):
        return repr(ob)
    elif pd is not None and isinstance(ob, pd.Index):
        return h_pandas_index(ob)
    elif pd is not None and isinstance(ob, pd.Series):
        return h_pandas_series(ob)
    elif pd is not None and isinstance(ob, pd.DataFrame):
        return h_pandas_dataframe(ob)
    elif hasattr(ob, '__mars_tokenize__'):
        return h(ob.__mars_tokenize__())

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


cdef h_pandas_index(ob):
    if isinstance(ob, pd.RangeIndex):
        # for range index, there is no need to get the values
        return h_iterative([ob.name, getattr(ob, 'names', None),
                            slice(ob._start, ob._stop, ob._step)])
    else:
        return h_iterative([ob.name, getattr(ob, 'names', None), ob.values])


cdef h_pandas_series(ob):
    return h_iterative([ob.name, ob.dtype, ob.values, ob.index])


cdef h_pandas_dataframe(ob):
    l = [block.values for block in ob._data.blocks]
    l.extend([ob.columns, ob.index])
    return h_iterative(l)


__all__ = ['to_str', 'to_binary', 'to_text', 'tokenize', 'tokenize_int']
