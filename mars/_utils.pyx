# distutils: language = c++
# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

import collections
import importlib
import itertools
import os
import pickle
import pkgutil
import time
import types
import uuid
import warnings
from datetime import date, datetime, timedelta, tzinfo
from enum import Enum
from functools import lru_cache, partial
from random import getrandbits
from weakref import WeakSet

import numpy as np
import pandas as pd
import cloudpickle
cimport cython
from cpython cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint_fast64_t, uint32_t, uint8_t
from libc.stdlib cimport malloc, free
from .lib.cython.libcpp cimport mt19937_64
try:
    from pandas.tseries.offsets import Tick as PDTick
except ImportError:
    PDTick = None

from .lib.mmh3 import hash as mmh_hash, hash_bytes as mmh_hash_bytes, \
    hash_from_buffer as mmh3_hash_from_buffer

cdef bint _has_cupy = bool(pkgutil.find_loader('cupy'))
cdef bint _has_cudf = bool(pkgutil.find_loader('cudf'))
cdef bint _has_sqlalchemy = bool(pkgutil.find_loader('sqlalchemy'))
cdef bint _has_interval_array_inclusive = hasattr(
    pd.arrays.IntervalArray, "inclusive"
)


cdef extern from "MurmurHash3.h":
    void MurmurHash3_x64_128(const void * key, Py_ssize_t len, uint32_t seed, void * out)


cdef bytes _get_mars_key(const uint8_t[:] bufferview):
    cdef const uint8_t *data = &bufferview[0]
    cdef uint8_t out[16]
    MurmurHash3_x64_128(data, len(bufferview), 0, out)
    out[0] |= 0xC0
    return PyBytes_FromStringAndSize(<char*>out, 16)


cpdef str to_str(s, encoding='utf-8'):
    if type(s) is str:
        return <str>s
    elif isinstance(s, bytes):
        return (<bytes>s).decode(encoding)
    elif isinstance(s, str):
        return str(s)
    elif s is None:
        return s
    else:
        raise TypeError(f"Could not convert from {s} to str.")


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
        raise TypeError(f"Could not convert from {s} to bytes.")


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
        raise TypeError(f"Could not convert from {s} to unicode.")


_type_dispatchers = WeakSet()


NamedType = collections.namedtuple("NamedType", ["name", "type_"])


cdef class TypeDispatcher:
    def __init__(self):
        self._handlers = dict()
        self._lazy_handlers = dict()
        # store inherited handlers to facilitate unregistering
        self._inherit_handlers = dict()

        _type_dispatchers.add(self)

    cpdef void register(self, object type_, object handler):
        if isinstance(type_, str):
            self._lazy_handlers[type_] = handler
        elif type(type_) is not NamedType and isinstance(type_, tuple):
            for t in type_:
                self.register(t, handler)
        else:
            self._handlers[type_] = handler

    cpdef void unregister(self, object type_):
        if type(type_) is not NamedType and isinstance(type_, tuple):
            for t in type_:
                self.unregister(t)
        else:
            self._lazy_handlers.pop(type_, None)
            self._handlers.pop(type_, None)
            self._inherit_handlers.clear()

    cdef _reload_lazy_handlers(self):
        for k, v in self._lazy_handlers.items():
            mod_name, obj_name = k.rsplit('.', 1)
            with warnings.catch_warnings():
                # the lazy imported cudf will warn no device found,
                # when we set visible device to -1 for CPU processes,
                # ignore the warning to not distract users
                warnings.simplefilter("ignore")
                mod = importlib.import_module(mod_name, __name__)
            self.register(getattr(mod, obj_name), v)
        self._lazy_handlers = dict()

    cpdef get_handler(self, object type_):
        try:
            return self._handlers[type_]
        except KeyError:
            pass

        try:
            return self._inherit_handlers[type_]
        except KeyError:
            self._reload_lazy_handlers()
            if type(type_) is NamedType:
                named_type = partial(NamedType, type_.name)
                mro = itertools.chain(
                    *zip(map(named_type, type_.type_.__mro__),
                         type_.type_.__mro__)
                )
            else:
                mro = type_.__mro__
            for clz in mro:
                # only lookup self._handlers for mro clz
                handler = self._handlers.get(clz)
                if handler is not None:
                    self._inherit_handlers[type_] = handler
                    return handler
            raise KeyError(f'Cannot dispatch type {type_}')

    def __call__(self, object obj, *args, **kwargs):
        return self.get_handler(type(obj))(obj, *args, **kwargs)

    @staticmethod
    def reload_all_lazy_handlers():
        for dispatcher in _type_dispatchers:
            (<TypeDispatcher>dispatcher)._reload_lazy_handlers()


cdef inline build_canonical_bytes(tuple args, kwargs):
    if kwargs:
        args = args + (kwargs,)
    return pickle.dumps(tokenize_handler(args))


def tokenize(*args, **kwargs):
    return _get_mars_key(build_canonical_bytes(args, kwargs)).hex()


def tokenize_int(*args, **kwargs):
    return mmh_hash(build_canonical_bytes(args, kwargs))


cdef class Tokenizer(TypeDispatcher):
    def __call__(self, object obj, *args, **kwargs):
        try:
            return self.get_handler(type(obj))(obj, *args, **kwargs)
        except KeyError:
            if hasattr(obj, '__mars_tokenize__') and not isinstance(obj, type):
                if len(args) == 0 and len(kwargs) == 0:
                    return obj.__mars_tokenize__()
                else:
                    obj = obj.__mars_tokenize__()
                    return self.get_handler(type(obj))(obj, *args, **kwargs)
            if callable(obj):
                if PDTick is not None and not isinstance(obj, PDTick):
                    return tokenize_function(obj)

            try:
                return cloudpickle.dumps(obj)
            except:
                raise TypeError(f'Cannot generate token for {obj}, type: {type(obj)}') from None


cdef inline list iterative_tokenize(object ob):
    cdef list dq = [ob]
    cdef int dq_pos = 0
    cdef list h_list = []
    while dq_pos < len(dq):
        x = dq[dq_pos]
        dq_pos += 1
        if type(x) in _primitive_types:
            h_list.append(x)
        elif isinstance(x, (list, tuple)):
            dq.extend(x)
        elif isinstance(x, set):
            dq.extend(sorted(x))
        elif isinstance(x, dict):
            dq.extend(sorted(x.items()))
        else:
            h_list.append(tokenize_handler(x))

        if dq_pos >= 64 and len(dq) < dq_pos * 2:  # pragma: no cover
            dq = dq[dq_pos:]
            dq_pos = 0
    return h_list


cdef inline tuple tokenize_numpy(ob):
    cdef int offset

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
            data = mmh_hash_bytes('-'.join(ob.flat).encode('utf-8', errors='surrogatepass'))
        except UnicodeDecodeError:
            data = mmh_hash_bytes(b'-'.join([to_binary(x) for x in ob.flat]))
        except TypeError:
            try:
                data = mmh_hash_bytes(pickle.dumps(ob, pickle.HIGHEST_PROTOCOL))
            except:
                # nothing can do, generate uuid
                data = uuid.uuid4().hex
    else:
        try:
            data = mmh_hash_bytes(ob.ravel().view('i1').data)
        except (BufferError, AttributeError, ValueError):
            data = mmh_hash_bytes(ob.copy().ravel().view('i1').data)
    return data, ob.dtype, ob.shape, ob.strides


cdef inline _extract_range_index_attr(object range_index, str attr):
    try:
        return getattr(range_index, attr)
    except AttributeError:  # pragma: no cover
        return getattr(range_index, '_' + attr)


cdef list tokenize_pandas_index(ob):
    cdef long long start
    cdef long long stop
    cdef long long end
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


cdef list tokenize_pandas_categorical(ob):
    l = ob.to_list()
    l.append(ob.shape)
    return iterative_tokenize(l)


cdef list tokenize_pd_extension_dtype(ob):
    return iterative_tokenize([ob.name])


cdef list tokenize_categories_dtype(ob):
    return iterative_tokenize([ob.categories, ob.ordered])


cdef list tokenize_interval_dtype(ob):
    return iterative_tokenize([type(ob).__name__, ob.subtype])


cdef list tokenize_pandas_time_arrays(ob):
    return iterative_tokenize([ob.asi8, ob.dtype])


cdef list tokenize_pandas_tick(ob):
    return iterative_tokenize([ob.freqstr])


cdef list tokenize_pandas_interval_arrays(ob):  # pragma: no cover
    if _has_interval_array_inclusive:
        return iterative_tokenize([ob.left, ob.right, ob.inclusive])
    else:
        return iterative_tokenize([ob.left, ob.right, ob.closed])


cdef list tokenize_sqlalchemy_data_type(ob):
    return iterative_tokenize([repr(ob)])


cdef list tokenize_sqlalchemy_selectable(ob):
    return iterative_tokenize([str(ob)])


cdef list tokenize_enum(ob):
    cls = type(ob)
    return iterative_tokenize([id(cls), cls.__name__, ob.name])


@lru_cache(500)
def tokenize_function(ob):
    if isinstance(ob, partial):
        args = iterative_tokenize(ob.args)
        keywords = iterative_tokenize(ob.keywords.items()) if ob.keywords else None
        return tokenize_function(ob.func), args, keywords
    else:
        try:
            if isinstance(ob, types.FunctionType):
                return iterative_tokenize([pickle.dumps(ob, protocol=0), id(ob)])
            else:
                return pickle.dumps(ob, protocol=0)
        except:
            pass
        try:
            return cloudpickle.dumps(ob, protocol=0)
        except:
            return str(ob)


@lru_cache(500)
def tokenize_pickled_with_cache(ob):
    return pickle.dumps(ob)


def tokenize_cupy(ob):
    from .serialization import serialize
    header, _buffers = serialize(ob)
    return iterative_tokenize([header, ob.data.ptr])


def tokenize_cudf(ob):
    from .serialization import serialize
    header, buffers = serialize(ob)
    return iterative_tokenize([header] + [(buf.ptr, buf.size) for buf in buffers])


cdef Tokenizer tokenize_handler = Tokenizer()

cdef set _primitive_types = {
    int, float, str, unicode, bytes, complex, type(None), type, slice, date, datetime, timedelta
}
for t in _primitive_types:
    tokenize_handler.register(t, lambda ob: ob)

for t in (np.dtype, np.generic):
    tokenize_handler.register(t, lambda ob: ob)

for t in (list, tuple, dict, set):
    tokenize_handler.register(t, iterative_tokenize)

tokenize_handler.register(np.ndarray, tokenize_numpy)
tokenize_handler.register(np.random.RandomState, lambda ob: iterative_tokenize(ob.get_state()))
tokenize_handler.register(memoryview, lambda ob: mmh3_hash_from_buffer(ob))
tokenize_handler.register(Enum, tokenize_enum)
tokenize_handler.register(pd.Index, tokenize_pandas_index)
tokenize_handler.register(pd.Series, tokenize_pandas_series)
tokenize_handler.register(pd.DataFrame, tokenize_pandas_dataframe)
tokenize_handler.register(pd.Categorical, tokenize_pandas_categorical)
tokenize_handler.register(pd.CategoricalDtype, tokenize_categories_dtype)
tokenize_handler.register(pd.IntervalDtype, tokenize_interval_dtype)
tokenize_handler.register(tzinfo, tokenize_pickled_with_cache)
tokenize_handler.register(pd.arrays.DatetimeArray, tokenize_pandas_time_arrays)
tokenize_handler.register(pd.arrays.TimedeltaArray, tokenize_pandas_time_arrays)
tokenize_handler.register(pd.arrays.PeriodArray, tokenize_pandas_time_arrays)
tokenize_handler.register(pd.arrays.IntervalArray, tokenize_pandas_interval_arrays)
tokenize_handler.register(pd.api.extensions.ExtensionDtype, tokenize_pd_extension_dtype)
if _has_cupy:
    tokenize_handler.register('cupy.ndarray', tokenize_cupy)
if _has_cudf:
    tokenize_handler.register('cudf.DataFrame', tokenize_cudf)
    tokenize_handler.register('cudf.Series', tokenize_cudf)
    tokenize_handler.register('cudf.Index', tokenize_cudf)

if PDTick is not None:
    tokenize_handler.register(PDTick, tokenize_pandas_tick)
if _has_sqlalchemy:
    tokenize_handler.register(
        "sqlalchemy.sql.sqltypes.TypeEngine", tokenize_sqlalchemy_data_type
    )
    tokenize_handler.register(
        "sqlalchemy.sql.Selectable", tokenize_sqlalchemy_selectable
    )

cpdef register_tokenizer(cls, handler):
    tokenize_handler.register(cls, handler)


@cython.nonecheck(False)
@cython.cdivision(True)
cpdef long long ceildiv(long long x, long long y) nogil:
    return x // y + (x % y != 0)


cdef class Timer:
    cdef object _start
    cdef readonly object duration

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *_):
        self.duration = time.time() - self._start


cdef mt19937_64 _rnd_gen
cdef bint _rnd_is_seed_set = False


cpdef void reset_id_random_seed() except *:
    cdef bytes seed_bytes
    global _rnd_is_seed_set

    seed_bytes = getrandbits(64).to_bytes(8, "little")
    _rnd_gen.seed((<uint_fast64_t *><char *>seed_bytes)[0])
    _rnd_is_seed_set = True


cpdef bytes new_random_id(int byte_len):
    cdef uint_fast64_t *res_ptr
    cdef uint_fast64_t res_data[4]
    cdef int i, qw_num = byte_len >> 3
    cdef bytes res

    if not _rnd_is_seed_set:
        reset_id_random_seed()

    if (qw_num << 3) < byte_len:
        qw_num += 1

    if qw_num <= 4:
        # use stack memory to accelerate
        res_ptr = res_data
    else:
        res_ptr = <uint_fast64_t *>malloc(qw_num << 3)

    try:
        for i in range(qw_num):
            res_ptr[i] = _rnd_gen()
        return <bytes>((<char *>&(res_ptr[0]))[:byte_len])
    finally:
        # free memory if allocated by malloc
        if res_ptr != res_data:
            free(res_ptr)


__all__ = ['to_str', 'to_binary', 'to_text', 'TypeDispatcher', 'tokenize', 'tokenize_int',
           'register_tokenizer', 'ceildiv', 'Timer', 'reset_id_random_seed', 'new_random_id']
