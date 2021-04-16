#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import asyncio
import dataclasses
import functools
import importlib
import inspect
import io
import itertools
import logging
import numbers
import os
import pickle
import pkgutil
import random
import shutil
import socket
import struct
import sys
import threading
import time
import warnings
import weakref
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Callable, Optional

import numpy as np
import pandas as pd
import cloudpickle

from ._utils import to_binary, to_str, to_text, TypeDispatcher, \
    tokenize, tokenize_int, register_tokenizer, insert_reversed_tuple, ceildiv
from .config import options
from .lib.tblib import Traceback

logger = logging.getLogger(__name__)
random.seed(int(time.time()) * os.getpid())

OBJECT_FIELD_OVERHEAD = 50

# make flake8 happy by referencing these imports
TypeDispatcher = TypeDispatcher
tokenize = tokenize
register_tokenizer = register_tokenizer
insert_reversed_tuple = insert_reversed_tuple
ceildiv = ceildiv


# fix encoding conversion problem under windows
if sys.platform == 'win32':  # pragma: no cover
    def _replace_default_encoding(func):
        def _fun(s, encoding=None):
            encoding = encoding or getattr(sys.stdout, 'encoding', None) or 'mbcs'
            return func(s, encoding=encoding)

        _fun.__name__ = func.__name__
        _fun.__doc__ = func.__doc__
        return _fun

    to_binary = _replace_default_encoding(to_binary)
    to_text = _replace_default_encoding(to_text)
    to_str = _replace_default_encoding(to_str)


class AttributeDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(
                f"'AttributeDict' object has no attribute {item}")


def on_serialize_shape(shape):
    if shape:
        return tuple(s if not np.isnan(s) else -1 for s in shape)
    return shape


def on_deserialize_shape(shape):
    if shape:
        return tuple(s if s != -1 else np.nan for s in shape)
    return shape


def on_serialize_numpy_type(value):
    if value is pd.NaT:
        value = None
    return value.item() if isinstance(value, np.generic) else value


def on_serialize_nsplits(value):
    if value is None:
        return None
    new_nsplits = []
    for dim_splits in value:
        new_nsplits.append(tuple(None if np.isnan(v) else v for v in dim_splits))
    return tuple(new_nsplits)


_memory_size_indices = {'': 0, 'k': 1, 'm': 2, 'g': 3, 't': 4}


def calc_size_by_str(value, total):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    mem_limit, is_percent = parse_readable_size(value)
    if is_percent:
        return int(total * mem_limit)
    else:
        return int(mem_limit)


def parse_readable_size(value):
    if isinstance(value, numbers.Number):
        return float(value), False

    value = value.strip().lower()
    num_pos = 0
    while num_pos < len(value) and value[num_pos] in '0123456789.-':
        num_pos += 1

    value, suffix = value[:num_pos], value[num_pos:]
    suffix = suffix.strip()
    if suffix.endswith('%'):
        return float(value) / 100, True

    try:
        return float(value) * (1024 ** _memory_size_indices[suffix[:1]]), False
    except (ValueError, KeyError):
        raise ValueError(f'Unknown limitation value: {value}')


def readable_size(size, trunc=False):
    if size < 1024:
        ret_size = size
        size_unit = ''
    elif 1024 <= size < 1024 ** 2:
        ret_size = size * 1.0 / 1024
        size_unit = 'K'
    elif 1024 ** 2 <= size < 1024 ** 3:
        ret_size = size * 1.0 / (1024 ** 2)
        size_unit = 'M'
    elif 1024 ** 3 <= size < 1024 ** 4:
        ret_size = size * 1.0 / (1024 ** 3)
        size_unit = 'G'
    else:
        ret_size = size * 1.0 / (1024 ** 4)
        size_unit = 'T'

    if not trunc:
        return '{0:.2f}{1}'.format(ret_size, size_unit)
    else:
        return f'{int(ret_size)}{size_unit}'


_git_info = None


def git_info():
    from ._version import get_git_info

    global _git_info
    if _git_info is not None:
        if _git_info == ':INVALID:':
            return None
        else:
            return _git_info

    git_tuple = get_git_info()
    if git_tuple is None:
        _git_info = ':INVALID:'
        return None
    else:
        _git_info = git_tuple
        return git_tuple


LOW_PORT_BOUND = 10000
HIGH_PORT_BOUND = 65535
_local_occupied_ports = set()


def _get_ports_from_netstat():
    import subprocess
    while True:
        p = subprocess.Popen('netstat -a -n -p tcp'.split(), stdout=subprocess.PIPE)
        try:
            p.wait(5)
            break
        except subprocess.TimeoutExpired:
            p.terminate()
            continue
    occupied = set()
    for line in p.stdout:
        line = to_str(line)
        if '.' not in line:
            continue
        for part in line.split():
            if '.' in part:
                _, port_str = part.rsplit('.', 1)
                if port_str == '*':
                    continue
                port = int(port_str)
                if LOW_PORT_BOUND <= port <= HIGH_PORT_BOUND:
                    occupied.add(int(port_str))
                break
    p.stdout.close()
    return occupied


def get_next_port(typ=None, occupy=True):
    import psutil
    try:
        conns = psutil.net_connections()
        typ = typ or socket.SOCK_STREAM
        occupied = set(sc.laddr.port for sc in conns
                       if sc.type == typ and LOW_PORT_BOUND <= sc.laddr.port <= HIGH_PORT_BOUND)
    except psutil.AccessDenied:
        occupied = _get_ports_from_netstat()

    occupied.update(_local_occupied_ports)
    randn = struct.unpack('<Q', os.urandom(8))[0]
    idx = int(randn % (1 + HIGH_PORT_BOUND - LOW_PORT_BOUND - len(occupied)))
    for i in range(LOW_PORT_BOUND, HIGH_PORT_BOUND + 1):
        if i in occupied:
            continue
        if idx == 0:
            if occupy:
                _local_occupied_ports.add(i)
            return i
        idx -= 1
    raise SystemError('No ports available.')


@functools.lru_cache(200)
def mod_hash(val, modulus):
    return tokenize_int(val) % modulus


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


def debug(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            print(f"execute function {func} with args {args} and kwargs {kwargs}")
            return func(*args, **kwargs)
        except NotImplementedError:
            return NotImplemented

    return wrapper


def async_debug(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            print(f"Start to execute function {func} with args {args} and kwargs {kwargs}")
            result = await func(*args, **kwargs)
            print(f"Finished executing function {func} with args {args} and kwargs {kwargs}")
            return result
        except NotImplementedError:
            return NotImplemented

    return wrapper


def lazy_import(name, package=None, globals=None, locals=None, rename=None):
    rename = rename or name
    prefix_name = name.split('.', 1)[0]

    class LazyModule(object):
        def __getattr__(self, item):
            if item.startswith('_pytest') or item in ('__bases__', '__test__'):
                raise AttributeError(item)

            real_mod = importlib.import_module(name, package=package)
            if globals is not None and rename in globals:
                globals[rename] = real_mod
            elif locals is not None:
                locals[rename] = real_mod
            return getattr(real_mod, item)

    if pkgutil.find_loader(prefix_name) is not None:
        return LazyModule()
    else:
        return None


def serialize_serializable(serializable, compress=False, serialize_method=None):
    from .serialization import serialize

    serialize_method = serialize_method or options.serialize_method
    assert serialize_method == 'pickle'

    bio = io.BytesIO()
    header, buffers = serialize(serializable)
    header['buf_sizes'] = [getattr(buf, 'nbytes', len(buf))
                           for buf in buffers]
    s_header = pickle.dumps(header)
    bio.write(struct.pack('<Q', len(s_header)))
    bio.write(s_header)
    for buf in buffers:
        bio.write(buf)
    ser_graph = bio.getvalue()

    if compress:
        ser_graph = zlib.compress(ser_graph)
    return ser_graph


serialize_graph = serialize_serializable


def deserialize_serializable(ser_serializable):
    from .serialization import deserialize

    bio = io.BytesIO(ser_serializable)
    s_header_length = struct.unpack('Q', bio.read(8))[0]
    header2 = pickle.loads(bio.read(s_header_length))
    buffers2 = [bio.read(s) for s in header2['buf_sizes']]
    return deserialize(header2, buffers2)


deserialize_graph = deserialize_serializable


def register_mars_serializer_on_ray(obj_type):
    from mars.serialization import serialize, deserialize

    def deserializer(to_deserialize):
        return deserialize(*to_deserialize)

    register_ray_serializer(obj_type, serializer=serialize, deserializer=deserializer)


def register_ray_serializer(obj_type, serializer=None, deserializer=None):
    ray = lazy_import("ray")
    if ray:
        print(f"register {obj_type}")
        try:
            ray.register_custom_serializer(
                obj_type, serializer=serializer, deserializer=deserializer)
        except AttributeError:  # ray >= 1.0
            try:
                from ray.worker import global_worker

                global_worker.check_connected()
                context = global_worker.get_serialization_context()
                context.register_custom_serializer(
                    obj_type, serializer=serializer, deserializer=deserializer)
            except AttributeError:  # ray >= 1.2.0
                ray.util.register_serializer(
                    obj_type, serializer=serializer, deserializer=deserializer)


def calc_data_size(dt, shape=None):
    if dt is None:
        return 0

    if isinstance(dt, tuple):
        return sum(calc_data_size(c) for c in dt)

    shape = getattr(dt, 'shape', None) or shape
    if hasattr(dt, 'memory_usage') or hasattr(dt, 'groupby_obj'):
        return sys.getsizeof(dt)
    if hasattr(dt, 'nbytes'):
        return max(sys.getsizeof(dt), dt.nbytes)
    if hasattr(dt, 'shape') and len(dt.shape) == 0:
        return 0
    if hasattr(dt, 'dtypes') and shape is not None:
        size = shape[0] * sum(dtype.itemsize for dtype in dt.dtypes)
        try:
            index_value_value = dt.index_value.value
            if hasattr(index_value_value, 'dtype'):
                size += calc_data_size(index_value_value, shape=shape)
        except AttributeError:
            pass
        return size
    if hasattr(dt, 'dtype') and shape is not None:
        return shape[0] * dt.dtype.itemsize

    # object chunk
    return sys.getsizeof(dt)


def _get_mod_logger():
    mod_logger = None
    cur_frame = inspect.currentframe()
    while cur_frame.f_globals.get('__name__') == __name__:
        cur_frame = cur_frame.f_back
    frame_globals = cur_frame.f_globals
    for logger_name in ('logger', 'LOG', 'LOGGER'):
        if logger_name in frame_globals:
            mod_logger = frame_globals[logger_name]
            break
    return mod_logger


def log_unhandled(func):
    mod_logger = _get_mod_logger()
    if not mod_logger:
        return func

    func_name = getattr(func, '__qualname__', func.__module__ + func.__name__)
    func_args = inspect.getfullargspec(func)

    @functools.wraps(func)
    def _wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:  # noqa: E722
            kwcopy = kwargs.copy()
            kwcopy.update(zip(func_args.args, args))
            if getattr(func, '__closure__', None) is not None:
                kwargs.update(zip(
                    func.__code__.co_freevars + getattr(func.__code__, 'co_cellvars', ()),
                    [getattr(c, 'cell_contents', None) for c in func.__closure__],
                ))

            messages = []
            for k, v in kwcopy.items():
                if 'key' in k:
                    messages.append(f'{k}={v}')

            err_msg = f'Unexpected exception occurred in {func_name}.'
            if messages:
                err_msg += ' ' + ' '.join(messages)
            mod_logger.exception(err_msg)
            raise
    return _wrapped


_internal_mode = threading.local()


def is_eager_mode():
    in_kernel = is_kernel_mode()
    if not in_kernel:
        return options.eager_mode
    else:
        # in kernel, eager mode always False
        return False


def is_kernel_mode():
    try:
        return bool(_internal_mode.kernel)
    except AttributeError:
        _internal_mode.kernel = None
        return bool(_internal_mode)


def is_build_mode():
    return bool(getattr(_internal_mode, 'build', False))


class _EnterModeFuncWrapper:
    def __init__(self, mode_name_to_value):
        self.mode_name_to_value = mode_name_to_value

        # as the wrapper may enter for many times
        # record old values for each time
        self.mode_name_to_value_list = list()

    def __enter__(self):
        mode_name_to_old_value = dict()
        for mode_name, value in self.mode_name_to_value.items():
            # record mode's old values
            mode_name_to_old_value[mode_name] = \
                getattr(_internal_mode, mode_name, None)
            if value is None:
                continue
            # set value
            setattr(_internal_mode, mode_name, value)
        self.mode_name_to_value_list.append(mode_name_to_old_value)

    def __exit__(self, *_):
        mode_name_to_old_value = self.mode_name_to_value_list.pop()
        for mode_name in self.mode_name_to_value.keys():
            # set back old values
            setattr(_internal_mode, mode_name,
                    mode_name_to_old_value[mode_name])

    def __call__(self, func):
        @functools.wraps(func)
        def _inner(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return _inner


def enter_mode(kernel=None, build=None):
    mode_name_to_value = {
        'kernel': kernel,
        'build': build,
    }

    return _EnterModeFuncWrapper(mode_name_to_value)


def build_tileable_graph(tileables, executed_tileable_keys, graph=None):
    from .core import TileableGraph, TileableGraphBuilder
    from .core.operand import Fetch

    with enter_mode(build=True):
        node_to_copy = weakref.WeakKeyDictionary()
        node_to_fetch = weakref.WeakKeyDictionary()
        copied = weakref.WeakSet()

        def replace_with_fetch_or_copy(n):
            n = n.data if hasattr(n, 'data') else n
            if n in copied:
                return n
            if n.key in executed_tileable_keys:
                if n not in node_to_fetch:
                    c = node_to_copy[n] = node_to_fetch[n] = build_fetch(n).data
                    copied.add(c)
                return node_to_fetch[n]
            if n not in node_to_copy:
                copy_op = n.op.copy()
                params = []
                for o in n.op.outputs:
                    p = o.params.copy()
                    p.update(o.extra_params)
                    p['_key'] = o.key
                    if isinstance(o.op, Fetch):
                        # chunks may be generated in the remote functions,
                        # thus bring chunks and nsplits for serialization
                        p['chunks'] = o.chunks
                        p['nsplits'] = o.nsplits
                    params.append(p)
                copies = copy_op.new_tileables([replace_with_fetch_or_copy(inp) for inp in n.inputs],
                                               kws=params, output_limit=len(params))
                for o, copy in zip(n.op.outputs, copies):
                    node_to_copy[o] = copy.data
                    copied.add(copy.data)
            return node_to_copy[n]

        graph = TileableGraph(tileables)
        tileable_graph_builder = TileableGraphBuilder(
            graph, node_processor=replace_with_fetch_or_copy)
        return next(iter(tileable_graph_builder.build()))


def build_exc_info(exc_type, *args, **kwargs):
    try:
        raise exc_type(*args, **kwargs)
    except exc_type:
        exc_info = sys.exc_info()
        tb = exc_info[-1]
        back_frame = tb.tb_frame.f_back
        tb_builder = object.__new__(Traceback)
        tb_builder.tb_frame = back_frame
        tb_builder.tb_lineno = back_frame.f_lineno
        tb_builder.tb_next = tb.tb_next
        return exc_info[:2] + (tb_builder.as_traceback(),)


class BlacklistSet(object):
    def __init__(self, expire_time):
        self._key_time = dict()
        self._expire_time = expire_time

    def add(self, key):
        self._key_time[key] = time.time()

    def remove(self, key):
        t = self._key_time[key]
        del self._key_time[key]
        if t < time.time() - self._expire_time:
            raise KeyError(key)

    def update(self, keys):
        t = time.time()
        for k in keys:
            self._key_time[k] = t

    def __contains__(self, item):
        try:
            if self._key_time[item] >= time.time() - self._expire_time:
                return True
            else:
                del self._key_time[item]
                return False
        except KeyError:
            return False

    def __iter__(self):
        rmv_list = []
        exp_time = time.time() - self._expire_time
        for k, t in self._key_time.items():
            if t >= exp_time:
                yield k
            else:
                rmv_list.append(k)
        for k in rmv_list:
            del self._key_time[k]


def build_fetch_chunk(chunk, input_chunk_keys=None, **kwargs):
    from .core.operand import ShuffleProxy

    chunk_op = chunk.op
    params = chunk.params.copy()

    if isinstance(chunk_op, ShuffleProxy):
        # for shuffle nodes, we build FetchShuffle chunks
        # to replace ShuffleProxy
        source_keys, source_idxes, source_mappers = [], [], []
        for pinp in chunk.inputs:
            if input_chunk_keys is not None and pinp.key not in input_chunk_keys:
                continue
            source_keys.append(pinp.key)
            source_idxes.append(pinp.index)
            source_mappers.append(get_chunk_mapper_id(pinp))
        op = chunk_op.get_fetch_op_cls(chunk)(
            source_keys=source_keys, source_idxes=source_idxes,
            source_mappers=source_mappers)
    else:
        # for non-shuffle nodes, we build Fetch chunks
        # to replace original chunk
        op = chunk_op.get_fetch_op_cls(chunk)(sparse=chunk.op.sparse)
    return op.new_chunk(None, kws=[params], _key=chunk.key, _id=chunk.id, **kwargs)


def build_fetch_tileable(tileable):
    if tileable.is_coarse():
        chunks = None
    else:
        chunks = []
        for c in tileable.chunks:
            fetch_chunk = build_fetch_chunk(c, index=c.index)
            chunks.append(fetch_chunk)

    tileable_op = tileable.op
    params = tileable.params.copy()

    new_op = tileable_op.get_fetch_op_cls(tileable)(_id=tileable_op.id)
    return new_op.new_tileables(None, chunks=chunks, nsplits=tileable.nsplits,
                                _key=tileable.key, _id=tileable.id, **params)[0]


def build_fetch(entity):
    from .core import CHUNK_TYPE, ENTITY_TYPE
    if isinstance(entity, CHUNK_TYPE):
        return build_fetch_chunk(entity)
    elif isinstance(entity, ENTITY_TYPE):
        return build_fetch_tileable(entity)
    else:
        raise TypeError(f'Type {type(entity)} not supported')


def get_chunk_mapper_id(chunk):
    op = chunk.op
    try:
        return op.mapper_id
    except AttributeError:
        from .core.operand import Fuse
        if isinstance(op, Fuse):
            return chunk.composed[-1].op.mapper_id
        else:  # pragma: no cover
            raise


def get_chunk_reducer_index(chunk):
    op = chunk.op
    try:
        return op.reducer_index
    except AttributeError:
        from .core.operand import Fuse
        if isinstance(op, Fuse):
            return chunk.composed[0].op.reducer_index
        else:  # pragma: no cover
            raise


def merge_chunks(chunk_results):
    """
    Concatenate chunk results according to index.
    :param chunk_results: list of tuple, {(chunk_idx, chunk_result), ...,}
    :return:
    """
    from .lib.sparse import SparseNDArray
    from .tensor.array_utils import get_array_module

    chunk_results = sorted(chunk_results, key=lambda x: x[0])
    v = chunk_results[0][1]
    if isinstance(v, (np.ndarray, SparseNDArray)):
        xp = get_array_module(v)
        ndim = v.ndim
        for i in range(ndim - 1):
            new_chunks = []
            for idx, cs in itertools.groupby(chunk_results, key=lambda t: t[0][:-1]):
                new_chunks.append((idx, xp.concatenate([c[1] for c in cs], axis=ndim - i - 1)))
            chunk_results = new_chunks
        concat_result = xp.concatenate([c[1] for c in chunk_results])
        return concat_result
    elif isinstance(v, pd.DataFrame):
        concats = []
        for _, cs in itertools.groupby(chunk_results, key=lambda t: t[0][0]):
            concats.append(pd.concat([c[1] for c in cs], axis='columns'))
        return pd.concat(concats, axis='index')
    elif isinstance(v, pd.Series):
        return pd.concat([c[1] for c in chunk_results])
    else:
        result = None
        for cr in chunk_results:
            if not cr[1]:
                continue
            if result is None:
                result = cr[1]
            else:
                raise TypeError(f'unsupported type {type(v)}')
        return result


def calc_nsplits(chunk_idx_to_shape):
    """
    Calculate a tiled entity's nsplits
    :param chunk_idx_to_shape: Dict type, {chunk_idx: chunk_shape}
    :return: nsplits
    """
    ndim = len(next(iter(chunk_idx_to_shape)))
    tileable_nsplits = []
    # for each dimension, record chunk shape whose index is zero on other dimensions
    for i in range(ndim):
        splits = []
        for index, shape in chunk_idx_to_shape.items():
            if all(idx == 0 for j, idx in enumerate(index) if j != i):
                splits.append(shape[i])
        tileable_nsplits.append(tuple(splits))
    return tuple(tileable_nsplits)


def sort_dataframe_result(df, result):
    """ sort DataFrame on client according to `should_be_monotonic` attribute """
    if hasattr(df, 'index_value'):
        if getattr(df.index_value, 'should_be_monotonic', False):
            result.sort_index(inplace=True)
        if hasattr(df, 'columns_value'):
            if getattr(df.columns_value, 'should_be_monotonic', False):
                result.sort_index(axis=1, inplace=True)
    return result


def numpy_dtype_from_descr_json(obj):
    """
    Construct numpy dtype from it's np.dtype.descr.

    The dtype can be trivial, but can also be very complex (nested) record type. In that
    case, the tuple in `descr` will be made as `list`, which can be understood by `np.dtype()`.
    This utility helps the reconstruct work.
    """
    if isinstance(obj, list):
        return np.dtype([(k, numpy_dtype_from_descr_json(v)) for k, v in obj])
    return obj


def check_chunks_unknown_shape(tileables, error_cls):
    for t in tileables:
        for ns in t.nsplits:
            if any(np.isnan(s) for s in ns):
                raise error_cls(f'Input tileable {t} has chunks with unknown shape')


def has_unknown_shape(tiled):
    if getattr(tiled, 'shape', None) is None:
        return False
    if any(pd.isnull(s) for s in tiled.shape):
        return True
    if any(pd.isnull(s) for s in itertools.chain(*tiled.nsplits)):
        return True
    return False


def sbytes(x):
    # NB: bytes() in Python 3 has different semantic with Python 2, see: help(bytes)
    from numbers import Number
    if x is None or isinstance(x, Number):
        return bytes(str(x), encoding='ascii')
    elif isinstance(x, list):
        return bytes('[' + ', '.join([str(k) for k in x]) + ']', encoding='utf-8')
    elif isinstance(x, tuple):
        return bytes('(' + ', '.join([str(k) for k in x]) + ')', encoding='utf-8')
    elif isinstance(x, str):
        return bytes(x, encoding='utf-8')
    else:
        return bytes(x)


def kill_process_tree(pid, include_parent=True):
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    plasma_sock_dir = None
    try:
        children = proc.children(recursive=True)
    except psutil.NoSuchProcess:  # pragma: no cover
        return

    if include_parent:
        children.append(proc)
    for p in children:
        try:
            if 'plasma' in p.name():
                plasma_sock_dir = next((conn.laddr for conn in p.connections('unix')
                                        if 'plasma' in conn.laddr), None)
            p.kill()
        except psutil.NoSuchProcess:  # pragma: no cover
            pass
    if plasma_sock_dir:
        shutil.rmtree(plasma_sock_dir, ignore_errors=True)


def copy_tileables(tileables: List, **kwargs):
    inputs = kwargs.pop('inputs', None)
    copy_key = kwargs.pop('copy_key', True)
    copy_id = kwargs.pop('copy_id', True)
    if kwargs:
        raise TypeError(f"got un unexpected keyword argument '{next(iter(kwargs))}'")
    if len(tileables) > 1:
        # cannot handle tileables with different operands here
        # try to copy separately if so
        if len({t.op for t in tileables}) != 1:
            raise TypeError("All tileables' operands should be same.")

    op = tileables[0].op.copy().reset_key()
    if copy_key:
        op._key = tileables[0].op.key
    kws = []
    for t in tileables:
        params = t.params.copy()
        if copy_key:
            params['_key'] = t.key
        if copy_id:
            params['_id'] = t.id
        params.update(t.extra_params)
        kws.append(params)
    inputs = inputs or op.inputs
    return op.new_tileables(inputs, kws=kws, output_limit=len(kws))


def require_not_none(obj):
    def wrap(func):
        if obj is not None:
            return func
        else:
            return
    return wrap


def require_module(module: str):
    def wrap(func):
        try:
            importlib.import_module(module)

            @functools.wraps(func)
            def inner(*args, **kwargs):
                return func(*args, **kwargs)
            return inner
        except ImportError:
            return
    return wrap


def ignore_warning(func):
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return func(*args, **kwargs)
    return inner


def flatten(nested_iterable):
    """
    Flatten a nested iterable into a list.

    Parameters
    ----------
    nested_iterable : list or tuple
        an iterable which can contain other iterables

    Returns
    -------
    flattened : list

    Examples
    --------
    >>> flatten([[0, 1], [2, 3]])
    [0, 1, 2, 3]
    >>> flatten([[0, 1], [[3], [4, 5]]])
    [0, 1, 3, 4, 5]
    """

    flattened = []
    stack = list(nested_iterable)[::-1]
    while len(stack) > 0:
        inp = stack.pop()
        if isinstance(inp, (tuple, list)):
            stack.extend(inp[::-1])
        else:
            flattened.append(inp)
    return flattened


def stack_back(flattened, raw):
    """
    Organize a new iterable from a flattened list according to raw iterable.

    Parameters
    ----------
    flattened : list
        flattened list
    raw: list
        raw iterable

    Returns
    -------
    ret : list

    Examples
    --------
    >>> raw = [[0, 1], [2, [3, 4]]]
    >>> flattened = flatten(raw)
    >>> flattened
    [0, 1, 2, 3, 4]
    >>> a = [f + 1 for f in flattened]
    >>> a
    [1, 2, 3, 4, 5]
    >>> stack_back(a, raw)
    [[1, 2], [3, [4, 5]]]
    """
    flattened_iter = iter(flattened)
    result = list()

    def _stack(container, items):
        for item in items:
            if not isinstance(item, (list, tuple)):
                container.append(next(flattened_iter))
            else:
                new_container = list()
                container.append(new_container)
                _stack(new_container, item)

        return container

    return _stack(result, raw)


def recursive_tile(tileable):
    q = [tileable]
    while q:
        t = q[-1]
        cs = [c for c in t.inputs if c.is_coarse()]
        if cs:
            q.extend(cs)
            continue
        t._inplace_tile()
        q.pop()

    return tileable


def replace_inputs(obj, old, new):
    new_inputs = []
    for inp in obj.inputs or []:
        if inp is old:
            new_inputs.append(new)
        else:
            new_inputs.append(inp)
    obj.inputs = new_inputs


def build_fuse_chunk(fused_chunks, fuse_op_cls, op_kw=None, chunk_kw=None):
    from .core.graph import ChunkGraph

    fuse_graph = ChunkGraph(fused_chunks)
    for i, fuse_chunk in enumerate(fused_chunks):
        fuse_graph.add_node(fuse_chunk)
        if i > 0:
            fuse_graph.add_edge(fused_chunks[i - 1], fuse_chunk)

    head_chunk = fused_chunks[0]
    tail_chunk = fused_chunks[-1]
    tail_chunk_op = tail_chunk.op
    fuse_op = fuse_op_cls(sparse=tail_chunk_op.sparse, gpu=tail_chunk_op.gpu,
                          _key=tail_chunk_op.key, fuse_graph=fuse_graph,
                          **(op_kw or dict()))
    return fuse_op.new_chunk(
        head_chunk.inputs, kws=[tail_chunk.params], _key=tail_chunk.key,
        _chunk=tail_chunk, **(chunk_kw or dict()))


def adapt_mars_docstring(doc):
    """
    Adapt numpy-style docstrings to Mars docstring.

    This util function will add Mars imports, replace object references
    and add execute calls. Note that check is needed after replacement.
    """
    if doc is None:
        return None

    lines = []
    first_prompt = True
    prev_prompt = False
    has_numpy = 'np.' in doc
    has_pandas = 'pd.' in doc

    for line in doc.splitlines():
        sp = line.strip()
        if sp.startswith('>>>') or sp.startswith('...'):
            prev_prompt = True
            if first_prompt:
                first_prompt = False
                indent = ''.join(itertools.takewhile(lambda x: x in (' ', '\t'), line))
                if has_numpy:
                    lines.extend([indent + '>>> import mars.tensor as mt'])
                if has_pandas:
                    lines.extend([indent + '>>> import mars.dataframe as md'])
            line = line.replace('np.', 'mt.').replace('pd.', 'md.')
        elif prev_prompt:
            prev_prompt = False
            if sp:
                lines[-1] += '.execute()'
        lines.append(line)
    return '\n'.join(lines)


def prune_chunk_graph(chunk_graph, result_chunk_keys):
    from .core.operand import Fetch

    key_to_fetch_chunk = {c.key: c for c in chunk_graph
                          if isinstance(c.op, Fetch)}

    reverse_chunk_graph = chunk_graph.build_reversed()
    marked = set()
    for c in reverse_chunk_graph.topological_iter():
        if c.key in result_chunk_keys or \
                any(inp in marked for inp in reverse_chunk_graph.iter_predecessors(c)):
            for o in c.op.outputs:
                marked.add(o)
                if o.key in key_to_fetch_chunk:
                    # for multi outputs, if one of the output is replaced by fetch
                    # keep the fetch chunk as marked,
                    # or the node will be lost in the chunk graph and serialize would fail
                    marked.add(key_to_fetch_chunk[o.key])
    for n in list(chunk_graph):
        if n not in marked:
            chunk_graph.remove_node(n)

    chunk_graph.results = [r for r in chunk_graph.results
                           if r.key in result_chunk_keys]


@functools.lru_cache(500)
def serialize_function(function, pickle_protocol=None):
    return cloudpickle.dumps(function, protocol=pickle_protocol)


class FixedSizeFileObject:
    def __init__(self, file_obj, fixed_size):
        self._file_obj = file_obj
        self._cur = self._file_obj.tell()
        self._size = fixed_size
        self._end = self._cur + self._size

    def _get_size(self, size):
        max_size = self._end - self._cur
        if size is None:
            return max_size
        else:
            return min(max_size, size)

    def read(self, size=None):
        result = self._file_obj.read(self._get_size(size))
        self._cur = self._file_obj.tell()
        return result

    def read1(self, size=None):
        return self.read(size)

    def readline(self, size=None):
        result = self._file_obj.readline(self._get_size(size))
        self._cur = self._file_obj.tell()
        return result

    def readlines(self, size=None):
        result = self._file_obj.readlines(self._get_size(size))
        self._cur = self._file_obj.tell()
        return result

    def seek(self, offset):
        self._cur = offset
        return self._file_obj.seek(offset)

    def tell(self):
        return self._file_obj.tell()

    def __next__(self):
        while True:
            result = self.readline()
            if len(result) == 0:
                raise StopIteration
            else:
                return result

    def __iter__(self):
        while True:
            try:
                yield next(self)
            except StopIteration:
                return

    def __getattr__(self, item):  # pragma: no cover
        return getattr(self._file_obj, item)


def is_object_dtype(dtype):
    try:
        return np.issubdtype(dtype, np.object_) \
               or np.issubdtype(dtype, np.unicode_) \
               or np.issubdtype(dtype, np.bytes_)
    except TypeError:  # pragma: no cover
        return False


def calc_object_overhead(chunk, shape):
    from .dataframe.core import DATAFRAME_CHUNK_TYPE, SERIES_CHUNK_TYPE, INDEX_CHUNK_TYPE

    if not shape or np.isnan(shape[0]) or getattr(chunk, 'dtypes', None) is None:
        return 0

    if isinstance(chunk, DATAFRAME_CHUNK_TYPE) \
            and chunk.dtypes is not None:
        n_strings = len([dt for dt in chunk.dtypes if is_object_dtype(dt)])
        if is_object_dtype(getattr(chunk.index_value.value, 'dtype', None)):
            n_strings += 1
    elif isinstance(chunk, SERIES_CHUNK_TYPE) \
            and chunk.dtype is not None:
        n_strings = 1 if is_object_dtype(chunk.dtype) else 0
        if is_object_dtype(getattr(chunk.index_value.value, 'dtype', None)):
            n_strings += 1
    elif isinstance(chunk, INDEX_CHUNK_TYPE) \
            and chunk.dtype is not None:
        n_strings = 1 if is_object_dtype(chunk.dtype) else 0
    else:
        n_strings = 0
    return n_strings * shape[0] * OBJECT_FIELD_OVERHEAD


def arrow_array_to_objects(obj):
    from .dataframe.arrays import ArrowDtype

    if isinstance(obj, pd.DataFrame):
        if any(isinstance(dt, ArrowDtype) for dt in obj.dtypes):
            # ArrowDtype exists
            result = pd.DataFrame(columns=obj.columns)
            for i, dtype in enumerate(obj.dtypes):
                if isinstance(dtype, ArrowDtype):
                    result.iloc[:, i] = pd.Series(obj.iloc[:, i].to_numpy(),
                                                  index=obj.index)
                else:
                    result.iloc[:, i] = obj.iloc[:, i]
            obj = result
    elif isinstance(obj, pd.Series):
        if isinstance(obj.dtype, ArrowDtype):
            obj = pd.Series(obj.to_numpy(), index=obj.index, name=obj.name)
    return obj


def enter_current_session(func):
    @functools.wraps(func)
    def wrapped(cls, ctx, op):
        from .session import Session
        from .context import ContextBase

        # skip in some test cases
        if not hasattr(ctx, 'get_current_session'):
            return func(cls, ctx, op)

        session = ctx.get_current_session()
        prev_default_session = Session.default
        session.as_default()

        try:
            if isinstance(ctx, ContextBase):
                with ctx:
                    result = func(cls, ctx, op)
            else:
                result = func(cls, ctx, op)
        finally:
            Session._set_default_session(prev_default_session)

        return result

    return wrapped


class Timer:
    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *_):
        end = time.time()
        self.duration = end - self._start


_io_quiet_local = threading.local()
_io_quiet_lock = threading.Lock()


class _QuietIOWrapper:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __getattr__(self, item):
        return getattr(self.wrapped, item)

    def write(self, d):
        if getattr(_io_quiet_local, 'is_wrapped', False):
            return 0
        return self.wrapped.write(d)


@contextmanager
def quiet_stdio():
    """Quiets standard outputs when inferring types of functions"""
    with _io_quiet_lock:
        _io_quiet_local.is_wrapped = True
        sys.stdout = _QuietIOWrapper(sys.stdout)
        sys.stderr = _QuietIOWrapper(sys.stderr)

    try:
        yield
    finally:
        with _io_quiet_lock:
            sys.stdout = sys.stdout.wrapped
            sys.stderr = sys.stderr.wrapped
            if not isinstance(sys.stdout, _QuietIOWrapper):
                _io_quiet_local.is_wrapped = False


def implements(f):
    def decorator(g):
        g.__doc__ = f.__doc__
        return g

    return decorator


def stringify_path(path: Union[str, os.PathLike]) -> str:
    """
    Convert *path* to a string or unicode path if possible.
    """
    if isinstance(path, str):
        return path

    # checking whether path implements the filesystem protocol
    try:
        return path.__fspath__()
    except AttributeError:
        raise TypeError("not a path-like object")


def find_objects(nested, types):
    found = []
    stack = [nested]

    while len(stack) > 0:
        it = stack.pop()
        if isinstance(it, types):
            found.append(it)
            continue

        if isinstance(it, (list, tuple, set)):
            stack.extend(list(it)[::-1])
        elif isinstance(it, dict):
            stack.extend(list(it.values())[::-1])

    return found


def replace_objects(nested, mapping):
    if not mapping:
        return nested

    if isinstance(nested, dict):
        vals = list(nested.values())
    else:
        vals = list(nested)

    new_vals = []
    for val in vals:
        if isinstance(val, (dict, list, tuple, set)):
            new_val = replace_objects(val, mapping)
        else:
            try:
                new_val = mapping.get(val, val)
            except TypeError:
                new_val = val
        new_vals.append(new_val)

    if isinstance(nested, dict):
        return type(nested)((k, v) for k, v in zip(nested.keys(), new_vals))
    else:
        return type(nested)(new_vals)


@dataclass
class _DelayedArgument:
    args: Tuple
    kwargs: Dict


class _ExtensibleCallable:
    func: Callable
    batch_func: Optional[Callable]
    is_async: bool

    def __call__(self, *args, **kwargs):
        if self.is_async:
            return self._async_call(*args, **kwargs)
        else:
            return self._sync_call(*args, **kwargs)

    async def _async_call(self, *args, **kwargs):
        try:
            return await self.func(*args, **kwargs)
        except NotImplementedError:
            if self.batch_func:
                return (await self.batch_func([args], [kwargs]))[0]
            raise

    def _sync_call(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except NotImplementedError:
            if self.batch_func:
                return self.batch_func([args], [kwargs])[0]
            raise


class _ExtensibleWrapper(_ExtensibleCallable):
    def __init__(self,
                 func: Callable,
                 batch_func: Callable = None,
                 is_async: bool = False):
        self.func = func
        self.batch_func = batch_func
        self.is_async = is_async

    @staticmethod
    def delay(*args, **kwargs):
        return _DelayedArgument(args=args, kwargs=kwargs)

    @staticmethod
    def _gen_args_kwargs_list(delays):
        args_list = list()
        kwargs_list = list()
        for delay in delays:
            args_list.append(delay.args)
            kwargs_list.append(delay.kwargs)
        return args_list, kwargs_list

    async def _async_batch(self, *delays):
        if self.batch_func:
            args_list, kwargs_list = self._gen_args_kwargs_list(delays)
            return await self.batch_func(args_list, kwargs_list)
        else:
            # this function has no batch implementation
            # call it separately
            coros = [self.func(*d.args, **d.kwargs)
                     for d in delays]
            return await asyncio.gather(*coros)

    def _sync_batch(self, *delays):
        if self.batch_func:
            args_list, kwargs_list = self._gen_args_kwargs_list(delays)
            return self.batch_func(args_list, kwargs_list)
        else:
            # this function has no batch implementation
            # call it separately
            return [self.func(*d.args, **d.kwargs) for d in delays]

    def batch(self, *delays):
        if self.is_async:
            return self._async_batch(*delays)
        else:
            return self._sync_batch(*delays)


class _ExtensibleAccessor(_ExtensibleCallable):
    func: Callable
    batch_func: Optional[Callable]

    def __init__(self, func: Callable):
        self.func = func
        self.batch_func = None
        self.is_async = asyncio.iscoroutinefunction(self.func)

    def batch(self, func: Callable):
        self.batch_func = func
        return self

    def __get__(self, instance, owner):
        if instance is None:
            # calling from class
            return self.func

        func = self.func.__get__(instance, owner)
        batch_func = self.batch_func.__get__(instance, owner) \
            if self.batch_func is not None else None

        return _ExtensibleWrapper(func, batch_func=batch_func,
                                  is_async=self.is_async)


def extensible(func: Callable):
    """
    `extensible` means this func could be functionality extended,
    especially for batch operations.

    Consider remote function calls, each function may have operations
    like opening file, closing file, batching them can help to reduce the cost,
    especially for remote function calls.

    Parameters
    ----------
    func : callable
        Function

    Returns
    -------
    func
    """
    return _ExtensibleAccessor(func)


# from https://github.com/ericvsmith/dataclasses/blob/master/dataclass_tools.py
# released under Apache License 2.0
def dataslots(cls):
    # Need to create a new class, since we can't set __slots__
    #  after a class has been created.

    # Make sure __slots__ isn't already set.
    if '__slots__' in cls.__dict__:  # pragma: no cover
        raise TypeError(f'{cls.__name__} already specifies __slots__')

    # Create a new dict for our new class.
    cls_dict = dict(cls.__dict__)
    field_names = tuple(f.name for f in dataclasses.fields(cls))
    cls_dict['__slots__'] = field_names
    for field_name in field_names:
        # Remove our attributes, if present. They'll still be
        #  available in _MARKER.
        cls_dict.pop(field_name, None)
    # Remove __dict__ itself.
    cls_dict.pop('__dict__', None)
    # And finally create the class.
    qualname = getattr(cls, '__qualname__', None)
    cls = type(cls)(cls.__name__, cls.__bases__, cls_dict)
    if qualname is not None:
        cls.__qualname__ = qualname
    return cls
