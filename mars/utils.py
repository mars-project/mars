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

import functools
import importlib
import inspect
import json
import logging
import numbers
import os
import pkgutil
import random
import shutil
import socket
import struct
import sys
import time
import zlib
import threading
import itertools
import weakref
import warnings

import numpy as np
import pandas as pd

from .compat import irange, functools32, getargspec
from ._utils import to_binary, to_str, to_text, tokenize, tokenize_int, register_tokenizer,\
    insert_reversed_tuple, ceildiv
from .config import options
from .lib.tblib import Traceback

logger = logging.getLogger(__name__)
random.seed(int(time.time()) * os.getpid())


# make flake8 happy by referencing these imports
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
                "'AttributeDict' object has no attribute {0}".format(item))


def on_serialize_shape(shape):
    if shape:
        return tuple(s if not np.isnan(s) else -1 for s in shape)
    return shape


def on_deserialize_shape(shape):
    if shape:
        return tuple(s if s != -1 else np.nan for s in shape)
    return shape


def on_serialize_numpy_type(value):
    return value.item() if isinstance(value, np.generic) else value


def on_serialize_nsplits(value):
    if value is None:
        return None
    new_nsplits = []
    for dim_splits in value:
        new_nsplits.append(tuple(None if np.isnan(v) else v for v in dim_splits))
    return tuple(new_nsplits)


_memory_size_indices = {'': 0, 'k': 1, 'm': 2, 'g': 3, 't': 4}


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
        raise ValueError('Unknown limitation value: {0}'.format(value))


def readable_size(size):
    if size < 1024:
        return size
    elif 1024 <= size < 1024 ** 2:
        return '{0:.2f}K'.format(size * 1.0 / 1024)
    elif 1024 ** 2 <= size < 1024 ** 3:
        return '{0:.2f}M'.format(size * 1.0 / (1024 ** 2))
    elif 1024 ** 3 <= size < 1024 ** 4:
        return '{0:.2f}G'.format(size * 1.0 / (1024 ** 3))
    else:
        return '{0:.2f}T'.format(size * 1.0 / (1024 ** 4))


_commit_hash, _commit_ref = None, None


def git_info():
    from ._version import get_git_info

    global _commit_hash, _commit_ref
    if _commit_ref is not None:
        if _commit_hash is None:
            return None
        return _commit_hash, _commit_ref

    git_tuple = get_git_info()
    if git_tuple is None:
        _commit_ref, _commit_hash = ':INVALID:', None
        return None
    else:
        _commit_hash, _commit_ref = git_tuple
        return git_tuple


LOW_PORT_BOUND = 10000
HIGH_PORT_BOUND = 65535
_local_occupied_ports = set()


def _get_ports_from_netstat():
    import psutil
    import subprocess
    while True:
        p = subprocess.Popen('netstat -a -n -p tcp'.split(), stdout=subprocess.PIPE)
        # in python 2, subprocess does not support waiting for fixed seconds
        ps_proc = psutil.Process(p.pid)
        try:
            ps_proc.wait(5)
            break
        except:  # noqa: E721  # pragma: no cover
            ps_proc.terminate()
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


def get_next_port(typ=None):
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
    for i in irange(LOW_PORT_BOUND, HIGH_PORT_BOUND + 1):
        if i in occupied:
            continue
        if idx == 0:
            _local_occupied_ports.add(i)
            return i
        idx -= 1
    raise SystemError('No ports available.')


@functools32.lru_cache(200)
def mod_hash(val, modulus):
    return tokenize_int(val) % modulus


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


def lazy_import(name, package=None, globals=None, locals=None, rename=None):
    rename = rename or name
    prefix_name = name.split('.', 1)[0]

    class LazyModule(object):
        def __getattr__(self, item):
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


def serialize_graph(graph, compress=False):
    ser_graph = graph.to_pb().SerializeToString()
    if compress:
        ser_graph = zlib.compress(ser_graph)
    return ser_graph


def deserialize_graph(ser_graph, graph_cls=None):
    from google.protobuf.message import DecodeError
    from .serialize.protos.graph_pb2 import GraphDef
    from .graph import DirectedGraph
    graph_cls = graph_cls or DirectedGraph
    ser_graph_bin = to_binary(ser_graph)
    g = GraphDef()
    try:
        ser_graph = ser_graph
        g.ParseFromString(ser_graph_bin)
        return graph_cls.from_pb(g)
    except DecodeError:
        pass

    try:
        ser_graph_bin = zlib.decompress(ser_graph_bin)
        g.ParseFromString(ser_graph_bin)
        return graph_cls.from_pb(g)
    except (zlib.error, DecodeError):
        json_obj = json.loads(to_str(ser_graph))
        return graph_cls.from_json(json_obj)


if sys.version_info[0] < 3:
    def wraps(fun):
        if isinstance(fun, functools.partial):
            return lambda f: f
        return functools.wraps(fun)
else:
    wraps = functools.wraps


def calc_data_size(dt):
    if dt is None:
        return 0

    if isinstance(dt, tuple):
        return sum(calc_data_size(c) for c in dt)

    if hasattr(dt, 'nbytes'):
        return max(sys.getsizeof(dt), dt.nbytes)
    if hasattr(dt, 'shape') and len(dt.shape) == 0:
        return 0
    if hasattr(dt, 'memory_usage'):
        return sys.getsizeof(dt)
    if hasattr(dt, 'dtypes') and hasattr(dt, 'shape'):
        return dt.shape[0] * sum(dtype.itemsize for dtype in dt.dtypes)
    if hasattr(dt, 'dtype') and hasattr(dt, 'shape'):
        return dt.shape[0] * dt.dtype.itemsize

    # object chunk
    return sys.getsizeof(dt)


def get_shuffle_input_keys_idxes(chunk):
    from .operands import ShuffleProxy

    if isinstance(chunk.op, ShuffleProxy):
        return [inp.key for inp in chunk.inputs], [inp.index for inp in chunk.inputs]
    else:
        return chunk.op.to_fetch_keys, chunk.op.to_fetch_idxes


def _get_mod_logger():
    mod_logger = None
    frame_globals = inspect.currentframe().f_back.f_globals
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
    func_args = getargspec(func)

    @wraps(func)
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
                    messages.append('%s=%r' % (k, v))

            err_msg = 'Unexpected exception occurred in %s.' % func_name
            if messages:
                err_msg += ' ' + ' '.join(messages)
            mod_logger.exception(err_msg)
            raise
    return _wrapped


_kernel_mode = threading.local()


def is_eager_mode():
    if getattr(_kernel_mode, 'eager', None) is None:
        return options.eager_mode
    return _kernel_mode.eager


def kernel_mode(func):
    """
    A decorator for kernel functions.
    When eager mode is on, expressions will be executed after `new_entities`, however
    `new_entities` is also called in `Executor` and `OperandTilesHandler`, this decorator
    provides an options context for kernel functions to avoid execution.
    """

    def _wrapped(*args, **kwargs):
        try:
            enter_eager_count = getattr(_kernel_mode, 'eager_count', 0)
            if enter_eager_count == 0:
                _kernel_mode.eager = False
            _kernel_mode.eager_count = enter_eager_count + 1
            return func(*args, **kwargs)
        finally:
            _kernel_mode.eager_count -= 1
            if _kernel_mode.eager_count == 0:
                _kernel_mode.eager = None

    return _wrapped


def build_tileable_graph(tileables, executed_tileable_keys, graph=None):
    from .tiles import TileableGraphBuilder

    with build_mode():
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
                    params.append(p)
                copies = copy_op.new_tileables([replace_with_fetch_or_copy(inp) for inp in n.inputs],
                                               kws=params, output_limit=len(params))
                for o, copy in zip(n.op.outputs, copies):
                    node_to_copy[o] = copy.data
                    copied.add(copy.data)
            return node_to_copy[n]

        tileable_graph_builder = TileableGraphBuilder(
            graph=graph, node_processor=replace_with_fetch_or_copy)
        return tileable_graph_builder.build(tileables)


_build_mode = threading.local()


class BuildMode(object):
    def __init__(self):
        self.is_build_mode = False
        self._old_mode = None
        self._enter_times = 0

    def __enter__(self):
        if self._enter_times == 0:
            # check to prevent nested enter and exit
            self._old_mode = self.is_build_mode
            self.is_build_mode = True
        self._enter_times += 1
        return self

    def __exit__(self, *_):
        self._enter_times -= 1
        if self._enter_times == 0:
            self.is_build_mode = self._old_mode
            self._old_mode = None


def build_mode():
    ret = getattr(_build_mode, 'build_mode', None)
    if ret is None:
        ret = BuildMode()
        _build_mode.build_mode = ret

    return ret


def enter_build_mode(func):
    """
    Decorator version of build_mode.

    :param func: function
    :return: the result of function
    """
    def inner(*args, **kwargs):
        with build_mode():
            return func(*args, **kwargs)
    return inner


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
    from .operands import ShuffleProxy

    chunk_op = chunk.op
    params = chunk.params.copy()

    if isinstance(chunk_op, ShuffleProxy):
        # for shuffle nodes, we build FetchShuffle chunks
        # to replace ShuffleProxy
        to_fetch_keys = [pinp.key for pinp in chunk.inputs
                         if input_chunk_keys is None or pinp.key in input_chunk_keys]
        to_fetch_idxes = [pinp.index for pinp in chunk.inputs]
        op = chunk_op.get_fetch_op_cls(chunk)(to_fetch_keys=to_fetch_keys, to_fetch_idxes=to_fetch_idxes)
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
    from .core import Chunk, ChunkData
    if isinstance(entity, (Chunk, ChunkData)):
        return build_fetch_chunk(entity)
    elif hasattr(entity, 'tiles'):
        return build_fetch_tileable(entity)
    else:
        raise TypeError('Type %s not supported' % type(entity).__name__)


def build_fuse_chunk(fused_chunks, **kwargs):
    head_chunk = fused_chunks[0]
    tail_chunk = fused_chunks[-1]
    chunk_op = tail_chunk.op
    params = tail_chunk.params.copy()

    fuse_op = chunk_op.get_fuse_op_cls(tail_chunk)(
        sparse=chunk_op.sparse, _key=chunk_op.key, _gpu=tail_chunk.op.gpu,
        _operands=[c.op for c in fused_chunks])
    return fuse_op.new_chunk(
        head_chunk.inputs, kws=[params], _key=tail_chunk.key, _composed=fused_chunks, **kwargs)


def get_chunk_shuffle_key(chunk):
    op = chunk.op
    try:
        return op.shuffle_key
    except AttributeError:
        from .operands import Fuse
        if isinstance(op, Fuse):
            return chunk.composed[0].op.shuffle_key
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
                raise TypeError('unsupported type %s' % type(v))
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
                raise error_cls(
                    'Input tileable {} has chunks with unknown shape'.format(t))


def has_unknown_shape(tiled):
    if getattr(tiled, 'shape', None) is None:
        return False
    if any(np.isnan(s) for s in tiled.shape):
        return True
    if any(np.isnan(s) for s in itertools.chain(*tiled.nsplits)):
        return True
    return False


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
    children = proc.children(recursive=True)
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


def copy_tileables(tileables, **kwargs):
    inputs = kwargs.pop('inputs', None)
    copy_key = kwargs.pop('copy_key', True)
    copy_id = kwargs.pop('copy_id', True)
    if kwargs:
        raise TypeError("got un unexpected "
                        "keyword argument '{}'".format(next(iter(kwargs))))
    if len(tileables) > 1:
        # cannot handle tileables with different operands here
        # try to copy separately if so
        if len({t.op for t in tileables}) != 1:
            raise TypeError("All tileables' operands should be same.")

    op = tileables[0].op.copy().reset_key()
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
            def inner(*args, **kwargs):
                return func(*args, **kwargs)
            return inner
        else:
            return
    return wrap


def require_module(module):
    def wrap(func):
        try:
            importlib.import_module(module)

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
