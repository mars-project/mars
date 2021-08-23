#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import asyncio
import dataclasses
import functools
import importlib
import io
import itertools
import logging
import numbers
import operator
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
import zlib
from abc import ABC
from contextlib import contextmanager
from typing import Any, List, Dict, Set, Tuple, Type, Union, Callable, Optional

import numpy as np
import pandas as pd

from ._utils import to_binary, to_str, to_text, TypeDispatcher, \
    tokenize, tokenize_int, register_tokenizer, insert_reversed_tuple, ceildiv
from .typing import ChunkType, TileableType, EntityType, OperandType

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


def on_serialize_shape(shape: Tuple[int]):
    if shape:
        return tuple(s if not np.isnan(s) else -1 for s in shape)
    return shape


def on_deserialize_shape(shape: Tuple[int]):
    if shape:
        return tuple(s if s != -1 else np.nan for s in shape)
    return shape


def on_serialize_numpy_type(value: np.dtype):
    if value is pd.NaT:
        value = None
    return value.item() if isinstance(value, np.generic) else value


def on_serialize_nsplits(value: Tuple[Tuple[int]]):
    if value is None:
        return None
    new_nsplits = []
    for dim_splits in value:
        new_nsplits.append(tuple(None if np.isnan(v) else v for v in dim_splits))
    return tuple(new_nsplits)


_memory_size_indices = {'': 0, 'k': 1, 'm': 2, 'g': 3, 't': 4}


def calc_size_by_str(value: Union[str, int],
                     total: Union[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    mem_limit, is_percent = parse_readable_size(value)
    if is_percent:
        return int(total * mem_limit)
    else:
        return int(mem_limit)


def parse_readable_size(value: Union[str, int, float]) -> Tuple[float, bool]:
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


def readable_size(size: int, trunc: bool = False) -> str:
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


def _get_ports_from_netstat() -> Set[int]:
    import subprocess
    while True:
        p = subprocess.Popen('netstat -a -n -p tcp'.split(), stdout=subprocess.PIPE)
        try:
            outs, _ = p.communicate(timeout=5)
            outs = outs.split(to_binary(os.linesep))
            occupied = set()
            for line in outs:
                if b'.' not in line:
                    continue
                line = to_str(line)
                for part in line.split():
                    # in windows, netstat uses ':' to separate host and port
                    part = part.replace(':', '.')
                    if '.' in part:
                        _, port_str = part.rsplit('.', 1)
                        if port_str == '*':
                            continue
                        port = int(port_str)
                        if LOW_PORT_BOUND <= port <= HIGH_PORT_BOUND:
                            occupied.add(int(port_str))
                        break
            return occupied
        except subprocess.TimeoutExpired:
            p.kill()
            continue


def get_next_port(typ: int = None,
                  occupy: bool = True) -> int:
    import psutil
    if sys.platform.lower().startswith('win'):
        occupied = _get_ports_from_netstat()
    else:
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
def mod_hash(val: Any, modulus: int):
    return tokenize_int(val) % modulus


class classproperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


def lazy_import(name: str,
                package: str = None,
                globals: Dict = None,  # pylint: disable=redefined-builtin
                locals: Dict = None,  # pylint: disable=redefined-builtin
                rename: str = None):
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


def serialize_serializable(serializable, compress: bool = False):
    from .serialization import serialize

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


def deserialize_serializable(ser_serializable: bytes):
    from .serialization import deserialize

    bio = io.BytesIO(ser_serializable)
    s_header_length = struct.unpack('Q', bio.read(8))[0]
    header2 = pickle.loads(bio.read(s_header_length))
    buffers2 = [bio.read(s) for s in header2['buf_sizes']]
    return deserialize(header2, buffers2)


def register_ray_serializer(obj_type, serializer=None, deserializer=None):
    ray = lazy_import("ray")
    if ray:
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


def calc_data_size(dt: Any, shape: Tuple[int] = None) -> int:
    from .dataframe.core import IndexValue

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
            if hasattr(index_value_value, 'dtype') \
                    and not isinstance(index_value_value, IndexValue.RangeIndex):
                size += calc_data_size(index_value_value, shape=shape)
        except AttributeError:
            pass
        return size
    if hasattr(dt, 'dtype') and shape is not None:
        return shape[0] * dt.dtype.itemsize

    # object chunk
    return sys.getsizeof(dt)


def build_fetch_chunk(chunk: ChunkType,
                      input_chunk_keys: List[str] = None,
                      **kwargs) -> ChunkType:
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
            source_mappers=source_mappers, gpu=chunk.op.gpu)
    else:
        # for non-shuffle nodes, we build Fetch chunks
        # to replace original chunk
        op = chunk_op.get_fetch_op_cls(chunk)(sparse=chunk.op.sparse, gpu=chunk.op.gpu)
    return op.new_chunk(None, kws=[params], _key=chunk.key, _id=chunk.id, **kwargs)


def build_fetch_tileable(tileable: TileableType) -> TileableType:
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


def build_fetch(entity: EntityType) -> EntityType:
    from .core import CHUNK_TYPE, ENTITY_TYPE
    if isinstance(entity, CHUNK_TYPE):
        return build_fetch_chunk(entity)
    elif isinstance(entity, ENTITY_TYPE):
        return build_fetch_tileable(entity)
    else:
        raise TypeError(f'Type {type(entity)} not supported')


def get_chunk_mapper_id(chunk: ChunkType) -> str:
    op = chunk.op
    try:
        return op.mapper_id
    except AttributeError:
        from .core.operand import Fuse
        if isinstance(op, Fuse):
            return chunk.composed[-1].op.mapper_id
        else:  # pragma: no cover
            raise


def get_chunk_reducer_index(chunk: ChunkType) -> Tuple[int]:
    op = chunk.op
    try:
        return op.reducer_index
    except AttributeError:
        from .core.operand import Fuse
        if isinstance(op, Fuse):
            return chunk.composed[0].op.reducer_index
        else:  # pragma: no cover
            raise


def merge_chunks(chunk_results: List[Tuple[Tuple[int], Any]]) -> Any:
    """
    Concatenate chunk results according to index.

    Parameters
    ----------
    chunk_results : list of tuple, {(chunk_idx, chunk_result), ...,}

    Returns
    -------
    Data
    """
    from .dataframe.utils import is_dataframe, is_index, is_series, get_xdf
    from .lib.groupby_wrapper import GroupByWrapper
    from .tensor.array_utils import get_array_module, is_array

    chunk_results = sorted(chunk_results, key=operator.itemgetter(0))
    v = chunk_results[0][1]
    if len(chunk_results) == 1 and not (chunk_results[0][0]):
        return v
    if is_array(v):
        xp = get_array_module(v)
        ndim = v.ndim
        for i in range(ndim - 1):
            new_chunks = []
            for idx, cs in itertools.groupby(chunk_results, key=lambda t: t[0][:-1]):
                new_chunks.append((idx, xp.concatenate([c[1] for c in cs], axis=ndim - i - 1)))
            chunk_results = new_chunks
        to_concat = [c[1] for c in chunk_results]
        if len(to_concat) == 1:
            return to_concat[0]
        concat_result = xp.concatenate(to_concat)
        return concat_result
    elif is_dataframe(v):
        xdf = get_xdf(v)
        concats = []
        for _, cs in itertools.groupby(chunk_results, key=lambda t: t[0][0]):
            concats.append(xdf.concat([c[1] for c in cs], axis='columns'))
        return xdf.concat(concats, axis='index')
    elif is_series(v):
        xdf = get_xdf(v)
        return xdf.concat([c[1] for c in chunk_results])
    elif is_index(v):
        xdf = get_xdf(v)
        df = xdf.concat([xdf.DataFrame(index=r[1])
                        for r in chunk_results])
        return df.index
    elif isinstance(v, pd.Categorical):
        categories = [r[1] for r in chunk_results]
        arrays = [np.asarray(r) for r in categories]
        array = np.concatenate(arrays)
        return pd.Categorical(array, categories=categories[0].categories,
                              ordered=categories[0].ordered)
    elif isinstance(v, GroupByWrapper):
        df = pd.concat([r[1].obj for r in chunk_results], axis=0)
        if not isinstance(v.keys, list):
            keys = v.keys
        else:
            keys = []
            for idx, k in enumerate(v.keys):
                if isinstance(k, pd.Series):
                    keys.append(pd.concat([r[1].keys[idx] for r in chunk_results]))
                else:
                    keys.append(k)
        grouped = GroupByWrapper(df, None, keys=keys, axis=v.axis, level=v.level,
                                 exclusions=v.exclusions, selection=v.selection,
                                 as_index=v.as_index, sort=v.sort,
                                 group_keys=v.group_keys, squeeze=v.squeeze,
                                 observed=v.observed, mutated=v.mutated)
        return grouped.groupby_obj
    elif isinstance(v, (str, bytes, memoryview)):
        result = [r[1] for r in chunk_results]
        if len(result) == 1:
            return result[0]
        return result
    else:
        result = None
        for cr in chunk_results:
            if cr[1] is None:
                continue
            if isinstance(cr[1], dict) and not cr[1]:
                continue
            if result is None:
                result = cr[1]
                result = result.item() if hasattr(result, 'item') else result
            else:
                raise TypeError(f'unsupported type {type(v)}')
        return result


def calc_nsplits(chunk_idx_to_shape: Dict[Tuple[int], Tuple[int]]) -> Tuple[Tuple[int]]:
    """
    Calculate a tiled entity's nsplits.

    Parameters
    ----------
    chunk_idx_to_shape : Dict type, {chunk_idx: chunk_shape}

    Returns
    -------
    nsplits
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


def sort_dataframe_result(df, result: pd.DataFrame) -> pd.DataFrame:
    """ sort DataFrame on client according to `should_be_monotonic` attribute """
    if hasattr(df, 'index_value'):
        if getattr(df.index_value, 'should_be_monotonic', False):
            try:
                result.sort_index(inplace=True)
            except TypeError:  # pragma: no cover
                # cudf doesn't support inplace
                result = result.sort_index()
        if hasattr(df, 'columns_value'):
            if getattr(df.columns_value, 'should_be_monotonic', False):
                try:
                    result.sort_index(axis=1, inplace=True)
                except TypeError:  # pragma: no cover
                    # cudf doesn't support inplace
                    result = result.sort_index(axis=1)
    return result


def numpy_dtype_from_descr_json(obj: Union[list, np.dtype]) -> np.dtype:
    """
    Construct numpy dtype from it's np.dtype.descr.

    The dtype can be trivial, but can also be very complex (nested) record type. In that
    case, the tuple in `descr` will be made as `list`, which can be understood by `np.dtype()`.
    This utility helps the reconstruct work.
    """
    if isinstance(obj, list):
        return np.dtype([(k, numpy_dtype_from_descr_json(v)) for k, v in obj])
    return obj


def has_unknown_shape(*tiled_tileables: TileableType) -> bool:
    for tileable in tiled_tileables:
        if getattr(tileable, 'shape', None) is None:
            continue
        if any(pd.isnull(s) for s in tileable.shape):
            return True
        if any(pd.isnull(s) for s in itertools.chain(*tileable.nsplits)):
            return True
    return False


def sbytes(x: Any) -> bytes:
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


def kill_process_tree(pid: int , include_parent: bool = True):
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
                try:
                    plasma_sock_dir = next((conn.laddr for conn in p.connections('unix')
                                            if 'plasma' in conn.laddr), None)
                except psutil.AccessDenied:
                    pass
            p.kill()
        except psutil.NoSuchProcess:  # pragma: no cover
            pass
    if plasma_sock_dir:
        shutil.rmtree(plasma_sock_dir, ignore_errors=True)


def copy_tileables(tileables: List[TileableType], **kwargs):
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


def require_not_none(obj: Any):
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


def ignore_warning(func: Callable):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return func(*args, **kwargs)
    return inner


def flatten(nested_iterable: Union[List, Tuple]) -> List:
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


def stack_back(flattened: List,
               raw: Union[List, Tuple]) -> Union[List, Tuple]:
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


def build_fuse_chunk(fused_chunks: List[ChunkType],
                     fuse_op_cls: Type[OperandType],
                     op_kw: Dict = None,
                     chunk_kw: Dict = None) -> ChunkType:
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


def adapt_mars_docstring(doc: str) -> str:
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


def is_object_dtype(dtype: np.dtype) -> bool:
    try:
        return np.issubdtype(dtype, np.object_) \
               or np.issubdtype(dtype, np.unicode_) \
               or np.issubdtype(dtype, np.bytes_)
    except TypeError:  # pragma: no cover
        return False


def get_dtype(dtype: Union[np.dtype, pd.api.extensions.ExtensionDtype]):
    if pd.api.types.is_extension_array_dtype(dtype):
        return dtype
    else:
        return np.dtype(dtype)


def calc_object_overhead(chunk: ChunkType,
                         shape: Tuple[int]) -> int:
    from .dataframe.core import DATAFRAME_CHUNK_TYPE, SERIES_CHUNK_TYPE, INDEX_CHUNK_TYPE

    if not shape or np.isnan(shape[0]) or getattr(chunk, 'dtypes', None) is None:
        return 0

    if isinstance(chunk, DATAFRAME_CHUNK_TYPE) \
            and chunk.dtypes is not None:
        n_strings = len([dt for dt in chunk.dtypes if is_object_dtype(dt)])
        if chunk.index_value \
                and is_object_dtype(getattr(chunk.index_value.value, 'dtype', None)):
            n_strings += 1
    elif isinstance(chunk, SERIES_CHUNK_TYPE) \
            and chunk.dtype is not None:
        n_strings = 1 if is_object_dtype(chunk.dtype) else 0
        if chunk.index_value \
                and is_object_dtype(getattr(chunk.index_value.value, 'dtype', None)):
            n_strings += 1
    elif isinstance(chunk, INDEX_CHUNK_TYPE) \
            and chunk.dtype is not None:
        n_strings = 1 if is_object_dtype(chunk.dtype) else 0
    else:
        n_strings = 0
    return n_strings * shape[0] * OBJECT_FIELD_OVERHEAD


def arrow_array_to_objects(obj: Union[pd.DataFrame, pd.Series]) \
        -> Union[pd.DataFrame, pd.Series]:
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


def enter_current_session(func: Callable):
    @functools.wraps(func)
    def wrapped(cls, ctx, op):
        from .deploy.oscar.session import AbstractSession, get_default_session

        # skip in some test cases
        if not hasattr(ctx, 'get_current_session'):
            return func(cls, ctx, op)

        session = ctx.get_current_session()
        prev_default_session = get_default_session()
        session.as_default()

        try:
            result = func(cls, ctx, op)
        finally:
            if prev_default_session:
                prev_default_session.as_default()
            else:
                AbstractSession.reset_default()

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


def implements(f: Callable):
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


def find_objects(nested: Union[List, Dict],
                 types: Union[Type, Tuple[Type]]) -> List:
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


def replace_objects(nested: Union[List, Dict],
                    mapping: Dict) -> Union[List, Dict]:
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


def get_params_fields(chunk):
    from .dataframe.core import DATAFRAME_CHUNK_TYPE, \
        DATAFRAME_GROUPBY_CHUNK_TYPE, SERIES_GROUPBY_CHUNK_TYPE

    fields = list(chunk.params)
    if isinstance(chunk, DATAFRAME_CHUNK_TYPE):
        fields.remove('dtypes')
        fields.remove('columns_value')
    elif isinstance(chunk, DATAFRAME_GROUPBY_CHUNK_TYPE):
        fields.remove('dtypes')
        fields.remove('key_dtypes')
        fields.remove('columns_value')
    elif isinstance(chunk, SERIES_GROUPBY_CHUNK_TYPE):
        fields.remove('key_dtypes')

    return fields


# Please refer to https://bugs.python.org/issue41451
try:
    class _Dummy(ABC):
        __slots__ = ('__weakref__',)
    abc_type_require_weakref_slot = True
except TypeError:
    abc_type_require_weakref_slot = False


def patch_asyncio_task_create_time():  # pragma: no cover
    new_loop = False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        new_loop = True
    loop_class = loop.__class__
    # Save raw loop_class.create_task and make multiple apply idempotent
    loop_create_task = getattr(patch_asyncio_task_create_time, 'loop_create_task', loop_class.create_task)
    patch_asyncio_task_create_time.loop_create_task = loop_create_task

    def new_loop_create_task(*args, **kwargs):
        task = loop_create_task(*args, **kwargs)
        task.__mars_asyncio_task_create_time__ = time.time()
        return task

    if loop_create_task is not new_loop_create_task:
        loop_class.create_task = new_loop_create_task
    if not new_loop and loop.create_task is not new_loop_create_task:
        loop.create_task = functools.partial(new_loop_create_task, loop)


async def asyncio_task_timeout_detector(
        check_interval: int, task_timeout_seconds: int, task_exclude_filters: List[str]):
    task_exclude_filters.append('asyncio_task_timeout_detector')
    while True:  # pragma: no cover
        await asyncio.sleep(check_interval)
        loop = asyncio.get_running_loop()
        current_time = time.time()  # avoid invoke `time.time()` frequently if we have plenty of unfinished tasks.
        for task in asyncio.all_tasks(loop=loop):
            # Some task may be create before `patch_asyncio_task_create_time` applied, take them as never timeout.
            create_time = getattr(task, '__mars_asyncio_task_create_time__', current_time)
            if current_time - create_time >= task_timeout_seconds:
                stack = io.StringIO()
                task.print_stack(file=stack)
                task_str = str(task)
                if any(excluded_task in task_str for excluded_task in task_exclude_filters):
                    continue
                logger.warning('''Task %s in event loop %s doesn't finish in %s seconds. %s''',
                               task, loop, time.time() - create_time, stack.getvalue())


def register_asyncio_task_timeout_detector(
        check_interval: int = None,
        task_timeout_seconds: int = None,
        task_exclude_filters: List[str] = None) -> Optional[asyncio.Task]:  # pragma: no cover
    """Register a asyncio task which print timeout task periodically."""
    check_interval = check_interval or int(os.environ.get('MARS_DEBUG_ASYNCIO_TASK_TIMEOUT_CHECK_INTERVAL', -1))
    if check_interval > 0:
        patch_asyncio_task_create_time()
        task_timeout_seconds = task_timeout_seconds or int(
            os.environ.get('MARS_DEBUG_ASYNCIO_TASK_TIMEOUT_SECONDS', check_interval))
        if not task_exclude_filters:
            # Ignore mars/oscar by default since it has some long-running coroutines.
            task_exclude_filters = os.environ.get('MARS_DEBUG_ASYNCIO_TASK_EXCLUDE_FILTERS', 'mars/oscar')
            task_exclude_filters = task_exclude_filters.split(';')
        if sys.version_info[:2] < (3, 7):
            logger.warning('asyncio tasks timeout detector is not supported under python %s', sys.version)
        else:
            loop = asyncio.get_running_loop()
            logger.info('Create asyncio tasks timeout detector with check_interval %s task_timeout_seconds %s '
                        'task_exclude_filters %s', check_interval, task_timeout_seconds, task_exclude_filters)
            return loop.create_task(asyncio_task_timeout_detector(
                check_interval, task_timeout_seconds, task_exclude_filters))
    else:
        return None


def ensure_own_data(data: np.ndarray) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        return data
    if not data.flags['OWNDATA']:
        return data.copy()
    else:
        return data


def get_chunk_key_to_data_keys(chunk_graph):
    from .core.operand import FetchShuffle, MapReduceOperand, OperandStage

    chunk_key_to_data_keys = dict()
    for chunk in chunk_graph:
        if chunk.key in chunk_key_to_data_keys:
            continue
        if not isinstance(chunk.op, FetchShuffle):
            chunk_key_to_data_keys[chunk.key] = [chunk.key]
        else:
            keys = []
            for succ in chunk_graph.iter_successors(chunk):
                if isinstance(succ.op, MapReduceOperand) and \
                        succ.op.stage == OperandStage.reduce:
                    for key in succ.op.get_dependent_data_keys():
                        if key not in keys:
                            keys.append(key)
            chunk_key_to_data_keys[chunk.key] = keys
    return chunk_key_to_data_keys


class ModulePlaceholder:
    def __init__(self, mod_name: str):
        self._mod_name = mod_name

    def _raises(self):
        raise AttributeError(f'{self._mod_name} is required but not installed.')

    def __getattr__(self, key):
        self._raises()

    def __call__(self, *_args, **_kwargs):
        self._raises()
