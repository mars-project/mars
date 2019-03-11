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

import base64
import functools
import inspect
import json
import logging
import numbers
import os
import random
import socket
import struct
import sys
import time
import zlib
from hashlib import md5

import numpy as np

from .compat import irange, functools32, getargspec
from .utils_c import to_binary, to_str, to_text, tokenize

logger = logging.getLogger(__name__)
random.seed(int(time.time()) * os.getpid())


tokenize = tokenize


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


def get_gpu_used_memory(device_id):
    import pynvml

    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used


def parse_memory_limit(value):
    if isinstance(value, numbers.Number):
        return float(value), False
    elif value.endswith('%'):
        return float(value[:-1]) / 100, True
    elif value.lower().endswith('t'):
        return float(value[:-1]) * (1024 ** 4), False
    elif value.lower().endswith('g'):
        return float(value[:-1]) * (1024 ** 3), False
    elif value.lower().endswith('m'):
        return float(value[:-1]) * (1024 ** 2), False
    elif value.lower().endswith('k'):
        return float(value[:-1]) * 1024, False
    else:
        raise ValueError('Unknown limitation value: {0}'.format(value))


def readable_size(size):
    if size < 1024:
        return size
    elif 1024 <= size < 1024 ** 2:
        return '{0:.2f}K'.format(size / 1024)
    elif 1024 ** 2 <= size < 1024 ** 3:
        return '{0:.2f}M'.format(size / (1024 ** 2))
    elif 1024 ** 3 <= size < 1024 ** 4:
        return '{0:.2f}G'.format(size / (1024 ** 3))
    else:
        return '{0:.2f}T'.format(size / (1024 ** 4))


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
    import subprocess
    p = subprocess.Popen('netstat -a -n -p tcp'.split(), stdout=subprocess.PIPE)
    p.wait()
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
    return int(md5(to_binary(val)).hexdigest(), 16) % modulus


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


def serialize_graph(graph, compress=False):
    ser_graph = graph.to_pb().SerializeToString()
    if compress:
        ser_graph = zlib.compress(ser_graph)
    return base64.b64encode(ser_graph)


def deserialize_graph(graph_b64, graph_cls=None):
    from .serialize.protos.graph_pb2 import GraphDef
    from .graph import DirectedGraph
    graph_cls = graph_cls or DirectedGraph
    try:
        json_obj = json.loads(to_str(graph_b64))
        return graph_cls.from_json(json_obj)
    except (SyntaxError, ValueError):
        g = GraphDef()
        ser_graph = base64.b64decode(graph_b64)
        try:
            ser_graph = zlib.decompress(ser_graph)
        except zlib.error:
            pass
        g.ParseFromString(ser_graph)
        return graph_cls.from_pb(g)


def merge_tensor_chunks(input_tensor, ctx):
    from .tensor.execution.core import Executor
    from .tensor.expressions.datasource import TensorFetchChunk

    if len(input_tensor.chunks) == 1:
        return ctx[input_tensor.chunks[0].key]

    chunks = []
    for c in input_tensor.chunks:
        op = TensorFetchChunk(dtype=c.dtype, to_fetch_key=c.key, sparse=c.op.sparse)
        chunk = op.new_chunk(None, c.shape, index=c.index, _key=c.key)
        chunks.append(chunk)

    new_op = TensorFetchChunk(dtype=input_tensor.dtype, to_fetch_key=input_tensor.key,
                              sparse=input_tensor.op.sparse)
    tensor = new_op.new_tensor(None, input_tensor.shape, chunks=chunks,
                               nsplits=input_tensor.nsplits)

    executor = Executor(storage=ctx)
    concat_result = executor.execute_tensor(tensor, concat=True)
    return concat_result[0]


if sys.version_info[0] < 3:
    def wraps(fun):
        if isinstance(fun, functools.partial):
            return lambda f: f
        return functools.wraps(fun)
else:
    wraps = functools.wraps


def calc_data_size(dt):
    if isinstance(dt, tuple):
        return sum(c.nbytes for c in dt)
    else:
        return dt.nbytes


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
