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
import collections
import functools
import inspect
import json
import logging
import numbers
import os
import random
import socket
import struct
import subprocess
import sys
import time
import uuid
from collections import deque
from hashlib import md5
from datetime import date, datetime, timedelta

import numpy as np

from .compat import six, irange, functools32, getargspec
from .utils_c import *

logger = logging.getLogger(__name__)
random.seed(int(time.time()) * os.getpid())


if 'to_binary' not in globals():
    def to_binary(text, encoding='utf-8'):
        if text is None:
            return text
        if isinstance(text, six.text_type):
            return text.encode(encoding)
        elif isinstance(text, (six.binary_type, bytearray)):
            return bytes(text)
        else:
            return str(text).encode(encoding) if six.PY3 else str(text)


if 'to_text' not in globals():
    def to_text(binary, encoding='utf-8'):
        if binary is None:
            return binary
        if isinstance(binary, (six.binary_type, bytearray)):
            return binary.decode(encoding)
        elif isinstance(binary, six.text_type):
            return binary
        else:
            return str(binary) if six.PY3 else str(binary).decode(encoding)


if 'to_str' not in globals():
    def to_str(text, encoding='utf-8'):
        return to_text(text, encoding=encoding) if six.PY3 else to_binary(text, encoding=encoding)


def build_id(prefix=''):
    return prefix + '-' + str(uuid.uuid1())


# fix encoding conversion problem under windows
if sys.platform == 'win32':
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


if 'tokenize' not in globals():
    def tokenize(*args, **kwargs):
        try:
            import numpy as np

            def h_numpy(ob):
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
                        data = md5(b'-'.join([six.text_type(item).encode('utf-8') for item in
                                              ob.flat])).hexdigest()
                else:
                    try:
                        data = md5(ob.ravel().view('i1').data).hexdigest()
                    except (BufferError, AttributeError, ValueError):
                        data = md5(ob.copy().ravel().view('i1').data).hexdigest()
                return data, ob.dtype, ob.shape, ob.strides
        except ImportError:
            np = None
            h_numpy = None

        def h(ob):
            try:
                return h_non_iterative(ob)
            except TypeError:
                if isinstance(ob, dict):
                    return list, h_iterative(sorted(list(ob.items()), key=str))
                if isinstance(ob, set):
                    return list, h_iterative(sorted(ob, key=str))
                if isinstance(ob, (tuple, list)):
                    return type(ob), h_iterative(ob)
                raise TypeError('Cannot generate token for %s, type: %s' % (ob, type(ob)))

        def h_iterative(ob):
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

        def h_non_iterative(ob):
            if isinstance(ob, (int, float, str, six.text_type, bytes,
                               type(None), type, slice, date, datetime, timedelta)):
                return ob
            if isinstance(ob, six.integer_types+(complex,)):
                return ob
            if hasattr(ob, 'key'):
                return ob.key
            # numpy relative
            if h_numpy and isinstance(ob, np.ndarray):
                return h_numpy(ob)
            if np and isinstance(ob, (np.dtype, np.generic)):
                return repr(ob)

            raise TypeError

        def h_old(ob):
            if isinstance(ob, (int, float, str, six.text_type, bytes,
                               type(None), type, slice, date, datetime, timedelta)):
                return ob
            if isinstance(ob, six.integer_types+(complex,)):
                return ob
            if isinstance(ob, dict):
                return h_old(sorted(list(ob.items()), key=str))
            if isinstance(ob, (tuple, list)):
                return type(ob), list(h_old(it) for it in ob)
            if isinstance(ob, set):
                return h_old(sorted(ob, key=str))
            if hasattr(ob, 'key'):
                return ob.key
            # numpy relative
            if h_numpy and isinstance(ob, np.ndarray):
                return h_numpy(ob)
            if np and isinstance(ob, (np.dtype, np.generic)):
                return repr(ob)

            raise TypeError('Cannot generate token for %s, type: %s' % (ob, type(ob)))

        if kwargs:
            args = args + (kwargs,)
        return md5(str([h_old(arg) for arg in args]).encode('utf-8')).hexdigest()


class AttributeDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(
                "'AttributeDict' object has no attribute {0}".format(item))


def hashable(obj):
    if isinstance(obj, six.string_types):
        items = obj
    elif isinstance(obj, slice):
        items = (obj.start, obj.stop, obj.step)
    elif isinstance(obj, collections.Mapping):
        items = type(obj)((k, hashable(v)) for k, v in six.iteritems(obj))
    elif isinstance(obj, collections.Iterable):
        items = tuple(hashable(item) for item in obj)
    elif isinstance(obj, collections.Hashable):
        items = obj
    elif hasattr(obj, 'key'):
        items = obj.key
    else:
        raise TypeError(type(obj))

    return items


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


def get_next_port(typ=None):
    import psutil
    try:
        conns = psutil.net_connections()
        typ = typ or socket.SOCK_STREAM
        occupied = set(sc.laddr.port for sc in conns
                       if sc.type == typ and LOW_PORT_BOUND <= sc.laddr.port <= HIGH_PORT_BOUND)
    except psutil.AccessDenied:
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

    randn = struct.unpack('<Q', os.urandom(8))[0]
    idx = int(randn % (1 + HIGH_PORT_BOUND - LOW_PORT_BOUND - len(occupied)))
    for i in irange(LOW_PORT_BOUND, HIGH_PORT_BOUND + 1):
        if i in occupied:
            continue
        if idx == 0:
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


class PlasmaProcessHelper(object):
    def __init__(self, base_directory=None, proc_name=None, size=1024 ** 3,
                 socket='/tmp/plasma', one_mapped_file=False, mount_point=None, huge_pages=False):
        import pyarrow

        self._proc_name = proc_name
        self._size = size
        self._socket = socket
        self._mount_point = mount_point
        self._enable_huge_pages = huge_pages
        self._one_mapped_file = one_mapped_file

        if not base_directory:
            base_directory = pyarrow.__path__[0]
        self._base_dir = base_directory

        if proc_name is None:
            for pname in ('plasma_store', 'plasma_store_server'):
                if sys.platform == 'win32':
                    pname += '.exe'
                if os.path.exists(os.path.join(self._base_dir, pname)):
                    self._proc_name = pname
                    break

        self._process = None

    def run(self, proc_args=None):
        proc_args = proc_args or []
        if self._proc_name is None:
            raise RuntimeError('Plasma store not found.')
        args = [os.path.join(self._base_dir, self._proc_name), '-m', str(self._size),
                '-s', self._socket]
        if self._mount_point:
            args.extend(['-d', self._mount_point])
        if self._enable_huge_pages:
            args.append('-h')
        if self._one_mapped_file:
            args.append('-f')
        args.extend(proc_args)

        daemon = subprocess.Popen(args)
        logger.debug('Started %d' % daemon.pid)
        logger.debug('Params: %s' % args)
        time.sleep(2)
        self._process = daemon

    def stop(self):
        self._process.kill()


def serialize_graph(graph):
    return base64.b64encode(graph.to_pb().SerializeToString())


def deserialize_graph(graph_b64):
    from .serialize.protos.graph_pb2 import GraphDef
    from .graph import DirectedGraph
    try:
        json_obj = json.loads(to_str(graph_b64))
        return DirectedGraph.from_json(json_obj)
    except (SyntaxError, ValueError):
        g = GraphDef()
        g.ParseFromString(base64.b64decode(graph_b64))
        return DirectedGraph.from_pb(g)


def merge_tensor_chunks(input_tensor, ctx):
    from .tensor.execution.core import Executor
    from .tensor.expressions.datasource import TensorFetchChunk

    if len(input_tensor.chunks) == 1:
        return ctx[input_tensor.chunks[0].key]

    chunks = []
    for c in input_tensor.chunks:
        op = TensorFetchChunk(dtype=c.dtype, to_fetch_key=c.key)
        chunk = op.new_chunk(None, c.shape, index=c.index, _key=c.key)
        chunks.append(chunk)

    new_op = TensorFetchChunk(dtype=input_tensor.dtype, to_fetch_key=input_tensor.key)
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


def log_unhandled(func):
    frame_globals = inspect.currentframe().f_back.f_globals
    mod_logger = None
    for logger_name in ('logger', 'LOG', 'LOGGER'):
        if logger_name in frame_globals:
            mod_logger = frame_globals[logger_name]
            break
    if not mod_logger:
        return func

    func_name = getattr(func, '__qualname__', func.__module__ + func.__name__)
    func_args = getargspec(func)

    @wraps(func)
    def _wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            kwcopy = kwargs.copy()
            kwcopy.update(zip(func_args.args, args))

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
