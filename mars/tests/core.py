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
import itertools
import os
import logging
import shutil
import subprocess
import tempfile
import time
import unittest
from collections.abc import Iterable
from weakref import ReferenceType

import numpy as np
import pandas as pd

from mars.serialize import serializes, deserializes, \
    ProtobufSerializeProvider, JsonSerializeProvider
from mars.utils import lazy_import
from mars.executor import Executor, GraphExecution

try:
    import pytest
except ImportError:
    pytest = None

from unittest import mock
_mock = mock

cupy = lazy_import('cupy', globals=globals())
cudf = lazy_import('cudf', globals=globals())

logger = logging.getLogger(__name__)


class TestCase(unittest.TestCase):
    pass


class MultiGetDict(dict):
    def __getitem__(self, item):
        if isinstance(item, tuple):
            return tuple(super(MultiGetDict, self).__getitem__(it)
                         for it in item)
        return super(MultiGetDict, self).__getitem__(item)


def parameterized(**params):
    """
    Parameterized testing decorator for unittest
    :param params: dictionary of params, e.g.:
    {
        'test1': {'param1': value1, 'param2': value2},
        'test2': {'param1': value1, 'param2': value2},
    }
    the test case's name will generated by dictionary keys.
    :return: Generated test classes
    """
    def _build_parameterized_case(cls, fun):
        @functools.wraps(fun)
        def test_fun(self, *args, **kwargs):
            for n, param in params.items():
                for pn, pv in param.items():
                    setattr(cls, pn, pv)
                try:
                    fun(self, *args, **kwargs)
                except:  # noqa: E722
                    logger.error('Failed when running %s with param %s', fun.__name__, n)
                    raise
        return test_fun

    def decorator(cls):
        fun_dict = dict()
        for fun_name, fun in cls.__dict__.items():
            if callable(fun) and fun_name.startswith('test'):
                fun_dict[fun_name] = _build_parameterized_case(cls, fun)
        for fun_name, fun in fun_dict.items():
            setattr(cls, fun_name, fun)
        return cls

    return decorator


class TestBase(unittest.TestCase):
    def setUp(self):
        self.pb_serialize = lambda *args, **kw: \
            serializes(ProtobufSerializeProvider(), *args, **kw)
        self.pb_deserialize = lambda *args, **kw: \
            deserializes(ProtobufSerializeProvider(), *args, **kw)
        self.json_serialize = lambda *args, **kw: \
            serializes(JsonSerializeProvider(), *args, **kw)
        self.json_deserialize = lambda *args, **kw: \
            deserializes(JsonSerializeProvider(), *args, **kw)

    @classmethod
    def _serial(cls, obj):
        from mars.operands import Operand
        from mars.core import Entity, TileableData, Chunk, ChunkData

        if isinstance(obj, (Entity, Chunk)):
            obj = obj.data

        to_serials = set()

        def serial(ob):
            if ob in to_serials:
                return
            if isinstance(ob, TileableData):
                to_serials.add(ob)
                [serial(i) for i in (ob.chunks or [])]
                serial(ob.op)
            elif isinstance(ob, ChunkData):
                to_serials.add(ob)
                [serial(c) for c in (ob.composed or [])]
                serial(ob.op)
            else:
                assert isinstance(ob, Operand)
                to_serials.add(ob)
                [serial(i) for i in (ob.inputs or [])]
                [serial(i) for i in (ob.outputs or []) if i is not None]

        serial(obj)
        return to_serials

    def _pb_serial(self, obj):
        to_serials = list(self._serial(obj))

        return MultiGetDict(zip(to_serials, self.pb_serialize(to_serials)))

    def _pb_deserial(self, serials_d):
        objs = list(serials_d)
        serials = list(serials_d[o] for o in objs)

        return MultiGetDict(zip(objs, self.pb_deserialize([type(o) for o in objs], serials)))

    def _json_serial(self, obj):
        to_serials = list(self._serial(obj))

        return MultiGetDict(zip(to_serials, self.json_serialize(to_serials)))

    def _json_deserial(self, serials_d):
        objs = list(serials_d)
        serials = list(serials_d[o] for o in objs)

        return MultiGetDict(zip(objs, self.json_deserialize([type(o) for o in objs], serials)))

    def base_equal(self, ob1, ob2):
        if type(ob1) != type(ob2):
            return False

        def cmp(obj1, obj2):
            if isinstance(obj1, np.ndarray):
                return np.array_equal(obj1, obj2)
            elif isinstance(obj1, Iterable) and \
                    not isinstance(obj1, str) and \
                    isinstance(obj2, Iterable) and \
                    not isinstance(obj2, str):
                return all(cmp(it1, it2) for it1, it2 in itertools.zip_longest(obj1, obj2))
            elif hasattr(obj1, 'key') and hasattr(obj2, 'key'):
                return obj1.key == obj2.key
            elif isinstance(obj1, ReferenceType) and isinstance(obj2, ReferenceType):
                return cmp(obj1(), obj2())
            else:
                return obj1 == obj2

        for slot in ob1.__slots__:
            if not cmp(getattr(ob1, slot, None), getattr(ob2, slot, None)):
                return False

        return True

    def assertBaseEqual(self, ob1, ob2):
        return self.assertTrue(self.base_equal(ob1, ob2))


class EtcdProcessHelper(object):
    # from https://github.com/jplana/python-etcd/blob/master/src/etcd/tests/integration/helpers.py
    # licensed under mit license
    def __init__(
            self,
            base_directory=None,
            proc_name='etcd',
            port_range_start=4001,
            internal_port_range_start=7001,
            cluster=False,
            tls=False
    ):
        if base_directory is None:
            import mars
            base_directory = os.path.join(os.path.dirname(os.path.dirname(mars.__file__)), 'bin')

        self.base_directory = base_directory
        self.proc_name = proc_name
        self.port_range_start = port_range_start
        self.internal_port_range_start = internal_port_range_start
        self.processes = {}
        self.cluster = cluster
        self.schema = 'http://'
        if tls:
            self.schema = 'https://'

    def is_installed(self):
        return os.path.exists(os.path.join(self.base_directory, self.proc_name))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dirs = list(tp[0] for tp in self.processes.values())
        self.stop()
        for temp_dir in dirs:
            try:
                shutil.rmtree(temp_dir)
            except OSError:
                pass

    def run(self, number=1, proc_args=None):
        proc_args = proc_args or []
        if number > 1:
            initial_cluster = ",".join([
                "test-node-{}={}127.0.0.1:{}".format(slot, 'http://', self.internal_port_range_start + slot)
                for slot in range(0, number)
            ])
            proc_args.extend([
                '-initial-cluster', initial_cluster,
                '-initial-cluster-state', 'new'
            ])
        else:
            proc_args.extend([
                '-initial-cluster', 'test-node-0=http://127.0.0.1:{}'.format(self.internal_port_range_start),
                '-initial-cluster-state', 'new'
            ])

        for i in range(0, number):
            self.add_one(i, proc_args)
        return self

    def stop(self):
        for key in [k for k in self.processes.keys()]:
            self.kill_one(key)

    def add_one(self, slot, proc_args=None):
        log = logging.getLogger()
        directory = tempfile.mkdtemp(
            dir=self.base_directory,
            prefix='etcd-gevent.%d-' % slot)

        log.debug('Created directory %s' % directory)
        client = '%s127.0.0.1:%d' % (self.schema, self.port_range_start + slot)
        peer = '%s127.0.0.1:%d' % ('http://', self.internal_port_range_start
                                   + slot)
        daemon_args = [
            self.proc_name,
            '-data-dir', directory,
            '-name', 'test-node-%d' % slot,
            '-initial-advertise-peer-urls', peer,
            '-listen-peer-urls', peer,
            '-advertise-client-urls', client,
            '-listen-client-urls', client
        ]

        env_path = os.environ.get('PATH', '')
        if self.base_directory not in env_path:
            os.environ['PATH'] = os.path.pathsep.join([env_path, self.base_directory])

        if proc_args:
            daemon_args.extend(proc_args)

        daemon = subprocess.Popen(daemon_args)
        log.debug('Started %d' % daemon.pid)
        log.debug('Params: %s' % daemon_args)
        time.sleep(2)
        self.processes[slot] = (directory, daemon)

    def kill_one(self, slot):
        log = logging.getLogger()
        data_dir, process = self.processes.pop(slot)
        process.kill()
        time.sleep(2)
        log.debug('Killed etcd pid:%d', process.pid)
        shutil.rmtree(data_dir)
        log.debug('Removed directory %s' % data_dir)


def patch_method(method, *args, **kwargs):
    if hasattr(method, '__qualname__'):
        return mock.patch(method.__module__ + '.' + method.__qualname__, *args, **kwargs)
    elif hasattr(method, 'im_class'):
        return mock.patch('.'.join([method.im_class.__module__, method.im_class.__name__, method.__name__]),
                          *args, **kwargs)
    else:
        return mock.patch(method.__module__ + '.' + method.__name__, *args, **kwargs)


def require_cupy(func):
    if pytest:
        func = pytest.mark.cuda(func)
    func = unittest.skipIf(cupy is None, reason='cupy not installed')(func)
    return func


def require_cudf(func):
    if pytest:
        func = pytest.mark.cuda(func)
    func = unittest.skipIf(cudf is None, reason='cudf not installed')(func)
    return func


def create_actor_pool(*args, **kwargs):
    import gevent.socket
    from mars.actors import create_actor_pool as new_actor_pool

    address = kwargs.pop('address', None)
    if not address:
        return new_actor_pool(*args, **kwargs)

    if isinstance(address, str):
        host = address.rsplit(':')[0]
        port = int(address.rsplit(':', 1)[1])
    else:
        host = '127.0.0.1'
        port = np.random.randint(10000, 65535)
    it = itertools.count(port)

    for _ in range(5):
        try:
            address = '{0}:{1}'.format(host, next(it))
            return new_actor_pool(address, *args, **kwargs)
        except (OSError, gevent.socket.error):
            continue
    raise OSError("Failed to create actor pool")


class MarsObjectCheckMixin:
    @staticmethod
    def assert_shape_consistent(expected_shape, real_shape):
        assert len(expected_shape) == len(real_shape)
        for e, r in zip(expected_shape, real_shape):
            if not np.isnan(e) and e != r:
                raise AssertionError('shape in metadata %r is not consistent with real shape %r'
                                     % (expected_shape, real_shape))

    @staticmethod
    def assert_dtype_consistent(expected_dtype, real_dtype):
        if expected_dtype != real_dtype:
            if expected_dtype is None:
                raise AssertionError('Expected dtype cannot be None')
            if not np.can_cast(expected_dtype, real_dtype):
                raise AssertionError('dtype in metadata %r cannot cast to real dtype %r'
                                     % (expected_dtype, real_dtype))

    @classmethod
    def assert_tensor_consistent(cls, expected, real):
        from mars.lib.sparse import SparseNDArray
        if isinstance(real, (str, int, bool, float, complex)):
            real = np.array([real])[0]
        if not isinstance(real, (np.generic, np.ndarray, SparseNDArray)):
            raise AssertionError('Type of real value (%r) not one of '
                                 '(np.generic, np.array, SparseNDArray)' % type(real))
        cls.assert_dtype_consistent(expected.dtype, real.dtype)
        cls.assert_shape_consistent(expected.shape, real.shape)

    @classmethod
    def assert_index_consistent(cls, expected_index_value, real_index):
        if expected_index_value.has_value():
            pd.testing.assert_index_equal(expected_index_value.to_pandas(), real_index)

    @classmethod
    def assert_dataframe_consistent(cls, expected, real):
        if not isinstance(real, pd.DataFrame):
            raise AssertionError('Type of real value (%r) not DataFrame' % type(real))
        cls.assert_shape_consistent(expected.shape, real.shape)
        pd.testing.assert_index_equal(expected.dtypes.index, real.dtypes.index)

        try:
            for expected_dtype, real_dtype in zip(expected.dtypes, real.dtypes):
                cls.assert_dtype_consistent(expected_dtype, real_dtype)
        except AssertionError:
            raise AssertionError('dtypes in metadata %r cannot cast to real dtype %r'
                                 % (expected.dtypes, real.dtypes))

        cls.assert_index_consistent(expected.columns_value, real.columns)
        cls.assert_index_consistent(expected.index_value, real.index)

    @classmethod
    def assert_series_consistent(cls, expected, real):
        if not isinstance(real, pd.Series):
            raise AssertionError('Type of real value (%r) not Series' % type(real))
        cls.assert_shape_consistent(expected.shape, real.shape)

        if expected.name != real.name:
            raise AssertionError('series name in metadata %r is not equal to real name %r'
                                 % (expected.name, real.name))

        cls.assert_dtype_consistent(expected.dtype, real.dtype)
        cls.assert_index_consistent(expected.index_value, real.index)

    @classmethod
    def assert_object_consistent(cls, expected, real):
        from mars.tensor.core import TENSOR_TYPE
        from mars.dataframe.core import DATAFRAME_TYPE, SERIES_TYPE

        from mars.tensor.core import CHUNK_TYPE as TENSOR_CHUNK_TYPE
        from mars.dataframe.core import DATAFRAME_CHUNK_TYPE, SERIES_CHUNK_TYPE

        if isinstance(expected, (TENSOR_TYPE, TENSOR_CHUNK_TYPE)):
            cls.assert_tensor_consistent(expected, real)
        elif isinstance(expected, (DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)):
            cls.assert_dataframe_consistent(expected, real)
        elif isinstance(expected, (SERIES_TYPE, SERIES_CHUNK_TYPE)):
            cls.assert_series_consistent(expected, real)


class GraphExecutionWithChunkCheck(MarsObjectCheckMixin, GraphExecution):
    def _execute_operand(self, op):
        super()._execute_operand(op)
        if self._mock:
            return
        for o in op.outputs:
            if o.key not in self._key_set:
                continue
            self.assert_object_consistent(o, self._chunk_results[o.key])


class ExecutorForTest(MarsObjectCheckMixin, Executor):
    """
    Mostly identical to normal executor, difference is that when executing graph,
    graph will be serialized then deserialized by Protocol Buffers and JSON both.
    """
    __test__ = False
    _graph_execution_cls = GraphExecutionWithChunkCheck

    def execute_graph(self, graph, keys, **kw):
        if 'NO_SERIALIZE_IN_TEST_EXECUTOR' not in os.environ:
            graph = type(graph).from_json(graph.to_json())
            graph = type(graph).from_pb(graph.to_pb())
        return super().execute_graph(graph, keys, **kw)

    def execute_tileable(self, tileable, *args, **kwargs):
        result = super().execute_tileable(tileable, *args, **kwargs)
        self.assert_object_consistent(tileable, result)
        return result

    def execute_tileables(self, tileables, *args, **kwargs):
        results = super().execute_tileables(tileables, *args, **kwargs)
        for tileable, result in zip(tileables, results):
            self.assert_object_consistent(tileable, result)
        return results
