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
import functools
import itertools
import os
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections.abc import Iterable
from weakref import ReferenceType

import numpy as np
import pandas as pd

from mars.context import LocalContext
from mars.core import OBJECT_TYPE
from mars.executor import Executor, GraphExecution
from mars.graph import SerializableGraph
from mars.optimizes.chunk_graph.fuse import Fusion
from mars.utils import lazy_import

try:
    import pytest
except ImportError:
    pytest = None
try:
    from flaky import flaky
except ImportError:
    def flaky(o=None, **_):
        if o is not None:
            return o

        def ident(x):
            return x
        return ident

if sys.version_info < (3, 8):
    import mock
else:
    from unittest import mock
_mock = mock

cupy = lazy_import('cupy', globals=globals())
cudf = lazy_import('cudf', globals=globals())
ray = lazy_import('ray', globals=globals())

logger = logging.getLogger(__name__)


class TestCase(unittest.TestCase):
    pass


class MultiGetDict(dict):
    def __getitem__(self, item):
        if isinstance(item, tuple):
            return tuple(super(MultiGetDict, self).__getitem__(it)
                         for it in item)
        return super(MultiGetDict, self).__getitem__(item)


def parameterized(defaults=None, **params):
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
                actual_param = (defaults or dict()).copy()
                actual_param.update(param)
                for pn, pv in actual_param.items():
                    setattr(cls, pn, pv)
                try:
                    fun(self, *args, **kwargs)
                except:  # noqa: E722
                    logger.error('Failed when running %s with param %s', fun.__qualname__, n)
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
        from mars.serialize import serializes, deserializes, \
            ProtobufSerializeProvider, JsonSerializeProvider

        self.pb_serialize = lambda *args, **kw: \
            serializes(ProtobufSerializeProvider(), *args, **kw)
        self.pb_deserialize = lambda *args, **kw: \
            deserializes(ProtobufSerializeProvider(), *args, **kw)
        self.json_serialize = lambda *args, **kw: \
            serializes(JsonSerializeProvider(), *args, **kw)
        self.json_deserialize = lambda *args, **kw: \
            deserializes(JsonSerializeProvider(), *args, **kw)

    @classmethod
    def _create_test_context(cls, executor=None):
        d = {'executor': executor}

        class MockSession:
            def __init__(self):
                self.executor = d['executor']

        ctx = LocalContext(MockSession())
        new_executor = d['executor'] = \
            ExecutorForTest('numpy', storage=ctx)

        return ctx, new_executor

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
            elif isinstance(obj1, SerializableGraph) and isinstance(obj2, SerializableGraph):
                return cmp(obj1.nodes, obj2.nodes)
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
                f"test-node-{slot}=http://127.0.0.1:{self.internal_port_range_start + slot}"
                for slot in range(0, number)
            ])
            proc_args.extend([
                '-initial-cluster', initial_cluster,
                '-initial-cluster-state', 'new'
            ])
        else:
            proc_args.extend([
                '-initial-cluster', f'test-node-0=http://127.0.0.1:{self.internal_port_range_start}',
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
            prefix=f'etcd-gevent.{slot}-')

        log.debug(f'Created directory {directory}')
        client = f'{self.schema}127.0.0.1:{self.port_range_start + slot}'
        peer = f'http://127.0.0.1:{self.internal_port_range_start + slot}'
        daemon_args = [
            self.proc_name,
            '-data-dir', directory,
            '-name', f'test-node-{slot}',
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
        log.debug(f'Started {daemon.pid}')
        log.debug(f'Params: {daemon_args}')
        time.sleep(2)
        self.processes[slot] = (directory, daemon)

    def kill_one(self, slot):
        log = logging.getLogger()
        data_dir, process = self.processes.pop(slot)
        process.kill()
        time.sleep(2)
        log.debug(f'Killed etcd pid: {process.pid}')
        shutil.rmtree(data_dir)
        log.debug(f'Removed directory {data_dir}')


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


def require_ray(func):
    if pytest:
        func = pytest.mark.ray(func)
    func = unittest.skipIf(ray is None, reason='ray not installed')(func)
    return func


def require_hadoop(func):
    if pytest:
        func = pytest.mark.hadoop(func)
    func = unittest.skipIf(not os.environ.get('WITH_HADOOP'), 'Only run when hadoop is installed')(func)
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
            address = f'{host}:{next(it)}'
            return new_actor_pool(address, *args, **kwargs)
        except (OSError, gevent.socket.error):
            continue
    raise OSError("Failed to create actor pool")


def assert_groupby_equal(left, right, sort_keys=False, sort_index=True, with_selection=False):
    if hasattr(left, 'groupby_obj'):
        left = left.groupby_obj
    if hasattr(right, 'groupby_obj'):
        right = right.groupby_obj

    if type(left) is not type(right):
        raise AssertionError(f'Type of groupby not consistent: {type(left)} != {type(right)}')

    left_selection = getattr(left, '_selection', None)
    right_selection = getattr(right, '_selection', None)
    if sort_keys:
        left = sorted(left, key=lambda p: p[0])
        right = sorted(right, key=lambda p: p[0])
    else:
        left, right = list(left), list(right)
    if sort_index:
        left = [(k, v.sort_index()) for k, v in left]
        right = [(k, v.sort_index()) for k, v in right]

    if len(left) != len(right):
        raise AssertionError(f'Count of groupby keys not consistent: {len(left)} != {len(right)}')

    left_keys = [p[0] for p in left]
    right_keys = [p[0] for p in right]
    if left_keys != right_keys:
        raise AssertionError(f'Group keys not consistent: {left_keys!r} != {right_keys!r}')
    for (left_key, left_frame), (right_key, right_frame) in zip(left, right):
        if with_selection:
            if left_selection and isinstance(left_frame, pd.DataFrame):
                left_frame = left_frame[left_selection]
            if right_selection and isinstance(right_frame, pd.DataFrame):
                right_frame = right_frame[right_selection]

        if isinstance(left_frame, pd.DataFrame):
            pd.testing.assert_frame_equal(left_frame, right_frame)
        else:
            pd.testing.assert_series_equal(left_frame, right_frame)


_check_options = dict()
_check_args = ['check_all', 'check_series_name', 'check_index_name', 'check_dtypes',
               'check_dtype', 'check_shape', 'check_nsplits']


class MarsObjectCheckMixin:
    @staticmethod
    def adapt_index_value(value):
        if hasattr(value, 'to_pandas'):
            return value.to_pandas()
        return value

    @staticmethod
    def assert_shape_consistent(expected_shape, real_shape):
        if not _check_options['check_shape']:
            return

        if len(expected_shape) != len(real_shape):
            raise AssertionError('ndim in metadata %r is not consistent with real ndim %r'
                                 % (len(expected_shape), len(real_shape)))
        for e, r in zip(expected_shape, real_shape):
            if not np.isnan(e) and e != r:
                raise AssertionError('shape in metadata %r is not consistent with real shape %r'
                                     % (expected_shape, real_shape))

    @staticmethod
    def assert_dtype_consistent(expected_dtype, real_dtype):
        if isinstance(real_dtype, pd.DatetimeTZDtype):
            real_dtype = real_dtype.base
        if expected_dtype != real_dtype:
            if expected_dtype == np.dtype('O') and real_dtype.type is np.str_:
                # real dtype is string, this matches expectation
                return
            if expected_dtype is None:
                raise AssertionError('Expected dtype cannot be None')
            if not np.can_cast(real_dtype, expected_dtype) and not np.can_cast(expected_dtype, real_dtype):
                raise AssertionError('cannot cast between dtype of real dtype %r and dtype %r defined in metadata'
                                     % (real_dtype, expected_dtype))

    @classmethod
    def assert_tensor_consistent(cls, expected, real):
        from mars.lib.sparse import SparseNDArray
        np_types = (np.generic, np.ndarray, SparseNDArray)
        if cupy is not None:
            np_types += (cupy.ndarray,)

        if isinstance(real, (str, int, bool, float, complex)):
            real = np.array([real])[0]
        if not isinstance(real, np_types):
            raise AssertionError(f'Type of real value ({type(real)}) not one of {np_types!r}')
        if not hasattr(expected, 'dtype'):
            return
        cls.assert_dtype_consistent(expected.dtype, real.dtype)
        cls.assert_shape_consistent(expected.shape, real.shape)

    @classmethod
    def assert_index_value_consistent(cls, expected_index_value, real_index):
        if expected_index_value.has_value():
            expected_index = expected_index_value.to_pandas()
            try:
                pd.testing.assert_index_equal(expected_index, cls.adapt_index_value(real_index))
            except AssertionError as e:
                raise AssertionError(
                    f'Index of real value ({expected_index}) not equal to ({real_index})') from e

    @classmethod
    def assert_dataframe_consistent(cls, expected, real):
        dataframe_types = (pd.DataFrame,)
        if cudf is not None:
            dataframe_types += (cudf.DataFrame,)

        if not isinstance(real, dataframe_types):
            raise AssertionError(f'Type of real value ({type(real)}) not DataFrame')
        cls.assert_shape_consistent(expected.shape, real.shape)
        if not np.isnan(expected.shape[1]):  # ignore when columns length is nan
            pd.testing.assert_index_equal(expected.dtypes.index, cls.adapt_index_value(real.dtypes.index))

            if _check_options['check_dtypes']:
                try:
                    for expected_dtype, real_dtype in zip(expected.dtypes, real.dtypes):
                        cls.assert_dtype_consistent(expected_dtype, real_dtype)
                except AssertionError:
                    raise AssertionError('dtypes in metadata %r cannot cast to real dtype %r'
                                         % (expected.dtypes, real.dtypes))

        cls.assert_index_value_consistent(expected.columns_value, real.columns)
        cls.assert_index_value_consistent(expected.index_value, real.index)

    @classmethod
    def assert_series_consistent(cls, expected, real):
        series_types = (pd.Series,)
        if cudf is not None:
            series_types += (cudf.Series,)

        if not isinstance(real, series_types):
            raise AssertionError(f'Type of real value ({type(real)}) not Series')
        cls.assert_shape_consistent(expected.shape, real.shape)

        if _check_options['check_series_name'] and expected.name != real.name:
            raise AssertionError(f'series name in metadata {expected.name} '
                                 f'is not equal to real name {real.name}')

        cls.assert_dtype_consistent(expected.dtype, real.dtype)
        cls.assert_index_value_consistent(expected.index_value, real.index)

    @classmethod
    def assert_groupby_consistent(cls, expected, real):
        from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
        from mars.dataframe.core import DATAFRAME_GROUPBY_TYPE, SERIES_GROUPBY_TYPE
        from mars.dataframe.core import DATAFRAME_GROUPBY_CHUNK_TYPE, SERIES_GROUPBY_CHUNK_TYPE

        df_groupby_types = (DataFrameGroupBy,)
        series_groupby_types = (SeriesGroupBy,)

        try:
            from cudf.core.groupby.groupby import DataFrameGroupBy as CUDataFrameGroupBy, \
                SeriesGroupBy as CUSeriesGroupBy
            df_groupby_types += (CUDataFrameGroupBy,)
            series_groupby_types += (CUSeriesGroupBy,)
        except ImportError:
            pass

        if isinstance(expected, (DATAFRAME_GROUPBY_TYPE, DATAFRAME_GROUPBY_CHUNK_TYPE)) \
                and isinstance(real, df_groupby_types):
            selection = getattr(real, '_selection', None)
            if not selection:
                cls.assert_dataframe_consistent(expected, real.obj)
            else:
                cls.assert_dataframe_consistent(expected, real.obj[selection])
        elif isinstance(expected, (SERIES_GROUPBY_TYPE, SERIES_GROUPBY_CHUNK_TYPE)) \
                and isinstance(real, series_groupby_types):
            cls.assert_series_consistent(expected, real.obj)
        else:
            raise AssertionError('GroupBy type not consistent. Expecting %r but receive %r'
                                 % (type(expected), type(real)))

    @classmethod
    def assert_index_consistent(cls, expected, real):
        index_types = (pd.Index,)
        if cudf is not None:
            index_types += (cudf.Index,)

        if not isinstance(real, index_types):
            raise AssertionError(f'Type of real value ({type(real)}) not Index')
        cls.assert_shape_consistent(expected.shape, real.shape)

        if _check_options['check_series_name'] and expected.name != real.name:
            raise AssertionError('series name in metadata %r is not equal to real name %r'
                                 % (expected.name, real.name))

        cls.assert_dtype_consistent(expected.dtype, real.dtype)
        cls.assert_index_value_consistent(expected.index_value, real)

    @classmethod
    def assert_categorical_consistent(cls, expected, real):
        if not isinstance(real, pd.Categorical):
            raise AssertionError(f'Type of real value ({type(real)}) not Categorical')
        cls.assert_dtype_consistent(expected.dtype, real.dtype)
        cls.assert_shape_consistent(expected.shape, real.shape)
        cls.assert_index_value_consistent(expected.categories_value, real.categories)

    @classmethod
    def assert_object_consistent(cls, expected, real):
        from mars.tensor.core import TENSOR_TYPE
        from mars.dataframe.core import DATAFRAME_TYPE, SERIES_TYPE, GROUPBY_TYPE, \
            INDEX_TYPE, CATEGORICAL_TYPE

        from mars.tensor.core import TENSOR_CHUNK_TYPE
        from mars.dataframe.core import DATAFRAME_CHUNK_TYPE, SERIES_CHUNK_TYPE, \
            GROUPBY_CHUNK_TYPE, INDEX_CHUNK_TYPE, CATEGORICAL_CHUNK_TYPE

        if isinstance(expected, (TENSOR_TYPE, TENSOR_CHUNK_TYPE)):
            cls.assert_tensor_consistent(expected, real)
        elif isinstance(expected, (DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)):
            cls.assert_dataframe_consistent(expected, real)
        elif isinstance(expected, (SERIES_TYPE, SERIES_CHUNK_TYPE)):
            cls.assert_series_consistent(expected, real)
        elif isinstance(expected, (GROUPBY_TYPE, GROUPBY_CHUNK_TYPE)):
            cls.assert_groupby_consistent(expected, real)
        elif isinstance(expected, (INDEX_TYPE, INDEX_CHUNK_TYPE)):
            cls.assert_index_consistent(expected, real)
        elif isinstance(expected, (CATEGORICAL_TYPE, CATEGORICAL_CHUNK_TYPE)):
            cls.assert_categorical_consistent(expected, real)


class GraphExecutionWithChunkCheck(MarsObjectCheckMixin, GraphExecution):
    def _execute_operand(self, op):
        super()._execute_operand(op)
        if self._mock:
            return
        if _check_options.get('check_all', True):
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

    def __init__(self, *args, **kwargs):
        from mars.serialize.core import get_serializables

        super().__init__(*args, **kwargs)
        self._raw_chunk_shapes = dict()
        self._tileable_checked = dict()
        if not hasattr(type(self), '_serializables_snapshot'):
            type(self)._serializables_snapshot = get_serializables()

    @staticmethod
    def _extract_check_options(kw_dict):
        for key in _check_args:
            _check_options[key] = kw_dict.pop(key, True)

    def _check_nsplits(self, tileable):
        from mars.tiles import get_tiled
        tiled = get_tiled(tileable)
        if tiled.nsplits == () and len(tiled.chunks) == 1:
            return

        nsplit_chunk_shape = tuple(len(s) for s in tiled.nsplits)
        if nsplit_chunk_shape != tiled.chunk_shape:
            raise AssertionError('Operand %r: shape of nsplits %r not consistent with chunk shape %r'
                                 % (tiled.op, nsplit_chunk_shape, tiled.chunk_shape)) from None

        nsplit_shape = tuple(np.sum(s) for s in tiled.nsplits)
        try:
            self.assert_shape_consistent(nsplit_shape, tiled.shape)
        except AssertionError:
            raise AssertionError('Operand %r: shape computed from nsplits %r -> %r not consistent with real shape %r'
                                 % (tiled.op, tiled.nsplits, nsplit_shape, tiled.shape)) from None

        for c in tiled.chunks:
            try:
                tiled_c = tiled.cix[c.index]
            except ValueError as ex:
                raise AssertionError('Operand %r: Malformed index %r, nsplits is %r. Raw error is %r'
                                     % (c.op, c.index, tiled.nsplits, ex)) from None

            if tiled_c is not c:
                raise AssertionError('Operand %r: Cannot spot chunk via index %r, nsplits is %r'
                                     % (c.op, c.index, tiled.nsplits))
        for cid, shape in enumerate(itertools.product(*tiled.nsplits)):
            chunk_shape = self._raw_chunk_shapes.get(tiled.chunks[cid].key) or tiled.chunks[cid].shape
            if len(shape) != len(chunk_shape):
                raise AssertionError('Operand %r: Shape in nsplits %r does not meet shape in chunk %r'
                                     % (tiled.chunks[cid].op, shape, chunk_shape))
            for s1, s2 in zip(shape, chunk_shape):
                if (not (np.isnan(s1) and np.isnan(s2))) and s1 != s2:
                    raise AssertionError('Operand %r: Shape in nsplits %r does not meet shape in chunk %r'
                                         % (tiled.chunks[cid].op, shape, chunk_shape))

    def execute_graph(self, graph, keys, **kw):
        if 'NO_SERIALIZE_IN_TEST_EXECUTOR' not in os.environ:
            raw_graph = graph
            graph = type(graph).from_json(json.loads(json.dumps(graph.to_json())))
            graph = type(graph).from_pb(graph.to_pb())
            if kw.get('compose', True):
                # decompose the raw graph
                # due to the reason that, now after fuse,
                # the inputs of node's op may be fuse,
                # call optimize to decompose back
                raw_graph.decompose()
                Fusion.check_graph(raw_graph)

        # record shapes generated in tile
        for n in graph:
            self._raw_chunk_shapes[n.key] = getattr(n, 'shape', None)
        return super().execute_graph(graph, keys, **kw)

    def _update_tileable_and_chunk_shape(self, tileable_graph, chunk_result, failed_ops):
        for n in tileable_graph:
            if _check_options['check_nsplits'] and n.op not in failed_ops \
                    and n.key not in self._tileable_checked and not isinstance(n, OBJECT_TYPE):
                self._check_nsplits(n)
                self._tileable_checked[n.key] = True
        return super()._update_tileable_and_chunk_shape(tileable_graph, chunk_result, failed_ops)

    def _check_serializable_registration(self):
        from mars.serialize.core import get_serializables

        cur_serializables = get_serializables()
        if len(cur_serializables) == len(self._serializables_snapshot):
            return
        unregistered_set = set(cur_serializables.keys()) - set(self._serializables_snapshot.keys())
        raise AssertionError('Operands %r not registered on initialization'
                             % ([cur_serializables[k] for k in unregistered_set],))

    def execute_tileable(self, tileable, *args, **kwargs):
        self._extract_check_options(kwargs)

        result = super().execute_tileable(tileable, *args, **kwargs)

        if _check_options.get('check_all', True):
            if not isinstance(tileable, OBJECT_TYPE) and tileable.key not in self._tileable_checked:
                if _check_options['check_nsplits']:
                    self._check_nsplits(tileable)

            # check returned type
            if kwargs.get('concat', False):
                self.assert_object_consistent(tileable, result[0])
        self._check_serializable_registration()
        return result

    execute_tensor = execute_tileable
    execute_dataframe = execute_tileable

    def execute_tileables(self, tileables, *args, **kwargs):
        self._extract_check_options(kwargs)

        tileables_to_check = tileables
        results = super().execute_tileables(tileables, *args, **kwargs)
        if results is None:
            # fetch = False
            tileables_to_check = [t for t in tileables if not isinstance(t, OBJECT_TYPE)]
            results = self.fetch_tileables(tileables_to_check)

        if _check_options.get('check_all', True):
            for tileable, result in zip(tileables_to_check, results):
                if not isinstance(tileable, OBJECT_TYPE) and tileable.key not in self._tileable_checked:
                    if _check_options['check_nsplits']:
                        self._check_nsplits(tileable)
                    self.assert_object_consistent(tileable, result)
        self._check_serializable_registration()
        return results

    def fetch_tileables(self, tileables, **kw):
        global _check_options
        old_options = _check_options.copy()
        try:
            _check_options['check_all'] = False
            kw.update(_check_options)
            return super().fetch_tileables(tileables, **kw)
        finally:
            _check_options = old_options

    execute_tensors = execute_tileables
    execute_dataframes = execute_tileables
