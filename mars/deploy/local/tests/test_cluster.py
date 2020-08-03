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
import logging
import os
import pickle
import sys
import time
import tempfile
import traceback
import unittest
import uuid

import cloudpickle
import numpy as np
import pandas as pd
try:
    import h5py
except ImportError:
    h5py = None
try:
    import sklearn
except ImportError:
    sklearn = None

from mars import tensor as mt
from mars import dataframe as md
from mars import remote as mr
from mars.config import options, option_context
from mars.deploy.local.core import new_cluster, LocalDistributedCluster, gen_endpoint
from mars.errors import ExecutionFailed
from mars.serialize import BytesField, Int64Field
from mars.tensor.operands import TensorOperand
from mars.tensor.arithmetic.core import TensorElementWise
from mars.tensor.arithmetic.abs import TensorAbs
from mars.session import new_session, Session, ClusterSession
from mars.scheduler import SessionManagerActor
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.worker.dispatcher import DispatchActor
from mars.web.session import Session as WebSession
from mars.context import get_context, RunningMode
from mars.utils import serialize_function
from mars.tests.core import mock, require_cudf

logger = logging.getLogger(__name__)
_exec_timeout = 120 if 'CI' in os.environ else -1


def _on_deserialize_fail(x):
    raise TypeError('intend to throw error on' + str(x))


class SerializeMustFailOperand(TensorOperand, TensorElementWise):
    _op_type_ = 356789

    _f = Int64Field('f', on_deserialize=_on_deserialize_fail)

    def __init__(self, f=None, **kw):
        super().__init__(_f=f, **kw)


class TileFailOperand(TensorAbs):
    _op_type_ = 198732951

    _exc_serial = BytesField('exc_serial')

    @classmethod
    def tile(cls, op):
        if op._exc_serial is not None:
            raise pickle.loads(op._exc_serial)
        return super().tile(op)


class ExecFailOperand(TensorAbs):
    _op_type_ = 196432154

    _exc_serial = BytesField('exc_serial')

    @classmethod
    def tile(cls, op):
        tileables = super().tile(op)
        # make sure chunks
        tileables[0]._shape = (np.nan, np.nan)
        return tileables

    @classmethod
    def execute(cls, ctx, op):
        if op._exc_serial is not None:
            raise pickle.loads(op._exc_serial)
        return super().execute(ctx, op)


class TileWithContextOperand(TensorAbs):
    _op_type_ = 9870102948

    _multiplier = Int64Field('multiplier')

    @classmethod
    def tile(cls, op):
        context = get_context()

        if context.running_mode != RunningMode.local_cluster:
            raise AssertionError

        inp_chunk = op.inputs[0].chunks[0]
        inp_size = context.get_chunk_metas([inp_chunk.key])[0].chunk_size
        chunk_op = op.copy().reset_key()
        chunk_op._multiplier = inp_size
        chunk = chunk_op.new_chunk([inp_chunk], shape=inp_chunk.shape)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=op.outputs[0].shape,
                                  order=op.outputs[0].order, nsplits=op.inputs[0].nsplits,
                                  chunks=[chunk])

    @classmethod
    def execute(cls, ctx, op):
        ctx[op.outputs[0].key] = ctx[op.inputs[0].key] * op._multiplier


@unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
@mock.patch('webbrowser.open_new_tab', new=lambda *_, **__: True)
class Test(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._old_default_cpu_usage = options.scheduler.default_cpu_usage
        options.scheduler.default_cpu_usage = 0

    def tearDown(self):
        super().tearDown()
        options.scheduler.default_cpu_usage = self._old_default_cpu_usage

    def testLocalCluster(self, *_):
        endpoint = gen_endpoint('0.0.0.0')
        with LocalDistributedCluster(endpoint, scheduler_n_process=2, worker_n_process=3,
                                     shared_memory='20M') as cluster:
            pool = cluster.pool

            self.assertTrue(pool.has_actor(pool.actor_ref(
                SchedulerClusterInfoActor.default_uid())))
            self.assertTrue(pool.has_actor(pool.actor_ref(SessionManagerActor.default_uid())))
            self.assertTrue(pool.has_actor(pool.actor_ref(DispatchActor.default_uid())))

            with new_session(endpoint) as session:
                api = session._api

                t = mt.ones((3, 3), chunk_size=2)
                result = session.run(t, timeout=_exec_timeout)

                np.testing.assert_array_equal(result, np.ones((3, 3)))

            self.assertNotIn(session._session_id, api.session_manager.get_sessions())

    def testLocalClusterWithWeb(self, *_):
        import psutil
        with new_cluster(scheduler_n_process=2, worker_n_process=3,
                         shared_memory='20M', web=True) as cluster:
            cluster_proc = psutil.Process(cluster._cluster_process.pid)
            web_proc = psutil.Process(cluster._web_process.pid)
            processes = list(cluster_proc.children(recursive=True)) + \
                list(web_proc.children(recursive=True))

            with cluster.session as session:
                t = mt.ones((3, 3), chunk_size=2)
                result = session.run(t, timeout=_exec_timeout)

                np.testing.assert_array_equal(result, np.ones((3, 3)))

            with new_session('http://' + cluster._web_endpoint) as session:
                t = mt.ones((3, 3), chunk_size=2)
                result = session.run(t, timeout=_exec_timeout)

                np.testing.assert_array_equal(result, np.ones((3, 3)))

        check_time = time.time()
        while any(p.is_running() for p in processes):
            time.sleep(0.1)
            if check_time + 10 < time.time():
                logger.error('Processes still running: %r',
                             [' '.join(p.cmdline()) for p in processes if p.is_running()])
                self.assertFalse(any(p.is_running() for p in processes))

    def testLocalClusterError(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=3,
                         shared_memory='20M', web=True, options={'scheduler.retry_num': 1}) as cluster:
            # Note that it is nested exception and we want to check the message
            # of the inner exeception, thus assertRaises won't work.

            with cluster.session as session:
                t = mt.array(["1", "2", "3", "4"])
                try:
                    session.run(t + 1)
                except:  # noqa: E722
                    etype, exp, tb = sys.exc_info()
                    self.assertEqual(etype, ExecutionFailed)
                    self.assertIsInstance(exp, ExecutionFailed)
                    formatted_tb = '\n'.join(traceback.format_exception(etype, exp, tb))
                    self.assertIn('TypeError', formatted_tb)
                    self.assertIn('ufunc', formatted_tb)
                    self.assertIn('add', formatted_tb)
                    self.assertIn('signature matching types', formatted_tb)

            with new_session('http://' + cluster._web_endpoint) as session:
                t = mt.array(["1", "2", "3", "4"])
                try:
                    session.run(t + 1)
                except:  # noqa: E722
                    etype, exp, tb = sys.exc_info()
                    self.assertEqual(etype, ExecutionFailed)
                    self.assertIsInstance(exp, ExecutionFailed)
                    formatted_tb = '\n'.join(traceback.format_exception(etype, exp, tb))
                    self.assertIn('TypeError', formatted_tb)
                    self.assertIn('ufunc', formatted_tb)
                    self.assertIn('add', formatted_tb)
                    self.assertIn('signature matching types', formatted_tb)

    def testNSchedulersNWorkers(self, *_):
        calc_cpu_cnt = functools.partial(lambda: 4)

        self.assertEqual(LocalDistributedCluster._calc_scheduler_worker_n_process(
            None, None, None, calc_cpu_count=calc_cpu_cnt), (2, 4))
        # scheduler and worker needs at least 2 processes
        self.assertEqual(LocalDistributedCluster._calc_scheduler_worker_n_process(
            1, None, None, calc_cpu_count=calc_cpu_cnt), (2, 2))
        self.assertEqual(LocalDistributedCluster._calc_scheduler_worker_n_process(
            3, None, None, calc_cpu_count=calc_cpu_cnt), (2, 2))
        self.assertEqual(LocalDistributedCluster._calc_scheduler_worker_n_process(
            5, None, None, calc_cpu_count=calc_cpu_cnt), (2, 3))
        self.assertEqual(LocalDistributedCluster._calc_scheduler_worker_n_process(
            None, 1, None, calc_cpu_count=calc_cpu_cnt), (1, 4))
        self.assertEqual(LocalDistributedCluster._calc_scheduler_worker_n_process(
            None, 3, None, calc_cpu_count=calc_cpu_cnt), (3, 4))
        self.assertEqual(LocalDistributedCluster._calc_scheduler_worker_n_process(
            None, None, 3, calc_cpu_count=calc_cpu_cnt), (2, 3))
        self.assertEqual(LocalDistributedCluster._calc_scheduler_worker_n_process(
            5, 3, 2, calc_cpu_count=calc_cpu_cnt), (3, 2))

    def testSingleOutputTensorExecute(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            self.assertIs(cluster.session, Session.default_or_local())

            t = mt.random.rand(10)
            r = t.sum()

            res = r.to_numpy()
            self.assertTrue(np.isscalar(res))
            self.assertLess(res, 10)

            t = mt.random.rand(10)
            r = t.sum() * 4 - 1

            res = r.to_numpy()
            self.assertLess(res, 39)

    def testMultipleOutputTensorExecute(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            session = cluster.session

            t = mt.random.rand(20, 5, chunk_size=5)
            r = mt.linalg.svd(t)

            res = session.run((t,) + r, timeout=_exec_timeout)

            U, s, V = res[1:]
            np.testing.assert_allclose(res[0], U.dot(np.diag(s).dot(V)))

            raw = np.random.rand(20, 5)

            # to test the fuse, the graph should be fused
            t = mt.array(raw)
            U, s, V = mt.linalg.svd(t)
            r = U.dot(mt.diag(s).dot(V))

            res = r.to_numpy()
            np.testing.assert_allclose(raw, res)

            # test submit part of svd outputs
            t = mt.array(raw)
            U, s, V = mt.linalg.svd(t)

            with new_session(cluster.endpoint) as session2:
                U_result, s_result = session2.run(U, s, timeout=_exec_timeout)
                U_expected, s_expectd, _ = np.linalg.svd(raw, full_matrices=False)

                np.testing.assert_allclose(U_result, U_expected)
                np.testing.assert_allclose(s_result, s_expectd)

            with new_session(cluster.endpoint) as session2:
                U_result, s_result = session2.run(U + 1, s + 1, timeout=_exec_timeout)
                U_expected, s_expectd, _ = np.linalg.svd(raw, full_matrices=False)

                np.testing.assert_allclose(U_result, U_expected + 1)
                np.testing.assert_allclose(s_result, s_expectd + 1)

            with new_session(cluster.endpoint) as session2:
                t = mt.array(raw)
                _, s, _ = mt.linalg.svd(t)
                del _

                s_result = session2.run(s, timeout=_exec_timeout)
                s_expected = np.linalg.svd(raw, full_matrices=False)[1]
                np.testing.assert_allclose(s_result, s_expected)

    def testIndexTensorExecute(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            session = cluster.session

            a = mt.random.rand(10, 5)
            idx = slice(0, 5), slice(0, 5)
            a[idx] = 2
            a_splits = mt.split(a, 2)
            r1, r2 = session.run(a_splits[0], a[idx], timeout=_exec_timeout)

            np.testing.assert_array_equal(r1, r2)
            np.testing.assert_array_equal(r1, np.ones((5, 5)) * 2)

            with new_session(cluster.endpoint) as session2:
                a = mt.random.rand(10, 5)
                idx = slice(0, 5), slice(0, 5)

                a[idx] = mt.ones((5, 5)) * 2
                r = session2.run(a[idx], timeout=_exec_timeout)

                np.testing.assert_array_equal(r, np.ones((5, 5)) * 2)

            with new_session(cluster.endpoint) as session3:
                a = mt.random.rand(100, 5)

                slice1 = a[:10]
                slice2 = a[10:20]
                r1, r2, expected = session3.run(slice1, slice2, a, timeout=_exec_timeout)

                np.testing.assert_array_equal(r1, expected[:10])
                np.testing.assert_array_equal(r2, expected[10:20])

            with new_session(cluster.endpoint) as session4:
                a = mt.random.rand(100, 5)

                a[:10] = mt.ones((10, 5))
                a[10:20] = 2
                r = session4.run(a, timeout=_exec_timeout)

                np.testing.assert_array_equal(r[:10], np.ones((10, 5)))
                np.testing.assert_array_equal(r[10:20], np.ones((10, 5)) * 2)

            with new_session(cluster.endpoint) as session5:
                raw = np.random.rand(10, 10)
                a = mt.tensor(raw, chunk_size=(5, 4))
                b = a[a.argmin(axis=1), mt.tensor(np.arange(10))]
                r = session5.run(b, timeout=_exec_timeout, compose=False)

                np.testing.assert_array_equal(r, raw[raw.argmin(axis=1), np.arange(10)])

    def testBoolIndexingExecute(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            a = mt.random.rand(8, 8, chunk_size=4)
            a[2:6, 2:6] = mt.ones((4, 4)) * 2
            b = a[a > 1]
            self.assertEqual(b.shape, (np.nan,))

            cluster.session.run(b, fetch=False, timeout=_exec_timeout)
            self.assertEqual(b.shape, (16,))

            c = b.reshape((4, 4))

            self.assertEqual(c.shape, (4, 4))

            with new_session('http://' + cluster._web_endpoint) as session2:
                a = mt.random.rand(8, 8, chunk_size=4)
                a[2:6, 2:6] = mt.ones((4, 4)) * 2
                b = a[a > 1]
                self.assertEqual(b.shape, (np.nan,))

                session2.run(b, fetch=False, timeout=_exec_timeout)
                self.assertEqual(b.shape, (16,))

                c = b.reshape((4, 4))
                self.assertEqual(c.shape, (4, 4))

            # test unknown-shape fusion
            with new_session('http://' + cluster._web_endpoint) as session2:
                a = mt.random.rand(6, 6, chunk_size=3)
                a[2:5, 2:5] = mt.ones((3, 3)) * 2
                b = (a[a > 1] - 1) * 2

                r = session2.run(b, timeout=_exec_timeout)
                np.testing.assert_array_equal(r, np.ones((9,)) * 2)

    def testExecutableTuple(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            with new_session('http://' + cluster._web_endpoint).as_default():
                a = mt.ones((20, 10), chunk_size=10)
                u, s, v = (mt.linalg.svd(a)).execute().fetch()
                np.testing.assert_allclose(u.dot(np.diag(s).dot(v)), np.ones((20, 10)))

    def testRerunTensor(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            session = cluster.session

            a = mt.ones((10, 10)) + 1
            result1 = session.run(a, timeout=_exec_timeout)
            np.testing.assert_array_equal(result1, np.ones((10, 10)) + 1)
            result2 = session.run(a, timeout=_exec_timeout)
            np.testing.assert_array_equal(result1, result2)

            with new_session(cluster.endpoint) as session2:
                a = mt.random.rand(10, 10)
                a_result1 = session2.run(a, timeout=_exec_timeout)
                b = mt.ones((10, 10))
                a_result2, b_result = session2.run(a, b, timeout=_exec_timeout)
                np.testing.assert_array_equal(a_result1, a_result2)
                np.testing.assert_array_equal(b_result, np.ones((10, 10)))

    def testRunWithoutFetch(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            session = cluster.session

            a = mt.ones((10, 20)) + 1
            self.assertIsNone(session.run(a, fetch=False, timeout=_exec_timeout))
            np.testing.assert_array_equal(a.to_numpy(session=session), np.ones((10, 20)) + 1)

    def testGraphFail(self, *_):
        op = SerializeMustFailOperand(f=3)
        tensor = op.new_tensor(None, (3, 3))

        try:
            raise ValueError
        except:  # noqa: E722
            exc = sys.exc_info()[1]

        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', modules=[__name__],
                         options={'scheduler.retry_num': 1}) as cluster:
            with self.assertRaises(ExecutionFailed):
                try:
                    cluster.session.run(tensor, timeout=_exec_timeout)
                except ExecutionFailed as ex:
                    self.assertIsInstance(ex.__cause__, TypeError)
                    raise

            data = mt.tensor(np.random.rand(10, 20))
            data2 = TileFailOperand(_exc_serial=pickle.dumps(exc)).new_tensor([data], shape=data.shape)
            with self.assertRaises(ExecutionFailed):
                try:
                    cluster.session.run(data2)
                except ExecutionFailed as ex:
                    self.assertIsInstance(ex.__cause__, ValueError)
                    raise

            data = mt.tensor(np.random.rand(20, 10))
            data2 = ExecFailOperand(_exc_serial=pickle.dumps(exc)).new_tensor([data], shape=data.shape)
            with self.assertRaises(ExecutionFailed):
                try:
                    cluster.session.run(data2)
                except ExecutionFailed as ex:
                    self.assertIsInstance(ex.__cause__, ValueError)
                    raise

    def testFetchTensor(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            session = cluster.session

            a1 = mt.ones((10, 20), chunk_size=8) + 1
            r1 = session.run(a1, timeout=_exec_timeout)
            r2 = session.fetch(a1)
            np.testing.assert_array_equal(r1, r2)

            r3 = session.run(a1 * 2, timeout=_exec_timeout)
            np.testing.assert_array_equal(r3, r1 * 2)

            a2 = mt.ones((10, 20), chunk_size=8) + 1
            r4 = session.run(a2, timeout=_exec_timeout)
            np.testing.assert_array_equal(r4, r1)

            del a1
            r4 = session.run(a2, timeout=_exec_timeout)
            np.testing.assert_array_equal(r4, r1)

            with new_session('http://' + cluster._web_endpoint) as session:
                a3 = mt.ones((5, 10), chunk_size=3) + 1
                r1 = session.run(a3, timeout=_exec_timeout)
                r2 = session.fetch(a3)
                np.testing.assert_array_equal(r1, r2)

                r3 = session.run(a3 * 2, timeout=_exec_timeout)
                np.testing.assert_array_equal(r3, r1 * 2)

                a4 = mt.ones((5, 10), chunk_size=3) + 1
                r4 = session.run(a4, timeout=_exec_timeout)
                np.testing.assert_array_equal(r4, r1)

                del a3
                r4 = session.run(a4, timeout=_exec_timeout)
                np.testing.assert_array_equal(r4, r1)

    def testFetchDataFrame(self, *_):
        from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df
        from mars.dataframe.arithmetic import add

        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            session = cluster.session

            data1 = pd.DataFrame(np.random.rand(10, 10))
            df1 = from_pandas_df(data1, chunk_size=5)
            data2 = pd.DataFrame(np.random.rand(10, 10))
            df2 = from_pandas_df(data2, chunk_size=6)

            df3 = add(df1, df2)

            r1 = session.run(df3, compose=False, timeout=_exec_timeout)
            r2 = session.fetch(df3)
            pd.testing.assert_frame_equal(r1, r2)

            data4 = pd.DataFrame(np.random.rand(10, 10))
            df4 = from_pandas_df(data4, chunk_size=6)

            df5 = add(df3, df4)

            r1 = session.run(df5, compose=False, timeout=_exec_timeout)
            r2 = session.fetch(df5)
            pd.testing.assert_frame_equal(r1, r2)

            df6 = df5.sum()
            r1 = session.run(df6, timeout=_exec_timeout)
            r2 = session.fetch(df6)
            pd.testing.assert_series_equal(r1, r2)

    def testMultiSessionDecref(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            session = cluster.session

            a = mt.ones((10, 20), chunk_size=8)
            b = mt.ones((10, 20), chunk_size=8)
            self.assertEqual(a.key, b.key)

            r1 = session.run(a, timeout=_exec_timeout)
            r1_fetch = session.fetch(a)
            np.testing.assert_array_equal(r1, r1_fetch)

            web_session = new_session('http://' + cluster._web_endpoint)
            r2 = web_session.run(a, timeout=_exec_timeout)
            r2_fetch = web_session.fetch(a)
            np.testing.assert_array_equal(r1, r2)
            np.testing.assert_array_equal(r2, r2_fetch)

            local_session = new_session()
            r3 = local_session.run(a)
            r3_fetch = local_session.fetch(a)
            np.testing.assert_array_equal(r1, r3)
            np.testing.assert_array_equal(r3, r3_fetch)

            del a
            self.assertEqual(len(local_session._sess._executor.chunk_result), 0)

            with self.assertRaises(ValueError):
                session.fetch(b)

            with self.assertRaises(ValueError):
                web_session.fetch(b)

    def testEagerMode(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:

            self.assertIsInstance(Session.default_or_local()._sess, ClusterSession)

            with option_context({'eager_mode': True}):
                a_data = np.random.rand(10, 10)

                a = mt.tensor(a_data, chunk_size=3)
                np.testing.assert_array_equal(a, a_data)

                r1 = a + 1
                expected1 = a_data + 1
                np.testing.assert_array_equal(r1, expected1)

                r2 = r1.dot(r1)
                expected2 = expected1.dot(expected1)
                np.testing.assert_array_almost_equal(r2, expected2)

            a = mt.ones((10, 10), chunk_size=3)
            with self.assertRaises(ValueError):
                a.fetch()

            r = a.dot(a)
            np.testing.assert_array_equal(r.to_numpy(), np.ones((10, 10)) * 10)

            with new_session('http://' + cluster._web_endpoint).as_default():
                self.assertIsInstance(Session.default_or_local()._sess, WebSession)

                with option_context({'eager_mode': True}):
                    a_data = np.random.rand(10, 10)

                    a = mt.tensor(a_data, chunk_size=3)
                    np.testing.assert_array_equal(a, a_data)

                    r1 = a + 1
                    expected1 = a_data + 1
                    np.testing.assert_array_equal(r1, expected1)

                    r2 = r1.dot(r1)
                    expected2 = expected1.dot(expected1)
                    np.testing.assert_array_almost_equal(r2, expected2)

                    web_session = Session.default_or_local()._sess
                    self.assertEqual(web_session.get_task_count(), 3)

                a = mt.ones((10, 10), chunk_size=3)
                with self.assertRaises(ValueError):
                    a.fetch()

                r = a.dot(a)
                np.testing.assert_array_equal(r.to_numpy(), np.ones((10, 10)) * 10)

            with new_session('http://' + cluster._web_endpoint).as_default():
                from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df
                from mars.dataframe.datasource.series import from_pandas as from_pandas_series
                from mars.dataframe.arithmetic import add

                self.assertIsInstance(Session.default_or_local()._sess, WebSession)

                with option_context({'eager_mode': True}):
                    data1 = pd.DataFrame(np.random.rand(10, 10), index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
                                         columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
                    df1 = from_pandas_df(data1, chunk_size=5)
                    pd.testing.assert_frame_equal(df1.fetch(), data1)

                    data2 = pd.DataFrame(np.random.rand(10, 10), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3],
                                         columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
                    df2 = from_pandas_df(data2, chunk_size=6)
                    pd.testing.assert_frame_equal(df2.fetch(), data2)

                    df3 = add(df1, df2)
                    pd.testing.assert_frame_equal(df3.fetch(), data1 + data2)

                    s1 = pd.Series(np.random.rand(10), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3])
                    series1 = from_pandas_series(s1)
                    pd.testing.assert_series_equal(series1.fetch(), s1)

                web_session = Session.default_or_local()._sess
                self.assertEqual(web_session.get_task_count(), 4)

    def testSparse(self, *_):
        import scipy.sparse as sps

        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=False) as cluster:
            session = cluster.session

            # calculate sparse with no element in matrix
            a = sps.csr_matrix((10000, 10000))
            b = sps.csr_matrix((10000, 1))
            t1 = mt.tensor(a)
            t2 = mt.tensor(b)
            session.run(t1 * t2, timeout=_exec_timeout)

    def testRunWithoutCompose(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=False) as cluster:
            session = cluster.session

            arr1 = (mt.ones((10, 10), chunk_size=3) + 1) * 2
            r1 = session.run(arr1, timeout=_exec_timeout)
            arr2 = (mt.ones((10, 10), chunk_size=4) + 1) * 2
            r2 = session.run(arr2, compose=False, timeout=_exec_timeout)
            np.testing.assert_array_equal(r1, r2)

    def testExistingOperand(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            session = cluster.session
            a = mt.ones((3, 3), chunk_size=2)
            r1 = session.run(a, compose=False, timeout=_exec_timeout)
            np.testing.assert_array_equal(r1, np.ones((3, 3)))

            b = mt.ones((4, 4), chunk_size=2) + 1
            r2 = session.run(b, compose=False, timeout=_exec_timeout)
            np.testing.assert_array_equal(r2, np.ones((4, 4)) + 1)

            del a
            b = mt.ones((3, 3), chunk_size=2)
            r2 = session.run(b, compose=False, timeout=_exec_timeout)
            np.testing.assert_array_equal(r2, np.ones((3, 3)))

            del b
            c = mt.ones((4, 4), chunk_size=2) + 1
            c = c.dot(c)
            r3 = session.run(c, compose=False, timeout=_exec_timeout)
            np.testing.assert_array_equal(r3, np.ones((4, 4)) * 16)

            d = mt.ones((5, 5), chunk_size=2)
            d = d.dot(d)
            r4 = session.run(d, compose=False, timeout=_exec_timeout)
            np.testing.assert_array_equal(r4, np.ones((5, 5)) * 5)

    def testTiledTensor(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            session = cluster.session
            a = mt.ones((10, 10), chunk_size=3)
            b = a.dot(a)
            b = b.tiles()

            r = session.run(b, timeout=_exec_timeout)
            np.testing.assert_array_equal(r, np.ones((10, 10)) * 10)

            a = a.tiles()
            b = a + 1

            r = session.run(b, timeout=_exec_timeout)
            np.testing.assert_array_equal(r, np.ones((10, 10)) + 1)

    def testFetchSlices(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            session = cluster.session
            a = mt.random.rand(10, 10, 10, chunk_size=3)

            r = session.run(a)

            r_slice1 = session.fetch(a[:2])
            np.testing.assert_array_equal(r[:2], r_slice1)

            r_slice2 = session.fetch(a[2:8, 2:8])
            np.testing.assert_array_equal(r[2:8, 2:8], r_slice2)

            r_slice3 = session.fetch(a[:, 2:])
            np.testing.assert_array_equal(r[:, 2:], r_slice3)

            r_slice4 = session.fetch(a[:, 2:, -5:])
            np.testing.assert_array_equal(r[:, 2:, -5:], r_slice4)

            r_slice5 = session.fetch(a[0])
            np.testing.assert_array_equal(r[0], r_slice5)

            # test repr
            with np.printoptions(threshold=100):
                raw = np.random.randint(1000, size=(3, 4, 6))
                b = mt.tensor(raw, chunk_size=3)
                self.assertEqual(repr(b.execute(session=session)),
                                 repr(raw))

            web_session = new_session('http://' + cluster._web_endpoint)
            r = web_session.run(a)

            r_slice1 = web_session.fetch(a[:2])
            np.testing.assert_array_equal(r[:2], r_slice1)

            r_slice2 = web_session.fetch(a[2:8, 2:8])
            np.testing.assert_array_equal(r[2:8, 2:8], r_slice2)

            r_slice3 = web_session.fetch(a[:, 2:])
            np.testing.assert_array_equal(r[:, 2:], r_slice3)

            r_slice4 = web_session.fetch(a[:, 2:, -5:])
            np.testing.assert_array_equal(r[:, 2:, -5:], r_slice4)

            r_slice5 = web_session.fetch(a[4])
            np.testing.assert_array_equal(r[4], r_slice5)

    def testFetchDataFrameSlices(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            session = cluster.session
            a = mt.random.rand(10, 10, chunk_size=3)
            df = md.DataFrame(a)

            r = session.run(df)

            r_slice1 = session.fetch(df.iloc[:2])
            pd.testing.assert_frame_equal(r.iloc[:2], r_slice1)

            r_slice2 = session.fetch(df.iloc[2:8, 2:8])
            pd.testing.assert_frame_equal(r.iloc[2:8, 2:8], r_slice2)

            r_slice3 = session.fetch(df.iloc[:, 2:])
            pd.testing.assert_frame_equal(r.iloc[:, 2:], r_slice3)

            r_slice4 = session.fetch(df.iloc[:, -5:])
            pd.testing.assert_frame_equal(r.iloc[:, -5:], r_slice4)

            r_slice5 = session.fetch(df.iloc[4])
            pd.testing.assert_series_equal(r.iloc[4], r_slice5)

            r_slice6 = session.fetch(df.iloc[6:9])
            pd.testing.assert_frame_equal(r.iloc[6:9], r_slice6)

            # test repr
            pdf = pd.DataFrame(np.random.randint(1000, size=(80, 10)))
            df2 = md.DataFrame(pdf, chunk_size=41)
            self.assertEqual(repr(df2.execute(session=session)), repr(pdf))

            ps = pdf[0]
            s = md.Series(ps, chunk_size=41)
            self.assertEqual(repr(s.execute(session=session)), repr(ps))

            web_session = new_session('http://' + cluster._web_endpoint)
            r = web_session.run(df)

            r_slice1 = web_session.fetch(df.iloc[:2])
            pd.testing.assert_frame_equal(r.iloc[:2], r_slice1)

            r_slice2 = web_session.fetch(df.iloc[2:8, 2:8])
            pd.testing.assert_frame_equal(r.iloc[2:8, 2:8], r_slice2)

            r_slice3 = web_session.fetch(df.iloc[:, 2:])
            pd.testing.assert_frame_equal(r.iloc[:, 2:], r_slice3)

            r_slice4 = web_session.fetch(df.iloc[:, -5:])
            pd.testing.assert_frame_equal(r.iloc[:, -5:], r_slice4)

            r_slice5 = web_session.fetch(df.iloc[4])
            pd.testing.assert_series_equal(r.iloc[4], r_slice5)

            r_slice6 = web_session.fetch(df.iloc[6:9])
            pd.testing.assert_frame_equal(r.iloc[6:9], r_slice6)

    def testClusterSession(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            sess1 = cluster.session
            sess2 = new_session(cluster.endpoint, session_id=sess1.session_id)

            self.assertNotEqual(sess1, sess2)
            self.assertEqual(sess1.session_id, sess2.session_id)

            session_id = str(uuid.uuid4())
            with self.assertRaises(ValueError) as cm:
                new_session(cluster.endpoint, session_id=session_id)

            expected_msg = "The session with id = %s doesn't exist" % session_id
            self.assertEqual(cm.exception.args[0], expected_msg)

            sess1.close()
            with self.assertRaises(ValueError) as cm:
                new_session(cluster.endpoint, session_id=sess1.session_id)

            expected_msg = "The session with id = %s doesn't exist" % sess1.session_id
            self.assertEqual(cm.exception.args[0], expected_msg)

            web_sess1 = new_session('http://' + cluster._web_endpoint)
            web_sess2 = new_session('http://' + cluster._web_endpoint, session_id=web_sess1.session_id)

            self.assertNotEqual(web_sess1, web_sess2)
            self.assertEqual(web_sess1.session_id, web_sess2.session_id)

            session_id = str(uuid.uuid4())
            with self.assertRaises(ValueError) as cm:
                new_session('http://' + cluster._web_endpoint, session_id=session_id)

            expected_msg = "The session with id = %s doesn't exist" % session_id
            self.assertEqual(cm.exception.args[0], expected_msg)

            web_sess1.close()
            with self.assertRaises(ValueError) as cm:
                new_session('http://' + cluster._web_endpoint, session_id=web_sess1.session_id)

            expected_msg = "The session with id = %s doesn't exist" % web_sess1.session_id
            self.assertEqual(cm.exception.args[0], expected_msg)

    def testTensorOrder(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            session = cluster.session

            data = np.asfortranarray(np.random.rand(10, 7))
            a = mt.asfortranarray(data, chunk_size=3)
            b = (a + 1) * 2
            res = session.run(b, timeout=_exec_timeout)
            expected = (data + 1) * 2

            np.testing.assert_array_equal(res, expected)
            self.assertEqual(res.flags['C_CONTIGUOUS'], expected.flags['C_CONTIGUOUS'])
            self.assertEqual(res.flags['F_CONTIGUOUS'], expected.flags['F_CONTIGUOUS'])

            c = b.reshape(7, 10, order='F')
            res = session.run(c, timeout=_exec_timeout)
            expected = ((data + 1) * 2).reshape((7, 10), order='F')

            np.testing.assert_array_equal(res, expected)
            self.assertEqual(res.flags['C_CONTIGUOUS'], expected.flags['C_CONTIGUOUS'])
            self.assertEqual(res.flags['F_CONTIGUOUS'], expected.flags['F_CONTIGUOUS'])

    def testIterativeDependency(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True):
            with tempfile.TemporaryDirectory() as d:
                file_path = os.path.join(d, 'test.csv')
                df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
                df.to_csv(file_path, index=False)

                mdf1 = md.read_csv(file_path, chunk_bytes=10)
                r1 = mdf1.iloc[:3].to_pandas()
                pd.testing.assert_frame_equal(df[:3], r1.reset_index(drop=True))

                mdf2 = md.read_csv(file_path, chunk_bytes=10)
                r2 = mdf2.iloc[:3].to_pandas()
                pd.testing.assert_frame_equal(df[:3], r2.reset_index(drop=True))

                f = mdf1[mdf1.a > mdf2.a]
                r3 = f.iloc[:3].to_pandas()
                pd.testing.assert_frame_equal(r3, df[df.a > df.a].reset_index(drop=True))

                mdf3 = md.read_csv(file_path, chunk_bytes=15, incremental_index=True)
                r4 = mdf3.to_pandas()
                pd.testing.assert_frame_equal(df, r4.reset_index(drop=True))

    def testDataFrameShuffle(self, *_):
        from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df
        from mars.dataframe.merge.merge import merge
        from mars.dataframe.utils import sort_dataframe_inplace

        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            session = cluster.session

            data1 = pd.DataFrame(np.arange(20).reshape((4, 5)) + 1, columns=['a', 'b', 'c', 'd', 'e'])
            data2 = pd.DataFrame(np.arange(20).reshape((5, 4)) + 1, columns=['a', 'b', 'x', 'y'])

            df1 = from_pandas_df(data1, chunk_size=2)
            df2 = from_pandas_df(data2, chunk_size=2)

            r1 = data1.merge(data2)
            r2 = session.run(merge(df1, df2), timeout=_exec_timeout)
            pd.testing.assert_frame_equal(sort_dataframe_inplace(r1, 0), sort_dataframe_inplace(r2, 0))

            r1 = data1.merge(data2, how='inner', on=['a', 'b'])
            r2 = session.run(merge(df1, df2, how='inner', on=['a', 'b']), timeout=_exec_timeout)
            pd.testing.assert_frame_equal(sort_dataframe_inplace(r1, 0), sort_dataframe_inplace(r2, 0))

            web_session = new_session('http://' + cluster._web_endpoint)

            r1 = data1.merge(data2)
            r2 = web_session.run(merge(df1, df2), timeout=_exec_timeout)
            pd.testing.assert_frame_equal(sort_dataframe_inplace(r1, 0), sort_dataframe_inplace(r2, 0))

            r1 = data1.merge(data2, how='inner', on=['a', 'b'])
            r2 = web_session.run(merge(df1, df2, how='inner', on=['a', 'b']), timeout=_exec_timeout)
            pd.testing.assert_frame_equal(sort_dataframe_inplace(r1, 0), sort_dataframe_inplace(r2, 0))

    @require_cudf
    def testCudaCluster(self, *_):
        from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df

        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            session = cluster.session
            pdf = pd.DataFrame(np.random.rand(20, 30), index=np.arange(20, 0, -1))
            df = from_pandas_df(pdf, chunk_size=(13, 21))
            cdf = df.to_gpu()
            result = session.run(cdf)
            pd.testing.assert_frame_equal(pdf, result)

    def testTileContextInLocalCluster(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', modules=[__name__], web=True) as cluster:
            session = cluster.session

            raw = np.random.rand(10, 20)
            data = mt.tensor(raw)

            session.run(data)

            data2 = TileWithContextOperand().new_tensor([data], shape=data.shape)

            result = session.run(data2)
            np.testing.assert_array_equal(raw * raw.nbytes, result)

    @unittest.skipIf(h5py is None, 'h5py not installed')
    def testStoreHDF5ForLocalCluster(self):
        with new_cluster(worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            session = cluster.session

            raw = np.random.RandomState(0).rand(10, 20)
            t = mt.tensor(raw, chunk_size=11)

            dataset = 'test_dataset'
            with tempfile.TemporaryDirectory() as d:
                filename = os.path.join(d, 'test_read_{}.hdf5'.format(int(time.time())))

                r = mt.tohdf5(filename, t, dataset=dataset)

                session.run(r, timeout=_exec_timeout)

                with h5py.File(filename, 'r') as f:
                    result = np.asarray(f[dataset])
                    np.testing.assert_array_equal(result, raw)

    def testRemoteFunctionInLocalCluster(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=3,
                         shared_memory='20M', modules=[__name__], web=True) as cluster:
            session = cluster.session

            def f(x):
                return x + 1

            def g(x, y):
                return x * y

            a = mr.spawn(f, 3)
            b = mr.spawn(f, 4)
            c = mr.spawn(g, (a, b))

            r = session.run(c, timeout=_exec_timeout)
            self.assertEqual(r, 20)

            e = mr.spawn(f, mr.spawn(f, 2))

            r = session.run(e, timeout=_exec_timeout)
            self.assertEqual(r, 4)

            session2 = new_session(cluster.endpoint)
            expect_session_id = session2.session_id

            def f2():
                session = Session.default
                assert isinstance(session._sess, ClusterSession)
                assert session._sess.session_id == expect_session_id

                t = mt.ones((3, 2))
                return t.sum().to_numpy()

            self.assertEqual(cloudpickle.loads(cloudpickle.dumps(Session.default)).session_id,
                             session.session_id)
            self.assertIsInstance(serialize_function(f2), bytes)

            d = mr.spawn(f2, retry_when_fail=False)

            r = session2.run(d, timeout=_exec_timeout)
            self.assertEqual(r, 6)

            # test input tileable
            def f(t, x):
                return (t * x).sum().to_numpy()

            rs = np.random.RandomState(0)
            raw = rs.rand(5, 4)

            t1 = mt.tensor(raw, chunk_size=3)
            t2 = t1.sum(axis=0)
            s = mr.spawn(f, args=(t2, 3), retry_when_fail=False)

            r = session.run(s, timeout=_exec_timeout)
            expected = (raw.sum(axis=0) * 3).sum()
            self.assertAlmostEqual(r, expected)

            # test named tileable
            session3 = new_session(cluster.endpoint)
            t = mt.ones((10, 10), chunk_size=3)
            session3.run(t, name='t_name')

            def f3():
                import mars.tensor as mt

                s = mt.named_tensor(name='t_name')
                return (s + 1).to_numpy()

            d = mr.spawn(f3, retry_when_fail=False)
            r = session3.run(d, timeout=_exec_timeout)
            np.testing.assert_array_equal(r, np.ones((10, 10)) + 1)

            # test tileable that executed
            session4 = new_session(cluster.endpoint)
            df1 = md.DataFrame(raw, chunk_size=3)
            df1 = df1[df1.iloc[:, 0] < 1.5]

            def f4(input_df):
                bonus = input_df.iloc[:, 0].fetch().sum()
                return input_df.sum().to_pandas() + bonus

            d = mr.spawn(f4, args=(df1,), retry_when_fail=False)
            r = session4.run(d, timeout=_exec_timeout)
            expected = pd.DataFrame(raw).sum() + raw[:, 0].sum()
            pd.testing.assert_series_equal(r, expected)

            # test tileable has unknown shape
            session5 = new_session(cluster.endpoint)

            def f5(t, x):
                assert all(not np.isnan(s) for s in t.shape)
                return (t * x).sum().to_numpy()

            rs = np.random.RandomState(0)
            raw = rs.rand(5, 4)

            t1 = mt.tensor(raw, chunk_size=3)
            t2 = t1[t1 < 0.5]
            s = mr.spawn(f5, args=(t2, 3))
            result = session5.run(s, timeout=_exec_timeout)
            expected = (raw[raw < 0.5] * 3).sum()
            self.assertAlmostEqual(result, expected)

    @unittest.skipIf(sklearn is None, 'sklearn not installed')
    def testLearnInLocalCluster(self, *_):
        from mars.learn.cluster import KMeans
        from mars.learn.neighbors import NearestNeighbors
        from sklearn.cluster import KMeans as SK_KMEANS
        from sklearn.neighbors import NearestNeighbors as SkNearestNeighbors

        with new_cluster(scheduler_n_process=2, worker_n_process=3, shared_memory='20M') as cluster:
            rs = np.random.RandomState(0)
            raw_X = rs.rand(10, 5)
            raw_Y = rs.rand(8, 5)

            X = mt.tensor(raw_X, chunk_size=7)
            Y = mt.tensor(raw_Y, chunk_size=(5, 3))
            nn = NearestNeighbors(n_neighbors=3)
            nn.fit(X)

            ret = nn.kneighbors(Y, session=cluster.session)

            snn = SkNearestNeighbors(n_neighbors=3)
            snn.fit(raw_X)
            expected = snn.kneighbors(raw_Y)

            result = [r.fetch() for r in ret]
            np.testing.assert_almost_equal(result[0], expected[0])
            np.testing.assert_almost_equal(result[1], expected[1])

            raw = np.array([[1, 2], [1, 4], [1, 0],
                            [10, 2], [10, 4], [10, 0]])
            X = mt.array(raw)
            kmeans = KMeans(n_clusters=2, random_state=0, init='k-means++').fit(X)
            sk_km_elkan = SK_KMEANS(n_clusters=2, random_state=0, init='k-means++').fit(raw)
            np.testing.assert_allclose(kmeans.cluster_centers_, sk_km_elkan.cluster_centers_)
