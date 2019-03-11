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

import functools
import logging
import sys
import time
import unittest

import numpy as np

from mars import tensor as mt
from mars.operands import Operand
from mars.tensor.expressions.arithmetic.core import TensorElementWise
from mars.serialize import Int64Field
from mars.session import new_session, Session
from mars.deploy.local.core import new_cluster, LocalDistributedCluster, gen_endpoint
from mars.deploy.local.session import LocalClusterSession
from mars.cluster_info import ClusterInfoActor
from mars.scheduler import SessionManagerActor
from mars.worker.dispatcher import DispatchActor
from mars.errors import ExecutionFailed
from mars.config import option_context
from mars.web.session import Session as WebSession

logger = logging.getLogger(__name__)


def _on_deserialize_fail(x):
    raise TypeError('intend to throw error on' + str(x))


class SerializeMustFailOperand(Operand, TensorElementWise):
    _op_type_ = 356789

    _f = Int64Field('f', on_deserialize=_on_deserialize_fail)

    def __init__(self, f=None, **kw):
        super(SerializeMustFailOperand, self).__init__(_f=f, **kw)


@unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
class Test(unittest.TestCase):
    def testLocalCluster(self):
        endpoint = gen_endpoint('0.0.0.0')
        with LocalDistributedCluster(endpoint, scheduler_n_process=2, worker_n_process=3,
                                     shared_memory='20M') as cluster:
            pool = cluster.pool

            self.assertTrue(pool.has_actor(pool.actor_ref(ClusterInfoActor.default_name())))
            self.assertTrue(pool.has_actor(pool.actor_ref(SessionManagerActor.default_name())))
            self.assertTrue(pool.has_actor(pool.actor_ref(DispatchActor.default_name())))

            with new_session(endpoint) as session:
                api = session._api

                t = mt.ones((3, 3), chunk_size=2)
                result = session.run(t)

                np.testing.assert_array_equal(result, np.ones((3, 3)))

            self.assertNotIn(session._session_id, api.session_manager.get_sessions())

    def testLocalClusterWithWeb(self):
        import psutil
        with new_cluster(scheduler_n_process=2, worker_n_process=3,
                         shared_memory='20M', web=True) as cluster:
            cluster_proc = psutil.Process(cluster._cluster_process.pid)
            web_proc = psutil.Process(cluster._web_process.pid)
            processes = list(cluster_proc.children(recursive=True)) + \
                list(web_proc.children(recursive=True))

            with cluster.session as session:
                t = mt.ones((3, 3), chunk_size=2)
                result = session.run(t)

                np.testing.assert_array_equal(result, np.ones((3, 3)))

            with new_session('http://' + cluster._web_endpoint) as session:
                t = mt.ones((3, 3), chunk_size=2)
                result = session.run(t)

                np.testing.assert_array_equal(result, np.ones((3, 3)))

        check_time = time.time()
        while any(p.is_running() for p in processes):
            time.sleep(0.1)
            if check_time + 10 < time.time():
                logger.error('Processes still running: %r',
                             [' '.join(p.cmdline()) for p in processes if p.is_running()])
                self.assertFalse(any(p.is_running() for p in processes))

    def testNSchedulersNWorkers(self):
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

    def testSingleOutputTensorExecute(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            self.assertIs(cluster.session, Session.default_or_local())

            t = mt.random.rand(10)
            r = t.sum()

            res = r.execute()
            self.assertTrue(np.isscalar(res))
            self.assertLess(res, 10)

            t = mt.random.rand(10)
            r = t.sum() * 4 - 1

            res = r.execute()
            self.assertLess(res, 39)

    def testMultipleOutputTensorExecute(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            session = cluster.session

            t = mt.random.rand(20, 5, chunk_size=5)
            r = mt.linalg.svd(t)

            res = session.run((t,) + r)

            U, s, V = res[1:]
            np.testing.assert_allclose(res[0], U.dot(np.diag(s).dot(V)))

            raw = np.random.rand(20, 5)

            # to test the fuse, the graph should be fused
            t = mt.array(raw)
            U, s, V = mt.linalg.svd(t)
            r = U.dot(mt.diag(s).dot(V))

            res = r.execute()
            np.testing.assert_allclose(raw, res)

            # test submit part of svd outputs
            t = mt.array(raw)
            U, s, V = mt.linalg.svd(t)

            with new_session(cluster.endpoint) as session2:
                U_result, s_result = session2.run(U, s)
                U_expected, s_expectd, _ = np.linalg.svd(raw, full_matrices=False)

                np.testing.assert_allclose(U_result, U_expected)
                np.testing.assert_allclose(s_result, s_expectd)

            with new_session(cluster.endpoint) as session2:
                U_result, s_result = session2.run(U + 1, s + 1)
                U_expected, s_expectd, _ = np.linalg.svd(raw, full_matrices=False)

                np.testing.assert_allclose(U_result, U_expected + 1)
                np.testing.assert_allclose(s_result, s_expectd + 1)

            with new_session(cluster.endpoint) as session2:
                t = mt.array(raw)
                _, s, _ = mt.linalg.svd(t)
                del _

                s_result = session2.run(s)
                s_expected = np.linalg.svd(raw, full_matrices=False)[1]
                np.testing.assert_allclose(s_result, s_expected)

    def testIndexTensorExecute(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            session = cluster.session

            a = mt.random.rand(10, 5)
            idx = slice(0, 5), slice(0, 5)
            a[idx] = 2
            a_splits = mt.split(a, 2)
            r1, r2 = session.run(a_splits[0], a[idx])

            np.testing.assert_array_equal(r1, r2)
            np.testing.assert_array_equal(r1, np.ones((5, 5)) * 2)

            with new_session(cluster.endpoint) as session2:
                a = mt.random.rand(10, 5)
                idx = slice(0, 5), slice(0, 5)

                a[idx] = mt.ones((5, 5)) * 2
                r = session2.run(a[idx])

                np.testing.assert_array_equal(r, np.ones((5, 5)) * 2)

            with new_session(cluster.endpoint) as session3:
                a = mt.random.rand(100, 5)

                slice1 = a[:10]
                slice2 = a[10:20]
                r1, r2, expected = session3.run(slice1, slice2, a)

                np.testing.assert_array_equal(r1, expected[:10])
                np.testing.assert_array_equal(r2, expected[10:20])

            with new_session(cluster.endpoint) as session4:
                a = mt.random.rand(100, 5)

                a[:10] = mt.ones((10, 5))
                a[10:20] = 2
                r = session4.run(a)

                np.testing.assert_array_equal(r[:10], np.ones((10, 5)))
                np.testing.assert_array_equal(r[10:20], np.ones((10, 5)) * 2)

    def testBoolIndexingExecute(self, *_):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            a = mt.random.rand(8, 8, chunk_size=4)
            a[2:6, 2:6] = mt.ones((4, 4)) * 2
            b = a[a > 1]
            self.assertEqual(b.shape, (np.nan,))

            cluster.session.run(b, fetch=False)
            self.assertEqual(b.shape, (16,))

            c = b.reshape((4, 4))

            self.assertEqual(c.shape, (4, 4))

            with new_session('http://' + cluster._web_endpoint) as session2:
                a = mt.random.rand(8, 8, chunk_size=4)
                a[2:6, 2:6] = mt.ones((4, 4)) * 2
                b = a[a > 1]
                self.assertEqual(b.shape, (np.nan,))

                session2.run(b, fetch=False)
                self.assertEqual(b.shape, (16,))

                c = b.reshape((4, 4))
                self.assertEqual(c.shape, (4, 4))

            # test unknown-shape fusion
            with new_session('http://' + cluster._web_endpoint) as session2:
                a = mt.random.rand(6, 6, chunk_size=3)
                a[2:5, 2:5] = mt.ones((3, 3)) * 2
                b = (a[a > 1] - 1) * 2

                r = session2.run(b)
                np.testing.assert_array_equal(r, np.ones((9,)) * 2)

    def testExecutableTuple(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            with new_session('http://' + cluster._web_endpoint).as_default():
                a = mt.ones((20, 10), chunk_size=10)
                u, s, v = (mt.linalg.svd(a)).execute()
                np.testing.assert_allclose(u.dot(np.diag(s).dot(v)), np.ones((20, 10)))

    def testRerunTensor(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            session = cluster.session

            a = mt.ones((10, 10)) + 1
            result1 = session.run(a)
            np.testing.assert_array_equal(result1, np.ones((10, 10)) + 1)
            result2 = session.run(a)
            np.testing.assert_array_equal(result1, result2)

            with new_session(cluster.endpoint) as session2:
                a = mt.random.rand(10, 10)
                a_result1 = session2.run(a)
                b = mt.ones((10, 10))
                a_result2, b_result = session2.run(a, b)
                np.testing.assert_array_equal(a_result1, a_result2)
                np.testing.assert_array_equal(b_result, np.ones((10, 10)))

    def testRunWithoutFetch(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            session = cluster.session

            a = mt.ones((10, 20)) + 1
            self.assertIsNone(session.run(a, fetch=False))
            np.testing.assert_array_equal(a.execute(session=session), np.ones((10, 20)) + 1)

    def testGraphFail(self):
        op = SerializeMustFailOperand(f=3)
        tensor = op.new_tensor(None, (3, 3))

        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            with self.assertRaises(ExecutionFailed):
                cluster.session.run(tensor)

    def testFetch(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            session = cluster.session

            a1 = mt.ones((10, 20), chunk_size=8) + 1
            r1 = session.run(a1)
            r2 = session.fetch(a1)
            np.testing.assert_array_equal(r1, r2)

            r3 = session.run(a1 * 2)
            np.testing.assert_array_equal(r3, r1 * 2)

            a2 = mt.ones((10, 20), chunk_size=8) + 1
            r4 = session.run(a2)
            np.testing.assert_array_equal(r4, r1)

            del a1
            r4 = session.run(a2)
            np.testing.assert_array_equal(r4, r1)

            with new_session('http://' + cluster._web_endpoint) as session:
                a3 = mt.ones((5, 10), chunk_size=3) + 1
                r1 = session.run(a3)
                r2 = session.fetch(a3)
                np.testing.assert_array_equal(r1, r2)

                r3 = session.run(a3 * 2)
                np.testing.assert_array_equal(r3, r1 * 2)

                a4 = mt.ones((5, 10), chunk_size=3) + 1
                r4 = session.run(a4)
                np.testing.assert_array_equal(r4, r1)

                del a3
                r4 = session.run(a4)
                np.testing.assert_array_equal(r4, r1)

    def testMultiSessionDecref(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            session = cluster.session

            a = mt.ones((10, 20), chunk_size=8)
            b = mt.ones((10, 20), chunk_size=8)
            self.assertEqual(a.key, b.key)

            r1 = session.run(a)
            r1_fetch = session.fetch(a)
            np.testing.assert_array_equal(r1, r1_fetch)

            web_session = new_session('http://' + cluster._web_endpoint)
            r2 = web_session.run(a)
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

    def testEagerMode(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:

            self.assertIsInstance(Session.default_or_local()._sess, LocalClusterSession)

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
            np.testing.assert_array_equal(r.execute(), np.ones((10, 10)) * 10)

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

                a = mt.ones((10, 10), chunk_size=3)
                with self.assertRaises(ValueError):
                    a.fetch()

                r = a.dot(a)
                np.testing.assert_array_equal(r.execute(), np.ones((10, 10)) * 10)

    def testSparse(self):
        import scipy.sparse as sps

        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            session = cluster.session

            # calculate sparse with no element in matrix
            a = sps.csr_matrix((10000, 10000))
            b = sps.csr_matrix((10000, 1))
            t1 = mt.tensor(a)
            t2 = mt.tensor(b)
            session.run(t1 * t2)
