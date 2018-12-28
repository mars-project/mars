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
import unittest
import sys

import numpy as np

from mars import tensor as mt
from mars.operands import Operand
from mars.tensor.expressions.arithmetic.core import TensorElementWise
from mars.serialize import Int64Field
from mars.config import options
from mars.session import new_session, Session
from mars.deploy.local.core import new_cluster, LocalDistributedCluster, gen_endpoint
from mars.cluster_info import ClusterInfoActor
from mars.scheduler.session import SessionManagerActor
from mars.worker.dispatcher import DispatchActor


def _on_deserialize_fail(x):
    raise TypeError('intend to throw error on' + str(x))


class SerializeMustFailOperand(Operand, TensorElementWise):
    _op_type_ = 356789

    _f = Int64Field('f', on_deserialize=_on_deserialize_fail)

    def __init__(self, f=None, **kw):
        super(SerializeMustFailOperand, self).__init__(_f=f, **kw)


@unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
class Test(unittest.TestCase):
    def setUp(self):
        self._old_cache_memory_limit = options.worker.cache_memory_limit
        options.worker.cache_memory_limit = '20M'

    def tearDown(self):
        options.worker.cache_memory_limit = self._old_cache_memory_limit

    def testLocalCluster(self):
        endpoint = gen_endpoint('0.0.0.0')
        with LocalDistributedCluster(endpoint, scheduler_n_process=2, worker_n_process=3) as cluster:
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
        with new_cluster(scheduler_n_process=2, worker_n_process=3, web=True) as cluster:
            with cluster.session as session:
                t = mt.ones((3, 3), chunk_size=2)
                result = session.run(t)

                np.testing.assert_array_equal(result, np.ones((3, 3)))

            with new_session('http://' + cluster._web_endpoint) as session:
                t = mt.ones((3, 3), chunk_size=2)
                result = session.run(t)

                np.testing.assert_array_equal(result, np.ones((3, 3)))

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
        with new_cluster(scheduler_n_process=2, worker_n_process=2) as cluster:
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
        with new_cluster(scheduler_n_process=2, worker_n_process=2) as cluster:
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

                s_result = session2.run(s)
                _, s_expected, _ = np.linalg.svd(raw, full_matrices=False)
                np.testing.assert_allclose(s_result, s_expected)

    def testIndexTensorExecute(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2) as cluster:
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

    def testExecutableTuple(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2, web=True) as cluster:
            with new_session('http://' + cluster._web_endpoint).as_default() as _:
                a = mt.ones((20, 10), chunk_size=10)
                u, s, v = (mt.linalg.svd(a)).execute()
                np.testing.assert_allclose(u.dot(np.diag(s).dot(v)), np.ones((20, 10)))

    def testGraphFail(self):
        op = SerializeMustFailOperand(f=3)
        tensor = op.new_tensor(None, (3, 3))

        with new_cluster(scheduler_n_process=2, worker_n_process=2) as cluster:
            with self.assertRaises(SystemError):
                cluster.session.run(tensor)
