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

import time
import unittest

import numpy as np

from mars import tensor as mt
from mars.config import options
from mars.session import new_session, Session
from mars.deploy.local.core import new_cluster, LocalDistributedCluster, gen_endpoint
from mars.cluster_info import ClusterInfoActor
from mars.scheduler.session import SessionManagerActor
from mars.worker.dispatcher import DispatchActor


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

                t = mt.ones((3, 3), chunks=2)
                result = session.run(t)

                np.testing.assert_array_equal(result, np.ones((3, 3)))

            self.assertNotIn(session._session_id, api.session_manager.get_sessions())

    def testLocalClusterWithWeb(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=3, web=True) as cluster:
            with cluster.session as session:
                t = mt.ones((3, 3), chunks=2)
                result = session.run(t)

                np.testing.assert_array_equal(result, np.ones((3, 3)))

            with new_session('http://' + cluster._web_endpoint) as session:
                t = mt.ones((3, 3), chunks=2)
                result = session.run(t)

                np.testing.assert_array_equal(result, np.ones((3, 3)))

    def testNSchedulersNWorkers(self):
        calc_cpu_cnt = lambda: 4

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

    def testTensorExecute(self):
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
