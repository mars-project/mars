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
from mars.session import new_session
from mars.deploy.local.core import new_cluster, LocalDistributedCluster, gen_endpoint
from mars.cluster_info import ClusterInfoActor
from mars.scheduler.session import SessionManagerActor
from mars.worker.dispatcher import DispatchActor


class Test(unittest.TestCase):
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
