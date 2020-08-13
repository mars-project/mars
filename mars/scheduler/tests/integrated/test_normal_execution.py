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

import logging
import operator
import os
import sys
import time
import unittest
from functools import reduce

import numpy as np
import pandas as pd

import mars.dataframe as md
import mars.tensor as mt
from mars.errors import ExecutionFailed
from mars.scheduler.resource import ResourceActor
from mars.scheduler.tests.integrated.base import SchedulerIntegratedTest
from mars.scheduler.tests.integrated.no_prepare_op import NoPrepareOperand
from mars.session import new_session
from mars.remote import spawn
from mars.tests.core import EtcdProcessHelper, require_cupy, require_cudf
from mars.context import DistributedContext

logger = logging.getLogger(__name__)


@unittest.skipIf(sys.platform == 'win32', "plasma don't support windows")
class Test(SchedulerIntegratedTest):
    def testMainTensorWithoutEtcd(self):
        self.start_processes()
        sess = new_session(self.session_manager_ref.address)

        a = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        b = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        c = (a * b * 2 + 1).sum()

        result = c.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        expected = (np.ones(a.shape) * 2 * 1 + 1) ** 2 * 2 + 1
        np.testing.assert_allclose(result, expected.sum())

        a = mt.ones((100, 50), chunk_size=35) * 2 + 1
        b = mt.ones((50, 200), chunk_size=35) * 2 + 1
        c = a.dot(b)
        result = c.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        np.testing.assert_allclose(result, np.ones((100, 200)) * 450)

        base_arr = np.random.random((100, 100))
        a = mt.array(base_arr)
        r = reduce(operator.add, [a[:10, :10] for _ in range(10)])
        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        expected = reduce(operator.add, [base_arr[:10, :10] for _ in range(10)])
        np.testing.assert_allclose(result, expected)

        a = mt.ones((31, 27), chunk_size=10)
        b = a.reshape(27, 31)
        b.op.extra_params['_reshape_with_shuffle'] = True
        r = b.sum(axis=1)
        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        np.testing.assert_allclose(result, np.ones((27, 31)).sum(axis=1))

        raw = np.random.RandomState(0).rand(10, 10)
        a = mt.tensor(raw, chunk_size=(5, 4))
        r = a[a.argmin(axis=1), mt.tensor(np.arange(10))]
        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        np.testing.assert_array_equal(result, raw[raw.argmin(axis=1), np.arange(10)])

    @unittest.skipIf('CI' not in os.environ and not EtcdProcessHelper().is_installed(),
                     'does not run without etcd')
    def testMainTensorWithEtcd(self):
        self.start_processes(etcd=True)
        sess = new_session(self.session_manager_ref.address)

        a = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        b = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        r = (a * b * 2 + 1).sum()
        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        expected = (np.ones(a.shape) * 2 * 1 + 1) ** 2 * 2 + 1
        np.testing.assert_allclose(result, expected.sum())

    @require_cupy
    @require_cudf
    def testMainTensorWithCuda(self):
        self.start_processes(cuda=True)
        sess = new_session(self.session_manager_ref.address)

        a = mt.ones((100, 100), chunk_size=30, gpu=True) * 2 * 1 + 1
        b = mt.ones((100, 100), chunk_size=30, gpu=True) * 2 * 1 + 1
        r = (a * b * 2 + 1).sum()
        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        expected = ((np.ones(a.shape) * 2 * 1 + 1) ** 2 * 2 + 1).sum()
        np.testing.assert_allclose(result, expected)

    def testMainDataFrameWithoutEtcd(self):
        self.start_processes(etcd=False, scheduler_args=['-Dscheduler.aggressive_assign=true'])
        sess = new_session(self.session_manager_ref.address)

        raw1 = pd.DataFrame(np.random.rand(10, 10))
        df1 = md.DataFrame(raw1, chunk_size=5)
        raw2 = pd.DataFrame(np.random.rand(10, 10))
        df2 = md.DataFrame(raw2, chunk_size=6)
        r = df1 + df2
        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        pd.testing.assert_frame_equal(result, raw1 + raw2)

        raw1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                            columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = md.DataFrame(raw1, chunk_size=(10, 5))
        raw2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                            columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = md.DataFrame(raw2, chunk_size=(10, 6))
        r = df1 + df2
        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        pd.testing.assert_frame_equal(result, raw1 + raw2)

        raw1 = pd.DataFrame(np.random.rand(10, 10), index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
                            columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = md.DataFrame(raw1, chunk_size=5)
        raw2 = pd.DataFrame(np.random.rand(10, 10), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3],
                            columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = md.DataFrame(raw2, chunk_size=6)
        r = df1 + df2
        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        pd.testing.assert_frame_equal(result, raw1 + raw2)

        raw1 = pd.DataFrame(np.random.rand(10, 10))
        raw1[0] = raw1[0].apply(str)
        df1 = md.DataFrame(raw1, chunk_size=5)
        r = df1.sort_values(0)
        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        pd.testing.assert_frame_equal(result, raw1.sort_values(0))

        rs = np.random.RandomState(0)
        raw2 = pd.DataFrame({'a': rs.rand(10),
                            'b': ['s%d' % rs.randint(1000) for _ in range(10)]
                            })
        raw2['b'] = raw2['b'].astype(md.ArrowStringDtype())
        mdf = md.DataFrame(raw2, chunk_size=3)
        df2 = mdf.sort_values(by='b')
        result = df2.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        expected = raw2.sort_values(by='b')
        pd.testing.assert_frame_equal(result, expected)

        s1 = pd.Series(np.random.rand(10), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3])
        series1 = md.Series(s1, chunk_size=6)
        result = series1.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        pd.testing.assert_series_equal(result, s1)

        data = pd.DataFrame(np.random.rand(10, 5), columns=['c1', 'c2', 'c3', 'c4', 'c5'])
        df3 = md.DataFrame(data, chunk_size=4)

        r = df3.reindex(index=mt.arange(10, 1, -1, chunk_size=3))

        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        expected = data.reindex(index=np.arange(10, 1, -1))
        pd.testing.assert_frame_equal(result, expected)

    def testIterativeTilingWithoutEtcd(self):
        self.start_processes(etcd=False)
        sess = new_session(self.session_manager_ref.address)
        actor_client = sess._api.actor_client
        session_ref = actor_client.actor_ref(self.session_manager_ref.create_session(sess.session_id))
        rs = np.random.RandomState(0)

        raw = rs.rand(100)
        a = mt.tensor(raw, chunk_size=10)
        a.sort()
        r = a[:5]
        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        expected = np.sort(raw)[:5]
        np.testing.assert_allclose(result, expected)

        graph_key = sess._get_tileable_graph_key(r.key)
        graph_ref = actor_client.actor_ref(session_ref.get_graph_refs()[graph_key])
        with self.assertRaises(KeyError):
            _, keys, _ = graph_ref.get_tileable_metas([a.key])[0]
            sess._api.fetch_chunk_data(sess.session_id, keys[0])

        raw1 = rs.rand(20)
        raw2 = rs.rand(20)
        a = mt.tensor(raw1, chunk_size=10)
        a.sort()
        b = mt.tensor(raw2, chunk_size=15) + 1
        c = mt.concatenate([a[:10], b])
        c.sort()
        r = c[:5]
        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        expected = np.sort(np.concatenate([np.sort(raw1)[:10], raw2 + 1]))[:5]
        np.testing.assert_allclose(result, expected)

        raw = rs.randint(100, size=(100,))
        a = mt.tensor(raw, chunk_size=53)
        a.sort()
        r = mt.histogram(a, bins='scott')
        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        expected = np.histogram(np.sort(raw), bins='scott')
        np.testing.assert_allclose(result[0], expected[0])
        np.testing.assert_allclose(result[1], expected[1])

    def testDistributedContext(self):
        self.start_processes(etcd=False)
        sess = new_session(self.session_manager_ref.address)
        rs = np.random.RandomState(0)
        context = DistributedContext(scheduler_address=self.session_manager_ref.address,
                                     session_id=sess.session_id)

        raw1 = rs.rand(10, 10)
        a = mt.tensor(raw1, chunk_size=4)
        a.execute(session=sess, timeout=self.timeout, name='test')

        tileable_infos = context.get_named_tileable_infos('test')
        self.assertEqual(a.key, tileable_infos.tileable_key)
        self.assertEqual(a.shape, tileable_infos.tileable_shape)

        nsplits = context.get_tileable_metas([a.key], filter_fields=['nsplits'])[0][0]
        self.assertEqual(((4, 4, 2), (4, 4, 2)), nsplits)

        r = context.get_tileable_data(a.key)
        np.testing.assert_array_equal(raw1, r)

        indexes = [slice(3, 9), slice(0, 7)]
        r = context.get_tileable_data(a.key, indexes)
        np.testing.assert_array_equal(raw1[tuple(indexes)], r)

        indexes = [[1, 4, 2, 4, 5], slice(None, None, None)]
        r = context.get_tileable_data(a.key, indexes)
        np.testing.assert_array_equal(raw1[tuple(indexes)], r)

        indexes = ([9, 1, 2, 0], [0, 0, 4, 4])
        r = context.get_tileable_data(a.key, indexes)
        np.testing.assert_array_equal(raw1[[9, 1, 2, 0], [0, 0, 4, 4]], r)

    def testOperandsWithoutPrepareInputs(self):
        self.start_processes(etcd=False, modules=['mars.scheduler.tests.integrated.no_prepare_op'])
        sess = new_session(self.session_manager_ref.address)

        actor_address = self.cluster_info.get_scheduler(ResourceActor.default_uid())
        resource_ref = sess._api.actor_client.actor_ref(ResourceActor.default_uid(), address=actor_address)
        worker_endpoints = resource_ref.get_worker_endpoints()

        t1 = mt.random.rand(10)
        t1.op._expect_worker = worker_endpoints[0]
        t2 = mt.random.rand(10)
        t2.op._expect_worker = worker_endpoints[1]

        t = NoPrepareOperand().new_tileable([t1, t2])
        t.op._prepare_inputs = [False, False]
        t.execute(session=sess, timeout=self.timeout)

    def testRemoteWithoutEtcd(self):
        from mars.scheduler.resource import ResourceActor
        from mars.worker.dispatcher import DispatchActor

        self.start_processes(etcd=False, modules=['mars.scheduler.tests.integrated.no_prepare_op'])
        sess = new_session(self.session_manager_ref.address)
        resource_ref = sess._api.actor_client.actor_ref(
            ResourceActor.default_uid(),
            address=self.cluster_info.get_scheduler(ResourceActor.default_uid())
        )
        worker_ips = resource_ref.get_worker_endpoints()

        rs = np.random.RandomState(0)
        raw1 = rs.rand(10, 10)
        raw2 = rs.rand(10, 10)

        def f_none(_x):
            return None

        r_none = spawn(f_none, raw1)
        result = r_none.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        self.assertIsNone(result)

        def f1(x):
            return x + 1

        def f2(x, y, z=None):
            return x * y * (z[0] + z[1])

        r1 = spawn(f1, raw1)
        r2 = spawn(f1, raw2)
        r3 = spawn(f2, (r1, r2), {'z': [r1, r2]})
        result = r3.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        expected = (raw1 + 1) * (raw2 + 1) * (raw1 + 1 + raw2 + 1)
        np.testing.assert_allclose(result, expected)

        def f(t, x):
            mul = (t * x).execute()
            return mul.sum().to_numpy()

        rs = np.random.RandomState(0)
        raw = rs.rand(5, 4)

        t1 = mt.tensor(raw, chunk_size=3)
        t2 = t1.sum(axis=0)
        s = spawn(f, args=(t2, 3))

        result = s.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        expected = (raw.sum(axis=0) * 3).sum()
        self.assertAlmostEqual(result, expected)

        time.sleep(1)
        for worker_ip in worker_ips:
            ref = sess._api.actor_client.actor_ref(DispatchActor.default_uid(), address=worker_ip)
            self.assertEqual(len(ref.get_slots('cpu')), 1)

    def testNoWorkerException(self):
        self.start_processes(etcd=False, n_workers=0)

        a = mt.ones((10, 10))
        b = mt.ones((10, 10))
        c = (a + b)

        endpoint = self.scheduler_endpoints[0]
        sess = new_session(endpoint)

        try:
            c.execute(session=sess, timeout=self.timeout)
        except ExecutionFailed as e:
            self.assertIsInstance(e.__cause__, RuntimeError)
