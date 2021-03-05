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

import itertools
import logging
import operator
import os
import sys
import tempfile
import time
import unittest
from functools import reduce

import numpy as np
import pandas as pd

import mars.dataframe as md
import mars.tensor as mt
from mars.errors import ExecutionFailed
from mars.scheduler.custom_log import CustomLogMetaActor
from mars.scheduler.resource import ResourceActor
from mars.scheduler.tests.integrated.base import SchedulerIntegratedTest
from mars.scheduler.tests.integrated.no_prepare_op import PureDependsOperand
from mars.session import new_session
from mars.remote import spawn, ExecutableTuple
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

        raw = np.random.RandomState(0).rand(1000)
        a = mt.tensor(raw, chunk_size=100)
        r = mt.median(a)
        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        np.testing.assert_array_equal(result, np.median(raw))

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

        # test binary arithmetics with different indices
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

        # test sort_values
        raw1 = pd.DataFrame(np.random.rand(10, 10))
        raw1[0] = raw1[0].apply(str)
        raw1.columns = pd.MultiIndex.from_product([list('AB'), list('CDEFG')])
        df1 = md.DataFrame(raw1, chunk_size=5)
        r = df1.sort_values([('A', 'C')])
        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        pd.testing.assert_frame_equal(result, raw1.sort_values([('A', 'C')]))

        rs = np.random.RandomState(0)
        raw2 = pd.DataFrame({'a': rs.rand(10),
                            'b': [f's{rs.randint(1000)}' for _ in range(10)]
                            })
        raw2['b'] = raw2['b'].astype(md.ArrowStringDtype())
        mdf = md.DataFrame(raw2, chunk_size=4)
        filtered = mdf[mdf['a'] > 0.5]
        df2 = filtered.sort_values(by='b')
        result = df2.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        expected = raw2[raw2['a'] > 0.5].sort_values(by='b')
        pd.testing.assert_frame_equal(result, expected)

        s1 = pd.Series(np.random.rand(10), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3])
        series1 = md.Series(s1, chunk_size=6)
        result = series1.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        pd.testing.assert_series_equal(result, s1)

        # test reindex
        data = pd.DataFrame(np.random.rand(10, 5), columns=['c1', 'c2', 'c3', 'c4', 'c5'])
        df3 = md.DataFrame(data, chunk_size=4)
        r = df3.reindex(index=mt.arange(10, 1, -1, chunk_size=3))

        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        expected = data.reindex(index=np.arange(10, 1, -1))
        pd.testing.assert_frame_equal(result, expected)

        # test rebalance
        df4 = md.DataFrame(data)
        r = df4.rebalance()

        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        pd.testing.assert_frame_equal(result, data)
        chunk_metas = sess.get_tileable_chunk_metas(r.key)
        workers = list(set(itertools.chain(*(m.workers for m in chunk_metas.values()))))
        self.assertEqual(len(workers), 2)

        # test nunique
        data = pd.DataFrame(np.random.randint(0, 10, (100, 5)),
                            columns=['c1', 'c2', 'c3', 'c4', 'c5'])
        df5 = md.DataFrame(data, chunk_size=4)
        r = df5.nunique()

        result = r.execute(session=sess, timeout=self.timeout).fetch(session=sess)
        expected = data.nunique()
        pd.testing.assert_series_equal(result, expected)

        # test re-execute df.groupby().agg().sort_values()
        rs = np.random.RandomState(0)
        data = pd.DataFrame({'col1': rs.rand(100), 'col2': rs.randint(10, size=100)})
        df6 = md.DataFrame(data, chunk_size=40)
        grouped = df6.groupby('col2', as_index=False)['col2'].agg({"cnt": "count"}) \
            .execute(session=sess, timeout=self.timeout)
        r = grouped.sort_values(by='cnt').head().execute(session=sess, timeout=self.timeout)
        result = r.fetch(session=sess)
        expected = data.groupby('col2', as_index=False)['col2'].agg({"cnt": "count"}) \
            .sort_values(by='cnt').head()
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
        r2 = df6.groupby('col2', as_index=False)['col2'].agg({"cnt": "count"}).sort_values(by='cnt').head() \
            .execute(session=sess, timeout=self.timeout)
        result = r2.fetch(session=sess)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

        # test groupby with sample
        src_data_list = []
        sample_count = 10
        for b in range(5):
            data_count = int(np.random.randint(40, 100))
            src_data_list.append(pd.DataFrame({
                'a': np.random.randint(0, 100, size=data_count),
                'b': np.array([b] * data_count),
                'c': np.random.randint(0, 100, size=data_count),
                'd': np.random.randint(0, 100, size=data_count),
            }))
        data = pd.concat(src_data_list)
        shuffle_idx = np.arange(len(data))
        np.random.shuffle(shuffle_idx)
        data = data.iloc[shuffle_idx].reset_index(drop=True)

        df7 = md.DataFrame(data, chunk_size=40)
        sampled = df7.groupby('b').sample(10)
        r = sampled.execute(session=sess, timeout=self.timeout)
        result = r.fetch(session=sess)
        self.assertFalse((result.groupby('b').count() - sample_count).any()[0])

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
            _, keys, _ = graph_ref.get_tileable_metas(
                [a.key], filter_fields=['nsplits', 'chunk_keys', 'chunk_indexes'])[0]
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

    def testOperandsWithPureDepends(self):
        self.start_processes(etcd=False, modules=['mars.scheduler.tests.integrated.no_prepare_op'])
        sess = new_session(self.session_manager_ref.address)

        actor_address = self.cluster_info.get_scheduler(ResourceActor.default_uid())
        resource_ref = sess._api.actor_client.actor_ref(ResourceActor.default_uid(), address=actor_address)
        worker_endpoints = resource_ref.get_worker_endpoints()

        t1 = mt.random.rand(10)
        t1.op._expect_worker = worker_endpoints[0]
        t2 = mt.random.rand(10)
        t2.op._expect_worker = worker_endpoints[1]

        t = PureDependsOperand().new_tileable([t1, t2])
        t.op._pure_depends = [True, True]
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

    def testFetchLogWithoutEtcd(self):
        # test fetch log
        with tempfile.TemporaryDirectory() as temp_dir:
            self.start_processes(etcd=False, modules=['mars.scheduler.tests.integrated.no_prepare_op'],
                                 scheduler_args=[f'-Dcustom_log_dir={temp_dir}'])
            sess = new_session(self.session_manager_ref.address)

            def f():
                print('test')

            r = spawn(f)
            r.execute(session=sess)

            custom_log_actor = sess._api.actor_client.actor_ref(
                CustomLogMetaActor.default_uid(),
                address=self.cluster_info.get_scheduler(CustomLogMetaActor.default_uid())
            )

            chunk_key_to_log_path = custom_log_actor.get_tileable_op_log_paths(
                sess.session_id, r.op.key)
            paths = list(chunk_key_to_log_path.values())
            self.assertEqual(len(paths), 1)
            log_path = paths[0][1]
            with open(log_path) as f:
                self.assertEqual(f.read().strip(), 'test')

            context = DistributedContext(scheduler_address=self.session_manager_ref.address,
                                         session_id=sess.session_id)
            log_result = context.fetch_tileable_op_logs(r.op.key)
            log = next(iter(log_result.values()))['log']
            self.assertEqual(log.strip(), 'test')

            log = r.fetch_log()
            self.assertEqual(str(log).strip(), 'test')

            # test multiple functions
            def f1(size):
                print('f1' * size)
                sys.stdout.flush()

            fs = ExecutableTuple([spawn(f1, 30), spawn(f1, 40)])
            fs.execute(session=sess)
            log = fs.fetch_log(offsets=20, sizes=10)
            self.assertEqual(str(log[0]).strip(), ('f1' * 30)[20:30])
            self.assertEqual(str(log[1]).strip(), ('f1' * 40)[20:30])
            self.assertGreater(len(log[0].offsets), 0)
            self.assertTrue(all(s > 0 for s in log[0].offsets))
            self.assertGreater(len(log[1].offsets), 0)
            self.assertTrue(all(s > 0 for s in log[1].offsets))
            self.assertGreater(len(log[0].chunk_op_keys), 0)

            # test negative offsets
            log = fs.fetch_log(offsets=-20, sizes=10)
            self.assertEqual(str(log[0]).strip(), ('f1' * 30 + '\n')[-20:-10])
            self.assertEqual(str(log[1]).strip(), ('f1' * 40 + '\n')[-20:-10])
            self.assertTrue(all(s > 0 for s in log[0].offsets))
            self.assertGreater(len(log[1].offsets), 0)
            self.assertTrue(all(s > 0 for s in log[1].offsets))
            self.assertGreater(len(log[0].chunk_op_keys), 0)

            # test negative offsets which represented in string
            log = fs.fetch_log(offsets='-0.02K', sizes='0.01K')
            self.assertEqual(str(log[0]).strip(), ('f1' * 30 + '\n')[-20:-10])
            self.assertEqual(str(log[1]).strip(), ('f1' * 40 + '\n')[-20:-10])
            self.assertTrue(all(s > 0 for s in log[0].offsets))
            self.assertGreater(len(log[1].offsets), 0)
            self.assertTrue(all(s > 0 for s in log[1].offsets))
            self.assertGreater(len(log[0].chunk_op_keys), 0)

            def test_nested():
                print('level0')
                fr = spawn(f1, 1)
                fr.execute()
                print(fr.fetch_log())

            r = spawn(test_nested)
            with self.assertRaises(ValueError):
                r.fetch_log()
            r.execute(session=sess)
            log = str(r.fetch_log())
            self.assertIn('level0', log)
            self.assertIn('f1', log)

            df = md.DataFrame(mt.random.rand(10, 3), chunk_size=5)

            def df_func(c):
                print('df func')
                return c

            df2 = df.map_chunk(df_func)
            df2.execute(session=sess)
            log = df2.fetch_log()
            self.assertIn('Chunk op key:', str(log))
            self.assertIn('df func', repr(log))
            self.assertEqual(len(str(df.fetch_log(session=sess))), 0)

            def f7(rndf):
                rm = spawn(f8, rndf)
                rm.execute()
                print(rm.fetch_log())

            def f8(_rndf):
                print('log_content')

            ds = [spawn(f7, n, retry_when_fail=False)
                  for n in np.random.rand(4)]
            xtp = ExecutableTuple(ds)
            xtp.execute(session=sess)
            for log in xtp.fetch_log(session=sess):
                self.assertEqual(str(log).strip(), 'log_content')

    def testNoWorkerException(self):
        self.start_processes(etcd=False, n_workers=0)

        a = mt.ones((10, 10))
        b = mt.ones((10, 10))
        c = (a + b)

        sess = new_session(self.session_manager_ref.address)

        try:
            c.execute(session=sess, timeout=self.timeout)
        except ExecutionFailed as e:
            self.assertIsInstance(e.__cause__, RuntimeError)
