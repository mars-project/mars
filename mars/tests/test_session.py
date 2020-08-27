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

import unittest

import numpy as np
import pandas as pd
try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

import mars.tensor as mt
import mars.dataframe as md
from mars.executor import register, Executor
from mars.tensor.datasource import ArrayDataSource
from mars.tiles import get_tiled
from mars.session import new_session, Session


class Test(unittest.TestCase):
    def testSessionExecute(self):
        a = mt.random.rand(10, 20)
        res = a.sum().to_numpy()
        self.assertTrue(np.isscalar(res))
        self.assertLess(res, 200)

    def testMultipleOutputExecute(self):
        data = np.random.random((5, 9))

        # test multiple outputs
        arr1 = mt.tensor(data.copy(), chunk_size=3)
        result = mt.modf(arr1).execute().fetch()
        expected = np.modf(data)

        np.testing.assert_array_equal(result[0], expected[0])
        np.testing.assert_array_equal(result[1], expected[1])

        # test 1 output
        arr2 = mt.tensor(data.copy(), chunk_size=3)
        result = ((arr2 + 1) * 2).to_numpy()
        expected = (data + 1) * 2

        np.testing.assert_array_equal(result, expected)

        # test multiple outputs, but only execute 1
        arr3 = mt.tensor(data.copy(), chunk_size=3)
        arrs = mt.split(arr3, 3, axis=1)
        result = arrs[0].to_numpy()
        expected = np.split(data, 3, axis=1)[0]

        np.testing.assert_array_equal(result, expected)

        # test multiple outputs, but only execute 1
        data = np.random.randint(0, 10, (5, 5))
        arr3 = (mt.tensor(data) + 1) * 2
        arrs = mt.linalg.qr(arr3)
        result = (arrs[0] + 1).to_numpy()
        expected = np.linalg.qr((data + 1) * 2)[0] + 1

        np.testing.assert_array_almost_equal(result, expected)

        result = (arrs[0] + 2).to_numpy()
        expected = np.linalg.qr((data + 1) * 2)[0] + 2

        np.testing.assert_array_almost_equal(result, expected)

        s = mt.shape(0)

        result = s.execute().fetch()
        expected = np.shape(0)
        self.assertEqual(result, expected)

    def testReExecuteSame(self):
        data = np.random.random((5, 9))

        # test run the same tensor
        arr4 = mt.tensor(data.copy(), chunk_size=3) + 1
        result1 = arr4.to_numpy()
        expected = data + 1

        np.testing.assert_array_equal(result1, expected)

        result2 = arr4.to_numpy()

        np.testing.assert_array_equal(result1, result2)

        # test run the same tensor with single chunk
        arr4 = mt.tensor(data.copy())
        result1 = arr4.to_numpy()
        expected = data

        np.testing.assert_array_equal(result1, expected)

        result2 = arr4.to_numpy()
        np.testing.assert_array_equal(result1, result2)

        # modify result
        sess = Session.default_or_local()
        executor = sess._sess._executor
        executor.chunk_result[get_tiled(arr4).chunks[0].key] = data + 2

        result3 = arr4.to_numpy()
        np.testing.assert_array_equal(result3, data + 2)

        # test run same key tensor
        arr5 = mt.ones((10, 10), chunk_size=3)
        result1 = arr5.to_numpy()

        del arr5
        arr6 = mt.ones((10, 10), chunk_size=3)
        result2 = arr6.to_numpy()

        np.testing.assert_array_equal(result1, result2)

    def testExecuteBothExecutedAndNot(self):
        data = np.random.random((5, 9))

        arr1 = mt.tensor(data, chunk_size=4) * 2
        arr2 = mt.tensor(data) + 1

        np.testing.assert_array_equal(arr2.to_numpy(), data + 1)

        # modify result
        sess = Session.default_or_local()
        executor = sess._sess._executor
        executor.chunk_result[get_tiled(arr2).chunks[0].key] = data + 2

        results = sess.run(arr1, arr2)
        np.testing.assert_array_equal(results[0], data * 2)
        np.testing.assert_array_equal(results[1], data + 2)

    def testTensorExecuteNotFetch(self):
        data = np.random.random((5, 9))
        sess = Session.default_or_local()

        arr1 = mt.tensor(data, chunk_size=2) * 2

        with self.assertRaises(ValueError):
            sess.fetch(arr1)

        self.assertIs(arr1.execute(), arr1)

        # modify result
        executor = sess._sess._executor
        executor.chunk_result[get_tiled(arr1).chunks[0].key] = data[:2, :2] * 3

        expected = data * 2
        expected[:2, :2] = data[:2, :2] * 3

        np.testing.assert_array_equal(arr1.to_numpy(), expected)

    def testDataFrameExecuteNotFetch(self):
        data1 = pd.DataFrame(np.random.random((5, 4)), columns=list('abcd'))
        sess = Session.default_or_local()

        df1 = md.DataFrame(data1, chunk_size=2)

        with self.assertRaises(ValueError):
            sess.fetch(df1)

        self.assertIs(df1.execute(), df1)

        # modify result
        executor = sess._sess._executor
        executor.chunk_result[get_tiled(df1).chunks[0].key] = data1.iloc[:2, :2] * 3

        expected = data1
        expected.iloc[:2, :2] = data1.iloc[:2, :2] * 3

        pd.testing.assert_frame_equal(df1.to_pandas(), expected)

    def testClosedSession(self):
        session = new_session()
        arr = mt.ones((10, 10))

        result = session.run(arr)

        np.testing.assert_array_equal(result, np.ones((10, 10)))

        session.close()
        with self.assertRaises(RuntimeError):
            session.run(arr)

        with self.assertRaises(RuntimeError):
            session.run(arr + 1)

    def testBoolIndexing(self):
        arr = mt.random.rand(10, 10, chunk_size=5)
        arr[3:8, 3:8] = mt.ones((5, 5))

        arr2 = arr[arr == 1]
        self.assertEqual(arr2.shape, (np.nan,))

        arr2.execute()
        self.assertEqual(arr2.shape, (25,))

        arr3 = arr2.reshape((5, 5))
        expected = np.ones((5, 5))
        np.testing.assert_array_equal(arr3.to_numpy(), expected)

    def testArrayProtocol(self):
        arr = mt.ones((10, 20))

        result = np.asarray(arr)
        np.testing.assert_array_equal(result, np.ones((10, 20)))

        arr2 = mt.ones((10, 20))

        result = np.asarray(arr2, mt.bool_)
        np.testing.assert_array_equal(result, np.ones((10, 20), dtype=np.bool_))

        arr3 = mt.ones((10, 20)).sum()

        result = np.asarray(arr3)
        np.testing.assert_array_equal(result, np.asarray(200))

        arr4 = mt.ones((10, 20)).sum()

        result = np.asarray(arr4, dtype=np.float_)
        np.testing.assert_array_equal(result, np.asarray(200, dtype=np.float_))

    def testRandomExecuteInSessions(self):
        arr = mt.random.rand(20, 20)

        sess1 = new_session()
        res1 = sess1.run(arr)

        sess2 = new_session()
        res2 = sess2.run(arr)

        np.testing.assert_array_equal(res1, res2)

    def testFetch(self):
        sess = new_session()

        arr1 = mt.ones((10, 5), chunk_size=3)

        r1 = sess.run(arr1)
        r2 = sess.run(arr1)
        np.testing.assert_array_equal(r1, r2)

        executor = sess._sess._executor
        executor.chunk_result[get_tiled(arr1).chunks[0].key] = np.ones((3, 3)) * 2
        r3 = sess.run(arr1 + 1)
        np.testing.assert_array_equal(r3[:3, :3], np.ones((3, 3)) * 3)

        # rerun to ensure arr1's chunk results still exist
        r4 = sess.run(arr1 + 1)
        np.testing.assert_array_equal(r4[:3, :3], np.ones((3, 3)) * 3)

        arr2 = mt.ones((10, 5), chunk_size=3)
        r5 = sess.run(arr2)
        np.testing.assert_array_equal(r5[:3, :3], np.ones((3, 3)) * 2)

        r6 = sess.run(arr2 + 1)
        np.testing.assert_array_equal(r6[:3, :3], np.ones((3, 3)) * 3)

        # test fetch multiple tensors
        raw = np.random.rand(5, 10)
        arr1 = mt.ones((5, 10), chunk_size=5)
        arr2 = mt.tensor(raw, chunk_size=3)
        arr3 = mt.sum(arr2)

        sess.run(arr1, arr2, arr3)

        fetch1, fetch2, fetch3 = sess.fetch(arr1, arr2, arr3)
        np.testing.assert_array_equal(fetch1, np.ones((5, 10)))
        np.testing.assert_array_equal(fetch2, raw)
        np.testing.assert_almost_equal(fetch3, raw.sum())

        fetch1, fetch2, fetch3 = sess.fetch([arr1, arr2, arr3])
        np.testing.assert_array_equal(fetch1, np.ones((5, 10)))
        np.testing.assert_array_equal(fetch2, raw)
        np.testing.assert_almost_equal(fetch3, raw.sum())

        raw = np.random.rand(5, 10)
        arr = mt.tensor(raw, chunk_size=5)
        s = arr.sum()

        self.assertAlmostEqual(s.execute().fetch(), raw.sum())

        def _execute_ds(*_):  # pragma: no cover
            raise ValueError('cannot run random again')

        try:
            register(ArrayDataSource, _execute_ds)

            self.assertAlmostEqual(s.fetch(), raw.sum())
        finally:
            del Executor._op_runners[ArrayDataSource]

    def testDecref(self):
        sess = new_session()

        arr1 = mt.ones((10, 5), chunk_size=3)
        arr2 = mt.ones((10, 5), chunk_size=3)
        sess.run(arr1)
        sess.run(arr2)
        sess.fetch(arr1)

        executor = sess._sess._executor

        self.assertEqual(len(executor.chunk_result), 4)  # 4 kinds of shapes
        del arr1
        self.assertEqual(len(executor.chunk_result), 4)
        del arr2
        self.assertEqual(len(executor.chunk_result), 0)

    def testWithoutCompose(self):
        sess = new_session()

        arr1 = (mt.ones((10, 10), chunk_size=3) + 1) * 2
        r1 = sess.run(arr1)
        arr2 = (mt.ones((10, 10), chunk_size=4) + 1) * 2
        r2 = sess.run(arr2, compose=False)
        np.testing.assert_array_equal(r1, r2)

    def testDataFrameCreate(self):
        sess = new_session()
        tensor = mt.ones((2, 2))
        df = md.DataFrame(tensor)
        df_result = sess.run(df)
        df2 = md.DataFrame(df)
        df2 = sess.run(df2)
        np.testing.assert_equal(df_result.values, np.ones((2, 2)))
        pd.testing.assert_frame_equal(df_result, df2)

        raw_a = np.random.rand(10)
        raw_b = np.random.randint(1000, size=10)
        df = md.DataFrame({'a': mt.tensor(raw_a), 'b': mt.tensor(raw_b)}, columns=['b', 'a'])
        df_result = sess.run(df)
        pd.testing.assert_frame_equal(
            df_result, pd.DataFrame({'a': raw_a, 'b': raw_b}, columns=['b', 'a']))

    def testDataFrameTensorConvert(self):
        # test from_tensor(), from_dataframe(), to_tensor(), to_dataframe()
        sess = new_session()
        tensor = mt.ones((2, 2))
        df = tensor.to_dataframe()
        np.testing.assert_equal(sess.run(df), np.ones((2, 2)))
        tensor2 = mt.from_dataframe(df)
        np.testing.assert_equal(sess.run(tensor2), np.ones((2, 2)))

        tensor3 = tensor2.from_dataframe(df)
        np.testing.assert_equal(sess.run(tensor3), np.ones((2, 2)))

        tensor4 = df.to_tensor()
        np.testing.assert_equal(sess.run(tensor4), np.ones((2, 2)))

        df = md.dataframe_from_tensor(tensor3)
        np.testing.assert_equal(sess.run(df).values, np.ones((2, 2)))

        df = df.from_tensor(tensor3)
        np.testing.assert_equal(sess.run(df).values, np.ones((2, 2)))

        # test raise error exception
        with self.assertRaises(TypeError):
            md.dataframe_from_tensor(mt.ones((1, 2, 3)))

        # test exception
        tensor = md.dataframe_from_tensor(mt.array([1, 2, 3]))
        np.testing.assert_equal(sess.run(tensor), np.array([1, 2, 3]).reshape(3, 1))

    def testDataFrameExecution(self):
        sess = new_session()

        raw = pd.DataFrame(np.random.rand(5, 3),
                           index=pd.date_range('2020-1-1', periods=5))

        for chunk_size in (3, 5):
            df = md.DataFrame(raw, chunk_size=chunk_size)

            r = df.loc['2020-1-2']
            result = sess.run(r)
            pd.testing.assert_series_equal(result, raw.loc['2020-1-2'])

    @unittest.skipIf(pa is not None, 'this test aims to test usage of ArrowDtype '
                                     'when pyarrow not installed')
    def testDataFrameWithArrowDtypeExecution(self):
        sess = new_session()

        # test ArrowDtype when pyarrow not installed
        raw = pd.DataFrame({'a': [f's{i}' for i in range(10)],
                            'b': np.random.rand(10)})
        df = md.DataFrame(raw, chunk_size=5)
        df['a'] = df['a'].astype('Arrow[string]')

        r = df.groupby('a').sum()  # can work for expression
        with self.assertRaises(ImportError):
            # cannot perform execution
            # due to the reason that pyarrow not installed
            _ = sess.run(r)

    def testFetchSlices(self):
        sess = new_session()

        arr1 = mt.random.rand(10, 8, chunk_size=3)
        r1 = sess.run(arr1)

        r2 = sess.fetch(arr1[:2, 3:9])
        np.testing.assert_array_equal(r2, r1[:2, 3:9])

        r3 = sess.fetch(arr1[0])
        np.testing.assert_array_equal(r3, r1[0])

    def testFetchDataFrameSlices(self):
        sess = new_session()

        arr1 = mt.random.rand(10, 8, chunk_size=3)
        df1 = md.DataFrame(arr1)
        r1 = sess.run(df1)

        r2 = sess.fetch(df1.iloc[:, :])
        pd.testing.assert_frame_equal(r2, r1.iloc[:, :])

        r3 = sess.fetch(df1.iloc[1])
        pd.testing.assert_series_equal(r3, r1.iloc[1])

        r4 = sess.fetch(df1.iloc[0, 2])
        self.assertEqual(r4, r1.iloc[0, 2])

    def testMultiOutputsOp(self):
        sess = new_session()

        rs = np.random.RandomState(0)
        raw = rs.rand(20, 5)
        a = mt.tensor(raw, chunk_size=5)
        q = mt.abs(mt.linalg.qr(a)[0])

        ret = sess.run(q)
        np.testing.assert_almost_equal(ret, np.abs(np.linalg.qr(raw)[0]))
        self.assertEqual(len(sess._sess.executor.chunk_result),
                         len(get_tiled(q).chunks))

    def testIterativeTiling(self):
        sess = new_session()

        rs = np.random.RandomState(0)
        raw = rs.rand(100)
        a = mt.tensor(raw, chunk_size=10)
        a.sort()
        c = a[:5]

        ret = sess.run(c)
        np.testing.assert_array_equal(ret, np.sort(raw)[:5])

        executor = sess._sess.executor
        self.assertEqual(len(executor.chunk_result), 1)
        executor.chunk_result.clear()

        raw1 = rs.rand(20)
        raw2 = rs.rand(20)
        a = mt.tensor(raw1, chunk_size=10)
        a.sort()
        b = mt.tensor(raw2, chunk_size=15) + 1
        c = mt.concatenate([a[:10], b])
        c.sort()
        d = c[:5]

        ret = sess.run(d)
        expected = np.sort(np.concatenate([np.sort(raw1)[:10], raw2 + 1]))[:5]
        np.testing.assert_array_equal(ret, expected)
        self.assertEqual(len(executor.chunk_result), len(get_tiled(d).chunks))

        raw = rs.rand(100)
        a = mt.tensor(raw, chunk_size=10)
        a.sort()
        b = a + 1
        c = b[:5]

        ret = sess.run([b, c])
        expected = np.sort(raw + 1)[:5]
        np.testing.assert_array_equal(ret[1], expected)

        raw = rs.randint(100, size=(100,))
        a = mt.tensor(raw, chunk_size=23)
        a.sort()
        b = mt.histogram(a, bins='stone')

        res = sess.run(b)
        expected = np.histogram(np.sort(raw), bins='stone')
        np.testing.assert_almost_equal(res[0], expected[0])
        np.testing.assert_almost_equal(res[1], expected[1])

    def testRepr(self):
        # test tensor repr
        with np.printoptions(threshold=100):
            arr = np.random.randint(1000, size=(11, 4, 13))

            t = mt.tensor(arr, chunk_size=3)

            result = repr(t.execute())
            expected = repr(arr)
            self.assertEqual(result, expected)

        for size in (5, 58, 60, 62, 64):
            pdf = pd.DataFrame(np.random.randint(1000, size=(size, 10)))

            # test DataFrame repr
            df = md.DataFrame(pdf, chunk_size=size//2)

            result = repr(df.execute())
            expected = repr(pdf)
            self.assertEqual(result, expected,
                             f'failed repr for DataFrame when size = {size}')

            # test DataFrame _repr_html_
            result = df.execute()._repr_html_()
            expected = pdf._repr_html_()
            self.assertEqual(result, expected,
                             f'failed repr html for DataFrame when size = {size}')

            # test Series repr
            ps = pdf[0]
            s = md.Series(ps, chunk_size=size//2)

            result = repr(s.execute())
            expected = repr(ps)
            self.assertEqual(result, expected,
                             f'failed repr for Series when size = {size}')

        # test Index repr
        pind = pd.date_range('2020-1-1', periods=10)
        ind = md.Index(pind, chunk_size=5)

        self.assertIn('DatetimeIndex', repr(ind.execute()))

        # test groupby repr
        df = md.DataFrame(pd.DataFrame(np.random.rand(100, 3), columns=list('abc')))
        grouped = df.groupby(['a', 'b']).execute()

        self.assertIn('DataFrameGroupBy', repr(grouped))

        # test Categorical repr
        c = md.qcut(range(5), 3)
        self.assertIn('Categorical', repr(c))
        self.assertIn('Categorical', str(c))
        self.assertEqual(repr(c.execute()), repr(pd.qcut(range(5), 3)))

    def testDataFrameIter(self):
        raw_data = pd.DataFrame(np.random.randint(1000, size=(20, 10)))
        df = md.DataFrame(raw_data, chunk_size=5)

        i = 0
        for result_row, expect_row in zip(df.iterrows(batch_size=15),
                                          raw_data.iterrows()):
            self.assertEqual(result_row[0], expect_row[0])
            pd.testing.assert_series_equal(result_row[1], expect_row[1])
            i += 1

        self.assertEqual(i, len(raw_data))

        i = 0
        for result_tup, expect_tup in zip(df.itertuples(batch_size=10),
                                          raw_data.itertuples()):
            self.assertEqual(result_tup, expect_tup)
            i += 1

        self.assertEqual(i, len(raw_data))

    def testNamed(self):
        rs = np.random.RandomState(0)
        raw = rs.rand(10, 10)

        sess = Session.default_or_local()

        # test named tensor
        t = mt.tensor(raw, chunk_size=3)
        name = 't_name'
        r1 = t.execute(name=name, session=sess)
        np.testing.assert_array_equal(r1, raw)

        t2 = mt.named_tensor(name=name, session=sess)
        r2 = (t2 + 1).execute(session=sess).fetch()
        np.testing.assert_array_equal(r2, raw + 1)

        # test named series
        name = 's_name'
        raw = pd.Series([1, 2, 3])
        s = md.Series(raw)
        r1 = s.execute(name=name, session=sess).fetch()
        pd.testing.assert_series_equal(r1, raw)

        s2 = md.named_series(name=name, session=sess)
        r2 = s2.execute(session=sess).fetch()
        pd.testing.assert_series_equal(r2, raw)

        # test dataframe
        name = 'd_name'
        raw = pd.DataFrame(np.random.rand(10, 3))
        d = md.DataFrame(raw, chunk_size=4)
        r1 = d.execute(name=name, session=sess).fetch()
        pd.testing.assert_frame_equal(r1, raw)

        d2 = md.named_dataframe(name=name, session=sess)
        r2 = d2.execute(session=sess).fetch()
        pd.testing.assert_frame_equal(r2, raw)
