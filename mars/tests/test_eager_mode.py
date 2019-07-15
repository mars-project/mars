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

import unittest

import numpy as np
import pandas as pd

import mars.tensor as mt
import mars.dataframe as md
from mars.config import option_context
from mars.dataframe.expressions.datasource.dataframe import from_pandas


class Test(unittest.TestCase):
    def testBaseExecute(self):
        with option_context({'eager_mode': True}):
            a_data = np.random.rand(10, 10)
            a = mt.tensor(a_data, chunk_size=3)
            np.testing.assert_array_equal(a.fetch(), a_data)

            r1 = a + 1
            np.testing.assert_array_equal(r1.fetch(), a_data + 1)

            r2 = 2 * r1
            np.testing.assert_array_equal(r2.fetch(), (a_data + 1) * 2)

            # test add with out
            b = mt.ones((10, 10), chunk_size=3)
            np.testing.assert_array_equal(b.fetch(), np.ones((10, 10)))

            mt.add(a, b, out=b)
            np.testing.assert_array_equal(b.fetch(), a_data + 1)

            # test tensor dot
            c_data1 = np.random.rand(10, 10)
            c_data2 = np.random.rand(10, 10)
            c1 = mt.tensor(c_data1, chunk_size=4)
            c2 = mt.tensor(c_data2, chunk_size=4)
            r3 = c1.dot(c2)
            np.testing.assert_array_almost_equal(r3.fetch(), c_data1.dot(c_data2))

    def testMultipleOutputExecute(self):
        with option_context({'eager_mode': True}):
            data = np.random.random((5, 9))

            arr1 = mt.tensor(data.copy(), chunk_size=3)
            result = mt.modf(arr1)
            expected = np.modf(data)

            np.testing.assert_array_equal(result[0].fetch(), expected[0])
            np.testing.assert_array_equal(result[1].fetch(), expected[1])

            arr3 = mt.tensor(data.copy(), chunk_size=3)
            result1, result2, result3 = mt.split(arr3, 3, axis=1)
            expected = np.split(data, 3, axis=1)

            np.testing.assert_array_equal(result1.fetch(), expected[0])
            np.testing.assert_array_equal(result2.fetch(), expected[1])
            np.testing.assert_array_equal(result3.fetch(), expected[2])

    def testMixedConfig(self):
        a = mt.ones((10, 10), chunk_size=3)
        with self.assertRaises(ValueError):
            a.fetch()

        with option_context({'eager_mode': True}):
            b = mt.ones((10, 10), chunk_size=(3, 4))
            np.testing.assert_array_equal(b.fetch(), np.ones((10, 10)))

            r = b + 1
            np.testing.assert_array_equal(r.fetch(), np.ones((10, 10)) * 2)

            r2 = b.dot(b)
            np.testing.assert_array_equal(r2.fetch(), np.ones((10, 10)) * 10)

        c = mt.ones((10, 10), chunk_size=3)
        with self.assertRaises(ValueError):
            c.fetch()
        np.testing.assert_array_equal(c.execute(), np.ones((10, 10)))

        r = c.dot(c)
        with self.assertRaises(ValueError):
            r.fetch()
        np.testing.assert_array_equal(r.execute(), np.ones((10, 10)) * 10)

    def testIndex(self):
        with option_context({'eager_mode': True}):
            a = mt.random.rand(10, 5, chunk_size=5)
            idx = slice(0, 5), slice(0, 5)
            a[idx] = 1
            np.testing.assert_array_equal(a.fetch()[idx], np.ones((5, 5)))

            split1, split2 = mt.split(a, 2)
            np.testing.assert_array_equal(split1.fetch(), np.ones((5, 5)))

            # test bool indexing
            a = mt.random.rand(8, 8, chunk_size=4)
            set_value = mt.ones((2, 2)) * 2
            a[4:6, 4:6] = set_value
            b = a[a > 1]
            self.assertEqual(b.shape, (4,))
            np.testing.assert_array_equal(b.fetch(), np.ones((4,)) * 2)

            c = b.reshape((2, 2))
            self.assertEqual(c.shape, (2, 2))
            np.testing.assert_array_equal(c.fetch(), np.ones((2, 2)) * 2)

    def testFetch(self):
        from mars.session import Session

        with option_context({'eager_mode': True}):
            arr1 = mt.ones((10, 5), chunk_size=4)
            np.testing.assert_array_equal(arr1, np.ones((10, 5)))

            sess = Session.default_or_local()
            executor = sess._sess._executor
            executor.chunk_result[arr1.chunks[0].key] = np.ones((4, 4)) * 2

            arr2 = mt.ones((10, 5), chunk_size=4) - 1
            result = arr2.fetch()
            np.testing.assert_array_equal(result[:4, :4], np.ones((4, 4)))
            np.testing.assert_array_equal(result[8:, :4], np.zeros((2, 4)))

        arr3 = mt.ones((10, 5), chunk_size=4) - 1

        with self.assertRaises(ValueError):
            arr3.fetch()

        result = arr3.execute()
        np.testing.assert_array_equal(result[:4, :4], np.ones((4, 4)))
        np.testing.assert_array_equal(result[8:, :4], np.zeros((2, 4)))

    def testKernelMode(self):
        from mars.session import Session

        t1 = mt.random.rand(10, 10, chunk_size=3)
        t2 = mt.ones((8, 8), chunk_size=6)

        with option_context({'eager_mode': True}):
            sess = Session()
            executor = sess._sess._executor

            t_tiled = t1.tiles()

            with self.assertRaises(ValueError):
                t_tiled.fetch()
            self.assertEqual(0, len(executor.chunk_result))

            result = sess.run(t2)
            self.assertEqual(4, len(executor.chunk_result))
            np.testing.assert_array_equal(result, np.ones((8, 8)))

    def testReprTensor(self):
        a = mt.ones((10, 10), chunk_size=3)
        self.assertIn(a.key, repr(a))

        with option_context({'eager_mode': True}):
            a = mt.ones((10, 10))
            self.assertIn(repr(np.ones((10, 10))), repr(a))
            self.assertIn(str(np.ones((10, 10))), str(a))

        self.assertNotIn(repr(np.ones((10, 10))), repr(a))
        self.assertNotIn(str(np.ones((10, 10))), str(a))

    def testReprDataFrame(self):
        a = md.DataFrame(np.ones((10, 10)), chunk_size=3)
        x = pd.DataFrame(np.ones((10, 10)))

        with option_context({'eager_mode': True}):
            a = md.DataFrame(np.ones((10, 10)), chunk_size=3)
            self.assertIn(repr(x), repr(a))
            self.assertIn(str(x), str(a))

        self.assertNotIn(repr(x), repr(a))
        self.assertNotIn(str(x), str(a))

    def testRuntimeError(self):
        from mars.utils import kernel_mode

        @kernel_mode
        def raise_error(*_):
            raise ValueError

        with option_context({'eager_mode': True}):
            a = mt.zeros((10, 10))
            with self.assertRaises(ValueError):
                raise_error(a)

            r = a + 1
            self.assertIn(repr(np.zeros((10, 10)) + 1), repr(r))
            np.testing.assert_array_equal(r.fetch(), np.zeros((10, 10)) + 1)

        a = mt.zeros((10, 10))
        with self.assertRaises(ValueError):
            a.fetch()

    def testView(self):
        with option_context({'eager_mode': True}):
            a = mt.ones((10, 20), chunk_size=3)
            b = a[0][1:4]
            b[1] = 10

            npa = np.ones((10, 20))
            npb = npa[0][1:4]
            npb[1] = 10

            np.testing.assert_array_equal(a.fetch(), npa)
            np.testing.assert_array_equal(b.fetch(), npb)

    def testDataFrame(self):
        with option_context({'eager_mode': True}):
            from mars.dataframe.expressions.arithmetic import add

            data1 = pd.DataFrame(np.random.rand(10, 10))
            df1 = from_pandas(data1, chunk_size=5)
            pd.testing.assert_frame_equal(df1.fetch(), data1)

            data2 = pd.DataFrame(np.random.rand(10, 10))
            df2 = from_pandas(data2, chunk_size=6)
            pd.testing.assert_frame_equal(df2.fetch(), data2)

            df3 = add(df1, df2)
            pd.testing.assert_frame_equal(df3.fetch(), data1 + data2)
