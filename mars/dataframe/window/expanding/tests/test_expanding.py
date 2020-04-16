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

import mars.dataframe as md
from mars.tests.core import ExecutorForTest


class Test(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.executor = ExecutorForTest()

    def testExpanding(self):
        df = pd.DataFrame(np.random.rand(4, 3), columns=list('abc'))
        df2 = md.DataFrame(df)

        with self.assertRaises(NotImplementedError):
            _ = df2.expanding(3, center=True)

        with self.assertRaises(NotImplementedError):
            _ = df2.expanding(3, axis=1)

        r = df2.expanding(3, center=False)
        expected = df.expanding(3, center=False)
        self.assertEqual(repr(r), repr(expected))

        self.assertIn('b', dir(r))

        with self.assertRaises(AttributeError):
            _ = r.d

        with self.assertRaises(KeyError):
            _ = r['d']

        with self.assertRaises(KeyError):
            _ = r['a', 'd']

        self.assertNotIn('a', dir(r.a))
        self.assertNotIn('c', dir(r['a', 'b']))

    def testExpandingAgg(self):
        df = pd.DataFrame(np.random.rand(4, 3), columns=list('abc'))
        df2 = md.DataFrame(df, chunk_size=3)

        r = df2.expanding(3).agg('max')
        expected = df.expanding(3).agg('max')

        self.assertEqual(r.shape, df.shape)
        self.assertIs(r.index_value, df2.index_value)
        pd.testing.assert_index_equal(r.columns_value.to_pandas(),
                                      expected.columns)
        pd.testing.assert_series_equal(r.dtypes, df2.dtypes)

        r = r.tiles()
        for c in r.chunks:
            self.assertEqual(c.shape, c.inputs[0].shape)
            self.assertIs(c.index_value, c.inputs[0].index_value)
            pd.testing.assert_index_equal(c.columns_value.to_pandas(),
                                          expected.columns)
            pd.testing.assert_series_equal(c.dtypes, expected.dtypes)
