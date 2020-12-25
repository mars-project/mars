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

import numpy as np
import pandas as pd

import mars.dataframe as md
import mars.tensor as mt
from mars.tests.core import TestBase


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.ctx, self.executor = self._create_test_context()

    def testDataFrameInitializer(self):
        # from tensor
        raw = np.random.rand(100, 10)
        tensor = mt.tensor(raw, chunk_size=7)
        r = md.DataFrame(tensor)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(result, pd.DataFrame(raw))

        r = md.DataFrame(tensor, chunk_size=13)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(result, pd.DataFrame(raw))

        # from Mars dataframe
        raw = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
        df = md.DataFrame(raw, chunk_size=15) * 2
        r = md.DataFrame(df, num_partitions=11)
        results = self.executor.execute_dataframe(r)
        self.assertEqual(len(results), 10)
        pd.testing.assert_frame_equal(pd.concat(results), raw * 2)

        # from tileable dict
        raw_dict = {
            'C': np.random.choice(['u', 'v', 'w'], size=(100,)),
            'A': pd.Series(np.random.rand(100)),
            'B': np.random.randint(0, 10, size=(100,)),
        }
        m_dict = raw_dict.copy()
        m_dict['A'] = md.Series(m_dict['A'])
        m_dict['B'] = mt.tensor(m_dict['B'])
        r = md.DataFrame(m_dict, columns=list('ABC'))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(result, pd.DataFrame(raw_dict, columns=list('ABC')))

        # from raw pandas initializer
        raw = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
        r = md.DataFrame(raw, num_partitions=10)
        results = self.executor.execute_dataframe(r)
        self.assertEqual(len(results), 10)
        pd.testing.assert_frame_equal(pd.concat(results), raw)

        # from mars series
        raw_s = np.random.rand(100)
        s = md.Series(raw_s, chunk_size=20)
        r = md.DataFrame(s, num_partitions=10)
        results = self.executor.execute_dataframe(r)
        self.assertEqual(len(results), 10)
        pd.testing.assert_frame_equal(pd.concat(results), pd.DataFrame(raw_s))

        # test check instance
        r = r * 2
        self.assertIsInstance(r, md.DataFrame)

    def testSeriesInitializer(self):
        # from tensor
        raw = np.random.rand(100)
        tensor = mt.tensor(raw, chunk_size=7)
        r = md.Series(tensor)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(result, pd.Series(raw))

        r = md.Series(tensor, chunk_size=13)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(result, pd.Series(raw))

        # from index
        raw = np.arange(100)
        np.random.shuffle(raw)
        raw = pd.Index(raw, name='idx_name')
        idx = md.Index(raw, chunk_size=7)
        r = md.Series(idx)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(result, pd.Series(raw))

        # from Mars series
        raw = pd.Series(np.random.rand(100), name='series_name')
        ms = md.Series(raw, chunk_size=15) * 2
        r = md.Series(ms, num_partitions=11)
        results = self.executor.execute_dataframe(r)
        self.assertEqual(len(results), 10)
        pd.testing.assert_series_equal(pd.concat(results), raw * 2)

        # from raw pandas initializer
        raw = pd.Series(np.random.rand(100), name='series_name')
        r = md.Series(raw, num_partitions=10)
        results = self.executor.execute_dataframe(r)
        self.assertEqual(len(results), 10)
        pd.testing.assert_series_equal(pd.concat(results), raw)

        # test check instance
        r = r * 2
        self.assertIsInstance(r, md.Series)

    def testIndexInitializer(self):
        def _concat_idx(results):
            s_results = [pd.Series(idx) for idx in results]
            return pd.Index(pd.concat(s_results))

        # from tensor
        raw = np.arange(100)
        np.random.shuffle(raw)
        tensor = mt.tensor(raw)
        r = md.Index(tensor, chunk_size=7)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_index_equal(result, pd.Index(raw))

        # from Mars index
        raw = np.arange(100)
        np.random.shuffle(raw)
        idx = md.Index(raw, chunk_size=7)
        r = md.Index(idx, num_partitions=11)
        results = self.executor.execute_dataframe(r)
        self.assertEqual(len(results), 10)
        pd.testing.assert_index_equal(_concat_idx(results), pd.Index(raw))

        # from pandas initializer
        raw = np.arange(100)
        np.random.shuffle(raw)
        raw_ser = pd.Series(raw, name='series_name')
        r = md.Index(raw_ser, chunk_size=7)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_index_equal(result, pd.Index(raw_ser))

        raw_idx = pd.Index(raw, name='idx_name')
        r = md.Index(raw_idx, num_partitions=10)
        results = self.executor.execute_dataframe(r)
        self.assertEqual(len(results), 10)
        pd.testing.assert_index_equal(_concat_idx(results), pd.Index(raw_idx))
