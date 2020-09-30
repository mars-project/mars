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
import os
from io import BytesIO

import pandas as pd
import numpy as np

import mars.dataframe as md
from mars.tests.core import TestBase


TEST_DIR = '/tmp/test'

csv_content = b"""
A,B,C,D,E
1.0,2020-01-01,1.0,3,foo
1.0,2020-01-01,1.0,3,foo
1.0,2020-01-01,1.0,3,foo
1.0,2020-01-01,1.0,3,foo
1.0,2020-01-01,1.0,3,foo
1.0,2020-01-01,1.0,3,foo
1.0,2020-01-01,1.0,3,foo
1.0,2020-01-01,1.0,3,foo
1.0,2020-01-01,1.0,3,foo
1.0,2020-01-01,1.0,3,foo
""".strip()


@unittest.skipIf(not os.environ.get('WITH_HADOOP'), 'Only run when hdfs is installed')
class TestHDFS(TestBase):
    def setUp(self):
        super().setUp()

        import pyarrow
        self.hdfs = pyarrow.hdfs.connect(host="localhost", port=8020)
        if self.hdfs.exists(TEST_DIR):
            self.hdfs.rm(TEST_DIR, recursive=True)

    def tearDown(self):
        if self.hdfs.exists(TEST_DIR):
            self.hdfs.rm(TEST_DIR, recursive=True)

    def testReadCSVExecution(self):
        with self.hdfs.open(f"{TEST_DIR}/simple_test.csv", "wb", replication=1) as f:
            f.write(b'name,amount,id\nAlice,100,1\nBob,200,2')

        df = md.read_csv(f'hdfs://localhost:8020{TEST_DIR}/simple_test.csv')
        expected = pd.read_csv(BytesIO(b'name,amount,id\nAlice,100,1\nBob,200,2'))
        res = df.to_pandas()
        pd.testing.assert_frame_equal(expected, res)

        with self.hdfs.open(f"{TEST_DIR}/chunk_test.csv", "wb", replication=1) as f:
            f.write(csv_content)

        df = md.read_csv(f'hdfs://localhost:8020{TEST_DIR}/chunk_test.csv', chunk_bytes=50)
        expected = pd.read_csv(BytesIO(csv_content))
        res = df.to_pandas()
        pd.testing.assert_frame_equal(expected.reset_index(drop=True), res.reset_index(drop=True))

    def testReadParquetExecution(self):
        test_df = pd.DataFrame({'a': np.arange(10).astype(np.int64, copy=False),
                                'b': [f's{i}' for i in range(10)],
                                'c': np.random.rand(10), })
        with self.hdfs.open(f"{TEST_DIR}/test.parquet", "wb", replication=1) as f:
            test_df.to_parquet(f, row_group_size=3)

        df = md.read_parquet(f'hdfs://localhost:8020{TEST_DIR}/test.parquet')
        res = df.to_pandas()
        pd.testing.assert_frame_equal(res, test_df)
