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

import sys
import unittest

import numpy as np

from mars.deploy.local.core import new_cluster
from mars.session import new_session

@unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
class Test(unittest.TestCase):
    def testMutableTensorCreateAndGet(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            with new_session(cluster.endpoint) as session:
                mut1 = session.create_mutable_tensor("test", (4, 5), dtype=np.double, chunk_size=3)
                mut2 = session.get_mutable_tensor("test")

                self.assertEqual(mut1.shape, (4, 5))
                self.assertEqual(mut1.dtype, np.double)
                self.assertEqual(mut1.nsplits, ((3, 1), (3, 2)))

                self.assertEqual(mut1.shape, mut2.shape)
                self.assertEqual(mut1.dtype, mut2.dtype)
                self.assertEqual(mut1.nsplits, mut2.nsplits)

                for chunk1, chunk2 in zip(mut2.chunks, mut2.chunks):
                    self.assertEqual(chunk1.key, chunk2.key)
                    self.assertEqual(chunk1.index, chunk2.index)
                    self.assertEqual(chunk1.shape, chunk2.shape)
                    self.assertEqual(chunk1.dtype, chunk2.dtype)

    def testMutableTensorWrite(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            with new_session(cluster.endpoint) as session:
                mut = session.create_mutable_tensor("test", (4, 5), dtype=np.double, chunk_size=3)

                # write [1:4, 2], and buffer is not full.
                chunk_records = mut._do_write((slice(1, 4, None), 2), 8)
                self.assertEqual(chunk_records, dict())
                chunk_records = mut._do_flush()

                result = chunk_records[mut.cix[(0, 0)].key]
                expected = np.array([[5, 8.], [8, 8.]])
                self.assertRecordsEqual(result, expected)

                result = chunk_records[mut.cix[(1, 0)].key]
                expected = np.array([[2, 8.]])
                self.assertRecordsEqual(result, expected)

                # write [2:4], and buffer is not full.
                chunk_records = mut._do_write(slice(2, 4, None), np.arange(10).reshape((2, 5)))
                self.assertEqual(chunk_records, dict())
                chunk_records = mut._do_flush()

                result = chunk_records[mut.cix[(0, 0)].key]
                expected = np.array([[6, 0.], [7, 1.], [8, 2.]])
                self.assertRecordsEqual(result, expected)

                result = chunk_records[mut.cix[(0, 1)].key]
                expected = np.array([[4, 3.], [5, 4.]])
                self.assertRecordsEqual(result, expected)

                result = chunk_records[mut.cix[(1, 0)].key]
                expected = np.array([[0, 5.], [1, 6.], [2, 7.]])
                self.assertRecordsEqual(result, expected)

                result = chunk_records[mut.cix[(1, 1)].key]
                expected = np.array([[0, 8.], [1, 9.]])
                self.assertRecordsEqual(result, expected)

                # mtensor[1], and buffer is not full.
                chunk_records = mut._do_write(1, np.arange(5))
                self.assertEqual(chunk_records, dict())
                chunk_records = mut._do_flush()

                result = chunk_records[mut.cix[(0, 0)].key]
                expected = np.array([[3, 0.], [4, 1.], [5, 2.]])
                self.assertRecordsEqual(result, expected)

                result = chunk_records[mut.cix[(0, 1)].key]
                expected = np.array([[2, 3.], [3, 4.]])
                self.assertRecordsEqual(result, expected)

                # write [:], and the first buffer is full.
                chunk_records = mut._do_write(slice(None, None, None), 999)

                result = chunk_records[mut.cix[(0, 0)].key]
                expected = np.array([[0, 999.], [1, 999.], [2, 999.], [3, 999.], [4, 999.],
                                     [5, 999.], [6, 999.], [7, 999.], [8, 999.]])
                self.assertRecordsEqual(result, expected)

                # check other chunks
                chunk_records = mut._do_flush()

                result = chunk_records[mut.cix[(0, 1)].key]
                expected = np.array([[0, 999.], [1, 999.], [2, 999.], [3, 999.], [4, 999.],
                                     [5, 999.]])
                self.assertRecordsEqual(result, expected)

                result = chunk_records[mut.cix[(1, 0)].key]
                expected = np.array([[0, 999.], [1, 999.], [2, 999.]])
                self.assertRecordsEqual(result, expected)

                result = chunk_records[mut.cix[(1, 1)].key]
                expected = np.array([[0, 999.], [1, 999.]])
                self.assertRecordsEqual(result, expected)


    def testMutableTensorSeal(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            session = cluster.session

            mut = session.create_mutable_tensor("test", (4, 5), dtype=np.int32, chunk_size=3)
            mut[1:4, 2] = 8
            mut[2:4] = np.arange(10).reshape(2, 5)
            mut[1] = np.arange(5)
            arr = mut.seal()

            expected = np.zeros((4, 5), dtype=np.int32)
            expected[1:4, 2] = 8
            expected[2:4] = np.arange(10).reshape(2, 5)
            expected[1] = np.arange(5)

            # check chunk properties
            for chunk1, chunk2 in zip(mut.chunks, arr.chunks):
                self.assertEqual(chunk1.key, chunk2.key)
                self.assertEqual(chunk1.index, chunk2.index)
                self.assertEqual(chunk1.shape, chunk2.shape)
                self.assertEqual(chunk1.dtype, chunk2.dtype)

            # check value
            np.testing.assert_array_equal(session.fetch(arr), expected)

            # check operations on the sealed tensor
            np.testing.assert_array_equal(session.run(arr + 1), expected + 1)
            np.testing.assert_array_equal(session.run(arr + arr), expected + expected)
            np.testing.assert_array_equal(session.run(arr.sum()), expected.sum())

    def assertRecordsEqual(self, records, expected):
        np.testing.assert_array_equal(records['index'], expected[:,0])
        np.testing.assert_array_equal(records['value'], expected[:,1])
