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
from mars.tests.core import mock


@unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
class Test(unittest.TestCase):

    @mock.patch('webbrowser.open_new_tab', new=lambda *_, **__: True)
    def testMutableTensorCreateAndGet(self):
        def testWithGivenSession(session):
            mut1 = session.create_mutable_tensor("test", (4, 5), dtype=np.double, chunk_size=3)
            mut2 = session.get_mutable_tensor("test")

            self.assertEqual(tuple(mut1.shape), (4, 5))
            self.assertEqual(mut1.dtype, np.double)
            self.assertEqual(mut1.nsplits, ((3, 1), (3, 2)))

            # mut1 and mut2 are not the same object, but has the same properties.
            self.assertNotEqual(mut1.id, mut2.id)
            self.assertEqual(mut1.shape, mut2.shape)
            self.assertEqual(mut1.dtype, mut2.dtype)
            self.assertEqual(mut1.nsplits, mut2.nsplits)

            for chunk1, chunk2 in zip(mut2.chunks, mut2.chunks):
                self.assertEqual(chunk1.key, chunk2.key)
                self.assertEqual(chunk1.index, chunk2.index)
                self.assertEqual(chunk1.shape, chunk2.shape)
                self.assertEqual(chunk1.dtype, chunk2.dtype)

        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            with new_session(cluster.endpoint).as_default() as session:
                testWithGivenSession(session)

            with new_session('http://' + cluster._web_endpoint).as_default() as web_session:
                testWithGivenSession(web_session)

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

                # write [1], and buffer is not full.
                chunk_records = mut._do_write(1, np.arange(5))
                self.assertEqual(chunk_records, dict())
                chunk_records = mut._do_flush()

                result = chunk_records[mut.cix[(0, 0)].key]
                expected = np.array([[3, 0.], [4, 1.], [5, 2.]])
                self.assertRecordsEqual(result, expected)

                result = chunk_records[mut.cix[(0, 1)].key]
                expected = np.array([[2, 3.], [3, 4.]])
                self.assertRecordsEqual(result, expected)

                # write [2, [0, 2, 4]] (fancy index), and buffer is not full.
                chunk_records = mut._do_write((2, [0, 2, 4]), np.array([11, 22, 33]))
                self.assertEqual(chunk_records, dict())
                chunk_records = mut._do_flush()

                result = chunk_records[mut.cix[(0, 0)].key]
                expected = np.array([[6, 11.], [8, 22.]])
                self.assertRecordsEqual(result, expected)

                result = chunk_records[mut.cix[(0, 1)].key]
                expected = np.array([[5, 33.]])
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

    @mock.patch('webbrowser.open_new_tab', new=lambda *_, **__: True)
    def testMutableTensorSeal(self):
        def testWithGivenSession(session):
            mut = session.create_mutable_tensor("test", (4, 5), dtype='int32', chunk_size=3)
            mut[1:4, 2] = 8
            mut[2:4] = np.arange(10).reshape(2, 5)
            mut[1] = np.arange(5)
            arr = mut.seal()

            expected = np.zeros((4, 5), dtype='int32')
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

        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            session = cluster.session.as_default()
            testWithGivenSession(session)

            with new_session('http://' + cluster._web_endpoint).as_default() as web_session:
                testWithGivenSession(web_session)

    @mock.patch('webbrowser.open_new_tab', new=lambda *_, **__: True)
    def testMutableTensorDuplicateName(self):
        def testWithGivenSession(session, check_message=True):
            session.create_mutable_tensor("test", (4, 5), dtype='int32')

            # The two unsealed mutable tensors cannot have the same name.
            with self.assertRaises(ValueError) as cm:
                session.create_mutable_tensor("test", (4, 5), dtype='int32')

            expected_msg = "The mutable tensor named 'test' already exists."

            if check_message:
                self.assertEqual(cm.exception.args[0], expected_msg)

        with new_session().as_default() as session:
            testWithGivenSession(session)

        with new_cluster(scheduler_n_process=2, worker_n_process=2, shared_memory='20M', web=True) as cluster:
            session = cluster.session.as_default()
            testWithGivenSession(session)

            with new_session('http://' + cluster._web_endpoint).as_default() as web_session:
                testWithGivenSession(web_session, check_message=False)

    @mock.patch('webbrowser.open_new_tab', new=lambda *_, **__: True)
    def testMutableTensorRaiseAfterSeal(self):
        def testWithGivenSession(session, check_message=True):
            mut = session.create_mutable_tensor("test", (4, 5), dtype='int32', chunk_size=3)
            mut.seal()

            expected_msg = "The mutable tensor named 'test' doesn't exist, or has already been sealed."

            # Cannot get after seal
            with self.assertRaises(ValueError) as cm:
                session.get_mutable_tensor("test")

            if check_message:
                self.assertEqual(cm.exception.args[0], expected_msg)

            # Cannot write after seal
            with self.assertRaises(ValueError) as cm:
                mut[:] = 111

            if check_message:
                self.assertEqual(cm.exception.args[0], expected_msg)

            # Cannot seal after seal
            with self.assertRaises(ValueError) as cm:
                session.seal(mut)

            if check_message:
                self.assertEqual(cm.exception.args[0], expected_msg)

        with new_session().as_default() as session:
            testWithGivenSession(session)

        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            session = cluster.session.as_default()
            testWithGivenSession(session)

            with new_session('http://' + cluster._web_endpoint).as_default() as web_session:
                testWithGivenSession(web_session, check_message=False)

    def testMutableTensorLocal(self):
        with new_session().as_default() as session:
            mut = session.create_mutable_tensor("test", (4, 5), dtype='int32', chunk_size=3)
            mut2 = session.get_mutable_tensor("test")

            # In local session, mut1 and mut2 should be the same object
            self.assertEqual(id(mut), id(mut2))

            # Check the property
            self.assertEqual(mut.shape, (4, 5))
            self.assertEqual(np.dtype(mut.dtype), np.int32)

            # Mutable tensor doesn't have chunks in local session
            self.assertEqual(mut.chunks, None)

            # Check write and seal
            mut[1:4, 2] = 8
            mut[2:4] = np.arange(10).reshape(2, 5)
            mut[1] = np.arange(5)
            arr = mut.seal()

            # The arr should be executed after seal
            self.assertIn(arr.key, session._sess.executed_tileables)

            # The arr should has chunks
            self.assertNotEqual(arr.chunks, None)

            expected = np.zeros((4, 5), dtype='int32')
            expected[1:4, 2] = 8
            expected[2:4] = np.arange(10).reshape(2, 5)
            expected[1] = np.arange(5)

            # Check the value
            np.testing.assert_array_equal(session.fetch(arr), expected)

            # check operations on the sealed tensor
            np.testing.assert_array_equal(session.run(arr + 1), expected + 1)
            np.testing.assert_array_equal(session.run(arr + arr), expected + expected)
            np.testing.assert_array_equal(session.run(arr.sum()), expected.sum())

    def assertRecordsEqual(self, records, expected):
        np.testing.assert_array_equal(records['index'], expected[:,0])
        np.testing.assert_array_equal(records['value'], expected[:,1])
