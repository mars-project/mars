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

import itertools
import os
import tempfile
import unittest.mock
from io import BytesIO

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from mars.dataframe import ArrowStringDtype, ArrowListDtype
from mars.errors import SerializationFailed
from mars.lib import sparse
from mars.lib.groupby_wrapper import wrapped_groupby
from mars.serialize import dataserializer
from mars.tests.core import assert_groupby_equal

try:
    import pyarrow
except ImportError:
    pyarrow = None
try:
    import scipy.sparse as sps
except ImportError:
    sps = None


class ClassToPickle:
    def __init__(self, a):
        self.a = a


class Test(unittest.TestCase):

    def testDataSerialize(self):
        for type_, compress in itertools.product(
                (None,) + tuple(dataserializer.SerialType.__members__.values()),
                (None,) + tuple(dataserializer.CompressType.__members__.values())):
            array = np.random.rand(1000, 100)
            assert_array_equal(array, dataserializer.loads(
                dataserializer.dumps(array, serial_type=type_, compress=compress)))

            array = np.random.rand(1000, 100)
            assert_array_equal(array, dataserializer.load(
                BytesIO(dataserializer.dumps(array, serial_type=type_, compress=compress))))

            array = np.random.rand(1000, 100).T  # test non c-contiguous
            assert_array_equal(array, dataserializer.loads(
                dataserializer.dumps(array, serial_type=type_, compress=compress)))

            array = np.float64(0.2345)
            assert_array_equal(array, dataserializer.loads(
                dataserializer.dumps(array, serial_type=type_, compress=compress)))

        # test non-serializable object
        if pyarrow:
            non_serial = type('non_serial', (object,), dict(nbytes=10))
            with self.assertRaises(SerializationFailed):
                dataserializer.dumps(non_serial())

        # test structured arrays.
        rec_dtype = np.dtype([('a', 'int64'), ('b', 'double'), ('c', '<U8')])
        array = np.ones((100,), dtype=rec_dtype)
        array_loaded = dataserializer.loads(dataserializer.dumps(array))
        self.assertEqual(array.dtype, array_loaded.dtype)
        assert_array_equal(array, array_loaded)

        fn = os.path.join(tempfile.gettempdir(), f'test_dump_file_{id(self)}.bin')
        try:
            array = np.random.rand(1000, 100).T  # test non c-contiguous
            with open(fn, 'wb') as dump_file:
                dataserializer.dump(array, dump_file)
            with open(fn, 'rb') as dump_file:
                assert_array_equal(array, dataserializer.load(dump_file))

            with open(fn, 'wb') as dump_file:
                dataserializer.dump(array, dump_file,
                                    compress=dataserializer.CompressType.LZ4)
            with open(fn, 'rb') as dump_file:
                assert_array_equal(array, dataserializer.load(dump_file))

            with open(fn, 'wb') as dump_file:
                dataserializer.dump(array, dump_file,
                                    compress=dataserializer.CompressType.GZIP)
            with open(fn, 'rb') as dump_file:
                assert_array_equal(array, dataserializer.load(dump_file))
        finally:
            if os.path.exists(fn):
                os.unlink(fn)

        # test sparse
        if sps:
            mat = sparse.SparseMatrix(sps.random(100, 100, 0.1, format='csr'))
            des_mat = dataserializer.loads(dataserializer.dumps(mat))
            self.assertTrue((mat.spmatrix != des_mat.spmatrix).nnz == 0)

            des_mat = dataserializer.loads(dataserializer.dumps(
                mat, compress=dataserializer.CompressType.LZ4))
            self.assertTrue((mat.spmatrix != des_mat.spmatrix).nnz == 0)

            des_mat = dataserializer.loads(dataserializer.dumps(
                mat, compress=dataserializer.CompressType.GZIP))
            self.assertTrue((mat.spmatrix != des_mat.spmatrix).nnz == 0)

            vector = sparse.SparseVector(sps.csr_matrix(np.random.rand(2)), shape=(2,))
            des_vector = dataserializer.loads(dataserializer.dumps(vector))
            self.assertTrue((vector.spmatrix != des_vector.spmatrix).nnz == 0)

            des_vector = dataserializer.loads(dataserializer.dumps(
                vector, compress=dataserializer.CompressType.LZ4))
            self.assertTrue((vector.spmatrix != des_vector.spmatrix).nnz == 0)

            des_vector = dataserializer.loads(dataserializer.dumps(
                vector, compress=dataserializer.CompressType.GZIP))
            self.assertTrue((vector.spmatrix != des_vector.spmatrix).nnz == 0)

        # test groupby
        df1 = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                            'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                            'c': list('aabaaddce')})
        grouped = wrapped_groupby(df1, 'b')
        restored = dataserializer.loads(dataserializer.dumps(grouped))
        assert_groupby_equal(grouped, restored.groupby_obj)

        grouped = wrapped_groupby(df1, 'b').c
        restored = dataserializer.loads(dataserializer.dumps(grouped))
        assert_groupby_equal(grouped, restored.groupby_obj)

        grouped = wrapped_groupby(df1, 'b')
        getattr(grouped, 'indices')
        restored = dataserializer.loads(dataserializer.dumps(grouped))
        assert_groupby_equal(grouped, restored.groupby_obj)

        grouped = wrapped_groupby(df1.b, lambda x: x % 2)
        restored = dataserializer.loads(dataserializer.dumps(grouped))
        assert_groupby_equal(grouped, restored.groupby_obj)

        grouped = wrapped_groupby(df1.b, lambda x: x % 2)
        getattr(grouped, 'indices')
        restored = dataserializer.loads(dataserializer.dumps(grouped))
        assert_groupby_equal(grouped, restored.groupby_obj)

        # test categorical
        s = np.random.RandomState(0).random(10)
        cat = pd.cut(s, [0.3, 0.5, 0.8])
        self.assertIsInstance(cat, pd.Categorical)
        des_cat = dataserializer.loads(dataserializer.dumps(cat))
        self.assertEqual(len(cat), len(des_cat))
        for c, dc in zip(cat, des_cat):
            np.testing.assert_equal(c, dc)

        # test IntervalIndex
        s = pd.interval_range(10, 100, 3)
        dest_s = dataserializer.loads((dataserializer.dumps(s)))
        pd.testing.assert_index_equal(s, dest_s)

        # test complex
        s = complex(10 + 5j)
        dest_s = dataserializer.loads((dataserializer.dumps(s)))
        self.assertIs(type(s), type(dest_s))
        self.assertEqual(s, dest_s)

        s = np.complex64(10 + 5j)
        dest_s = dataserializer.loads((dataserializer.dumps(s)))
        self.assertIs(type(s), type(dest_s))
        self.assertEqual(s, dest_s)

        # test pickle
        d = ClassToPickle(dict(a=1, b='uvw'))
        dest_d = dataserializer.loads((dataserializer.dumps(d)))
        self.assertIs(type(d), type(dest_d))
        self.assertEqual(d.a, dest_d.a)

        # test ndarray with negative strides
        arr = np.zeros((5, 6, 3))
        arr2 = arr[:, :, ::-1]
        dest_arr2 = dataserializer.loads(dataserializer.dumps(arr2))
        np.testing.assert_array_equal(arr2, dest_arr2)

        # test ArrowArray
        df = pd.DataFrame({'a': ['s1', 's2', 's3'],
                           'b': [['s1', 's2'], ['s3'], ['s4', 's5']]})
        df['a'] = df['a'].astype(ArrowStringDtype())
        df['b'] = df['b'].astype(ArrowListDtype(str))
        dest_df = dataserializer.loads(dataserializer.dumps(df))
        self.assertIs(type(df), type(dest_df))
        pd.testing.assert_frame_equal(df, dest_df)

        # test DataFrame with SparseDtype
        s = pd.Series([1, 2, np.nan, np.nan, 3]).astype(
            pd.SparseDtype(np.dtype(np.float64), np.nan))
        dest_s = dataserializer.loads((dataserializer.dumps(s)))
        pd.testing.assert_series_equal(s, dest_s)
        df = pd.DataFrame({'s': s})
        dest_df = dataserializer.loads((dataserializer.dumps(df)))
        pd.testing.assert_frame_equal(df, dest_df)

    @unittest.skipIf(pyarrow is None, 'PyArrow is not installed.')
    def testArrowSerialize(self):
        array = np.random.rand(1000, 100)
        assert_array_equal(array, dataserializer.deserialize(dataserializer.serialize(array).to_buffer()))

        if sps:
            mat = sparse.SparseMatrix(sps.random(100, 100, 0.1, format='csr'))
            des_mat = dataserializer.deserialize(dataserializer.serialize(mat).to_buffer())
            self.assertTrue((mat.spmatrix != des_mat.spmatrix).nnz == 0)

            array = np.random.rand(1000, 100)
            mat = sparse.SparseMatrix(sps.random(100, 100, 0.1, format='csr'))
            tp = (array, mat)
            des_tp = dataserializer.deserialize(dataserializer.serialize(tp).to_buffer())
            assert_array_equal(tp[0], des_tp[0])
            self.assertTrue((tp[1].spmatrix != des_tp[1].spmatrix).nnz == 0)
