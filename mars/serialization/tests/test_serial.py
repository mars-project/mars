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
from collections import OrderedDict

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
except ImportError:
    pa = None
try:
    import scipy.sparse as sps
except ImportError:
    sps = None

from mars.lib.sparse import SparseMatrix
from mars.serialization import serialize, deserialize
from mars.tests.core import require_cupy, require_cudf
from mars.utils import lazy_import

cupy = lazy_import('cupy', globals=globals())
cudf = lazy_import('cudf', globals=globals())


class CustomList(list):
    pass


class Test(unittest.TestCase):
    def testCore(self):
        test_vals = [
            False,
            123,
            3.567,
            3.5 + 4.3j,
            b'abcd',
            'abcd',
            ['uvw', ('mno', 'sdaf'), 4, 6.7],
            CustomList([3, 4, CustomList([5, 6])]),
            {'abc': 5.6, 'def': [3.4]},
            OrderedDict([('abcd', 5.6)])
        ]
        for val in test_vals:
            deserialized = deserialize(*serialize(val))
            self.assertEqual(type(val), type(deserialized))
            self.assertEqual(val, deserialized)

    def testNumpy(self):
        test_vals = [
            np.array(np.random.rand(100, 100)),
            np.array(np.random.rand(100, 100).T),
            np.array(['a', 'bcd', None]),
        ]
        for val in test_vals:
            deserialized = deserialize(*serialize(val))
            self.assertEqual(type(val), type(deserialized))
            np.testing.assert_equal(val, deserialized)
            if val.flags.f_contiguous:
                self.assertTrue(deserialized.flags.f_contiguous)

    def testPandas(self):
        val = pd.Series([1, 2, 3, 4])
        pd.testing.assert_series_equal(val, deserialize(*serialize(val)))

        val = pd.DataFrame({
            'a': np.random.rand(1000),
            'b': np.random.choice(list('abcd'), size=(1000,)),
            'c': np.random.randint(0, 100, size=(1000,)),
        })
        pd.testing.assert_frame_equal(val, deserialize(*serialize(val)))

    @unittest.skipIf(pa is None, 'need pyarrow to run the cases')
    def testArrow(self):
        test_df = pd.DataFrame({
            'a': np.random.rand(1000),
            'b': np.random.choice(list('abcd'), size=(1000,)),
            'c': np.random.randint(0, 100, size=(1000,)),
        })
        test_vals = [
            pa.RecordBatch.from_pandas(test_df),
            pa.Table.from_pandas(test_df),
        ]
        for val in test_vals:
            deserialized = deserialize(*serialize(val))
            self.assertEqual(type(val), type(deserialized))
            np.testing.assert_equal(val, deserialized)

    @require_cupy
    def testCupy(self):
        test_vals = [
            cupy.array(np.random.rand(100, 100)),
            cupy.array(np.random.rand(100, 100).T),
        ]
        for val in test_vals:
            deserialized = deserialize(*serialize(val))
            self.assertEqual(type(val), type(deserialized))
            cupy.testing.assert_array_equal(val, deserialized)

    @require_cudf
    def testCudf(self):
        test_df = cudf.DataFrame(pd.DataFrame({
            'a': np.random.rand(1000),
            'b': np.random.choice(list('abcd'), size=(1000,)),
            'c': np.random.randint(0, 100, size=(1000,)),
        }))
        cudf.testing.assert_frame_equal(test_df, deserialize(*serialize(test_df)))

    @unittest.skipIf(sps is None, 'need scipy to run the test')
    def testScipySparse(self):
        val = sps.random(100, 100, 0.1, format='csr')
        deserial = deserialize(*serialize(val))
        self.assertTrue((val != deserial).nnz == 0)

    @unittest.skipIf(sps is None, 'need scipy to run the test')
    def testMarsSparse(self):
        val = SparseMatrix(sps.random(100, 100, 0.1, format='csr'))
        deserial = deserialize(*serialize(val))
        self.assertTrue((val.spmatrix != deserial.spmatrix).nnz == 0)
