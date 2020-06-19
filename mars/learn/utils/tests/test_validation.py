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

from itertools import product
import unittest

import numpy as np

import mars.tensor as mt
import mars.dataframe as md
from mars.session import new_session
from mars.tensor.core import Tensor
from mars.tests.core import ExecutorForTest

try:
    import scipy.sparse as sp
    import sklearn
    from sklearn.utils.estimator_checks import NotAnArray
    from sklearn.utils.mocking import MockDataFrame
    from sklearn.utils._testing import assert_raise_message, assert_raises_regex
    import pytest

    from mars.learn.utils.validation import check_array, check_consistent_length
except ImportError:
    sklearn = None


@unittest.skipIf(sklearn is None, 'scikit-learn not installed')
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.executor = self.session._sess._executor = \
            ExecutorForTest('numpy', storage=self.session._sess._context)

    def tearDown(self) -> None:
        self.session._sess._executor = self._old_executor

    def test_ordering(self):
        # Check that ordering is enforced correctly by validation utilities.
        # We need to check each validation utility, because a 'copy' without
        # 'order=K' will kill the ordering.
        X = mt.ones((10, 5))
        for A in X, X.T:
            for copy in (True, False):
                B = check_array(A, order='C', copy=copy)
                self.assertTrue(B.flags['C_CONTIGUOUS'])
                B = check_array(A, order='F', copy=copy)
                self.assertTrue(B.flags['F_CONTIGUOUS'])
                if copy:
                    self.assertIsNot(A, B)

    def test_check_array(self):
        # accept_sparse == False
        # raise error on sparse inputs
        X = [[1, 2], [3, 4]]
        X_csr = sp.csr_matrix(X)
        with self.assertRaises(TypeError):
            check_array(X_csr)
        X_csr = mt.tensor(sp.csr_matrix(X))
        with self.assertRaises(TypeError):
            check_array(X_csr)
        # ensure_2d=False
        X_array = check_array([0, 1, 2], ensure_2d=False)
        self.assertEqual(X_array.ndim, 1)
        # ensure_2d=True with 1d array
        assert_raise_message(ValueError, 'Expected 2D array, got 1D array instead',
                             check_array, [0, 1, 2], ensure_2d=True)
        assert_raise_message(ValueError, 'Expected 2D array, got 1D array instead',
                             check_array, mt.tensor([0, 1, 2]), ensure_2d=True)
        # ensure_2d=True with scalar array
        assert_raise_message(ValueError,
                             'Expected 2D array, got scalar array instead',
                             check_array, 10, ensure_2d=True)
        # don't allow ndim > 3
        X_ndim = mt.arange(8).reshape(2, 2, 2)
        with self.assertRaises(ValueError):
            check_array(X_ndim)
        check_array(X_ndim, allow_nd=True)  # doesn't raise

        # dtype and order enforcement.
        X_C = mt.arange(4).reshape(2, 2).copy("C")
        X_F = X_C.copy("F")
        X_int = X_C.astype(mt.int)
        X_float = X_C.astype(mt.float)
        Xs = [X_C, X_F, X_int, X_float]
        dtypes = [mt.int32, mt.int, mt.float, mt.float32, None, mt.bool, object]
        orders = ['C', 'F', None]
        copys = [True, False]

        for X, dtype, order, copy in product(Xs, dtypes, orders, copys):
            X_checked = check_array(X, dtype=dtype, order=order, copy=copy,
                                    force_all_finite=False)
            if dtype is not None:
                self.assertEqual(X_checked.dtype, dtype)
            else:
                self.assertEqual(X_checked.dtype, X.dtype)
            if order == 'C':
                assert X_checked.flags['C_CONTIGUOUS']
                assert not X_checked.flags['F_CONTIGUOUS']
            elif order == 'F':
                assert X_checked.flags['F_CONTIGUOUS']
                assert not X_checked.flags['C_CONTIGUOUS']
            if copy:
                assert X is not X_checked
            else:
                # doesn't copy if it was already good
                if (X.dtype == X_checked.dtype and
                        X_checked.flags['C_CONTIGUOUS'] == X.flags['C_CONTIGUOUS']
                        and X_checked.flags['F_CONTIGUOUS'] == X.flags['F_CONTIGUOUS']):
                    assert X is X_checked

        # other input formats
        # convert lists to arrays
        X_dense = check_array([[1, 2], [3, 4]])
        assert isinstance(X_dense, Tensor)
        # raise on too deep lists
        with self.assertRaises(ValueError):
            check_array(X_ndim.to_numpy().tolist())
        check_array(X_ndim.to_numpy().tolist(), allow_nd=True)  # doesn't raise
        # convert weird stuff to arrays
        X_no_array = NotAnArray(X_dense.to_numpy())
        result = check_array(X_no_array)
        assert isinstance(result, Tensor)

        # deprecation warning if string-like array with dtype="numeric"
        expected_warn_regex = r"converted to decimal numbers if dtype='numeric'"
        X_str = [['11', '12'], ['13', 'xx']]
        for X in [X_str, mt.array(X_str, dtype='U'), mt.array(X_str, dtype='S')]:
            with pytest.warns(FutureWarning, match=expected_warn_regex):
                check_array(X, dtype="numeric")

        # deprecation warning if byte-like array with dtype="numeric"
        X_bytes = [[b'a', b'b'], [b'c', b'd']]
        for X in [X_bytes, mt.array(X_bytes, dtype='V1')]:
            with pytest.warns(FutureWarning, match=expected_warn_regex):
                check_array(X, dtype="numeric")

        # test finite
        X = [[1.0, np.nan], [2.0, 3.0]]
        with self.assertRaises(ValueError):
            _ = check_array(X).execute()

    def test_check_array_pandas_dtype_object_conversion(self):
        # test that data-frame like objects with dtype object
        # get converted
        X = mt.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=mt.object)
        X_df = MockDataFrame(X)
        self.assertEqual(check_array(X_df).dtype.kind, "f")
        self.assertEqual(check_array(X_df, ensure_2d=False).dtype.kind, "f")
        # smoke-test against dataframes with column named "dtype"
        X_df.dtype = "Hans"
        self.assertEqual(check_array(X_df, ensure_2d=False).dtype.kind, "f")

    def test_check_array_from_dataframe(self):
        X = md.DataFrame({'a': [1.0, 2.0, 3.0]})
        self.assertEqual(check_array(X).dtype.kind, 'f')

    def test_check_array_accept_sparse_type_exception(self):
        X = [[1, 2], [3, 4]]
        X_csr = sp.csr_matrix(X)

        msg = ("A sparse tensor was passed, but dense data is required. "
               "Use X.todense() to convert to a dense tensor.")
        assert_raise_message(TypeError, msg,
                             check_array, X_csr, accept_sparse=False)

        msg = ("When providing 'accept_sparse' as a tuple or list, "
               "it must contain at least one string value.")
        assert_raise_message(ValueError, msg.format([]),
                             check_array, X_csr, accept_sparse=[])
        assert_raise_message(ValueError, msg.format(()),
                             check_array, X_csr, accept_sparse=())

        with self.assertRaises(ValueError):
            check_array(X_csr, accept_sparse=object)

    def test_check_array_accept_sparse_no_exception(self):
        X = [[1, 2], [3, 4]]
        X_csr = sp.csr_matrix(X)

        array = check_array(X_csr, accept_sparse=True)
        self.assertIsInstance(array, Tensor)
        self.assertTrue(array.issparse())

    def test_check_array_min_samples_and_features_messages(_):
        # empty list is considered 2D by default:
        msg = "0 feature(s) (shape=(1, 0)) while a minimum of 1 is required."
        assert_raise_message(ValueError, msg, check_array, [[]])

        # If considered a 1D collection when ensure_2d=False, then the minimum
        # number of samples will break:
        msg = "0 sample(s) (shape=(0,)) while a minimum of 1 is required."
        assert_raise_message(ValueError, msg, check_array, [], ensure_2d=False)

        # Invalid edge case when checking the default minimum sample of a scalar
        msg = "Singleton array array(42) cannot be considered a valid collection."
        assert_raise_message(TypeError, msg, check_array, 42, ensure_2d=False)

    def test_check_array_complex_data_error(_):
        X = mt.array([[1 + 2j, 3 + 4j, 5 + 7j], [2 + 3j, 4 + 5j, 6 + 7j]])
        assert_raises_regex(
            ValueError, "Complex data not supported", check_array, X)

        # list of lists
        X = [[1 + 2j, 3 + 4j, 5 + 7j], [2 + 3j, 4 + 5j, 6 + 7j]]
        assert_raises_regex(
            ValueError, "Complex data not supported", check_array, X)

        # tuple of tuples
        X = ((1 + 2j, 3 + 4j, 5 + 7j), (2 + 3j, 4 + 5j, 6 + 7j))
        assert_raises_regex(
            ValueError, "Complex data not supported", check_array, X)

        # list of np arrays
        X = [mt.array([1 + 2j, 3 + 4j, 5 + 7j]),
             mt.array([2 + 3j, 4 + 5j, 6 + 7j])]
        assert_raises_regex(
            ValueError, "Complex data not supported", check_array, X)

        # tuple of np arrays
        X = (mt.array([1 + 2j, 3 + 4j, 5 + 7j]),
             mt.array([2 + 3j, 4 + 5j, 6 + 7j]))
        assert_raises_regex(
            ValueError, "Complex data not supported", check_array, X)

        # dataframe
        X = MockDataFrame(
            mt.array([[1 + 2j, 3 + 4j, 5 + 7j], [2 + 3j, 4 + 5j, 6 + 7j]]))
        assert_raises_regex(
            ValueError, "Complex data not supported", check_array, X)

        # sparse matrix
        X = sp.coo_matrix([[0, 1 + 2j], [0, 0]])
        assert_raises_regex(
            ValueError, "Complex data not supported", check_array, X)

    def testCheckConsistentLength(self):
        t = mt.random.RandomState(0).rand(10, 5)
        t2 = t[t[:, 0] < 0.5]
        t3 = t[t[:, 1] < 0.1]

        check_consistent_length(t2, t2.copy())
        with self.assertRaises(ValueError):
            check_consistent_length(t2, t3)
