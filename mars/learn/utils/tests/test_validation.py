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

import warnings
from itertools import product
import unittest

import mars.tensor as mt
from mars.tensor.core import Tensor

try:
    import scipy.sparse as sp
    import sklearn
    from sklearn.utils.testing import assert_raise_message
    from sklearn.utils.estimator_checks import NotAnArray

    from mars.learn.utils.validation import check_array
except ImportError:
    sklearn = None


@unittest.skipIf(sklearn is None, 'scikit-learn not installed')
class Test(unittest.TestCase):
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
            X_checked = check_array(X, dtype=dtype, order=order, copy=copy)
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

        # # allowed sparse != None
        # X_csc = sp.csc_matrix(X_C)
        # X_coo = X_csc.tocoo()
        # X_dok = X_csc.todok()
        # X_int = X_csc.astype(mt.int)
        # X_float = X_csc.astype(mt.float)
        #
        # Xs = [X_csc, X_coo, X_dok, X_int, X_float]
        # accept_sparses = [['csr', 'coo'], ['coo', 'dok']]
        # for X, dtype, accept_sparse, copy in product(Xs, dtypes, accept_sparses,
        #                                              copys):
        #     with warnings.catch_warnings(record=True) as w:
        #         X_checked = check_array(X, dtype=dtype,
        #                                 accept_sparse=accept_sparse, copy=copy)
        #     if (dtype is object or sp.isspmatrix_dok(X)) and len(w):
        #         message = str(w[0].message)
        #         messages = ["object dtype is not supported by sparse matrices",
        #                     "Can't check dok sparse matrix for nan or inf."]
        #         assert message in messages
        #     else:
        #         self.assertEqual(len(w), 0)
        #     if dtype is not None:
        #         self.assertEqual(X_checked.dtype, dtype)
        #     else:
        #         self.assertEqual(X_checked.dtype, X.dtype)
        #     if X.format in accept_sparse:
        #         # no change if allowed
        #         self.assertEqual(X.format, X_checked.format)
        #     else:
        #         # got converted
        #         self.assertEqual(X_checked.format, accept_sparse[0])
        #     if copy:
        #         assert X is not X_checked
        #     else:
        #         # doesn't copy if it was already good
        #         if X.dtype == X_checked.dtype and X.format == X_checked.format:
        #             assert X is X_checked

        # other input formats
        # convert lists to arrays
        X_dense = check_array([[1, 2], [3, 4]])
        assert isinstance(X_dense, Tensor)
        # raise on too deep lists
        with self.assertRaises(ValueError):
            check_array(X_ndim.execute().tolist())
        check_array(X_ndim.execute().tolist(), allow_nd=True)  # doesn't raise
        # convert weird stuff to arrays
        X_no_array = NotAnArray(X_dense.execute())
        result = check_array(X_no_array)
        assert isinstance(result, Tensor)

        # deprecation warning if string-like array with dtype="numeric"
        expected_warn_regex = r"converted to decimal numbers if dtype='numeric'"
        X_str = [['11', '12'], ['13', 'xx']]
        for X in [X_str, mt.array(X_str, dtype='U'), mt.array(X_str, dtype='S')]:
            with self.assertWarnsRegex(FutureWarning, expected_warn_regex):
                check_array(X, dtype="numeric")

        # deprecation warning if byte-like array with dtype="numeric"
        X_bytes = [[b'a', b'b'], [b'c', b'd']]
        for X in [X_bytes, mt.array(X_bytes, dtype='V1')]:
            with self.assertWarnsRegex(FutureWarning, expected_warn_regex):
                check_array(X, dtype="numeric")
