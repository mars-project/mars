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

from mars.tensor.operands import TensorOperand, TensorOperandMixin
from mars.tensor import tensor as astensor
from mars.learn.utils import check_array


class PairwiseDistances(TensorOperand, TensorOperandMixin):
    _op_module_ = 'learn'

    @staticmethod
    def _return_float_dtype(X, Y):
        """
        1. If dtype of X and Y is float32, then dtype float32 is returned.
        2. Else dtype float is returned.
        """

        X = astensor(X)

        if Y is None:
            Y_dtype = X.dtype
        else:
            Y = astensor(Y)
            Y_dtype = Y.dtype

        if X.dtype == Y_dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float

        return X, Y, dtype

    @staticmethod
    def check_pairwise_arrays(X, Y, precomputed=False, dtype=None):
        X, Y, dtype_float = PairwiseDistances._return_float_dtype(X, Y)

        estimator = 'check_pairwise_arrays'
        if dtype is None:
            dtype = dtype_float

        if Y is X or Y is None:
            X = Y = check_array(X, accept_sparse=True, dtype=dtype,
                                estimator=estimator)
        else:
            X = check_array(X, accept_sparse=True, dtype=dtype,
                            estimator=estimator)
            Y = check_array(Y, accept_sparse=True, dtype=dtype,
                            estimator=estimator)

        if precomputed:
            if X.shape[1] != Y.shape[0]:
                raise ValueError("Precomputed metric requires shape "
                                 "(n_queries, n_indexed). Got (%d, %d) "
                                 "for %d indexed." %
                                 (X.shape[0], X.shape[1], Y.shape[0]))
        elif X.shape[1] != Y.shape[1]:
            raise ValueError("Incompatible dimension for X and Y matrices: "
                             "X.shape[1] == %d while Y.shape[1] == %d" % (
                                 X.shape[1], Y.shape[1]))

        return X, Y
