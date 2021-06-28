# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

from .... import opcodes as OperandDef
from ....core import recursive_tile
from ....serialization.serializables import KeyField
from .... import tensor as mt
from ....tensor.core import TensorOrder
from ...preprocessing import normalize
from .core import PairwiseDistances


class CosineDistances(PairwiseDistances):
    _op_type_ = OperandDef.PAIRWISE_COSINE_DISTANCES

    _x = KeyField('x')
    _y = KeyField('y')

    def __init__(self, x=None, y=None, **kw):
        super().__init__(_x=x, _y=y, **kw)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._x = self._inputs[0]
        self._y = self._inputs[1]

    def __call__(self, x, y=None):
        x, y = self.check_pairwise_arrays(x, y)
        return self.new_tensor([x, y], shape=(x.shape[0], y.shape[0]),
                               order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op):
        x, y = op.x, op.y
        if x is y:
            S = cosine_similarity(x)
        else:
            S = cosine_similarity(x, y)
        S = (S * -1) + 1
        S = mt.clip(S, 0, 2)
        if x is y:
            mt.fill_diagonal(S, 0.0)
        return [(yield from recursive_tile(S))]


def cosine_similarity(X, Y=None, dense_output=True):
    """Compute cosine similarity between samples in X and Y.

    Cosine similarity, or the cosine kernel, computes similarity as the
    normalized dot product of X and Y:

        K(X, Y) = <X, Y> / (||X||*||Y||)

    On L2-normalized data, this function is equivalent to linear_kernel.

    Read more in the :ref:`User Guide <cosine_similarity>`.

    Parameters
    ----------
    X : Tensor or sparse tensor, shape: (n_samples_X, n_features)
        Input data.

    Y : Tensor or sparse tensor, shape: (n_samples_Y, n_features)
        Input data. If ``None``, the output will be the pairwise
        similarities between all samples in ``X``.

    dense_output : boolean (optional), default True
        Whether to return dense output even when the input is sparse. If
        ``False``, the output is sparse if both input tensors are sparse.

    Returns
    -------
    kernel matrix : Tensor
        A tensor with shape (n_samples_X, n_samples_Y).
    """
    X, Y = PairwiseDistances.check_pairwise_arrays(X, Y)

    X_normalized = normalize(X, copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y, copy=True)

    K = X_normalized.dot(Y_normalized.T)
    if dense_output:
        K = K.todense()
    return K


def cosine_distances(X, Y=None):
    """Compute cosine distance between samples in X and Y.

    Cosine distance is defined as 1.0 minus the cosine similarity.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array_like, sparse matrix
        with shape (n_samples_X, n_features).

    Y : array_like, sparse matrix (optional)
        with shape (n_samples_Y, n_features).

    Returns
    -------
    distance matrix : Tensor
        A tensor with shape (n_samples_X, n_samples_Y).

    See also
    --------
    mars.learn.metrics.pairwise.cosine_similarity
    mars.tensor.spatial.distance.cosine : dense matrices only
    """
    op = CosineDistances(x=X, y=Y, dtype=np.dtype(np.float64))
    return op(X, y=Y)
