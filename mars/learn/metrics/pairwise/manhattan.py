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
try:
    from sklearn.metrics.pairwise import manhattan_distances as sklearn_manhattan_distances
except ImportError:  # pragma: no cover
    sklearn_manhattan_distances = None

from .... import opcodes as OperandDef
from ....core import recursive_tile
from ....serialization.serializables import KeyField, BoolField
from ....tensor.core import TensorOrder
from ....tensor.arithmetic import abs as mt_abs
from ....tensor.spatial.distance import cdist
from ....tensor.array_utils import as_same_device, device
from ....utils import ensure_own_data
from .core import PairwiseDistances


class ManhattanDistances(PairwiseDistances):
    _op_type_ = OperandDef.PAIRWISE_MANHATTAN_DISTANCES

    _x = KeyField('x')
    _y = KeyField('y')
    _sum_over_features = BoolField('sum_over_features')

    def __init__(self, x=None, y=None, sum_over_features=None,
                 use_sklearn=None, **kw):
        super().__init__(_x=x, _y=y, _sum_over_features=sum_over_features,
                         _use_sklearn=use_sklearn, **kw)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def sum_over_features(self):
        return self._sum_over_features

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._x = self._inputs[0]
        self._y = self._inputs[1]

    def __call__(self, X, Y=None):
        X, Y = self.check_pairwise_arrays(X, Y)
        if self._y is None:
            self._y = Y

        if (X.issparse() or Y.issparse()) and not self._sum_over_features:
            raise TypeError(f"sum_over_features={self._sum_over_features} not supported"
                            " for sparse matrices")

        if not self._sum_over_features:
            shape = (X.shape[0] * Y.shape[0], X.shape[1])
        else:
            shape = (X.shape[0], Y.shape[0])

        return self.new_tensor([X, Y], shape=shape,
                               order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op):
        x, y = op.x, op.y

        if len(x.chunks) == 1 and len(y.chunks) == 1:
            return cls._tile_one_chunk(op)

        if x.issparse() or y.issparse():
            assert op.sum_over_features
            return cls._tile_chunks(op, x, y)
        elif op.sum_over_features:
            # if x, y are not sparse and `sum_over_features` is True
            # just use cdist
            return [(yield from recursive_tile(cdist(x, y, 'cityblock')))]
        else:
            d = x[:, np.newaxis, :] - y[np.newaxis, :, :]
            d = mt_abs(d)
            d = d.reshape((-1, x.shape[1]))
            return [(yield from recursive_tile(d))]

    @classmethod
    def execute(cls, ctx, op):
        (x, y), device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)
        out = op.outputs[0]

        with device(device_id):
            if sklearn_manhattan_distances is not None:
                ctx[out.key] = sklearn_manhattan_distances(
                    ensure_own_data(x), ensure_own_data(y),
                    sum_over_features=op.sum_over_features)
            else:  # pragma: no cover
                # we cannot support sparse
                raise NotImplementedError('cannot support calculate manhattan '
                                          'distances on GPU')


def manhattan_distances(X, Y=None, sum_over_features=True):
    """ Compute the L1 distances between the vectors in X and Y.

    With sum_over_features equal to False it returns the componentwise
    distances.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array_like
        A tensor with shape (n_samples_X, n_features).

    Y : array_like, optional
        A tensor with shape (n_samples_Y, n_features).

    sum_over_features : bool, default=True
        If True the function returns the pairwise distance matrix
        else it returns the componentwise L1 pairwise-distances.
        Not supported for sparse matrix inputs.

    Returns
    -------
    D : Tensor
        If sum_over_features is False shape is
        (n_samples_X * n_samples_Y, n_features) and D contains the
        componentwise L1 pairwise-distances (ie. absolute difference),
        else shape is (n_samples_X, n_samples_Y) and D contains
        the pairwise L1 distances.

    Examples
    --------
    >>> from mars.learn.metrics.pairwise import manhattan_distances
    >>> manhattan_distances([[3]], [[3]]).execute() #doctest:+ELLIPSIS
    array([[0.]])
    >>> manhattan_distances([[3]], [[2]]).execute() #doctest:+ELLIPSIS
    array([[1.]])
    >>> manhattan_distances([[2]], [[3]]).execute() #doctest:+ELLIPSIS
    array([[1.]])
    >>> manhattan_distances([[1, 2], [3, 4]],\
         [[1, 2], [0, 3]]).execute() #doctest:+ELLIPSIS
    array([[0., 2.],
           [4., 4.]])
    >>> import mars.tensor as mt
    >>> X = mt.ones((1, 2))
    >>> y = mt.full((2, 2), 2.)
    >>> manhattan_distances(X, y, sum_over_features=False).execute() #doctest:+ELLIPSIS
    array([[1., 1.],
           [1., 1.]])
    """
    op = ManhattanDistances(x=X, y=Y, sum_over_features=sum_over_features,
                            dtype=np.dtype(np.float64))
    return op(X, Y=Y)
