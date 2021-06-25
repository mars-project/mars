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
    from sklearn.neighbors import DistanceMetric as SklearnDistanceMetric
except ImportError:  # pragma: no cover
    SklearnDistanceMetric = None

from .... import opcodes as OperandDef
from ....core import recursive_tile
from ....serialization.serializables import KeyField, BoolField
from ....tensor.core import TensorOrder
from ....tensor.indexing import fill_diagonal
from ....tensor.array_utils import as_same_device, device
from .core import PairwiseDistances


class HaversineDistances(PairwiseDistances):
    _op_type_ = OperandDef.PAIRWISE_HAVERSINE_DISTANCES

    _x = KeyField('x')
    _y = KeyField('y')
    # for test purpose
    _use_sklearn = BoolField('use_sklearn')

    def __init__(self, x=None, y=None, use_sklearn=None, **kw):
        super().__init__(_x=x, _y=y, _use_sklearn=use_sklearn, **kw)
        if self._use_sklearn is None:
            # if not set use_sklearn, will try to use sklearn by default
            self._use_sklearn = True

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def use_sklearn(self):
        return self._use_sklearn

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._x = self._inputs[0]
        self._y = self._inputs[1]

    def __call__(self, X, Y=None):
        X, Y = self.check_pairwise_arrays(X, Y)
        if self._y is None:
            self._y = Y

        if X.shape[1] != 2 or Y.shape[1] != 2:
            raise ValueError('Haversine distance only valid in 2 dimensions')
        if X.issparse() or Y.issparse():
            raise TypeError('Haversine distance requires inputs dense')

        return self.new_tensor([X, Y], shape=(X.shape[0], Y.shape[0]),
                               order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op):
        x, y = op.x, op.y
        y_is_x = y is x

        if len(x.chunks) == 1 and len(y.chunks) == 1:
            return cls._tile_one_chunk(op)

        x, y = yield from cls._rechunk_cols_into_one(x, y)
        ret, = cls._tile_chunks(op, x, y)
        if y_is_x:
            fill_diagonal(ret, 0)
        return [(yield from recursive_tile(ret))]

    @classmethod
    def execute(cls, ctx, op):
        (x, y), device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            if xp is np and op.use_sklearn and SklearnDistanceMetric is not None:
                # CPU and sklearn installed, delegate computation to sklearn
                d = SklearnDistanceMetric.get_metric('haversine').pairwise(x, y)
            else:
                # try to leverage xp(np, cp) to perform computation
                sin_0 = xp.sin(0.5 * (x[:, [0]] - y[:, 0]))
                sin_1 = xp.sin(0.5 * (x[:, [1]] - y[:, 1]))
                d = 2 * xp.arcsin(xp.sqrt(
                    sin_0 * sin_0 + xp.cos(x[:, [0]]) * xp.cos(y[:, 0]) * sin_1 * sin_1))

            ctx[op.outputs[0].key] = d


def haversine_distances(X, Y=None):
    """Compute the Haversine distance between samples in X and Y

    The Haversine (or great circle) distance is the angular distance between
    two points on the surface of a sphere. The first distance of each point is
    assumed to be the latitude, the second is the longitude, given in radians.
    The dimension of the data must be 2.

    .. math::
       D(x, y) = 2\\arcsin[\\sqrt{\\sin^2((x1 - y1) / 2)
                                + \\cos(x1)\\cos(y1)\\sin^2((x2 - y2) / 2)}]

    Parameters
    ----------
    X : array_like, shape (n_samples_1, 2)

    Y : array_like, shape (n_samples_2, 2), optional

    Returns
    -------
    distance : {Tensor}, shape (n_samples_1, n_samples_2)

    Notes
    -----
    As the Earth is nearly spherical, the haversine formula provides a good
    approximation of the distance between two points of the Earth surface, with
    a less than 1% error on average.

    Examples
    --------
    We want to calculate the distance between the Ezeiza Airport
    (Buenos Aires, Argentina) and the Charles de Gaulle Airport (Paris, France)

    >>> from mars.learn.metrics.pairwise import haversine_distances
    >>> bsas = [-34.83333, -58.5166646]
    >>> paris = [49.0083899664, 2.53844117956]
    >>> result = haversine_distances([bsas, paris])
    >>> (result * 6371000/1000).execute()  # multiply by Earth radius to get kilometers
    array([[    0.        , 11279.45379464],
           [11279.45379464,     0.        ]])
    """
    op = HaversineDistances(x=X, y=Y, dtype=np.dtype(np.float64))
    return op(X, Y=Y)
