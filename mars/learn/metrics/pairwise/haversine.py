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

import numpy as np
try:
    from sklearn.neighbors import DistanceMetric as SklearnDistanceMetric
except ImportError:  # pragma: no cover
    SklearnDistanceMetric = None

from .... import opcodes as OperandDef
from ....serialize import KeyField, BoolField
from ....tensor.core import TensorOrder
from ....tensor.indexing import fill_diagonal
from ....tensor.array_utils import as_same_device, device
from ....tensor.utils import recursive_tile
from ....tiles import TilesError
from ....utils import check_chunks_unknown_shape
from .core import PairwiseDistances


class HaversineDistances(PairwiseDistances):
    _op_type_ = OperandDef.PAIRWISE_HAVERSINE_DISTANCES

    _x = KeyField('x')
    _y = KeyField('y')
    # for test purpose
    _use_sklearn = BoolField('use_sklearn')

    def __init__(self, x=None, y=None, dtype=None, gpu=None,
                 use_sklearn=None, **kw):
        super().__init__(_x=x, _y=y, _dtype=dtype, _gpu=gpu,
                         _use_sklearn=use_sklearn, **kw)
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
            raise TypeError('haversine distance requires inputs dense')

        return self.new_tensor([X, Y], shape=(X.shape[0], Y.shape[0]),
                               order=TensorOrder.C_ORDER)

    @classmethod
    def _tile_one_chunk(cls, op):
        out = op.outputs[0]
        chunk_op = op.copy().reset_key()
        chunk = chunk_op.new_chunk([op.x.chunks[0], op.y.chunks[0]],
                                    shape=out.shape, order=out.order,
                                    index=(0, 0))
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=out.shape,
                                  order=out.order, chunks=[chunk],
                                  nsplits=tuple((s,) for s in out.shape))

    @classmethod
    def _tile_chunks(cls, op, x, y):
        out = op.outputs[0]
        out_chunks = []
        for idx in itertools.product(range(x.chunk_shape[0]),
                                     range(y.chunk_shape[0])):
            xi, yi = idx

            chunk_op = op.copy().reset_key()
            chunk_inputs = [x.cix[xi, 0], y.cix[yi, 0]]
            out_chunk = chunk_op.new_chunk(
                chunk_inputs, shape=(chunk_inputs[0].shape[0],
                                     chunk_inputs[1].shape[0],),
                order=out.order, index=idx)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=out.shape,
                                  order=out.order, chunks=out_chunks,
                                  nsplits=(x.nsplits[0], y.nsplits[0]))

    @classmethod
    def tile(cls, op):
        x, y = op.x, op.y
        y_is_x = y is x

        if len(x.chunks) == 1 and len(y.chunks) == 1:
            return cls._tile_one_chunk(op)

        if x.chunk_shape[1] != 1 or y.chunk_shape[1] != 1:
            check_chunks_unknown_shape([x, y], TilesError)

            x = x.rechunk({1: x.shape[1]})._inplace_tile()
            if y_is_x:
                y = x
            else:
                y = y.rechunk({1: y.shape[1]})._inplace_tile()

        ret, = cls._tile_chunks(op, x, y)
        if y_is_x:
            fill_diagonal(ret, 0)
        return [recursive_tile(ret)]

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
    op = HaversineDistances(x=X, y=Y, dtype=np.dtype(np.float64))
    return op(X, Y=Y)
