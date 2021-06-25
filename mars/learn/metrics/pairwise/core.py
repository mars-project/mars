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

import itertools

import numpy as np

from ....core import recursive_tile
from ....serialization.serializables import Int64Field
from ....tensor.operands import TensorOperand, TensorOperandMixin
from ....tensor import tensor as astensor
from ....utils import has_unknown_shape
from ...utils import check_array


class PairwiseDistances(TensorOperand, TensorOperandMixin):
    _op_module_ = 'learn'

    chunk_store_limit = Int64Field('chunk_store_limit')

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
            dtype = float

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
                                 f"(n_queries, n_indexed). Got ({X.shape[0]}, {X.shape[1]}) "
                                 f"for {Y.shape[0]} indexed.")
        elif X.shape[1] != Y.shape[1]:
            raise ValueError("Incompatible dimension for X and Y matrices: "
                             f"X.shape[1] == {X.shape[1]} while Y.shape[1] == {Y.shape[1]}")

        return X, Y

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
    def _rechunk_cols_into_one(cls, x, y):
        y_is_x = y is x
        if x.chunk_shape[1] != 1 or y.chunk_shape[1] != 1:
            if has_unknown_shape([x, y]):
                yield

            x = yield from recursive_tile(x.rechunk({1: x.shape[1]}))
            if y_is_x:
                y = x
            else:
                y = yield from recursive_tile(y.rechunk({1: y.shape[1]}))

        return x, y

    @classmethod
    def _adjust_chunk_sizes(cls, op, X, Y, out):
        max_x_chunk_size = max(X.nsplits[0])
        max_y_chunk_size = max(Y.nsplits[0])
        itemsize = out.dtype.itemsize
        max_chunk_bytes = max_x_chunk_size * max_y_chunk_size * itemsize
        chunk_store_limit = op.chunk_store_limit * 2  # scale 2 times
        if max_chunk_bytes > chunk_store_limit:
            adjust_succeeded = False
            # chunk is too huge, try to rechunk X and Y
            if X.shape[0] > Y.shape[0]:
                # y is smaller, rechunk y is more efficient
                expected_y_chunk_size = max(int(chunk_store_limit / itemsize / max_x_chunk_size), 1)
                if max_x_chunk_size * expected_y_chunk_size * itemsize <= chunk_store_limit:
                    adjust_succeeded = True
                    Y = yield from recursive_tile(
                        Y.rechunk({0: expected_y_chunk_size}))
            else:
                # x is smaller, rechunk x is more efficient
                expected_x_chunk_size = max(int(chunk_store_limit / itemsize / max_y_chunk_size), 1)
                if max_y_chunk_size * expected_x_chunk_size * itemsize <= chunk_store_limit:
                    adjust_succeeded = True
                    X = yield from recursive_tile(X.rechunk({0: expected_x_chunk_size}))

            if not adjust_succeeded:
                expected_chunk_size = max(int(np.sqrt(chunk_store_limit / itemsize)), 1)
                X = yield from recursive_tile(X.rechunk({0: expected_chunk_size}))
                Y = yield from recursive_tile(Y.rechunk({0: expected_chunk_size}))

        return X, Y
