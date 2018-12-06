#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import numpy as np
from numpy.linalg import LinAlgError

from .... import operands
from ..core import TensorOperandMixin
from ..datasource import tensor as astensor


class TensorSolveTriangular(operands.SolveTriangular, TensorOperandMixin):
    def __init__(self, lower=None, dtype=None, **kw):
        super(TensorSolveTriangular, self).__init__(_lower=lower, _dtype=dtype, **kw)

    def _set_inputs(self, inputs):
        super(TensorSolveTriangular, self)._set_inputs(inputs)
        self._a, self._b = self._inputs

    def __call__(self, a, b):
        shape = (a.shape[1],) if len(b.shape) == 1 else (a.shape[1], b.shape[1])
        return self.new_tensor([a, b], shape)

    @classmethod
    def tile(cls, op):
        from ..arithmetic.subtract import TensorSubtract
        from ..arithmetic.utils import tree_add
        from .dot import TensorDot

        a, b = op.a, op.b
        if a.nsplits[0] != a.nsplits[1]:
            raise LinAlgError("matrix a's splits of all axis must be equal, "
                              'Use rechunk method to change the splits')
        if a.nsplits[1] != b.nsplits[0]:
            raise LinAlgError("matrix a's splits of axis 1 and matrix b's splits of axis 0 must be equal, "
                              'Use rechunk method to change the splits')

        b_multi_dim = b.ndim > 1
        b_hsplits = b.chunk_shape[1] if b_multi_dim else 1

        def _x_shape(a_shape, b_shape):
            return (a_shape[1],) if len(b_shape) == 1 else (a_shape[1], b_shape[1])

        def _dot_shape(a_shape, b_shape):
            return (a_shape[0],) if len(b_shape) == 1 else (a_shape[0], b_shape[1])

        lower = op.lower
        out_chunks = {}
        if lower:
            for i in range(a.chunk_shape[0]):
                target_a = a.cix[i, i]
                for j in range(b_hsplits):
                    idx = (i, j) if b_multi_dim else (i,)
                    target_b = b.cix[idx]
                    if i > 0:
                        prev_chunks = []
                        for k in range(i):
                            a_chunk, b_chunk = a.cix[i, k], out_chunks[(k, j) if b_multi_dim else (k,)]
                            prev_chunk = TensorDot(dtype=op.dtype).new_chunk(
                                [a_chunk, b_chunk], _dot_shape(a_chunk.shape, b_chunk.shape))
                            prev_chunks.append(prev_chunk)
                        if len(prev_chunks) == 1:
                            s = prev_chunks[0]
                        else:
                            s = tree_add(prev_chunks[0].dtype, prev_chunks,
                                         None, prev_chunks[0].shape)
                        target_b = TensorSubtract(dtype=op.dtype).new_chunk(
                            [target_b, s, None, None], target_b.shape)
                    out_chunk = TensorSolveTriangular(lower=lower).new_chunk(
                        [target_a, target_b], _x_shape(target_a.shape, target_b.shape), index=idx)
                    out_chunks[out_chunk.index] = out_chunk
        else:
            for i in range(a.chunk_shape[0] - 1, -1, -1):
                target_a = a.cix[i, i]
                for j in range(b_hsplits):
                    idx = (i, j) if b_multi_dim else (i,)
                    target_b = b.cix[idx]
                    if i < a.chunk_shape[0] - 1:
                        prev_chunks = []
                        for k in range(i + 1, a.chunk_shape[0]):
                            a_chunk, b_chunk = a.cix[i, k], out_chunks[(k, j) if b_multi_dim else (k,)]
                            prev_chunk = TensorDot(dtype=op.dtype).new_chunk(
                                [a_chunk, b_chunk], _dot_shape(a_chunk.shape, b_chunk.shape))
                            prev_chunks.append(prev_chunk)
                        if len(prev_chunks) == 1:
                            s = prev_chunks[0]
                        else:
                            s = tree_add(prev_chunks[0].dtype, prev_chunks,
                                         None, prev_chunks[0].shape)
                        target_b = TensorSubtract(dtype=op.dtype).new_chunk(
                            [target_b, s, None, None], target_b.shape)
                    out_chunk = TensorSolveTriangular(lower=lower).new_chunk(
                        [target_a, target_b], _x_shape(target_a.shape, target_b.shape), index=idx)
                    out_chunks[out_chunk.index] = out_chunk
        new_op = op.copy()
        nsplits = (a.nsplits[0],) if b.ndim == 1 else (a.nsplits[0], b.nsplits[1])
        return new_op.new_tensors(op.inputs, op.outputs[0].shape, chunks=list(out_chunks.values()), nsplits=nsplits)


def solve_triangular(a, b, lower=False):
    """
    Solve the equation `a x = b` for `x`, assuming a is a triangular matrix.
    Parameters
    ----------
    a : (M, M) array_like
        A triangular matrix
    b : (M,) or (M, N) array_like
        Right-hand side matrix in `a x = b`
    lower : bool, optional
        Use only data contained in the lower triangle of `a`.
        Default is to use upper triangle.

    Returns
    -------
    x : (M,) or (M, N) ndarray
        Solution to the system `a x = b`.  Shape of return matches `b`.

    Examples
    --------
    Solve the lower triangular system a x = b, where::
             [3  0  0  0]       [4]
        a =  [2  1  0  0]   b = [2]
             [1  0  1  0]       [4]
             [1  1  1  1]       [2]
    >>> import mars.tensor as mt
    >>> a = mt.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
    >>> b = mt.array([4, 2, 4, 2])
    >>> x = mt.linalg.solve_triangular(a, b, lower=True)
    >>> x.execute()
    array([ 1.33333333, -0.66666667,  2.66666667, -1.33333333])
    >>> a.dot(x).execute()  # Check the result
    array([ 4.,  2.,  4.,  2.])
    """
    import scipy.linalg

    a = astensor(a)
    b = astensor(b)

    if a.ndim != 2:
        raise LinAlgError('a must be 2 dimensional')
    if b.ndim <= 2:
        if a.shape[1] != b.shape[0]:
            raise LinAlgError('a.shape[1] and b.shape[0] must be equal')
    else:
        raise LinAlgError('b must be 1 or 2 dimensional')

    tiny_x = scipy.linalg.solve_triangular(np.array([[2, 0], [2, 1]], dtype=a.dtype),
                                           np.array([[2], [3]], dtype=b.dtype))
    op = TensorSolveTriangular(lower=lower, dtype=tiny_x.dtype)
    return op(a, b)
