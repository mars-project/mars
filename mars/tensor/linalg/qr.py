#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from numpy.linalg import LinAlgError

from ... import opcodes as OperandDef
from ...serialization.serializables import KeyField, StringField
from ...core import ExecutableTuple
from ..array_utils import device, as_same_device
from ..datasource import tensor as astensor
from ..operands import TensorHasInput, TensorOperandMixin
from ..core import TensorOrder
from .core import SFQR, TSQR


class TensorQR(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.QR

    _input = KeyField('input')
    _method = StringField('method')

    def __init__(self, method=None, **kw):
        super().__init__(_method=method, **kw)

    @property
    def method(self):
        return self._method

    @property
    def output_limit(self):
        return 2

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, a):
        a = astensor(a)

        if a.ndim != 2:
            raise LinAlgError(f'{a.ndim}-dimensional tensor given. '
                              'Tensor must be two-dimensional')

        tiny_q, tiny_r = np.linalg.qr(np.ones((1, 1), dtype=a.dtype))

        x, y = a.shape
        q_shape, r_shape = (a.shape, (y, y)) if x > y else ((x, x), a.shape)
        q, r = self.new_tensors([a],
                                kws=[{'side': 'q', 'dtype': tiny_q.dtype,
                                      'shape': q_shape, 'order': TensorOrder.C_ORDER},
                                     {'side': 'r', 'dtype': tiny_r.dtype,
                                      'shape': r_shape, 'order': TensorOrder.C_ORDER}])
        return ExecutableTuple([q, r])

    @classmethod
    def tile(cls, op):
        q, r = op.outputs
        q_dtype, r_dtype = q.dtype, r.dtype
        q_shape, r_shape = q.shape, r.shape
        in_tensor = op.input
        if in_tensor.chunk_shape == (1, 1):
            in_chunk = in_tensor.chunks[0]
            chunk_op = op.copy().reset_key()
            qr_chunks = chunk_op.new_chunks([in_chunk], kws=[
                {'side': 'q', 'shape': q_shape, 'index': in_chunk.index},
                {'side': 'r', 'shape': r_shape, 'index': in_chunk.index}
            ])
            q_chunk, r_chunk = qr_chunks

            new_op = op.copy()
            kws = [
                {'chunks': [q_chunk], 'nsplits': ((q_shape[0],), (q_shape[1],)),
                 'dtype': q_dtype, 'shape': q_shape, 'order': q.order},
                {'chunks': [r_chunk], 'nsplits': ((r_shape[0],), (r_shape[1],)),
                 'dtype': r_dtype, 'shape': r_shape, 'order': r.order}
            ]
            return new_op.new_tensors(op.inputs, kws=kws)
        elif op.method == 'tsqr':
            return (yield from TSQR.tile(op))
        elif op.method == 'sfqr':
            return (yield from SFQR.tile(op))
        else:
            raise NotImplementedError('Only tsqr method supported for now')

    @classmethod
    def execute(cls, ctx, op):
        (a,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            q, r = xp.linalg.qr(a)
            qc, rc = op.outputs
            ctx[qc.key] = q
            ctx[rc.key] = r


def qr(a, method='tsqr'):
    """
    Compute the qr factorization of a matrix.

    Factor the matrix `a` as *qr*, where `q` is orthonormal and `r` is
    upper-triangular.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Matrix to be factored.
    method: {'tsqr', 'sfqr'}, optional
        method to calculate qr factorization, tsqr as default

        TSQR is presented in:

            A. Benson, D. Gleich, and J. Demmel.
            Direct QR factorizations for tall-and-skinny matrices in
            MapReduce architectures.
            IEEE International Conference on Big Data, 2013.
            http://arxiv.org/abs/1301.1071

        FSQR is a QR decomposition for fat and short matrix:
            A = [A1, A2, A3, ...], A1 may be decomposed as A1 = Q1 * R1,
            for A = Q * R, Q = Q1, R = [R1, R2, R3, ...] where A2 = Q1 * R2, A3 = Q1 * R3, ...

    Returns
    -------
    q : Tensor of float or complex, optional
        A matrix with orthonormal columns. When mode = 'complete' the
        result is an orthogonal/unitary matrix depending on whether or not
        a is real/complex. The determinant may be either +/- 1 in that
        case.
    r : Tensor of float or complex, optional
        The upper-triangular matrix.

    Raises
    ------
    LinAlgError
        If factoring fails.

    Notes
    -----
    For more information on the qr factorization, see for example:
    http://en.wikipedia.org/wiki/QR_factorization

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.random.randn(9, 6)
    >>> q, r = mt.linalg.qr(a)
    >>> mt.allclose(a, mt.dot(q, r)).execute()  # a does equal qr
    True

    """
    op = TensorQR(method=method)
    return op(a)
