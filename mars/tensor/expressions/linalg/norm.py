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

import itertools
from collections import Iterable

import numpy as np

from .... import opcodes as OperandDef
from ....serialize import ValueType, KeyField, AnyField, TupleField, BoolField
from ..utils import recursive_tile
from ..core import TensorHasInput, TensorOperandMixin
from ..arithmetic import sqrt
from ..datasource import tensor as astensor
from .svd import svd


class TensorNorm(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.NORM

    _input = KeyField('input')
    _ord = AnyField('ord')
    _axis = TupleField('axis', ValueType.int32)
    _keepdims = BoolField('keepdims')

    def __init__(self, ord=None, axis=None, keepdims=None, dtype=None, sparse=False, **kw):
        super(TensorNorm, self).__init__(_ord=ord, _axis=axis, _keepdims=keepdims,
                                         _dtype=dtype, _sparse=sparse, **kw)

    @property
    def ord(self):
        return getattr(self, '_ord', None)

    @property
    def axis(self):
        return self._axis

    @property
    def keepdims(self):
        return self._keepdims

    def _set_inputs(self, inputs):
        super(TensorNorm, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, x):
        r = x.astype(self.dtype)
        shape = self._norm(r, self._ord, self._axis, self._keepdims).shape
        return self.new_tensor([x], shape)

    @classmethod
    def tile(cls, op):
        x = op.input
        axis = op.axis
        ord = op.ord
        keepdims = op.keepdims

        axis_chunk_shapes = tuple(x.chunk_shape[i] for i in axis)
        can_apply_norm = all(s == 1 for s in axis_chunk_shapes)

        if can_apply_norm:
            axis_set = set(axis)
            get_shape = lambda shape: tuple(s if i not in axis_set else 1 for i, s in enumerate(shape)
                                            if i not in axis_set or keepdims)

            out_chunk_shape = get_shape(x.chunk_shape)
            out_chunks = []
            for idx in itertools.product(*[range(s) for s in out_chunk_shape]):
                idx_iter = iter(idx)
                in_idx = tuple(0 if i in axis_set and not keepdims else next(idx_iter)
                               for i in range(x.ndim))

                c = x.cix[in_idx]
                chunk_op = op.copy().reset_key()
                out_chunk = chunk_op.new_chunk([c], shape=get_shape(c.shape), index=idx)
                out_chunks.append(out_chunk)

            nsplits = [tuple(c.shape[i] for c in out_chunks
                             if all(idx == 0 for j, idx in enumerate(c.index) if j != i))
                       for i in range(len(out_chunks[0].shape))]
            new_op = op.copy()
            return new_op.new_tensors(op.inputs, op.outputs[0].shape, chunks=out_chunks, nsplits=nsplits)

        r = cls._norm(x.astype(op.outputs[0].dtype), ord, axis, keepdims)
        recursive_tile(r)
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape, chunks=r.chunks, nsplits=r.nsplits)

    @staticmethod
    def _norm(r, ord, axis, keepdims):
        if ord is None:
            return sqrt((abs(r) ** 2).sum(axis=axis, keepdims=keepdims))
        elif ord == 'nuc':
            if len(axis) == 1:
                raise ValueError('Invalid norm order for vectors.')
            return svd(r)[1][np.newaxis].sum(keepdims=keepdims)
        elif ord == np.inf:
            if r.ndim > 2:
                raise ValueError('Improper number of dimensions to norm.')
            r = abs(r)
            if len(axis) == 1:
                return r.max(axis=axis, keepdims=keepdims)
            else:
                return r.sum(axis=axis[1], keepdims=keepdims).max(keepdims=keepdims)
        elif ord == -np.inf:
            if r.ndim > 2:
                raise ValueError('Improper number of dimensions to norm.')
            r = abs(r)
            if len(axis) == 1:
                return r.min(axis=axis, keepdims=keepdims)
            else:
                return r.sum(axis=axis[1], keepdims=keepdims).min(keepdims=keepdims)
        elif ord == 0:
            if r.ndim > 2:
                raise ValueError('Improper number of dimensions to norm.')
            if len(axis) == 2:
                raise ValueError('Invalid norm order for matrices.')
            return (r != 0).astype(r.dtype).sum(axis=axis, keepdims=keepdims)
        elif ord == 1:
            if r.ndim > 2:
                raise ValueError('Improper number of dimensions to norm.')
            r = abs(r)
            if len(axis) == 1:
                return r.sum(axis=axis, keepdims=keepdims)
            else:
                return r.sum(axis=axis[0], keepdims=keepdims).max(keepdims=keepdims)
        elif ord == -1 and len(axis) == 2:
            if r.ndim > 2:
                raise ValueError('Improper number of dimensions to norm.')
            return abs(r).sum(axis=axis[0], keepdims=keepdims).min(keepdims=keepdims)
        elif ord == 2 and len(axis) == 2:
            return svd(r)[1][np.newaxis].max(keepdims=keepdims)
        elif ord == -2 and len(axis) == 2:
            return svd(r)[1][np.newaxis].min(keepdims=keepdims)
        else:
            if len(axis) == 2:
                raise ValueError('Invalid norm order for matrices.')

            return (abs(r) ** ord).sum(axis=axis, keepdims=keepdims) ** (1.0 / ord)


def norm(x, ord=None, axis=None, keepdims=False):
    r"""
    Matrix or vector norm.

    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    Parameters
    ----------
    x : array_like
        Input tensor.  If `axis` is None, `x` must be 1-D or 2-D.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Notes``). inf means mars tensor's
        `inf` object.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

    Returns
    -------
    n : float or Tensor
        Norm of the matrix or vector(s).

    Notes
    -----
    For values of ``ord <= 0``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.

    The following norms can be calculated:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    The nuclear norm is the sum of the singular values.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    Examples
    --------
    >>> from mars.tensor import linalg as LA
    >>> import mars.tensor as mt
    >>> a = mt.arange(9) - 4
    >>> a.execute()
    array([-4, -3, -2, -1,  0,  1,  2,  3,  4])
    >>> b = a.reshape((3, 3))
    >>> b.execute()
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]])

    >>> LA.norm(a).execute()
    7.745966692414834
    >>> LA.norm(b).execute()
    7.745966692414834
    >>> LA.norm(b, 'fro').execute()
    7.745966692414834
    >>> LA.norm(a, mt.inf).execute()
    4.0
    >>> LA.norm(b, mt.inf).execute()
    9.0
    >>> LA.norm(a, -mt.inf).execute()
    0.0
    >>> LA.norm(b, -mt.inf).execute()
    2.0

    >>> LA.norm(a, 1).execute()
    20.0
    >>> LA.norm(b, 1).execute()
    7.0
    >>> LA.norm(a, -1).execute()
    0.0
    >>> LA.norm(b, -1).execute()
    6.0
    >>> LA.norm(a, 2).execute()
    7.745966692414834
    >>> LA.norm(b, 2).execute()
    7.3484692283495345

    >>> LA.norm(a, -2).execute()
    0.0
    >>> LA.norm(b, -2).execute()
    4.351066026358965e-18
    >>> LA.norm(a, 3).execute()
    5.8480354764257312
    >>> LA.norm(a, -3).execute()
    0.0

    Using the `axis` argument to compute vector norms:

    >>> c = mt.array([[ 1, 2, 3],
    ...               [-1, 1, 4]])
    >>> LA.norm(c, axis=0).execute()
    array([ 1.41421356,  2.23606798,  5.        ])
    >>> LA.norm(c, axis=1).execute()
    array([ 3.74165739,  4.24264069])
    >>> LA.norm(c, ord=1, axis=1).execute()
    array([ 6.,  6.])

    Using the `axis` argument to compute matrix norms:

    >>> m = mt.arange(8).reshape(2,2,2)
    >>> LA.norm(m, axis=(1,2)).execute()
    array([  3.74165739,  11.22497216])
    >>> LA.norm(m[0, :, :]).execute(), LA.norm(m[1, :, :]).execute()
    (3.7416573867739413, 11.224972160321824)

    """
    x = astensor(x)

    if ord == 'fro':
        ord = None
    if axis is not None:
        if isinstance(axis, Iterable):
            axis = tuple(axis)
        else:
            axis = (axis,)
    else:
        axis = tuple(range(x.ndim))

    op = TensorNorm(ord=ord, axis=axis, keepdims=keepdims,
                    dtype=np.result_type(x.dtype, np.float_), sparse=x.issparse())
    return op(x)
