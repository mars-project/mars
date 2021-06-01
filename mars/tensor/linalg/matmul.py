#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from ... import opcodes as OperandDef
from ...serialization.serializables import KeyField, StringField
from ...utils import has_unknown_shape
from ..core import Tensor, TensorOrder
from ..utils import broadcast_shape, check_out_param, unify_chunks, check_order
from ..array_utils import device, as_same_device, is_sparse_module
from ..operands import TensorOperand, TensorOperandMixin
from ..arithmetic.utils import chunk_tree_add
from ..datasource import tensor as astensor


class TensorMatmul(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.MATMUL

    _a = KeyField('a')
    _b = KeyField('b')
    _casting = StringField('casting')
    _order = StringField('order')

    def __init__(self, casting=None, order=None,  **kw):
        super().__init__(_casting=casting, _order=order, **kw)
        if self._casting is None:
            self._casting = 'same_kind'
        if self._order is None:
            self._order = 'K'
        check_order(self._order)

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def casting(self):
        return self._casting

    @property
    def order(self):
        return self._order

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._a = self._inputs[0]
        self._b = self._inputs[1]

    def _calc_order(self, a, b, out):
        if out is not None:
            return out.order

        if self._order in 'A':
            if a.order == TensorOrder.C_ORDER or b.order == TensorOrder.C_ORDER:
                return TensorOrder.C_ORDER
            else:
                return TensorOrder.F_ORDER
        elif self._order in 'CK':
            return TensorOrder.C_ORDER
        else:
            return TensorOrder.F_ORDER

    def __call__(self, a, b, out=None):
        from ..base import broadcast_to

        if a.ndim == 0 or b.ndim == 0:
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        if out is not None and not isinstance(out, Tensor):
            raise TypeError(f'out must be a Tensor, got {type(out)} instead')

        a_is_1d = False
        if a.ndim == 1:
            a_is_1d = True
            a = a[np.newaxis, :]

        b_is_1d = False
        if b.ndim == 1:
            b_is_1d = True
            b = b[:, np.newaxis]

        if a.ndim < b.ndim:
            a = a[(b.ndim - a.ndim) * (np.newaxis,)]
        elif a.ndim > b.ndim:
            b = b[(a.ndim - b.ndim) * (np.newaxis,)]

        if a.shape[-1] != b.shape[-2]:
            raise ValueError(f'shape {a.shape} and {b.shape} not aligned: '
                             f'{a.shape[-1]} (dim {a.ndim - 1}) != {b.shape[-2]} (dim {b.ndim - 2})')

        shape = broadcast_shape(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-1])
        order = self._calc_order(a, b, out)
        t = self.new_tensor([a, b], shape, order=order)

        if a_is_1d:
            t = t[..., 0, :]
        if b_is_1d:
            t = t[..., 0]

        if out is not None:
            check_out_param(out, t, self._casting)
            t = broadcast_to(t, out.shape)
            out.data = t.data
            return out

        return t

    @classmethod
    def tile(cls, op):
        a, b = op.inputs
        tensor = op.outputs[0]
        # the axes to align on
        a_axes = list(range(a.ndim - 2))[::-1] + [tensor.ndim - 2, tensor.ndim - 1]
        b_axes = list(range(b.ndim - 2))[::-1] + [tensor.ndim - 1, tensor.ndim]
        if has_unknown_shape(a, b):
            yield
        a, b = yield from unify_chunks((a, a_axes), (b, b_axes))

        get_nsplit = lambda i: a.nsplits[i] if a.nsplits[i] != (1,) else b.nsplits[i]
        get_idx = lambda ch, idx: tuple(0 if ch.nsplits[j] == (1,) else ix for j, ix in enumerate(idx))

        prefix_idxes = [range(len(get_nsplit(i))) for i in range(a.ndim - 2)]
        out_idxes = prefix_idxes + [range(len(a.nsplits[-2])), range(len(b.nsplits[-1]))]

        out_chunks = []
        for out_idx in itertools.product(*out_idxes):
            chunks = []
            get_s = lambda x, idx: x[idx] if x != (1,) else x[0]
            shape = tuple(max(get_s(a_s, j), get_s(b_s, j))
                          for a_s, b_s, j in zip(a.nsplits[:-2], b.nsplits[:-2], out_idx[:-2])) + \
                (get_s(a.nsplits[-2], out_idx[-2]), get_s(b.nsplits[-1], out_idx[-1]))

            for contract_idx in range(len(a.nsplits[-1])):
                a_idx = get_idx(a, out_idx[: a.ndim - 1] + (contract_idx,))
                a_chunk = a.cix[a_idx]
                b_idx = get_idx(b, out_idx[: b.ndim - 2] + (contract_idx,) + out_idx[-1:])
                b_chunk = b.cix[b_idx]
                chunk_op = op.copy().reset_key()
                c = chunk_op.new_chunk([a_chunk, b_chunk], shape=shape, order=tensor.order)
                chunks.append(c)

            if len(chunks) == 1:
                c = chunks[0]
                out_chunk_op = c.op.copy()
                out_chunk = out_chunk_op.new_chunk(out_chunk_op.inputs, shape=c.shape,
                                                   index=out_idx, order=tensor.order)
            else:
                out_chunk = chunk_tree_add(tensor.op.dtype, chunks, out_idx, shape, sparse=tensor.op.sparse)

            out_chunks.append(out_chunk)

        nsplits = tuple(get_nsplit(i) for i in range(a.ndim - 2)) + (a.nsplits[-2], b.nsplits[-1])
        new_op = op.copy()
        return new_op.new_tensors([a, b], tensor.shape, order=tensor.order,
                                  chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        (a, b), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            if not op.sparse and is_sparse_module(xp):
                # tell sparse to do calculation on numpy or cupy matmul
                ctx[op.outputs[0].key] = xp.matmul(a, b, sparse=False)
            else:
                try:
                    # `np.matmul` support `order` argument in version 1.16
                    ctx[op.outputs[0].key] = xp.matmul(a, b, casting=op.casting, order=op.order)
                except TypeError:  # pragma: no cover
                    ctx[op.outputs[0].key] = xp.matmul(a, b).astype(dtype=op.dtype,
                                                                    casting=op.casting, order=op.order)


def matmul(a, b, sparse=None, out=None, **kw):
    """
    Matrix product of two tensors.

    The behavior depends on the arguments in the following way.

    - If both arguments are 2-D they are multiplied like conventional
      matrices.
    - If either argument is N-D, N > 2, it is treated as a stack of
      matrices residing in the last two indexes and broadcast accordingly.
    - If the first argument is 1-D, it is promoted to a matrix by
      prepending a 1 to its dimensions. After matrix multiplication
      the prepended 1 is removed.
    - If the second argument is 1-D, it is promoted to a matrix by
      appending a 1 to its dimensions. After matrix multiplication
      the appended 1 is removed.

    Multiplication by a scalar is not allowed, use ``*`` instead. Note that
    multiplying a stack of matrices with a vector will result in a stack of
    vectors, but matmul will not recognize it as such.

    ``matmul`` differs from ``dot`` in two important ways.

    - Multiplication by scalars is not allowed.
    - Stacks of matrices are broadcast together as if the matrices
      were elements.

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.
    out : Tensor, optional
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right type,
        and its dtype must be the dtype that would be returned
        for `dot(a,b)`. This is a performance feature. Therefore, if these
        conditions are not met, an exception is raised, instead of attempting
        to be flexible.

    Returns
    -------
    output : Tensor
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        1-D arrays then a scalar is returned; otherwise an array is
        returned.  If `out` is given, then it is returned.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

        If scalar value is passed.

    See Also
    --------
    vdot : Complex-conjugating dot product.
    tensordot : Sum products over arbitrary axes.
    dot : alternative matrix product with different broadcasting rules.

    Notes
    -----
    The matmul function implements the semantics of the `@` operator introduced
    in Python 3.5 following PEP465.

    Examples
    --------
    For 2-D arrays it is the matrix product:

    >>> import mars.tensor as mt

    >>> a = [[1, 0], [0, 1]]
    >>> b = [[4, 1], [2, 2]]
    >>> mt.matmul(a, b).execute()
    array([[4, 1],
           [2, 2]])

    For 2-D mixed with 1-D, the result is the usual.

    >>> a = [[1, 0], [0, 1]]
    >>> b = [1, 2]
    >>> mt.matmul(a, b).execute()
    array([1, 2])
    >>> mt.matmul(b, a).execute()
    array([1, 2])


    Broadcasting is conventional for stacks of arrays

    >>> a = mt.arange(2*2*4).reshape((2,2,4))
    >>> b = mt.arange(2*2*4).reshape((2,4,2))
    >>> mt.matmul(a,b).shape
    (2, 2, 2)
    >>> mt.matmul(a,b)[0,1,1].execute()
    98
    >>> mt.sum(a[0,1,:] * b[0,:,1]).execute()
    98

    Vector, vector returns the scalar inner product, but neither argument
    is complex-conjugated:

    >>> mt.matmul([2j, 3j], [2j, 3j]).execute()
    (-13+0j)

    Scalar multiplication raises an error.

    >>> mt.matmul([1,2], 3)
    Traceback (most recent call last):
    ...
    ValueError: Scalar operands are not allowed, use '*' instead
    """
    a = astensor(a)
    b = astensor(b)

    sparse = sparse if sparse is not None else a.issparse() and b.issparse()
    op = TensorMatmul(dtype=np.promote_types(a.dtype, b.dtype), sparse=sparse, **kw)
    return op(a, b, out=out)
