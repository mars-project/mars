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

from .... import operands
from ....compat import izip
from ..utils import unify_chunks
from ..core import TensorOperandMixin
from ..arithmetic.utils import tree_add
from ..datasource import tensor as astensor


class TensorTensorDot(operands.TensorDot, TensorOperandMixin):
    def __init__(self, a_axes=None, b_axes=None, dtype=None, sparse=False, **kw):
        super(TensorTensorDot, self).__init__(_a_axes=a_axes, _b_axes=b_axes,
                                              _dtype=dtype, _sparse=sparse, **kw)

    def _set_inputs(self, inputs):
        super(TensorTensorDot, self)._set_inputs(inputs)
        self._a = self._inputs[0]
        self._b = self._inputs[1]

    def calc_shape(self, *inputs_shape):
        a_shape = inputs_shape[0]
        b_shape = inputs_shape[1]
        shape = tuple(s for i, s in enumerate(a_shape) if i not in set(self._a_axes)) + \
            tuple(s for i, s in enumerate(b_shape) if i not in set(self._b_axes))
        return shape

    def __call__(self, a, b):
        shape = tuple(s for i, s in enumerate(a.shape) if i not in set(self._a_axes)) + \
            tuple(s for i, s in enumerate(b.shape) if i not in set(self._b_axes))
        return self.new_tensor([a, b], shape)

    @classmethod
    def tile(cls, op):
        a, b, a_axes, b_axes = op.a, op.b, op.a_axes, op.b_axes

        c = itertools.count(max(a.ndim, b.ndim))
        a_ax = tuple(a_axes.index(i) if i in a_axes else next(c) for i in range(a.ndim))
        b_ax = tuple(b_axes.index(i) if i in b_axes else next(c) for i in range(b.ndim))
        a, b = unify_chunks((a, a_ax), (b, b_ax))

        a_output_indexes = [range(len(a.nsplits[i])) for i in range(a.ndim) if i not in a_axes]
        b_output_indexes = [range(len(b.nsplits[i])) for i in range(b.ndim) if i not in b_axes]
        output_axes = [(0, i) for i in range(a.ndim) if i not in a_axes] + \
                      [(1, i) for i in range(b.ndim) if i not in b_axes]

        out_chunks = []
        for out_idx in itertools.product(*itertools.chain(a_output_indexes, b_output_indexes)):
            a_indexes = [None] * a.ndim
            b_indexes = [None] * b.ndim
            tensor_shape = []
            for i, idx in enumerate(out_idx):
                t_idx, axis = output_axes[i]
                t = (a, b)[t_idx]
                (a_indexes if t_idx == 0 else b_indexes)[axis] = idx
                tensor_shape.append(t.nsplits[axis][idx])
            tensor_shape = tuple(tensor_shape)

            tensordot_chunks = []
            for contract_indexes in itertools.product(*[range(len(a.nsplits[ax])) for ax in a_axes]):
                a_indices, b_indices = list(a_indexes), list(b_indexes)
                for a_axis, contract_index in izip(a_axes, contract_indexes):
                    a_indices[a_axis] = contract_index
                for b_axis, contract_index in izip(b_axes, contract_indexes):
                    b_indices[b_axis] = contract_index

                tensordot_chunk_op = op.copy().reset_key()
                tensordot_chunk = tensordot_chunk_op.new_chunk(
                    [a.cix[tuple(a_indices)], b.cix[tuple(b_indices)]], tensor_shape)
                tensordot_chunks.append(tensordot_chunk)

            if len(tensordot_chunks) == 1:
                c = tensordot_chunks[0]
                chunk_op = c.op.copy()
                chunk = chunk_op.new_chunk(c.inputs, c.shape, index=out_idx)
            else:
                chunk = tree_add(op.dtype, tensordot_chunks, out_idx, tensor_shape, sparse=op.sparse)
            out_chunks.append(chunk)

        get_nsplits = lambda t_idx, i: (a, b)[t_idx].nsplits[i]
        nsplits = [get_nsplits(*it) for it in output_axes]
        new_op = op.copy()
        return new_op.new_tensors([a, b], op.outputs[0].shape,
                                  chunks=out_chunks, nsplits=nsplits)


def tensordot(a, b, axes=2, sparse=None):
    """
    Compute tensor dot product along specified axes for tensors >= 1-D.
    Given two tensors (arrays of dimension greater than or equal to one),
    `a` and `b`, and an array_like object containing two array_like
    objects, ``(a_axes, b_axes)``, sum the products of `a`'s and `b`'s
    elements (components) over the axes specified by ``a_axes`` and
    ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N``
    dimensions of `a` and the first ``N`` dimensions of `b` are summed
    over.
    Parameters
    ----------
    a, b : array_like, len(shape) >= 1
        Tensors to "dot".
    axes : int or (2,) array_like
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order. The sizes of the corresponding axes must match.
        * (2,) array_like
          Or, a list of axes to be summed over, first sequence applying to `a`,
          second to `b`. Both elements array_like must be of the same length.
    See Also
    --------
    dot, einsum
    Notes
    -----
    Three common use cases are:
        * ``axes = 0`` : tensor product :math:`a\\otimes b`
        * ``axes = 1`` : tensor dot product :math:`a\\cdot b`
        * ``axes = 2`` : (default) tensor double contraction :math:`a:b`
    When `axes` is integer_like, the sequence for evaluation will be: first
    the -Nth axis in `a` and 0th axis in `b`, and the -1th axis in `a` and
    Nth axis in `b` last.
    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.
    Examples
    --------
    >>> import mars.tensor as mt

    A "traditional" example:
    >>> a = mt.arange(60.).reshape(3,4,5)
    >>> b = mt.arange(24.).reshape(4,3,2)
    >>> c = mt.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c.shape
    (5, 2)
    >>> r = c.execute()
    >>> r
    array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])
    >>> # A slower but equivalent way of computing the same...
    >>> ra = np.arange(60.).reshape(3,4,5)
    >>> rb = np.arange(24.).reshape(4,3,2)
    >>> d = np.zeros((5,2))
    >>> for i in range(5):
    ...   for j in range(2):
    ...     for k in range(3):
    ...       for n in range(4):
    ...         d[i,j] += ra[k,n,i] * rb[n,k,j]
    >>> r == d
    array([[ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True]], dtype=bool)
    An extended example taking advantage of the overloading of + and \\*:
    >>> a = mt.array(range(1, 9))
    >>> a.shape = (2, 2, 2)
    >>> A = mt.array(('a', 'b', 'c', 'd'), dtype=object)
    >>> A.shape = (2, 2)
    >>> a.execute(); A.execute()
    array([[[1, 2],
            [3, 4]],
           [[5, 6],
            [7, 8]]])
    array([[a, b],
           [c, d]], dtype=object)
    >>> mt.tensordot(a, A).execute() # third argument default is 2 for double-contraction
    array([abbcccdddd, aaaaabbbbbbcccccccdddddddd], dtype=object)
    >>> mt.tensordot(a, A, 1).execute()
    array([[[acc, bdd],
            [aaacccc, bbbdddd]],
           [[aaaaacccccc, bbbbbdddddd],
            [aaaaaaacccccccc, bbbbbbbdddddddd]]], dtype=object)
    >>> mt.tensordot(a, A, 0).execute() # tensor product (result too long to incl.)
    array([[[[[a, b],
              [c, d]],
              ...
    >>> mt.tensordot(a, A, (0, 1)).execute()
    array([[[abbbbb, cddddd],
            [aabbbbbb, ccdddddd]],
           [[aaabbbbbbb, cccddddddd],
            [aaaabbbbbbbb, ccccdddddddd]]], dtype=object)
    >>> mt.tensordot(a, A, (2, 1)).execute()
    array([[[abb, cdd],
            [aaabbbb, cccdddd]],
           [[aaaaabbbbbb, cccccdddddd],
            [aaaaaaabbbbbbbb, cccccccdddddddd]]], dtype=object)
    >>> mt.tensordot(a, A, ((0, 1), (0, 1))).execute()
    array([abbbcccccddddddd, aabbbbccccccdddddddd], dtype=object)
    >>> mt.tensordot(a, A, ((2, 1), (1, 0))).execute()
    array([acccbbdddd, aaaaacccccccbbbbbbdddddddd], dtype=object)
    """
    a = astensor(a)
    b = astensor(b)

    if isinstance(axes, Iterable):
        a_axes, b_axes = axes
    else:
        a_axes = tuple(range(a.ndim - 1, a.ndim - axes - 1, -1))
        b_axes = tuple(range(0, axes))

    if isinstance(a_axes, Iterable):
        a_axes = tuple(a_axes)
    else:
        a_axes = (a_axes,)
    a_axes = tuple(axis if axis >= 0 else a.ndim + axis for axis in a_axes)
    if isinstance(b_axes, Iterable):
        b_axes = tuple(b_axes)
    else:
        b_axes = (b_axes,)
    b_axes = tuple(axis if axis >= 0 else b.ndim + axis for axis in b_axes)

    if a.shape and b.shape and \
            not np.array_equal(np.array(a.shape)[list(a_axes)], np.array(b.shape)[list(b_axes)]):
        raise ValueError('shape-mismatch for sum')

    sparse = sparse if sparse is not None else a.issparse() and b.issparse()
    op = TensorTensorDot(a_axes=a_axes, b_axes=b_axes, dtype=np.promote_types(a.dtype, b.dtype),
                         sparse=sparse)
    return op(a, b)
