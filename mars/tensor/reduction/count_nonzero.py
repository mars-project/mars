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

import numpy as np

from ... import opcodes as OperandDef
from ..array_utils import device, as_same_device
from .sum import TensorSum
from .core import TensorReduction, TensorReductionMixin


class TensorCountNonzero(TensorReduction, TensorReductionMixin):
    _op_type_ = OperandDef.COUNT_NONZERO

    def __init__(self, axis=None, dtype=np.intp, keepdims=None, combine_size=None, stage=None, **kw):
        stage = self._rewrite_stage(stage)
        super().__init__(_axis=axis, _dtype=dtype, _keepdims=keepdims,
                         _combine_size=combine_size, _stage=stage, **kw)

    @classmethod
    def execute_map(cls, ctx, op):
        (x,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], op.device, ret_extra=True)

        axis = cls.get_arg_axis(op.axis, op.inputs[0].ndim)
        keepdims = op.keepdims
        with device(device_id):
            nz = xp.count_nonzero(x, axis=axis)
            if keepdims:
                slcs = [slice(None)] * op.inputs[0].ndim
                for ax in op.axis:
                    slcs[ax] = np.newaxis
                nz = xp.asarray(nz)[tuple(slcs)]

            ctx[op.outputs[0].key] = nz

    @classmethod
    def execute_agg(cls, ctx, op):
        return TensorSum.execute_agg(ctx, op)

    @classmethod
    def execute_one_chunk(cls, ctx, op):
        a = ctx[op.inputs[0].key]
        (inp,), device_id, xp = as_same_device(
            [a], device=op.device, ret_extra=True)
        with device(device_id):
            ctx[op.outputs[0].key] = xp.count_nonzero(inp, axis=op.axis)


def count_nonzero(a, axis=None, combine_size=None):
    """
    Counts the number of non-zero values in the tensor ``a``.

    The word "non-zero" is in reference to the Python 2.x
    built-in method ``__nonzero__()`` (renamed ``__bool__()``
    in Python 3.x) of Python objects that tests an object's
    "truthfulness". For example, any number is considered
    truthful if it is nonzero, whereas any string is considered
    truthful if it is not the empty string. Thus, this function
    (recursively) counts how many elements in ``a`` (and in
    sub-tensors thereof) have their ``__nonzero__()`` or ``__bool__()``
    method evaluated to ``True``.

    Parameters
    ----------
    a : array_like
        The tensor for which to count non-zeros.
    axis : int or tuple, optional
        Axis or tuple of axes along which to count non-zeros.
        Default is None, meaning that non-zeros will be counted
        along a flattened version of ``a``.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    count : int or tensor of int
        Number of non-zero values in the array along a given axis.
        Otherwise, the total number of non-zero values in the tensor
        is returned.

    See Also
    --------
    nonzero : Return the coordinates of all the non-zero values.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.count_nonzero(mt.eye(4)).execute()
    4
    >>> mt.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]]).execute()
    5
    >>> mt.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]], axis=0).execute()
    array([1, 1, 1, 1, 1])
    >>> mt.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]], axis=1).execute()
    array([2, 3])

    """
    op = TensorCountNonzero(axis=axis, dtype=np.dtype(np.int_),
                            keepdims=None, combine_size=combine_size)
    return op(a)
