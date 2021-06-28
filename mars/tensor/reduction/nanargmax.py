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

from ... import opcodes as OperandDef
from ...serialization.serializables import AnyField, TupleField
from .core import TensorReduction, TensorArgReductionMixin


class TensorNanArgmax(TensorReduction, TensorArgReductionMixin):
    _op_type_ = OperandDef.NANARGMAX
    _func_name = 'nanargmax'
    _agg_func_name = 'nanmax'

    _offset = AnyField('offset')
    _total_shape = TupleField('total_shape')

    def __init__(self, axis=None, dtype=None, combine_size=None,
                 offset=None, total_shape=None, stage=None, **kw):
        if dtype is None:
            dtype = np.dtype(int)
        stage = self._rewrite_stage(stage)
        super().__init__(_axis=axis, _combine_size=combine_size,
                         _offset=offset, _total_shape=total_shape,
                         dtype=dtype, stage=stage, **kw)

    @property
    def offset(self):
        return getattr(self, '_offset', None)

    @property
    def total_shape(self):
        return getattr(self, '_total_shape', None)


def nanargmax(a, axis=None, out=None, combine_size=None):
    """
    Return the indices of the maximum values in the specified axis ignoring
    NaNs. For all-NaN slices ``ValueError`` is raised. Warning: the
    results cannot be trusted if a slice contains only NaNs and -Infs.


    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which to operate.  By default flattened input is used.
    out : Tensor, optional
        Alternate output tensor in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See `doc.ufuncs` for details.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    index_array : Tensor
        An tensor of indices or a single index value.

    See Also
    --------
    argmax, nanargmin

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([[mt.nan, 4], [2, 3]])
    >>> mt.argmax(a).execute()
    0
    >>> mt.nanargmax(a).execute()
    1
    >>> mt.nanargmax(a, axis=0).execute()
    array([1, 0])
    >>> mt.nanargmax(a, axis=1).execute()
    array([1, 1])

    """
    op = TensorNanArgmax(axis=axis, dtype=np.dtype(int), combine_size=combine_size)
    return op(a, out=out)
