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
from ...serialize import AnyField, TupleField
from .core import TensorReduction, TensorArgReductionMixin


class TensorArgmin(TensorReduction, TensorArgReductionMixin):
    _op_type_ = OperandDef.ARGMIN
    _func_name = 'argmin'
    _agg_func_name = 'min'

    _offset = AnyField('offset')
    _total_shape = TupleField('total_shape')

    def __init__(self, axis=None, dtype=np.dtype(int), combine_size=None,
                 offset=None, total_shape=None, stage=None, **kw):
        stage = self._rewrite_stage(stage)
        super().__init__(_axis=axis, _dtype=dtype, _combine_size=combine_size,
                         _offset=offset, _total_shape=total_shape, _stage=stage, **kw)

    @property
    def offset(self):
        return getattr(self, '_offset', None)

    @property
    def total_shape(self):
        return getattr(self, '_total_shape', None)


def argmin(a, axis=None, out=None, combine_size=None):
    """
    Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    a : array_like
        Input tensor.
    axis : int, optional
        By default, the index is into the flattened tensor, otherwise
        along the specified axis.
    out : Tensor, optional
        If provided, the result will be inserted into this tensor. It should
        be of the appropriate shape and dtype.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    index_array : Tensor of ints
        Tensor of indices into the tensor. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    Tensor.argmin, argmax
    amin : The minimum value along a given axis.
    unravel_index : Convert a flat index into an index tuple.

    Notes
    -----
    In case of multiple occurrences of the minimum values, the indices
    corresponding to the first occurrence are returned.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.arange(6).reshape(2,3)
    >>> a.execute()
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mt.argmin(a).execute()
    0
    >>> mt.argmin(a, axis=0).execute()
    array([0, 0, 0])
    >>> mt.argmin(a, axis=1).execute()
    array([0, 0])

    Indices of the minimum elements of a N-dimensional tensor:

    >>> ind = mt.unravel_index(mt.argmin(a, axis=None), a.shape)
    >>> ind.execute()
    (0, 0)
    >>> a[ind]  # TODO(jisheng): accomplish when fancy index on tensor is supported

    >>> b = mt.arange(6)
    >>> b[4] = 0
    >>> b.execute()
    array([0, 1, 2, 3, 0, 5])
    >>> mt.argmin(b).execute()  # Only the first occurrence is returned.
    0

    """
    op = TensorArgmin(axis=axis, dtype=np.dtype(int), combine_size=combine_size)
    return op(a, out=out)
