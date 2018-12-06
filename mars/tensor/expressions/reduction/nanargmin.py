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

from .... import operands
from .core import TensorArgReduction


class TensorNanArgminChunk(operands.NanArgminChunk, TensorArgReduction):
    def __init__(self, axis=None, dtype=np.dtype(int), keepdims=None,
                 combine_size=None, offset=None, total_shape=None,**kw):
        super(TensorNanArgminChunk, self).__init__(_axis=axis, _dtype=dtype,
                                                   _keepdims=keepdims, _combine_size=combine_size,
                                                   _offset=offset, _total_shape=total_shape, **kw)


class TensorNanArgminCombine(operands.NanArgminCombine, TensorArgReduction):
    def __init__(self, axis=None, dtype=np.dtype(int), keepdims=None, combine_size=None, **kw):
        super(TensorNanArgminCombine, self).__init__(_axis=axis, _dtype=dtype, _keepdims=keepdims,
                                                     _combine_size=combine_size, **kw)


class TensorNanArgmin(operands.NanArgmin, TensorArgReduction):
    def __init__(self, axis=None, dtype=np.dtype(int), keepdims=None, combine_size=None, **kw):
        super(TensorNanArgmin, self).__init__(_axis=axis, _dtype=dtype, _keepdims=keepdims,
                                              _combine_size=combine_size, **kw)

    @staticmethod
    def _get_op_types():
        return TensorNanArgminChunk, TensorNanArgmin, TensorNanArgminCombine


def nanargmin(a, axis=None, out=None, keepdims=None, combine_size=None):
    """
    Return the indices of the minimum values in the specified axis ignoring
    NaNs. For all-NaN slices ``ValueError`` is raised. Warning: the results
    cannot be trusted if a slice contains only NaNs and Infs.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which to operate.  By default flattened input is used.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    index_array : Tensor
        A tensor of indices or a single index value.

    See Also
    --------
    argmin, nanargmax

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([[mt.nan, 4], [2, 3]])
    >>> mt.argmin(a).execute()
    0
    >>> mt.nanargmin(a).execute()
    2
    >>> mt.nanargmin(a, axis=0).execute()
    array([1, 1])
    >>> mt.nanargmin(a, axis=1).execute()
    array([1, 0])

    """
    op = TensorNanArgmin(axis=axis, dtype=np.dtype(int), keepdims=keepdims, combine_size=combine_size)
    return op(a, out=out)
