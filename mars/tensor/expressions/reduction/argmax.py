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


class TensorArgmaxChunk(operands.ArgmaxChunk, TensorArgReduction):
    def __init__(self, axis=None, dtype=np.dtype(int), keepdims=None,
                 combine_size=None, offset=None, total_shape=None, **kw):
        super(TensorArgmaxChunk, self).__init__(_axis=axis, _dtype=dtype,
                                                _keepdims=keepdims, _combine_size=combine_size,
                                                _offset=offset, _total_shape=total_shape, **kw)


class TensorArgmaxCombine(operands.ArgmaxCombine, TensorArgReduction):
    def __init__(self, axis=None, dtype=np.dtype(int), keepdims=None, combine_size=None, **kw):
        super(TensorArgmaxCombine, self).__init__(_axis=axis, _dtype=dtype, _keepdims=keepdims,
                                                  _combine_size=combine_size, **kw)


class TensorArgmax(operands.Argmax, TensorArgReduction):
    def __init__(self, axis=None, dtype=np.dtype(int), keepdims=None, combine_size=None, **kw):
        super(TensorArgmax, self).__init__(_axis=axis, _dtype=dtype, _keepdims=keepdims,
                                           _combine_size=combine_size, **kw)

    @staticmethod
    def _get_op_types():
        return TensorArgmaxChunk, TensorArgmax, TensorArgmaxCombine


def argmax(a, axis=None, out=None, keepdims=None, combine_size=None):
    """
    Returns the indices of the maximum values along an axis.

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
    Tensor.argmax, argmin
    amax : The maximum value along a given axis.
    unravel_index : Convert a flat index into an index tuple.

    Notes
    -----
    In case of multiple occurrences of the maximum values, the indices
    corresponding to the first occurrence are returned.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.session import new_session

    >>> sess = new_session().as_default()

    >>> a = mt.arange(6).reshape(2,3)
    >>> a.execute()
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mt.argmax(a).execute()
    5
    >>> mt.argmax(a, axis=0).execute()
    array([1, 1, 1])
    >>> mt.argmax(a, axis=1).execute()
    array([2, 2])

    Indexes of the maximal elements of a N-dimensional tensor:

    >>> ind = mt.unravel_index(mt.argmax(a, axis=None), a.shape)
    >>> sess.run(ind)
    (1, 2)
    >>> a[ind].execute()  # TODO(jisheng): accomplish when fancy index on tensor is supported

    >>> b = mt.arange(6)
    >>> b[1] = 5
    >>> b.execute()
    array([0, 5, 2, 3, 4, 5])
    >>> mt.argmax(b).execute()  # Only the first occurrence is returned.
    1

    """
    op = TensorArgmax(axis=axis, dtype=np.dtype(int), keepdims=keepdims, combine_size=combine_size)
    return op(a, out=out)
