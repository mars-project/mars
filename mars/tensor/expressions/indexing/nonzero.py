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

from ....operands import Nonzero
from ...core import ExecutableTuple
from ..utils import recursive_tile
from ..core import TensorOperandMixin
from ..datasource import tensor as astensor
from .unravel_index import unravel_index


class TensorNonzero(Nonzero, TensorOperandMixin):
    def __init__(self, dtype=None, **kw):
        super(TensorNonzero, self).__init__(_dtype=dtype, **kw)

    def _set_inputs(self, inputs):
        super(TensorNonzero, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, a):
        return ExecutableTuple(self.new_tensors([a], shape=(np.nan,), output_limit=a.ndim))

    @classmethod
    def tile(cls, op):
        from ..datasource import arange

        in_tensor = op.input

        flattened = in_tensor.astype(bool).flatten()
        recursive_tile(flattened)
        indices = arange(flattened.size, dtype=np.intp, chunk_size=flattened.nsplits)
        indices = indices[flattened]
        dim_indices = unravel_index(indices, in_tensor.shape)
        [recursive_tile(ind) for ind in dim_indices]

        shapes = [t.shape for t in op.outputs]
        kws = [{'nsplits': ind.nsplits, 'chunks': ind.chunks} for ind in dim_indices]
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shapes, kws=kws, output_limit=len(shapes))


def nonzero(a):
    """
    Return the indices of the elements that are non-zero.

    Returns a tuple of tensors, one for each dimension of `a`,
    containing the indices of the non-zero elements in that
    dimension. The values in `a` are always tested and returned.
    The corresponding non-zero
    values can be obtained with::

        a[nonzero(a)]

    To group the indices by element, rather than dimension, use::

        transpose(nonzero(a))

    The result of this is always a 2-D array, with a row for
    each non-zero element.

    Parameters
    ----------
    a : array_like
        Input tensor.

    Returns
    -------
    tuple_of_arrays : tuple
        Indices of elements that are non-zero.

    See Also
    --------
    flatnonzero :
        Return indices that are non-zero in the flattened version of the input
        tensor.
    Tensor.nonzero :
        Equivalent tensor method.
    count_nonzero :
        Counts the number of non-zero elements in the input tensor.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.session import new_session

    >>> sess = new_session().as_default()

    >>> x = mt.array([[1,0,0], [0,2,0], [1,1,0]])
    >>> x.execute()
    array([[1, 0, 0],
           [0, 2, 0],
           [1, 1, 0]])
    >>> sess.run(mt.nonzero(x))
    (array([0, 1, 2, 2]), array([0, 1, 0, 1]))

    >>> x[mt.nonzero(x)].execute()  # TODO(jisheng): accomplish this after fancy indexing is supported

    >>> mt.transpose(mt.nonzero(x)).execute() # TODO(jisheng): accomplish this later

    A common use for ``nonzero`` is to find the indices of an array, where
    a condition is True.  Given an array `a`, the condition `a` > 3 is a
    boolean array and since False is interpreted as 0, np.nonzero(a > 3)
    yields the indices of the `a` where the condition is true.

    >>> a = mt.array([[1,2,3],[4,5,6],[7,8,9]])
    >>> (a > 3).execute()
    array([[False, False, False],
           [ True,  True,  True],
           [ True,  True,  True]])
    >>> sess.run(mt.nonzero(a > 3))
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    The ``nonzero`` method of the boolean array can also be called.

    >>> sess.run((a > 3).nonzero())
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    """
    a = astensor(a)
    op = TensorNonzero(dtype=np.dtype(np.intp))
    return op(a)
