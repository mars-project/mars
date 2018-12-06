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

from ....operands import Astype
from ..core import TensorOperandMixin


class TensorAstype(Astype, TensorOperandMixin):
    def __init__(self, dtype=None, casting=None, sparse=False, **kw):
        super(TensorAstype, self).__init__(_dtype=dtype, _casting=casting,
                                           _sparse=sparse, **kw)

    def _set_inputs(self, inputs):
        super(TensorAstype, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, tensor, copy=True):
        t = self.new_tensor([tensor], tensor.shape)
        if copy:
            return t
        tensor.data = t.data
        return tensor

    @classmethod
    def tile(cls, op):
        in_tensor = op.input

        out_chunks = []
        for c in in_tensor.chunks:
            chunk_op = op.copy().reset_key()
            chunk = chunk_op.new_chunk([c], c.shape, index=c.index)
            out_chunks.append(chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape, nsplits=in_tensor.nsplits,
                                  chunks=out_chunks)


def _astype(tensor, dtype, casting='unsafe', copy=True):
    """
    Copy of the tensor, cast to a specified type.

    Parameters
    ----------
    dtype : str or dtype
        Typecode or data-type to which the array is cast.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'unsafe'
        for backwards compatibility.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.
    copy : bool, optional
        By default, astype always returns a newly allocated array. If this
        is set to false, and the `dtype`, `order`, and `subok`
        requirements are satisfied, the input array is returned instead
        of a copy.

    Returns
    -------
    arr_t : Tensor
        Unless `copy` is False and the other conditions for returning the input
        array are satisfied (see description for `copy` input parameter), `arr_t`
        is a new tensor of the same shape as the input array, with dtype, order
        given by `dtype`, `order`.

    Notes
    -----
    astype method returns an error if the string
    dtype to cast to is not long enough in 'safe' casting mode to hold the max
    value of integer/float array that is being casted. Previously the casting
    was allowed even if the result was truncated.

    Raises
    ------
    ComplexWarning
        When casting from complex to float or int. To avoid this,
        one should use ``a.real.astype(t)``.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.array([1, 2, 2.5])
    >>> x.execute()
    array([ 1. ,  2. ,  2.5])

    >>> x.astype(int).execute()
    array([1, 2, 2])
    """
    dtype = np.dtype(dtype)
    if tensor.dtype == dtype:
        return tensor
    elif not np.can_cast(tensor.dtype, dtype, casting=casting):
        raise TypeError('Cannot cast array from {0!r} to {1!r} '
                        'according to the rule {2!s}'.format(
            tensor.dtype, dtype, casting))

    op = TensorAstype(dtype=dtype, casting=casting, sparse=tensor.issparse())
    return op(tensor, copy=copy)
