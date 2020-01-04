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
from ...serialize import KeyField, AnyField, BoolField
from ...lib.sparse.core import get_array_module
from ...utils import check_chunks_unknown_shape
from ...tiles import TilesError
from ..operands import TensorHasInput, TensorOperandMixin, Tensor
from ..datasource import tensor as astensor
from ..array_utils import as_same_device, device
from ..core import TensorOrder


class TensorDigitize(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.DIGITIZE

    _input = KeyField('input')
    _bins = AnyField('bins')
    _right = BoolField('right')

    def __init__(self, right=False, dtype=None, **kw):
        super().__init__(_right=right, _dtype=dtype, **kw)

    @property
    def bins(self):
        return self._bins

    @property
    def right(self):
        return self._right

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if len(inputs) > 1:
            self._bins = self._inputs[1]

    def __call__(self, x, bins):
        x = astensor(x)
        inputs = [x]
        if not isinstance(bins, Tensor):
            bins = get_array_module(bins).asarray(bins)
            self._bins = bins
        else:
            inputs.append(bins)
        self._dtype = np.digitize([0], np.empty(1, dtype=bins.dtype), right=self._right).dtype

        return self.new_tensor(inputs, x.shape, order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]
        in_tensor = op.input
        bins = op.bins
        if len(op.inputs) == 2:
            # bins is TensorData
            check_chunks_unknown_shape([bins], TilesError)
            bins = bins.rechunk(tensor.shape)._inplace_tile().chunks[0]

        out_chunks = []
        for c in in_tensor.chunks:
            input_chunks = [c]
            if len(op.inputs) == 2:
                input_chunks.append(bins)
            out_chunk = op.copy().reset_key().new_chunk(input_chunks, shape=c.shape,
                                                        index=c.index, order=tensor.order)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, tensor.shape, order=tensor.order,
                                  chunks=out_chunks, nsplits=in_tensor.nsplits)

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        x = inputs[0]
        if len(inputs) > 1:
            bins = inputs[1]
        else:
            bins = op.bins

        with device(device_id):
            ctx[op.outputs[0].key] = xp.digitize(x, bins=bins, right=op.right)


def digitize(x, bins, right=False):
    """
    Return the indices of the bins to which each value in input tensor belongs.

    Each index ``i`` returned is such that ``bins[i-1] <= x < bins[i]`` if
    `bins` is monotonically increasing, or ``bins[i-1] > x >= bins[i]`` if
    `bins` is monotonically decreasing. If values in `x` are beyond the
    bounds of `bins`, 0 or ``len(bins)`` is returned as appropriate. If right
    is True, then the right bin is closed so that the index ``i`` is such
    that ``bins[i-1] < x <= bins[i]`` or ``bins[i-1] >= x > bins[i]`` if `bins`
    is monotonically increasing or decreasing, respectively.

    Parameters
    ----------
    x : array_like
        Input tensor to be binned.
    bins : array_like
        Array of bins. It has to be 1-dimensional and monotonic.
    right : bool, optional
        Indicating whether the intervals include the right or the left bin
        edge. Default behavior is (right==False) indicating that the interval
        does not include the right edge. The left bin end is open in this
        case, i.e., bins[i-1] <= x < bins[i] is the default behavior for
        monotonically increasing bins.

    Returns
    -------
    out : Tensor of ints
        Output tensor of indices, of same shape as `x`.

    Raises
    ------
    ValueError
        If `bins` is not monotonic.
    TypeError
        If the type of the input is complex.

    See Also
    --------
    bincount, histogram, unique, searchsorted

    Notes
    -----
    If values in `x` are such that they fall outside the bin range,
    attempting to index `bins` with the indices that `digitize` returns
    will result in an IndexError.

    `mt.digitize` is  implemented in terms of `mt.searchsorted`. This means
    that a binary search is used to bin the values, which scales much better
    for larger number of bins than the previous linear search. It also removes
    the requirement for the input array to be 1-dimensional.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.array([0.2, 6.4, 3.0, 1.6])
    >>> bins = mt.array([0.0, 1.0, 2.5, 4.0, 10.0])
    >>> inds = mt.digitize(x, bins)
    >>> inds.execute()
    array([1, 4, 3, 2])

    >>> x = mt.array([1.2, 10.0, 12.4, 15.5, 20.])
    >>> bins = mt.array([0, 5, 10, 15, 20])
    >>> mt.digitize(x,bins,right=True).execute()
    array([1, 2, 3, 4, 4])
    >>> mt.digitize(x,bins,right=False).execute()
    array([1, 3, 3, 4, 5])
    """
    op = TensorDigitize(right=right)
    return op(x, bins)
