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
from ...serialize import KeyField, BoolField
from ...utils import check_chunks_unknown_shape
from ...tiles import TilesError
from ..operands import TensorOperand, TensorOperandMixin
from ..datasource import tensor as astensor
from ..array_utils import as_same_device, device
from ..core import TensorOrder
from .ravel import ravel


class TensorIsIn(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.ISIN

    _element = KeyField('element')
    _test_elements = KeyField('test_elements')
    _assume_unique = BoolField('assume_unique')
    _invert = BoolField('invert')

    def __init__(self, assume_unique=None, invert=None, dtype=None, **kw):
        dtype = np.dtype(bool) if dtype is None else dtype
        super().__init__(_assume_unique=assume_unique, _invert=invert, _dtype=dtype, **kw)

    @property
    def element(self):
        return self._element

    @property
    def test_elements(self):
        return self._test_elements

    @property
    def assume_unique(self):
        return self._assume_unique

    @property
    def invert(self):
        return self._invert

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._element = self._inputs[0]
        self._test_elements = self._inputs[1]

    def __call__(self, element, test_elements):
        element, test_elements = astensor(element), ravel(astensor(test_elements))

        return self.new_tensor([element, test_elements], element.shape, order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op):
        in_tensor = op.element
        test_elements = op.test_elements
        out_tensor = op.outputs[0]

        if len(test_elements.chunks) != 1:
            check_chunks_unknown_shape([test_elements], TilesError)
            test_elements = test_elements.rechunk(len(test_elements))._inplace_tile()
        test_elements_chunk = test_elements.chunks[0]

        out_chunks = []
        for c in in_tensor.chunks:
            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk([c, test_elements_chunk], shape=c.shape,
                                           index=c.index, order=out_tensor.order)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors([in_tensor, test_elements], out_tensor.shape,
                                  order=out_tensor.order, chunks=out_chunks,
                                  nsplits=in_tensor.nsplits)

    @classmethod
    def execute(cls, ctx, op):
        (element, test_elements), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            ctx[op.outputs[0].key] = xp.isin(element, test_elements,
                                             assume_unique=op.assume_unique,
                                             invert=op.invert)


def isin(element, test_elements, assume_unique=False, invert=False):
    """
    Calculates `element in test_elements`, broadcasting over `element` only.
    Returns a boolean array of the same shape as `element` that is True
    where an element of `element` is in `test_elements` and False otherwise.

    Parameters
    ----------
    element : array_like
        Input tensor.
    test_elements : array_like
        The values against which to test each value of `element`.
        This argument is flattened if it is a tensor or array_like.
        See notes for behavior with non-array-like parameters.
    assume_unique : bool, optional
        If True, the input tensors are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    invert : bool, optional
        If True, the values in the returned tensor are inverted, as if
        calculating `element not in test_elements`. Default is False.
        ``mt.isin(a, b, invert=True)`` is equivalent to (but faster
        than) ``mt.invert(mt.isin(a, b))``.

    Returns
    -------
    isin : Tensor, bool
        Has the same shape as `element`. The values `element[isin]`
        are in `test_elements`.

    See Also
    --------
    in1d                  : Flattened version of this function.

    Notes
    -----

    `isin` is an element-wise function version of the python keyword `in`.
    ``isin(a, b)`` is roughly equivalent to
    ``mt.array([item in b for item in a])`` if `a` and `b` are 1-D sequences.

    `element` and `test_elements` are converted to tensors if they are not
    already. If `test_elements` is a set (or other non-sequence collection)
    it will be converted to an object tensor with one element, rather than a
    tensor of the values contained in `test_elements`. This is a consequence
    of the `tensor` constructor's way of handling non-sequence collections.
    Converting the set to a list usually gives the desired behavior.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> element = 2*mt.arange(4).reshape((2, 2))
    >>> element.execute()
    array([[0, 2],
           [4, 6]])
    >>> test_elements = [1, 2, 4, 8]
    >>> mask = mt.isin(element, test_elements)
    >>> mask.execute()
    array([[ False,  True],
           [ True,  False]])
    >>> element[mask].execute()
    array([2, 4])
    >>> mask = mt.isin(element, test_elements, invert=True)
    >>> mask.execute()
    array([[ True, False],
           [ False, True]])
    >>> element[mask]
    array([0, 6])

    Because of how `array` handles sets, the following does not
    work as expected:

    >>> test_set = {1, 2, 4, 8}
    >>> mt.isin(element, test_set).execute()
    array([[ False, False],
           [ False, False]])

    Casting the set to a list gives the expected result:

    >>> mt.isin(element, list(test_set)).execute()
    array([[ False,  True],
           [ True,  False]])
    """
    op = TensorIsIn(assume_unique, invert)
    return op(element, test_elements)
