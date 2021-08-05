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

from typing import Union

import numpy as np

from ... import opcodes as OperandDef
from ...serialization.serializables import BoolField
from ...typing import TileableType
from ..operands import TensorOperand, TensorOperandMixin
from ..datasource import tensor as astensor
from ..array_utils import as_same_device, device
from ..core import TensorOrder


class TensorIsIn(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.ISIN

    assume_unique = BoolField('assume_unique')
    invert = BoolField('invert')

    def __call__(self, element, test_elements):
        self.dtype = np.dtype(bool)
        return self.new_tensor([element, test_elements],
                               shape=element.shape, order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op):
        from ..merge.stack import TensorStack
        from ..reduction import TensorAll, TensorAny

        ar1, ar2 = op.inputs
        invert = op.invert
        out = op.outputs[0]

        out_chunks = []
        for ar1_chunk in ar1.chunks:
            to_concat_chunks = []
            for ar2_chunk in ar2.chunks:
                chunk_op = op.copy().reset_key()
                out_chunk = chunk_op.new_chunk([ar1_chunk, ar2_chunk], dtype=out.dtype,
                                               shape=ar1_chunk.shape, order=out.order,
                                               index=ar1_chunk.index)
                to_concat_chunks.append(out_chunk)
            if len(to_concat_chunks) == 1:
                out_chunks.append(to_concat_chunks[0])
            else:
                # concat chunks
                concat_op = TensorStack(axis=0)
                shape = (len(to_concat_chunks),) + ar1_chunk.shape
                concat_chunk = concat_op.new_chunk(
                    to_concat_chunks, shape=shape,
                    dtype=out.dtype, order=out.order)
                if not invert:
                    chunk_op = TensorAny(axis=(0,), dtype=out.dtype)
                    out_chunk = chunk_op.new_chunk(
                        [concat_chunk], shape=ar1_chunk.shape,
                        dtype=out.dtype, order=out.order,
                        index=ar1_chunk.index)
                else:
                    chunk_op = TensorAll(axis=(0,), dtype=out.dtype)
                    out_chunk = chunk_op.new_chunk(
                        [concat_chunk], shape=ar1_chunk.shape,
                        dtype=out.dtype, order=out.order,
                        index=ar1_chunk.index)
                out_chunks.append(out_chunk)

        params = out.params.copy()
        params['nsplits'] = ar1.nsplits
        params['chunks'] = out_chunks
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        (element, test_elements), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            ctx[op.outputs[0].key] = xp.isin(element, test_elements,
                                             assume_unique=op.assume_unique,
                                             invert=op.invert)


def isin(element: Union[TileableType, np.ndarray],
         test_elements: Union[TileableType, np.ndarray, list],
         assume_unique: bool = False,
         invert: bool = False):
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
    element, test_elements = astensor(element), astensor(test_elements).ravel()
    op = TensorIsIn(assume_unique=assume_unique, invert=invert)
    return op(element, test_elements)
