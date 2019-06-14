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

from numbers import Integral

import numpy as np

from ....operands import IndexSetValue
from ....core import Base, Entity
from ...core import TENSOR_TYPE
from ..core import TensorOperandMixin
from ..utils import filter_inputs
from .core import process_index, get_index_and_shape
from .getitem import TensorIndex


class TensorIndexSetValue(IndexSetValue, TensorOperandMixin):
    def __init__(self, dtype=None, sparse=False, indexes=None, value=None, **kw):
        super(TensorIndexSetValue, self).__init__(_dtype=dtype, _sparse=sparse,
                                                  _indexes=indexes, _value=value, **kw)

    def _set_inputs(self, inputs):
        super(TensorIndexSetValue, self)._set_inputs(inputs)
        inputs_iter = iter(self._inputs[1:])
        new_indexes = [next(inputs_iter) if isinstance(index, (Base, Entity)) else index
                       for index in self._indexes]
        self._indexes = new_indexes
        self._value = next(inputs_iter) if isinstance(self._value, (Base, Entity)) else self._value

    def __call__(self, a, index, value):
        inputs = filter_inputs([a] + list(index) + [value])
        self._indexes = index
        self._value = value
        return self.new_tensor(inputs, a.shape)

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]
        value = op.value
        is_value_tensor = isinstance(value, TENSOR_TYPE)

        index_tensor_op = TensorIndex(dtype=tensor.dtype, sparse=op.sparse, indexes=op.indexes)
        index_tensor_inputs = filter_inputs([op.input] + op.indexes)
        index_tensor = index_tensor_op.new_tensor(index_tensor_inputs, tensor.shape).single_tiles()

        nsplits = index_tensor.nsplits
        if any(any(np.isnan(ns) for ns in nsplit) for nsplit in nsplits):
            raise NotImplementedError

        if is_value_tensor:
            value = op.value.rechunk(nsplits).single_tiles()

        chunk_mapping = {c.op.input.index: c for c in index_tensor.chunks}
        out_chunks = []
        for chunk in op.input.chunks:
            index_chunk = chunk_mapping.get(chunk.index)
            if index_chunk is None:
                out_chunks.append(chunk)
                continue

            value_chunk = value.cix[index_chunk.index] if is_value_tensor else value
            chunk_op = TensorIndexSetValue(dtype=op.dtype, sparse=op.sparse,
                                           indexes=index_chunk.op.indexes, value=value_chunk)
            chunk_inputs = filter_inputs([chunk] + index_chunk.op.indexes + [value_chunk])
            out_chunk = chunk_op.new_chunk(chunk_inputs, shape=chunk.shape, index=chunk.index)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, tensor.shape, chunks=out_chunks, nsplits=op.input.nsplits)


def _setitem(a, item, value):
    from ..base import broadcast_to

    index = process_index(a, item)
    index, shape = get_index_and_shape(a.shape, index)

    for ix in index:
        if not isinstance(ix, (slice, Integral)):
            raise NotImplementedError('Only slice or int supported by now, got {0}'.format(type(ix)))

    if np.isscalar(value):
        value = a.dtype.type(value)
    else:
        value = broadcast_to(value, shape).astype(a.dtype)

    op = TensorIndexSetValue(dtype=a.dtype, sparse=a.issparse(), indexes=index, value=value)
    ret = op(a, index, value)
    a.data = ret.data
