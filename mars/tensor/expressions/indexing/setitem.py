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
import contextlib

import numpy as np

from ....operands import IndexSetValue
from ....core import BaseWithKey, Entity
from ...core import TENSOR_TYPE, CHUNK_TYPE
from ..core import TensorOperandMixin
from .core import process_index, get_index_and_shape
from .getitem import TensorIndex


class TensorIndexSetValue(IndexSetValue, TensorOperandMixin):
    def __init__(self, dtype=None, sparse=False, **kw):
        super(TensorIndexSetValue, self).__init__(_dtype=dtype, _sparse=sparse, **kw)

    @contextlib.contextmanager
    def _handle_params(self, inputs, indexes, value):
        """
        TensorIndexSetValue operator is like Index operand, it has additional parameter `indexes` and `value`, all of
        them may be tensor type. As explained in TensorIndex, when indexes and value are not provided, we should get
        from operand itself and replace tensor-liked objects by iterating over inputs.
        """
        if indexes is not None and value is not None:
            self._indexes = indexes
            self._value = value

            indexes_inputs = [ind for ind in indexes if isinstance(ind, TENSOR_TYPE + CHUNK_TYPE)]
            inputs += indexes_inputs
            if isinstance(value, TENSOR_TYPE + CHUNK_TYPE):
                inputs += [value]
        yield inputs

        inputs_iter = iter(self._inputs[1:])
        new_indexes = [next(inputs_iter) if isinstance(index, (BaseWithKey, Entity)) else index
                       for index in self._indexes]
        self._indexes = new_indexes
        if isinstance(self._value, (BaseWithKey, Entity)):
            self._value = next(inputs_iter)

    def new_tensors(self, inputs, shape, **kw):
        indexes = kw.pop('indexes', None)
        value = kw.pop('value', None)
        with self._handle_params(inputs, indexes, value) as mix_inputs:
            return super(TensorIndexSetValue, self).new_tensors(mix_inputs, shape, **kw)

    def new_chunks(self, inputs, shape, **kw):
        indexes = kw.pop('indexes', None)
        value = kw.pop('value', None)
        with self._handle_params(inputs, indexes, value) as mix_inputs:
            return super(TensorIndexSetValue, self).new_chunks(mix_inputs, shape, **kw)

    def calc_shape(self, *inputs_shape):
        return inputs_shape[0]

    def __call__(self, a, index, value):
        return self.new_tensor([a], a.shape, indexes=index, value=value)

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]
        value = op.value
        is_value_tensor = isinstance(value, TENSOR_TYPE)

        index_tensor_op = TensorIndex(dtype=tensor.dtype, sparse=op.sparse)
        index_tensor = index_tensor_op.new_tensor([op.input], tensor.shape, indexes=op.indexes).single_tiles()

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
            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk([chunk], chunk.shape, indexes=index_chunk.op.indexes,
                                           value=value_chunk, index=chunk.index)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors([op.input], tensor.shape, indexes=op.indexes, value=op.value,
                                  chunks=out_chunks, nsplits=op.input.nsplits)


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

    op = TensorIndexSetValue(dtype=a.dtype, sparse=a.issparse())
    ret = op(a, index, value)
    a.data = ret.data
