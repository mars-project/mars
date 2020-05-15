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
from pandas.api.types import is_list_like

from ... import opcodes as OperandDef
from ...serialize import KeyField, AnyField
from ...tensor.core import TENSOR_TYPE
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape
from ..core import SERIES_TYPE, INDEX_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType


class DataFrameIsin(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.ISIN

    _input = KeyField('input')
    _values = AnyField('values')

    def __init__(self, values=None, object_type=None, **kw):
        if object_type is None:
            object_type = ObjectType.series
        super().__init__(_values=values, _object_type=object_type, **kw)

    @property
    def input(self):
        return self._input

    @property
    def values(self):
        return self._values

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if len(inputs) == 2:
            self._values = self._inputs[1]

    def __call__(self, elements):
        inputs = [elements]
        if isinstance(self._values, (SERIES_TYPE, TENSOR_TYPE, INDEX_TYPE)):
            inputs.append(self._values)
        return self.new_series(inputs, shape=elements.shape, dtype=np.dtype('bool'),
                               index_value=elements.index_value, name=elements.name)

    @classmethod
    def tile(cls, op):
        in_elements = op.input
        out_elements = op.outputs[0]

        values = op.values
        if len(op.inputs) == 2:
            # make sure arg has known shape when it's a md.Series
            check_chunks_unknown_shape([op.values], TilesError)
            values = op.values.rechunk(op.values.shape)._inplace_tile()

        out_chunks = []
        for chunk in in_elements.chunks:
            chunk_op = op.copy().reset_key()
            chunk_inputs = [chunk]
            if len(op.inputs) == 2:
                chunk_inputs.append(values.chunks[0])
            out_chunk = chunk_op.new_chunk(chunk_inputs, shape=chunk.shape,
                                           dtype=out_elements.dtype,
                                           index_value=chunk.index_value,
                                           name=out_elements.name,
                                           index=chunk.index)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_seriess(op.inputs, out_elements.shape,
                                  nsplits=in_elements.nsplits,
                                  chunks=out_chunks, dtype=out_elements.dtype,
                                  index_value=out_elements.index_value, name=out_elements.name)

    @classmethod
    def execute(cls, ctx, op):
        elements = ctx[op.inputs[0].key]
        if len(op.inputs) == 2:
            values = ctx[op.inputs[1].key]
        else:
            values = op.values
        try:
            ctx[op.outputs[0].key] = elements.isin(values)
        except ValueError:
            # buffer read-only
            ctx[op.outputs[0].key] = elements.copy().isin(values)


def isin(elements, values):
    # TODO(hetao): support more type combinations, for example, DataFrame.isin.
    if is_list_like(values):
        values = list(values)
    elif not isinstance(values, (SERIES_TYPE, TENSOR_TYPE, INDEX_TYPE)):
        raise TypeError('only list-like objects are allowed to be passed to isin(), '
                        'you passed a [{}]'.format(type(values)))
    if not isinstance(elements, SERIES_TYPE):  # pragma: no cover
        raise NotImplementedError('Unsupported parameter types: %s and %s' %
                                  (type(elements), type(values)))
    op = DataFrameIsin(values)
    return op(elements)
