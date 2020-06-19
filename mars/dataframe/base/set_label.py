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

from ... import opcodes
from ...core import Entity, Base
from ...serialize import KeyField, Int8Field, AnyField
from ...tiles import TilesError
from ..core import DATAFRAME_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index


class DataFrameSetLabel(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.SET_LABEL

    _input = KeyField('input')
    _axis = Int8Field('axis')
    _value = AnyField('value')

    def __init__(self, axis=None, value=None, **kw):
        super().__init__(_axis=axis, _value=value, **kw)

    @property
    def input(self):
        return self._input

    @property
    def axis(self):
        return self._axis

    @property
    def value(self):
        return self._value

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, inp):
        self._object_type = inp.op.object_type
        if not isinstance(inp, DATAFRAME_TYPE) or self._axis != 1:  # pragma: no cover
            raise NotImplementedError('Only support set DataFrame columns for now')
        if isinstance(self._value, (Base, Entity)):  # pragma: no cover
            label = 'index' if self._axis == 0 else 'columns'
            raise NotImplementedError(
                'Cannot set {} with Mars object: {}'.format(label, self._value))

        params = inp.params
        new_dtypes = inp.dtypes.copy()
        try:
            new_dtypes.index = self._value
        except ValueError:
            raise ValueError('ValueError: Length mismatch: Expected axis has {} elements, '
                             'new values have {} elements'.format(len(new_dtypes), len(self._value)))
        params['dtypes'] = new_dtypes
        columns_value = parse_index(new_dtypes.index, store_data=True)
        params['columns_value'] = columns_value
        return self.new_tileable([inp], kws=[params])

    @classmethod
    def tile(cls, op: "DataFrameSetLabel"):
        inp = op.input
        out = op.outputs[0]
        value = op.value

        if op.input.ndim == 2 and op.axis == 1:
            if any(np.isnan(s) for s in inp.nsplits[1]):  # pragma: no cover
                raise TilesError('chunks has unknown shape on columns')

            nsplits_acc = [0] + np.cumsum(inp.nsplits[1]).tolist()
            out_chunks = []

            for chunk in inp.chunks:
                chunk_op = op.copy().reset_key()
                i = chunk.index[1]
                chunk_columns = chunk_op._value = value[nsplits_acc[i]: nsplits_acc[i + 1]]
                chunk_params = chunk.params
                chunk_dtypes = chunk.dtypes.copy()
                chunk_dtypes.index = chunk_columns
                chunk_params['dtypes'] = chunk_dtypes
                chunk_params['columns_value'] = parse_index(chunk_dtypes.index, store_data=True)
                out_chunks.append(chunk_op.new_chunk([chunk], kws=[chunk_params]))

            new_op = op.copy()
            params = out.params
            params['chunks'] = out_chunks
            params['nsplits'] = inp.nsplits
            return new_op.new_tileables([inp], kws=[params])
        else:  # pragma: no cover
            raise NotImplementedError

    @classmethod
    def execute(cls, ctx, op):
        inp = ctx[op.input.key]

        if inp.ndim == 2 and op.axis == 1:
            out = inp.copy()
            out.columns = op.value
            ctx[op.outputs[0].key] = out
        else:  # pragma: no cover
            raise NotImplementedError
