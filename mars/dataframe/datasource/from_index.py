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

from ... import opcodes
from ...core import recursive_tile
from ...serialization.serializables import KeyField
from ..initializer import Index
from ..operands import DataFrameOperand, DataFrameOperandMixin


class SeriesFromIndex(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.SERIES_FROM_INDEX

    _input = KeyField('input')
    _index = KeyField('index')

    def __init__(self, input_=None, index=None, **kw):
        super().__init__(_input=input_, _index=index, **kw)

    @property
    def input(self):
        return self._input

    @property
    def index(self):
        return self._index

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if len(self._inputs) > 1:
            self._index = self._inputs[1]

    def __call__(self, index, new_index=None, name=None):
        inputs = [index]
        index_value = index.index_value
        if new_index is not None:
            inputs.append(new_index)
            index_value = new_index.index_value
        return self.new_series(inputs, shape=index.shape, dtype=index.dtype,
                               index_value=index_value, name=name)

    @classmethod
    def tile(cls, op: "SeriesFromIndex"):
        inp = op.input
        out = op.outputs[0]
        index = op.index

        if index is not None:
            index = yield from recursive_tile(
                op.index.rechunk({0: inp.nsplits[0]}))

        chunks = []
        for i, c in enumerate(inp.chunks):
            chunk_op = op.copy().reset_key()
            chunk_inputs = [c]
            chunk_index_value = c.index_value
            if index is not None:
                index_chunk = index.chunks[i]
                chunk_index_value = index_chunk.index_value
                chunk_inputs.append(index_chunk)
            chunk = chunk_op.new_chunk(chunk_inputs, shape=c.shape, dtype=c.dtype,
                                       index_value=chunk_index_value,
                                       name=out.name, index=c.index)
            chunks.append(chunk)

        new_op = op.copy()
        params = out.params
        params['chunks'] = chunks
        params['nsplits'] = inp.nsplits
        return new_op.new_tileables([inp], kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        out = op.outputs[0]
        inp = ctx[op.input.key]
        index = None
        if op.index is not None:
            index = ctx[op.index.key]
        ctx[out.key] = inp.to_series(index=index, name=out.name)


def series_from_index(ind, index=None, name=None):
    name = name or ind.name or 0
    if index is not None:
        index = Index(index)
    op = SeriesFromIndex(input_=ind, index=index)
    return op(ind, new_index=index, name=name)
