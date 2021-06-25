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

import pandas as pd

from ...config import options
from ...core import recursive_tile
from ...core.operand import OperandStage
from ...serialization.serializables import FieldTypes, Int32Field, Int64Field, \
    StringField, ListField, BoolField
from ...utils import ceildiv
from ..operands import DataFrameOperand
from ..utils import parse_index


class DataFrameSortOperand(DataFrameOperand):
    _axis = Int32Field('axis')
    _ascending = BoolField('ascending')
    _inplace = BoolField('inplace')
    _kind = StringField('kind')
    _na_position = StringField('na_position')
    _ignore_index = BoolField('ignore_index')
    _parallel_kind = StringField('parallel_kind')
    _psrs_kinds = ListField('psrs_kinds', FieldTypes.string)
    _nrows = Int64Field('nrows')

    def __init__(self, axis=None, ascending=None, inplace=None, kind=None, na_position=None,
                 ignore_index=None, parallel_kind=None, psrs_kinds=None,
                 nrows=None, gpu=False, **kw):
        super().__init__(_axis=axis, _ascending=ascending, _inplace=inplace, _kind=kind,
                         _na_position=na_position, _ignore_index=ignore_index,
                         _parallel_kind=parallel_kind, _psrs_kinds=psrs_kinds,
                         _nrows=nrows, _gpu=gpu, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def ascending(self):
        return self._ascending

    @property
    def inplace(self):
        return self._inplace

    @property
    def kind(self):
        return self._kind

    @property
    def na_position(self):
        return self._na_position

    @property
    def ignore_index(self):
        return self._ignore_index

    @property
    def parallel_kind(self):
        return self._parallel_kind

    @property
    def psrs_kinds(self):
        return self._psrs_kinds

    @property
    def nrows(self):
        return self._nrows

    @classmethod
    def _tile_head(cls, op: "DataFrameSortOperand"):
        from ..merge import DataFrameConcat

        inp = op.inputs[0]
        out = op.outputs[0]
        axis = op.axis
        assert axis == 0
        pd_index = out.index_value.to_pandas()
        combine_size = options.combine_size

        if inp.ndim == 2:
            if inp.chunk_shape[1 - axis] > 1:  # pragma: no cover
                if any(pd.isna(s) for s in inp.nsplits[1 - axis]):
                    yield
                inp = yield from recursive_tile(inp.rechunk({1 - axis: inp.shape[1 - axis]}))

        out_chunks = []
        for c in inp.chunks:
            chunk_op = op.copy().reset_key()
            chunk_op.stage = OperandStage.map
            chunk_params = c.params
            chunk_params['index_value'] = parse_index(pd_index, c)
            out_chunks.append(chunk_op.new_chunk([c], kws=[chunk_params]))

        while True:
            chunk_size = ceildiv(len(out_chunks), combine_size)
            combine_chunks = []
            for i in range(chunk_size):
                chunk_index = (i,) if inp.ndim == 1 else (i, 0)

                to_combine_chunks = out_chunks[i * combine_size: (i + 1) * combine_size]
                concat_params = to_combine_chunks[0].params
                concat_params['index'] = chunk_index
                shape = list(to_combine_chunks[0].shape)
                shape[0] = sum(c.shape[0] for c in to_combine_chunks)
                shape = tuple(shape)
                concat_params['shape'] = shape
                c = DataFrameConcat(axis=axis, output_types=op.output_types).new_chunk(
                    to_combine_chunks, kws=[concat_params])

                chunk_op = op.copy().reset_key()
                chunk_op.stage = OperandStage.combine \
                    if chunk_size > 1 else OperandStage.agg
                chunk_params = c.params
                chunk_params['index_value'] = parse_index(pd_index, c)
                chunk_params['shape'] = (min(shape[0], op.nrows),) + shape[1:]
                combine_chunks.append(chunk_op.new_chunk([c], kws=[chunk_params]))
            out_chunks = combine_chunks
            if chunk_size == 1:
                break

        new_op = op.copy()
        params = out.params
        params['nsplits'] = tuple((s,) for s in out.shape)
        params['chunks'] = out_chunks
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def _tile(cls, op):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def tile(cls, op: "DataFrameSortOperand"):
        if op.nrows is not None:
            return (yield from cls._tile_head(op))
        else:
            return (yield from cls._tile(op))
