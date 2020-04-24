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
import pandas as pd

from ... import opcodes
from ...operands import OperandStage
from ...serialize import KeyField, BoolField, Int64Field, StringField
from ...utils import recursive_tile
from ..core import Series
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import build_series, parse_index


class DataFrameValueCounts(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.VALUE_COUNTS

    _input = KeyField('input')
    _normalize = BoolField('normalize')
    _sort = BoolField('sort')
    _ascending = BoolField('ascending')
    _bins = Int64Field('bins')
    _dropna = BoolField('dropna')
    _method = StringField('method')

    def __init__(self, normalize=None, sort=None, ascending=None,
                 bins=None, dropna=None, method=None, stage=None, **kw):
        super().__init__(_normalize=normalize, _sort=sort, _ascending=ascending,
                         _bins=bins, _dropna=dropna, _method=method,
                         _stage=stage, **kw)
        self._object_type = ObjectType.series

    @property
    def input(self):
        return self._input

    @property
    def normalize(self):
        return self._normalize

    @property
    def sort(self):
        return self._sort

    @property
    def ascending(self):
        return self._ascending

    @property
    def bins(self):
        return self._bins

    @property
    def dropna(self):
        return self._dropna

    @property
    def method(self):
        return self._method

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, inp):
        test_series = build_series(inp).value_counts(normalize=self.normalize)
        if self._bins is not None:
            from .cut import cut

            # cut
            try:
                inp = cut(inp, self._bins, include_lowest=True)
            except TypeError:  # pragma: no cover
                raise TypeError("bins argument only works with numeric data.")

            self._bins = None
            return self.new_series([inp], shape=(np.nan,),
                                   index_value=parse_index(pd.CategoricalIndex([]),
                                                           inp, store_data=False),
                                   name=inp.name, dtype=test_series.dtype)
        else:
            return self.new_series([inp], shape=(np.nan,),
                                   index_value=parse_index(test_series.index, store_data=False),
                                   name=inp.name, dtype=test_series.dtype)

    @classmethod
    def tile(cls, op):
        from .cut import DataFrameCut

        out = op.outputs[0]

        if np.prod(op.input.chunk_shape) == 1:
            inp = op.input
            chunk_op = op.copy().reset_key()
            chunk_param = out.params
            chunk_param['index'] = (0,)
            chunk = chunk_op.new_chunk(inp.chunks, kws=[chunk_param])

            new_op = op.copy()
            param = out.params
            param['chunks'] = [chunk]
            param['nsplits'] = ((np.nan,),)
            return new_op.new_seriess(op.inputs, kws=[param])

        inp = Series(op.input)

        if op.dropna:
            inp = inp.dropna()

        inp = inp.groupby(inp).count(method=op.method)

        if op.normalize:
            inp = inp.truediv(inp.sum(), axis=0)

        if op.sort:
            inp = inp.sort_values(ascending=op.ascending)

        ret = recursive_tile(inp)

        if isinstance(op.input.op, DataFrameCut):
            # convert index to IntervalDtype
            chunks = []
            for c in ret.chunks:
                chunk_op = DataFrameValueCounts(stage=OperandStage.map)
                chunk_params = c.params
                chunk_params['index_value'] = parse_index(pd.IntervalIndex([]),
                                                          c, store_data=False)
                chunks.append(chunk_op.new_chunk([c], kws=[chunk_params]))

            new_op = op.copy()
            params = out.params
            params['chunks'] = chunks
            params['nsplits'] = ret.nsplits
            return new_op.new_seriess(out.inputs, kws=[params])

        return [ret]

    @classmethod
    def execute(cls, ctx, op: "DataFrameValueCounts"):
        if op.stage != OperandStage.map:
            result = ctx[op.input.key].value_counts(
                normalize=op.normalize, sort=op.sort, ascending=op.ascending,
                bins=op.bins, dropna=op.dropna)
        else:
            result = ctx[op.input.key]
        # convert CategoricalDtype which generated in `cut`
        # to IntervalDtype
        result.index = result.index.astype('interval')
        ctx[op.outputs[0].key] = result


def value_counts(series, normalize=False, sort=True, ascending=False,
                 bins=None, dropna=True, method='tree'):
    op = DataFrameValueCounts(normalize=normalize, sort=sort,
                              ascending=ascending, bins=bins,
                              dropna=dropna, method=method)
    return op(series)
