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

import pandas as pd

from ... import opcodes as OperandDef
from ...serialize import KeyField, StringField, TupleField, DictField, BoolField
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import build_empty_series


class SeriesDatetimeMethod(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATETIME_METHOD

    _input = KeyField('input')
    _method = StringField('method')
    _method_args = TupleField('method_args')
    _method_kwargs = DictField('method_kwargs')
    _is_property = BoolField('is_property')

    def __init__(self, method=None, method_args=None, method_kwargs=None,
                 is_property=None, stage=None, object_type=None, **kw):
        super().__init__(_method=method, _method_args=method_args,
                         _method_kwargs=method_kwargs, _is_property=is_property,
                         _stage=stage, _object_type=object_type, **kw)
        if self._object_type is None:
            self._object_type = ObjectType.series

    @property
    def input(self):
        return self._input

    @property
    def method(self):
        return self._method

    @property
    def method_args(self):
        return self._method_args

    @property
    def method_kwargs(self):
        return self._method_kwargs

    @property
    def is_property(self):
        return self._is_property

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, inp):
        return _datetime_method_to_handlers[self._method].call(self, inp)

    @classmethod
    def tile(cls, op):
        return _datetime_method_to_handlers[op.method].tile(op)

    @classmethod
    def execute(cls, ctx, op):
        return _datetime_method_to_handlers[op.method].execute(ctx, op)


class SeriesDatetimeMethodBaseHandler:
    @classmethod
    def call(cls, op, inp):
        empty_series = build_empty_series(inp.dtype)
        if op.is_property:
            test_obj = getattr(empty_series.dt, op.method)
        else:
            test_obj = getattr(empty_series.dt, op.method)(
                *op.method_args, **op.method_kwargs)
        dtype = test_obj.dtype
        return op.new_series([inp], shape=inp.shape,
                             dtype=dtype, index_value=inp.index_value,
                             name=inp.name)

    @classmethod
    def tile(cls, op):
        out = op.outputs[0]

        out_chunks = []
        for series_chunk in op.input.chunks:
            chunk_op = op.copy().reset_key()
            out_chunks.append(chunk_op.new_chunk(
                [series_chunk], shape=series_chunk.shape,
                dtype=out.dtype, index=series_chunk.index,
                index_value=series_chunk.index_value,
                name=series_chunk.name))

        params = out.params
        params['chunks'] = out_chunks
        params['nsplits'] = op.input.nsplits
        new_op = op.copy()
        return new_op.new_tileables([op.input], kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        inp = ctx[op.input.key]
        try:
            out = getattr(inp.dt, op.method)
        except ValueError:
            # fail due to buffer read-only
            out = getattr(inp.copy().dt, op.method)
        if not op.is_property:
            out = out(*op.method_args, **op.method_kwargs)
        ctx[op.outputs[0].key] = out


_datetime_method_to_handlers = {}
for method in dir(pd.Series.dt):
    if not method.startswith('_'):
        _datetime_method_to_handlers[method] = SeriesDatetimeMethodBaseHandler
