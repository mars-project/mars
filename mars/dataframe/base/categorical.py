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

import inspect

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import BoolField, DictField, StringField, TupleField
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_empty_series, parse_index


class SeriesCategoricalMethod(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.CATEGORICAL_METHOD

    method = StringField("method")
    method_args = TupleField("method_args")
    method_kwargs = DictField("method_kwargs")
    is_property = BoolField("is_property")

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)
        if not self.output_types:
            self.output_types = [OutputType.series]

    def __call__(self, inp):
        return _categorical_method_to_handlers[self.method].call(self, inp)

    @classmethod
    def tile(cls, op: "SeriesCategoricalMethod"):
        tiled = _categorical_method_to_handlers[op.method].tile(op)
        if inspect.isgenerator(tiled):
            return (yield from tiled)
        else:
            return tiled

    @classmethod
    def execute(cls, ctx, op):
        return _categorical_method_to_handlers[op.method].execute(ctx, op)


class SeriesCategoricalMethodBaseHandler:
    @classmethod
    def call(cls, op: "SeriesCategoricalMethod", inp):
        empty_series = build_empty_series(inp.dtype)
        rseries = getattr(empty_series.cat, op.method)
        if not op.is_property:
            rseries = rseries(*op.method_args, **op.method_kwargs)
        dtype = rseries.dtype
        return op.new_series(
            [inp],
            shape=inp.shape,
            dtype=dtype,
            index_value=inp.index_value,
            name=inp.name,
        )

    @classmethod
    def tile(cls, op: "SeriesCategoricalMethod"):
        out = op.outputs[0]
        out_chunks = []
        for series_chunk in op.inputs[0].chunks:
            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk(
                [series_chunk],
                shape=series_chunk.shape,
                dtype=out.dtype,
                index=series_chunk.index,
                index_value=series_chunk.index_value,
                name=series_chunk.name,
            )
            out_chunks.append(out_chunk)

        params = out.params
        params["chunks"] = out_chunks
        params["nsplits"] = op.inputs[0].nsplits
        new_op = op.copy()
        return new_op.new_tileables([op.inputs[0]], kws=[params])

    @classmethod
    def execute(cls, ctx, op: "SeriesCategoricalMethod"):
        inp = ctx[op.inputs[0].key]
        result = getattr(inp.cat, op.method)
        if not op.is_property:
            result = result(*op.method_args, **op.method_kwargs)
        ctx[op.outputs[0].key] = result


class GetCategoriesHandler(SeriesCategoricalMethodBaseHandler):
    @classmethod
    def call(cls, op: "SeriesCategoricalMethod", inp):
        dtype = inp.dtype.categories.dtype
        return op.new_index(
            [inp],
            shape=(np.nan,),
            dtype=dtype,
            index_value=parse_index(pd.Index([], dtype=dtype)),
        )

    @classmethod
    def tile(cls, op: "SeriesCategoricalMethod"):
        out = op.outputs[0]

        chunk_op = op.copy().reset_key()
        out_chunk = chunk_op.new_chunk(
            [op.inputs[0].chunks[0]],
            index=(0,),
            shape=out.shape,
            dtype=out.dtype,
            index_value=out.index_value,
        )

        params = out.params
        params["chunks"] = [out_chunk]
        params["nsplits"] = ((np.nan,),)
        new_op = op.copy()
        return new_op.new_tileables([op.inputs[0]], kws=[params])

    @classmethod
    def execute(cls, ctx, op: "SeriesCategoricalMethod"):
        inp = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = inp.cat.categories


_categorical_method_to_handlers = {}
_not_implements = ["categories", "ordered", "remove_unused_categories"]
# start to register handlers for categorical methods
# register special methods first
_categorical_method_to_handlers["_get_categories"] = GetCategoriesHandler
# then come to the normal methods
for method in dir(pd.Series.cat):
    if method.startswith("_") and method != "__getitem__":
        continue
    if method in _not_implements:
        continue
    if method in _categorical_method_to_handlers:
        continue
    _categorical_method_to_handlers[method] = SeriesCategoricalMethodBaseHandler
