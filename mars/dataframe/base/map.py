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

import inspect
from collections.abc import MutableMapping

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...serialize import KeyField, AnyField, StringField
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape
from ..core import SERIES_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType


class DataFrameMap(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.MAP

    _input = KeyField('input')
    _arg = AnyField('arg')
    _na_action = StringField('na_action')

    def __init__(self, arg=None, na_action=None, object_type=None, **kw):
        super().__init__(_arg=arg, _na_action=na_action,
                         _object_type=object_type, **kw)
        if self._object_type is None:
            self._object_type = ObjectType.series

    @property
    def input(self):
        return self._input

    @property
    def arg(self):
        return self._arg

    @property
    def na_action(self):
        return self._na_action

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if len(inputs) == 2:
            self._arg = self._inputs[1]

    def __call__(self, series, dtype):
        if dtype is None:
            inferred_dtype = None
            if callable(self._arg):
                # arg is a function, try to inspect the signature
                sig = inspect.signature(self._arg)
                return_type = sig.return_annotation
                if return_type is not inspect._empty:
                    inferred_dtype = np.dtype(return_type)
            else:
                if isinstance(self._arg, MutableMapping):
                    inferred_dtype = pd.Series(self._arg).dtype
                else:
                    inferred_dtype = self._arg.dtype
            if inferred_dtype is not None and np.issubdtype(inferred_dtype, np.number):
                if np.issubdtype(inferred_dtype, np.inexact):
                    # for the inexact e.g. float
                    # we can make the decision,
                    # but for int, due to the nan which may occur,
                    # we cannot infer the dtype
                    dtype = inferred_dtype
            else:
                dtype = inferred_dtype

        if dtype is None:
            raise ValueError('cannot infer dtype, '
                             'it needs to be specified manually for `map`')
        else:
            dtype = np.int64 if dtype is int else dtype
            dtype = np.dtype(dtype)

        inputs = [series]
        if isinstance(self._arg, SERIES_TYPE):
            inputs.append(self._arg)
        return self.new_series(inputs, shape=series.shape, dtype=dtype,
                               index_value=series.index_value, name=series.name)

    @classmethod
    def tile(cls, op):
        in_series = op.input
        out_series = op.outputs[0]

        arg = op.arg
        if len(op.inputs) == 2:
            # make sure arg has known shape when it's a md.Series
            check_chunks_unknown_shape([op.arg], TilesError)
            arg = op.arg.rechunk(op.arg.shape)._inplace_tile()

        out_chunks = []
        for chunk in in_series.chunks:
            chunk_op = op.copy().reset_key()
            chunk_inputs = [chunk]
            if len(op.inputs) == 2:
                chunk_inputs.append(arg.chunks[0])
            out_chunk = chunk_op.new_chunk(chunk_inputs, shape=chunk.shape,
                                           dtype=out_series.dtype,
                                           index_value=chunk.index_value,
                                           name=out_series.name,
                                           index=chunk.index)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        params = out_series.params
        params['chunks'] = out_chunks
        params['nsplits'] = in_series.nsplits
        return new_op.new_seriess(op.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        series = ctx[op.inputs[0].key]
        out = op.outputs[0]
        if len(op.inputs) == 2:
            arg = ctx[op.inputs[1].key]
        else:
            arg = op.arg

        ret = series.map(arg, na_action=op.na_action)
        if ret.dtype != out.dtype:
            ret = ret.astype(out.dtype)
        ctx[out.key] = ret


def map_(series, arg, na_action=None, dtype=None):
    op = DataFrameMap(arg=arg, na_action=na_action)
    return op(series, dtype=dtype)
