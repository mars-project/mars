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

from ... import opcodes as OperandDef
from ...core import OutputType, recursive_tile
from ...serialization.serializables import KeyField, StringField, \
    TupleField, DictField
from ...tensor import tensor as astensor
from ...tensor.core import TENSOR_TYPE
from ...utils import has_unknown_shape
from ..align import align_series_series
from ..core import SERIES_TYPE
from ..initializer import Series as asseries
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_empty_series, parse_index, infer_index_value


class SeriesStringMethod(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.STRING_METHOD

    _input = KeyField('input')
    _method = StringField('method')
    _method_args = TupleField('method_args')
    _method_kwargs = DictField('method_kwargs')

    def __init__(self, method=None, method_args=None, method_kwargs=None,
                 output_types=None, **kw):
        super().__init__(_method=method, _method_args=method_args,
                         _method_kwargs=method_kwargs,
                         _output_types=output_types, **kw)
        if not self.output_types:
            self.output_types = [OutputType.series]

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

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if len(self._inputs) == 2:
            # for method cat
            self._method_kwargs['others'] = self._inputs[1]

    def __call__(self, inp):
        return _string_method_to_handlers[self._method].call(self, inp)

    @classmethod
    def tile(cls, op):
        tiled = _string_method_to_handlers[op.method].tile(op)
        if inspect.isgenerator(tiled):
            return (yield from tiled)
        else:
            return tiled

    @classmethod
    def execute(cls, ctx, op):
        return _string_method_to_handlers[op.method].execute(ctx, op)


class SeriesStringMethodBaseHandler:
    @classmethod
    def call(cls, op, inp):
        empty_series = build_empty_series(inp.dtype)
        dtype = getattr(empty_series.str, op.method)(
            *op.method_args, **op.method_kwargs).dtype
        return op.new_series([inp], shape=inp.shape,
                             dtype=dtype,
                             index_value=inp.index_value,
                             name=inp.name)

    @classmethod
    def tile(cls, op):
        out = op.outputs[0]
        out_chunks = []
        for series_chunk in op.input.chunks:
            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk(
                [series_chunk], shape=series_chunk.shape,
                dtype=out.dtype, index=series_chunk.index,
                index_value=series_chunk.index_value,
                name=series_chunk.name)
            out_chunks.append(out_chunk)

        params = out.params
        params['chunks'] = out_chunks
        params['nsplits'] = op.input.nsplits
        new_op = op.copy()
        return new_op.new_tileables([op.input], kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        inp = ctx[op.input.key]
        ctx[op.outputs[0].key] = getattr(inp.str, op.method)(
            *op.method_args, **op.method_kwargs)


class SeriesStringSplitHandler(SeriesStringMethodBaseHandler):
    @classmethod
    def call(cls, op, inp):
        method_kwargs = op.method_kwargs
        if method_kwargs.get('expand', False) is False:
            return super().call(op, inp)
        n = method_kwargs.get('n', -1)
        # does not support if expand and n == -1
        if n == -1:  # pragma: no cover
            raise NotImplementedError('`n` needs to be specified when expand=True')

        op.output_types = [OutputType.dataframe]
        columns = pd.RangeIndex(n + 1)
        columns_value = parse_index(columns, store_data=True)
        dtypes = pd.Series([inp.dtype] * len(columns), index=columns)
        return op.new_dataframe([inp], shape=(inp.shape[0], len(columns)),
                                dtypes=dtypes, columns_value=columns_value,
                                index_value=inp.index_value)

    @classmethod
    def tile(cls, op):
        out = op.outputs[0]

        if out.op.output_types[0] == OutputType.series:
            return super().tile(op)

        out_chunks = []
        columns = out.columns_value.to_pandas()
        for series_chunk in op.input.chunks:
            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk(
                [series_chunk], shape=(series_chunk.shape[0], len(columns)),
                index=(series_chunk.index[0], 0),
                dtypes=out.dtypes, index_value=series_chunk.index_value,
                columns_value=out.columns_value
            )
            out_chunks.append(out_chunk)

        params = out.params
        params['chunks'] = out_chunks
        params['nsplits'] = (op.input.nsplits[0], (len(columns),))
        new_op = op.copy()
        return new_op.new_tileables([op.input], kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        inp = ctx[op.input.key]
        out = op.outputs[0]
        result = getattr(inp.str, op.method)(
            *op.method_args, **op.method_kwargs)
        if result.ndim == 2 and result.shape[1] < out.shape[1]:
            for i in range(result.shape[1], out.shape[1]):
                result[i] = None
        ctx[op.outputs[0].key] = result


class SeriesStringCatHandler(SeriesStringMethodBaseHandler):
    CAT_TYPE_ERROR = "others must be Series, Index, DataFrame, " \
                     "Tensor, np.ndarrary or list-like " \
                     "(either containing only strings or " \
                     "containing only objects of " \
                     "type Series/Index/Tensor/np.ndarray[1-dim])"
    CAT_LEN_ERROR = "If `others` contains arrays or lists (or other list-likes without an index), " \
                    "these must all be of the same length as the calling Series/Index."

    @classmethod
    def call(cls, op, inp):
        method_kwargs = op.method_kwargs
        others = method_kwargs.get('others')

        if others is None:
            from ..reduction import build_str_concat_object
            return build_str_concat_object(inp, sep=op.method_kwargs.get('sep'),
                                           na_rep=op.method_kwargs.get('na_rep'))
        elif isinstance(others, (tuple, list, np.ndarray, TENSOR_TYPE)):
            others = astensor(others, dtype=object)
            if others.ndim != 1:
                raise TypeError(cls.CAT_TYPE_ERROR)
            if not np.isnan(inp.shape[0]) and not np.isnan(others.shape[0]) and \
                    inp.shape[0] != others.shape[0]:
                raise ValueError(cls.CAT_LEN_ERROR)
            inputs = [inp]
            if isinstance(others, TENSOR_TYPE):
                inputs.append(others)
            return op.new_series(inputs, shape=inp.shape,
                                 dtype=inp.dtype,
                                 index_value=inp.index_value,
                                 name=inp.name)
        elif isinstance(others, (pd.Series, SERIES_TYPE)):
            others = asseries(others)
            if op.method_kwargs.get('join') != 'outer':  # pragma: no cover
                raise NotImplementedError('only outer join supported for now')
            return op.new_series([inp, others], shape=inp.shape,
                                 dtype=inp.dtype,
                                 index_value=infer_index_value(inp.index_value,
                                                               others.index_value),
                                 name=inp.name)
        elif isinstance(others, str) and op.method_kwargs.get('sep') is None:
            raise ValueError('Did you mean to supply a `sep` keyword?')
        else:
            raise TypeError(cls.CAT_TYPE_ERROR)

    @classmethod
    def tile(cls, op):
        inp = op.input
        out = op.outputs[0]

        # aggregation concat resulting in scalars is redirected
        assert out.ndim != 0

        if isinstance(op.inputs[1], TENSOR_TYPE):
            if has_unknown_shape(*op.inputs):
                yield
            # rechunk others as input
            others = yield from recursive_tile(
                op.inputs[1].rechunk(op.input.nsplits))
            out_chunks = []
            for c in inp.chunks:
                chunk_op = op.copy().reset_key()
                chunk_op._method_kwargs = op.method_kwargs.copy()
                out_chunk = chunk_op.new_chunk([c, others.cix[c.index]],
                                               dtype=c.dtype,
                                               index=c.index,
                                               shape=c.shape,
                                               index_value=c.index_value,
                                               name=c.name)
                out_chunks.append(out_chunk)
            new_op = op.copy()
            params = out.params
            params['nsplits'] = inp.nsplits
            params['chunks'] = out_chunks
            return new_op.new_tileables(op.inputs, kws=[params])
        elif isinstance(op.inputs[1], SERIES_TYPE):
            # both series
            out_chunks = []
            nsplits, _, left_chunks, right_chunks = align_series_series(*op.inputs)
            for left_chunk, right_chunk in zip(left_chunks, right_chunks):
                chunk_op = op.copy().reset_key()
                chunk_op._method_kwargs = op.method_kwargs.copy()
                params = left_chunk.params
                params['name'] = out.name
                out_chunk = chunk_op.new_chunk([left_chunk, right_chunk], **params)
                out_chunks.append(out_chunk)
            new_op = op.copy()
            params = out.params
            params['nsplits'] = nsplits
            params['chunks'] = out_chunks
            return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        inputs = [ctx[inp.key] for inp in op.inputs]
        method_kwargs = op.method_kwargs

        # aggregation concat is redirected and `others` is always defined
        assert len(inputs) > 1

        method_kwargs['others'] = inputs[1]
        ctx[op.outputs[0].key] = inputs[0].str.cat(**method_kwargs)


class SeriesStringExtractHandler(SeriesStringMethodBaseHandler):
    @classmethod
    def call(cls, op, inp):
        empty_series = build_empty_series(
            inp.dtype, index=inp.index_value.to_pandas()[:0])
        test_df = getattr(empty_series.str, op.method)(
            *op.method_args, **op.method_kwargs)
        if test_df.ndim == 1:
            return op.new_series([inp], shape=inp.shape,
                                 dtype=test_df.dtype,
                                 index_value=inp.index_value,
                                 name=inp.name)
        else:
            op.output_types = [OutputType.dataframe]
            if op.method == 'extractall':
                index_value = parse_index(test_df.index, inp)
                shape = (np.nan, test_df.shape[1])
            else:
                index_value = inp.index_value
                shape = (inp.shape[0], test_df.shape[1])
            return op.new_dataframe([inp], shape=shape,
                                    dtypes=test_df.dtypes,
                                    index_value=index_value,
                                    columns_value=parse_index(test_df.columns,
                                                              store_data=True))

    @classmethod
    def tile(cls, op):
        out = op.outputs[0]
        out_chunks = []
        for series_chunk in op.input.chunks:
            chunk_op = op.copy().reset_key()
            if out.ndim == 1:
                out_chunk = chunk_op.new_chunk(
                    [series_chunk], shape=series_chunk.shape,
                    index=series_chunk.index,
                    dtype=out.dtype, index_value=series_chunk.index_value,
                    name=out.name)
            else:
                if op.method == 'extract':
                    index_value = series_chunk.index_value
                    shape = (series_chunk.shape[0], out.shape[1])
                else:
                    index_value = parse_index(out.index_value.to_pandas()[:0],
                                              series_chunk)
                    shape = (np.nan, out.shape[1])
                out_chunk = chunk_op.new_chunk(
                    [series_chunk], shape=shape, index=(series_chunk.index[0], 0),
                    dtypes=out.dtypes, index_value=index_value,
                    columns_value=out.columns_value)
            out_chunks.append(out_chunk)

        out = op.outputs[0]
        params = out.params
        params['chunks'] = out_chunks
        if out.ndim == 1:
            params['nsplits'] = op.input.nsplits
        elif op.method == 'extract':
            params['nsplits'] = (op.input.nsplits[0], (out.shape[1],))
        else:
            params['nsplits'] = ((np.nan,) * len(op.input.nsplits[0]), (out.shape[1],))
        new_op = op.copy()
        return new_op.new_tileables([op.input], kws=[params])


_string_method_to_handlers = {}
_not_implements = ['get_dummies']
# start to register handlers for string methods
# register special methods first
_string_method_to_handlers['split'] = SeriesStringSplitHandler
_string_method_to_handlers['rsplit'] = SeriesStringSplitHandler
_string_method_to_handlers['cat'] = SeriesStringCatHandler
_string_method_to_handlers['extract'] = SeriesStringExtractHandler
_string_method_to_handlers['extractall'] = SeriesStringExtractHandler
# then come to the normal methods
for method in dir(pd.Series.str):
    if method.startswith('_') and method != '__getitem__':
        continue
    if method in _not_implements:
        continue
    if method in _string_method_to_handlers:
        continue
    _string_method_to_handlers[method] = SeriesStringMethodBaseHandler
