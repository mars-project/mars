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

import itertools
import copy

import numpy as np
import pandas as pd

from ...core import ENTITY_TYPE, recursive_tile
from ...serialization.serializables import AnyField, Float64Field
from ...tensor.core import TENSOR_TYPE, TENSOR_CHUNK_TYPE, ChunkData, Chunk
from ...tensor.datasource import tensor as astensor
from ...utils import classproperty, get_dtype
from ..align import align_series_series, align_dataframe_series, align_dataframe_dataframe
from ..core import DATAFRAME_TYPE, SERIES_TYPE, DATAFRAME_CHUNK_TYPE, SERIES_CHUNK_TYPE
from ..initializer import Series, DataFrame
from ..operands import DataFrameOperandMixin, DataFrameOperand
from ..ufunc.tensor import TensorUfuncMixin
from ..utils import parse_index, infer_dtypes, infer_dtype, infer_index_value, build_empty_df


class DataFrameBinOpMixin(DataFrameOperandMixin):

    @classmethod
    def _tile_both_dataframes(cls, op):
        # if both of the inputs are DataFrames, axis is just ignored
        left, right = op.lhs, op.rhs
        df = op.outputs[0]

        nsplits, out_shape, left_chunks, right_chunks = align_dataframe_dataframe(left, right)
        out_chunk_indexes = itertools.product(*(range(s) for s in out_shape))

        out_chunks = []
        for idx, left_chunk, right_chunk in zip(out_chunk_indexes, left_chunks, right_chunks):
            out_chunk = op.copy().reset_key().new_chunk([left_chunk, right_chunk],
                                                        shape=(nsplits[0][idx[0]], nsplits[1][idx[1]]),
                                                        index=idx)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, df.shape,
                                     nsplits=tuple(tuple(ns) for ns in nsplits),
                                     chunks=out_chunks, dtypes=df.dtypes,
                                     index_value=df.index_value, columns_value=df.columns_value)

    @classmethod
    def _tile_both_series(cls, op):
        left, right = op.lhs, op.rhs
        df = op.outputs[0]

        nsplits, out_shape, left_chunks, right_chunks = align_series_series(left, right)

        out_chunks = []
        for idx, left_chunk, right_chunk in zip(range(out_shape[0]), left_chunks, right_chunks):
            out_chunk = op.copy().reset_key().new_chunk([left_chunk, right_chunk],
                                                        shape=(nsplits[0][idx],), index=(idx,))
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_seriess(op.inputs, df.shape,
                                  nsplits=tuple(tuple(ns) for ns in nsplits),
                                  chunks=out_chunks, dtype=df.dtype,
                                  index_value=df.index_value, name=df.name)

    @classmethod
    def _tile_dataframe_series(cls, op):
        left, right = op.lhs, op.rhs
        df = op.outputs[0]

        nsplits, out_shape, left_chunks, right_chunks = align_dataframe_series(left, right, axis=op.axis)
        out_chunk_indexes = itertools.product(*(range(s) for s in out_shape))

        out_chunks = []
        for out_idx, df_chunk in zip(out_chunk_indexes, left_chunks):
            if op.axis == 'columns' or op.axis == 1:
                series_chunk = right_chunks[out_idx[1]]
                kw = {
                    'shape': (nsplits[0][out_idx[0]], nsplits[1][out_idx[1]]),
                    'index_value': df_chunk.index_value,
                    'dtypes_value': df_chunk.dtypes_value
                }
            else:
                series_chunk = right_chunks[out_idx[0]]
                kw = {
                    'shape': (nsplits[0][out_idx[0]], nsplits[1][out_idx[1]]),
                    'columns_value': df_chunk.columns_value,
                    'dtypes_value': df_chunk.dtypes_value
                }
            out_chunk = op.copy().reset_key().new_chunk([df_chunk, series_chunk], index=out_idx, **kw)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, df.shape,
                                     nsplits=tuple(tuple(ns) for ns in nsplits),
                                     chunks=out_chunks, dtypes=df.dtypes,
                                     index_value=df.index_value, columns_value=df.columns_value)

    @classmethod
    def _tile_series_dataframe(cls, op):
        left, right = op.lhs, op.rhs
        df = op.outputs[0]

        nsplits, out_shape, right_chunks, left_chunks = align_dataframe_series(right, left, axis=op.axis)
        out_chunk_indexes = itertools.product(*(range(s) for s in out_shape))

        out_chunks = []
        for out_idx, df_chunk in zip(out_chunk_indexes, right_chunks):
            if op.axis == 'columns' or op.axis == 1:
                series_chunk = left_chunks[out_idx[1]]
                kw = {
                    'shape': (df_chunk.shape[0], np.nan),
                    'index_value': df_chunk.index_value,
                    'dtypes_value': df_chunk.dtypes_value
                }
            else:
                series_chunk = left_chunks[out_idx[0]]
                kw = {
                    'shape': (df_chunk.shape[0], np.nan),
                    'index_value': df_chunk.index_value,
                    'dtypes_value': df_chunk.dtypes_value
                }
            out_chunk = op.copy().reset_key().new_chunk([series_chunk, df_chunk], index=out_idx, **kw)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, df.shape,
                                     nsplits=tuple(tuple(ns) for ns in nsplits),
                                     chunks=out_chunks, dtypes=df.dtypes,
                                     index_value=df.index_value, columns_value=df.columns_value)

    @classmethod
    def _tile_scalar(cls, op):
        tileable = op.rhs if pd.api.types.is_scalar(op.lhs) else op.lhs
        df = op.outputs[0]
        out_chunks = []
        for chunk in tileable.chunks:
            out_op = op.copy().reset_key()
            if isinstance(chunk, DATAFRAME_CHUNK_TYPE):
                out_chunk = out_op.new_chunk([chunk], shape=chunk.shape, index=chunk.index,
                                             dtypes=chunk.dtypes, index_value=chunk.index_value,
                                             columns_value=getattr(chunk, 'columns_value'))
            else:
                out_chunk = out_op.new_chunk([chunk], shape=chunk.shape, index=chunk.index, dtype=chunk.dtype,
                                             index_value=chunk.index_value, name=getattr(chunk, 'name'))
            out_chunks.append(out_chunk)

        new_op = op.copy()
        out = op.outputs[0]
        if isinstance(df, SERIES_TYPE):
            return new_op.new_seriess(op.inputs, df.shape, nsplits=tileable.nsplits, dtype=out.dtype,
                                      index_value=df.index_value, name=df.name, chunks=out_chunks)
        else:
            return new_op.new_dataframes(op.inputs, df.shape, nsplits=tileable.nsplits, dtypes=out.dtypes,
                                         index_value=df.index_value, columns_value=df.columns_value,
                                         chunks=out_chunks)

    @classmethod
    def _tile_with_tensor(cls, op):
        out = op.outputs[0]
        axis = op.axis
        if axis is None:
            axis = 0

        rhs_is_tensor = isinstance(op.rhs, TENSOR_TYPE)
        tensor, other = (op.rhs, op.lhs) if rhs_is_tensor else (op.lhs, op.rhs)
        if tensor.shape == other.shape:
            tensor = yield from recursive_tile(tensor.rechunk(other.nsplits))
        else:
            # shape differs only when dataframe add 1-d tensor, we need rechunk on columns axis.
            if axis in ['columns', 1] and other.ndim == 1:
                # force axis == 0 if it's Series other than DataFrame
                axis = 0
            rechunk_size = other.nsplits[1] if axis == 'columns' or axis == 1 else other.nsplits[0]
            if tensor.ndim > 0:
                tensor = yield from recursive_tile(tensor.rechunk((rechunk_size,)))

        out_chunks = []
        for out_index in itertools.product(*(map(range, other.chunk_shape))):
            tensor_chunk = tensor.cix[out_index[:tensor.ndim]]
            other_chunk = other.cix[out_index]
            out_op = op.copy().reset_key()
            inputs = [other_chunk, tensor_chunk] if rhs_is_tensor else [tensor_chunk, other_chunk]
            if isinstance(other_chunk, DATAFRAME_CHUNK_TYPE):
                cum_splits = [0] + np.cumsum(other.nsplits[1]).tolist()
                start = cum_splits[out_index[1]]
                end = cum_splits[out_index[1] + 1]
                chunk_dtypes = out.dtypes.iloc[start: end]
                out_chunk = out_op.new_chunk(inputs, shape=other_chunk.shape, index=other_chunk.index,
                                             dtypes=chunk_dtypes,
                                             index_value=other_chunk.index_value,
                                             columns_value=other_chunk.columns_value)
            else:
                out_chunk = out_op.new_chunk(inputs, shape=other_chunk.shape, index=other_chunk.index,
                                             dtype=out.dtype,
                                             index_value=other_chunk.index_value,
                                             name=other_chunk.name)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        if isinstance(other, SERIES_TYPE):
            return new_op.new_seriess(op.inputs, other.shape, nsplits=other.nsplits, dtype=out.dtype,
                                      name=other.name, index_value=other.index_value, chunks=out_chunks)
        else:
            return new_op.new_dataframes(op.inputs, other.shape, nsplits=other.nsplits, dtypes=out.dtypes,
                                         index_value=other.index_value, columns_value=other.columns_value,
                                         chunks=out_chunks)

    @classmethod
    def tile(cls, op):
        if len(op.inputs) < 2:
            return cls._tile_scalar(op)
        elif isinstance(op.inputs[0], DATAFRAME_TYPE) and isinstance(op.inputs[1], DATAFRAME_TYPE):
            return cls._tile_both_dataframes(op)
        elif isinstance(op.inputs[0], SERIES_TYPE) and isinstance(op.inputs[1], SERIES_TYPE):
            return cls._tile_both_series(op)
        elif isinstance(op.inputs[0], DATAFRAME_TYPE) and isinstance(op.inputs[1], SERIES_TYPE):
            return cls._tile_dataframe_series(op)
        elif isinstance(op.inputs[0], SERIES_TYPE) and isinstance(op.inputs[1], DATAFRAME_TYPE):
            return cls._tile_series_dataframe(op)
        elif isinstance(op.inputs[0], TENSOR_TYPE) or isinstance(op.inputs[1], TENSOR_TYPE):
            return (yield from cls._tile_with_tensor(op))

    @classmethod
    def execute(cls, ctx, op):
        if getattr(cls, '_func_name', None) is not None:
            if len(op.inputs) == 2:
                df, other = ctx[op.inputs[0].key], ctx[op.inputs[1].key]
                if isinstance(op.inputs[0], SERIES_CHUNK_TYPE) and \
                        isinstance(op.inputs[1], DATAFRAME_CHUNK_TYPE):
                    df, other = other, df
                    func_name = getattr(cls, '_rfunc_name')
                else:
                    func_name = getattr(cls, '_func_name')
            elif pd.api.types.is_scalar(op.lhs) or isinstance(op.lhs, np.ndarray):
                df = ctx[op.rhs.key]
                other = op.lhs
                func_name = getattr(cls, '_rfunc_name')
            else:
                df = ctx[op.lhs.key]
                other = op.rhs
                func_name = getattr(cls, '_func_name')
            if df.ndim == 2:
                kw = dict(axis=op.axis)
            else:
                kw = dict()
            if op.fill_value is not None:
                # comparison function like eq does not have `fill_value`
                kw['fill_value'] = op.fill_value
            if op.level is not None:
                # logical function like and may don't have `level` (for Series type)
                kw['level'] = op.level
            if hasattr(other, 'ndim') and other.ndim == 0:
                other = other.item()
            ctx[op.outputs[0].key] = getattr(df, func_name)(other, **kw)
        else:
            inputs_iter = iter(op.inputs)
            if not pd.api.types.is_scalar(op.lhs):
                lhs = ctx[next(inputs_iter).key]
            else:
                lhs = op.lhs
            if not pd.api.types.is_scalar(op.rhs):
                rhs = ctx[next(inputs_iter).key]
            else:
                rhs = op.rhs
            ctx[op.outputs[0].key] = cls._operator(lhs, rhs)  # pylint: disable=too-many-function-args

    @classproperty
    def _operator(self):
        raise NotImplementedError

    @classmethod
    def _calc_properties(cls, x1, x2=None, axis='columns'):
        if isinstance(x1, (DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)) and (
                x2 is None or pd.api.types.is_scalar(x2) or
                isinstance(x2, (TENSOR_TYPE, TENSOR_CHUNK_TYPE))):
            if x2 is None:
                dtypes = x1.dtypes
            elif pd.api.types.is_scalar(x2):
                dtypes = cls._operator(build_empty_df(x1.dtypes), x2).dtypes
            elif x1.dtypes is not None and isinstance(x2, TENSOR_TYPE):
                dtypes = pd.Series(
                    [infer_dtype(dt, x2.dtype, cls._operator) for dt in x1.dtypes],
                    index=x1.dtypes.index)
            else:
                dtypes = x1.dtypes
            return {'shape': x1.shape, 'dtypes': dtypes,
                    'columns_value': x1.columns_value, 'index_value': x1.index_value}

        if isinstance(x1, (SERIES_TYPE, SERIES_CHUNK_TYPE)) and (
                x2 is None or pd.api.types.is_scalar(x2) or
                isinstance(x2, (TENSOR_TYPE, TENSOR_CHUNK_TYPE))):
            x2_dtype = x2.dtype if hasattr(x2, 'dtype') else type(x2)
            x2_dtype = get_dtype(x2_dtype)
            dtype = infer_dtype(x1.dtype, x2_dtype, cls._operator)
            ret = {'shape': x1.shape, 'dtype': dtype, 'index_value': x1.index_value}
            if pd.api.types.is_scalar(x2) or (
                    hasattr(x2, 'ndim') and (
                    x2.ndim == 0 or x2.ndim == 1)):
                ret['name'] = x1.name
            return ret

        if isinstance(x1, (DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)) and isinstance(
                x2, (DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)):
            index_shape, column_shape, dtypes, columns, index = np.nan, np.nan, None, None, None

            if x1.columns_value is not None and x2.columns_value is not None and \
                    x1.columns_value.key == x2.columns_value.key:
                dtypes = pd.Series([infer_dtype(dt1, dt2, cls._operator) for dt1, dt2
                                    in zip(x1.dtypes, x2.dtypes)],
                                   index=x1.dtypes.index)
                columns = copy.copy(x1.columns_value)
                columns.value.should_be_monotonic = False
                column_shape = len(dtypes)
            elif x1.dtypes is not None and x2.dtypes is not None:
                dtypes = infer_dtypes(x1.dtypes, x2.dtypes, cls._operator)
                columns = parse_index(dtypes.index, store_data=True)
                columns.value.should_be_monotonic = True
                column_shape = len(dtypes)
            if x1.index_value is not None and x2.index_value is not None:
                if x1.index_value.key == x2.index_value.key:
                    index = copy.copy(x1.index_value)
                    index.value.should_be_monotonic = False
                    index_shape = x1.shape[0]
                else:
                    index = infer_index_value(x1.index_value, x2.index_value)
                    index.value.should_be_monotonic = True
                    if index.key == x1.index_value.key == x2.index_value.key and \
                            (not np.isnan(x1.shape[0]) or not np.isnan(x2.shape[0])):
                        index_shape = x1.shape[0] if not np.isnan(x1.shape[0]) else x2.shape[0]

            return {'shape': (index_shape, column_shape), 'dtypes': dtypes,
                    'columns_value': columns, 'index_value': index}

        if isinstance(x1, (DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)) and \
                isinstance(x2, (SERIES_TYPE, SERIES_CHUNK_TYPE)):
            if axis == 'columns' or axis == 1:
                index_shape = x1.shape[0]
                index = x1.index_value
                column_shape, dtypes, columns = np.nan, None, None
                if x1.columns_value is not None and x1.index_value is not None:
                    if x1.columns_value.key == x2.index_value.key:
                        dtypes = pd.Series([infer_dtype(dt, x2.dtype, cls._operator) for dt in x1.dtypes],
                                           index=x1.dtypes.index)
                        columns = copy.copy(x1.columns_value)
                        columns.value.should_be_monotonic = False
                        column_shape = len(dtypes)
                    else:  # pragma: no cover
                        dtypes = x1.dtypes  # FIXME
                        columns = infer_index_value(x1.columns_value, x2.index_value)
                        columns.value.should_be_monotonic = True
                        column_shape = np.nan
            else:
                assert axis == 'index' or axis == 0
                column_shape = x1.shape[1]
                columns = x1.columns_value
                dtypes = x1.dtypes
                index_shape, index = np.nan, None
                if x1.index_value is not None and x1.index_value is not None:
                    if x1.index_value.key == x2.index_value.key:
                        dtypes = pd.Series([infer_dtype(dt, x2.dtype, cls._operator) for dt in x1.dtypes],
                                           index=x1.dtypes.index)
                        index = copy.copy(x1.index_value)
                        index.value.should_be_monotonic = False
                        index_shape = x1.shape[0]
                    else:
                        if x1.dtypes is not None:
                            dtypes = pd.Series(
                                [infer_dtype(dt, x2.dtype, cls._operator) for dt in x1.dtypes],
                                index=x1.dtypes.index)
                        index = infer_index_value(x1.index_value, x2.index_value)
                        index.value.should_be_monotonic = True
                        index_shape = np.nan
            return {'shape': (index_shape, column_shape), 'dtypes': dtypes,
                    'columns_value': columns, 'index_value': index}

        if isinstance(x1, (SERIES_TYPE, SERIES_CHUNK_TYPE)) and \
                isinstance(x2, (SERIES_TYPE, SERIES_CHUNK_TYPE)):
            index_shape, dtype, index = np.nan, None, None

            dtype = infer_dtype(x1.dtype, x2.dtype, cls._operator)
            if x1.index_value is not None and x2.index_value is not None:
                if x1.index_value.key == x2.index_value.key:
                    index = copy.copy(x1.index_value)
                    index.value.should_be_monotonic = False
                    index_shape = x1.shape[0]
                else:
                    index = infer_index_value(x1.index_value, x2.index_value)
                    index.value.should_be_monotonic = True
                    if index.key == x1.index_value.key == x2.index_value.key and \
                            (not np.isnan(x1.shape[0]) or not np.isnan(x2.shape[0])):
                        index_shape = x1.shape[0] if not np.isnan(x1.shape[0]) else x2.shape[0]

            ret = {'shape': (index_shape,), 'dtype': dtype, 'index_value': index}
            if x1.name == x2.name:
                ret['name'] = x1.name
            return ret

        raise NotImplementedError('Unknown combination of parameters')

    def _new_chunks(self, inputs, kws=None, **kw):
        property_inputs = [
            inp for inp in inputs
            if isinstance(inp, (DATAFRAME_CHUNK_TYPE, SERIES_CHUNK_TYPE, TENSOR_CHUNK_TYPE))]
        if len(property_inputs) == 1:
            properties = self._calc_properties(*property_inputs)
        elif any(inp.ndim == 2 for inp in property_inputs):
            df1, df2 = property_inputs \
                if isinstance(property_inputs[0], DATAFRAME_CHUNK_TYPE) else \
                reversed(property_inputs)
            properties = self._calc_properties(df1, df2, axis=self.axis)
        else:
            if property_inputs[0].ndim < property_inputs[1].ndim or \
                    isinstance(property_inputs[0], (TENSOR_TYPE, TENSOR_CHUNK_TYPE)):
                property_inputs = reversed(property_inputs)
            properties = self._calc_properties(*property_inputs)

        inputs = [inp for inp in inputs if isinstance(inp, (Chunk, ChunkData))]

        shape = properties.pop('shape')
        if 'shape' in kw:
            shape = kw.pop('shape')

        for prop, value in properties.items():
            if kw.get(prop, None) is None:
                kw[prop] = value

        return super()._new_chunks(
            inputs, shape=shape, kws=kws, **kw)

    @staticmethod
    def _process_input(x):
        if isinstance(x, (DATAFRAME_TYPE, SERIES_TYPE)) or pd.api.types.is_scalar(x):
            return x
        elif isinstance(x, pd.Series):
            return Series(x)
        elif isinstance(x, pd.DataFrame):
            return DataFrame(x)
        elif isinstance(x, (list, tuple, np.ndarray, TENSOR_TYPE)):
            return astensor(x)
        raise NotImplementedError

    def _check_inputs(self, x1, x2):
        if isinstance(x1, TENSOR_TYPE) or isinstance(x2, TENSOR_TYPE):
            tensor, other = (x1, x2) if isinstance(x1, TENSOR_TYPE) else (x2, x1)
            if isinstance(other, DATAFRAME_TYPE):
                if self.axis == 'index' or self.axis == 0:
                    other_shape = tuple(reversed(other.shape))
                else:
                    other_shape = other.shape
                if tensor.ndim == 2 and tensor.shape != other_shape:
                    raise ValueError(
                        f'Unable to coerce to DataFrame, shape must be {other_shape}: '
                        f'given {tensor.shape}')
                elif tensor.ndim == 1 and tensor.shape[0] != other_shape[1]:
                    raise ValueError(
                        f'Unable to coerce to Series, length must be {other_shape[1]}: '
                        f'given {tensor.shape[0]}')
                elif tensor.ndim > 2:
                    raise ValueError('Unable to coerce to Series/DataFrame, dim must be <= 2')
            if isinstance(other, SERIES_TYPE):
                if tensor.ndim == 1 and (tensor.shape[0] != other.shape[0]):
                    raise ValueError(
                        f'Unable to coerce to Series, length must be {other.shape[0]}: '
                        f'given {tensor.shape[0]}')
                elif tensor.ndim > 1:
                    raise ValueError('Unable to coerce to Series, dim must be 1')

    def _call(self, x1, x2):
        self._check_inputs(x1, x2)
        if isinstance(x1, DATAFRAME_TYPE) or isinstance(x2, DATAFRAME_TYPE):
            df1, df2 = (x1, x2) if isinstance(x1, DATAFRAME_TYPE) else (x2, x1)
            kw = self._calc_properties(df1, df2, axis=self.axis)
            if not pd.api.types.is_scalar(df2):
                return self.new_dataframe([x1, x2], **kw)
            else:
                return self.new_dataframe([df1], **kw)
        if isinstance(x1, SERIES_TYPE) or isinstance(x2, SERIES_TYPE):
            s1, s2 = (x1, x2) if isinstance(x1, SERIES_TYPE) else (x2, x1)
            kw = self._calc_properties(s1, s2)
            if not pd.api.types.is_scalar(s2):
                return self.new_series([x1, x2], **kw)
            else:
                return self.new_series([s1], **kw)
        raise NotImplementedError('Only support add dataframe, series or scalar for now')

    def __call__(self, x1, x2):
        x1 = self._process_input(x1)
        x2 = self._process_input(x2)
        if isinstance(x1, SERIES_TYPE) and isinstance(x2, DATAFRAME_TYPE):
            # reject invoking series's op on dataframe
            raise NotImplementedError
        return self._call(x1, x2)

    def rcall(self, x1, x2):
        x1 = self._process_input(x1)
        x2 = self._process_input(x2)
        if isinstance(x1, SERIES_TYPE) and isinstance(x2, DATAFRAME_TYPE):
            # reject invoking series's op on dataframe
            raise NotImplementedError
        return self._call(x2, x1)


class DataFrameBinOp(DataFrameOperand, DataFrameBinOpMixin):
    _axis = AnyField('axis')
    _level = AnyField('level')
    _fill_value = Float64Field('fill_value')
    _lhs = AnyField('lhs')
    _rhs = AnyField('rhs')

    def __init__(self, axis=None, level=None, fill_value=None,
                 output_types=None, lhs=None, rhs=None, **kw):
        super().__init__(_axis=axis, _level=level, _fill_value=fill_value,
                         _output_types=output_types, _lhs=lhs, _rhs=rhs, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def level(self):
        return self._level

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if len(self._inputs) == 2:
            self._lhs = self._inputs[0]
            self._rhs = self._inputs[1]
        else:
            if isinstance(self._lhs, ENTITY_TYPE):
                self._lhs = self._inputs[0]
            elif pd.api.types.is_scalar(self._lhs):
                self._rhs = self._inputs[0]


class DataFrameUnaryOpMixin(DataFrameOperandMixin):
    __slots__ = ()

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        out_chunks = []
        index_dtypes_cache = dict()
        for in_chunk in in_df.chunks:
            out_op = op.copy().reset_key()
            if out_df.ndim == 2:
                try:
                    dtypes = index_dtypes_cache[in_chunk.index[1]]
                except KeyError:
                    dtypes = out_df.dtypes[in_chunk.columns_value.to_pandas()]
                    index_dtypes_cache[in_chunk.index[1]] = dtypes

                out_chunk = out_op.new_chunk([in_chunk], shape=in_chunk.shape,
                                             dtypes=dtypes,
                                             index=in_chunk.index,
                                             index_value=in_chunk.index_value,
                                             columns_value=in_chunk.columns_value)
            else:
                out_chunk = out_op.new_chunk([in_chunk], shape=in_chunk.shape,
                                             index=in_chunk.index,
                                             dtype=in_chunk.dtype,
                                             index_value=in_chunk.index_value,
                                             name=in_chunk.name)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        kw = out_df.params
        kw['nsplits'] = in_df.nsplits
        kw['chunks'] = out_chunks
        return new_op.new_tileables(op.inputs, kws=[kw])

    @classmethod
    def execute(cls, ctx, op):
        df = ctx[op.inputs[0].key]
        func_name = getattr(cls, '_func_name')
        if hasattr(df, func_name):
            ctx[op.outputs[0].key] = getattr(df, func_name)()
        else:
            ctx[op.outputs[0].key] = getattr(np, func_name)(df)


class DataFrameUnaryOp(DataFrameOperand, DataFrameUnaryOpMixin):
    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @classmethod
    def _get_output_dtype(cls, df):
        if df.ndim == 2:
            return df.dtypes
        else:
            return df.dtype

    def __call__(self, df):
        self.output_types = df.op.output_types
        if df.ndim == 2:
            return self.new_dataframe([df], shape=df.shape, dtypes=self._get_output_dtype(df),
                                      columns_value=df.columns_value,
                                      index_value=df.index_value)
        else:
            series = df
            return self.new_series([series], shape=series.shape, name=series.name,
                                   index_value=series.index_value,
                                   dtype=self._get_output_dtype(series))


class DataFrameUnaryUfunc(DataFrameUnaryOp, TensorUfuncMixin):
    pass


class DataFrameBinopUfunc(DataFrameBinOp, TensorUfuncMixin):
    pass
