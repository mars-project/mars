# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from pandas.core.dtypes.cast import find_common_type

from ...utils import classproperty
from ..align import align_series_series, align_dataframe_series, align_dataframe_dataframe
from ..core import DATAFRAME_TYPE, SERIES_TYPE, DATAFRAME_CHUNK_TYPE, SERIES_CHUNK_TYPE
from ..operands import DataFrameOperandMixin, ObjectType
from ..utils import parse_index, infer_dtypes, infer_index_value


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
                                                        shape=(np.nan, np.nan), index=idx)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, df.shape,
                                     nsplits=tuple(tuple(ns) for ns in nsplits),
                                     chunks=out_chunks, dtypes=df.dtypes,
                                     index_value=df.index_value, columns_value=df.columns)

    @classmethod
    def _tile_both_series(cls, op):
        left, right = op.lhs, op.rhs
        df = op.outputs[0]

        nsplits, out_shape, left_chunks, right_chunks = align_series_series(left, right)

        out_chunks = []
        for idx, left_chunk, right_chunk in zip(range(out_shape[0]), left_chunks, right_chunks):
            out_chunk = op.copy().reset_key().new_chunk([left_chunk, right_chunk],
                                                        shape=(np.nan,), index=(idx,))
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
                    'shape': (df_chunk.shape[0], np.nan),
                    'index_value': df_chunk.index_value,
                }
            else:
                series_chunk = right_chunks[out_idx[0]]
                kw = {
                    'shape': (np.nan, df_chunk.shape[1]),
                    'columns_value': df_chunk.columns,
                }
            out_chunk = op.copy().reset_key().new_chunk([df_chunk, series_chunk], index=out_idx, **kw)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, df.shape,
                                     nsplits=tuple(tuple(ns) for ns in nsplits),
                                     chunks=out_chunks, dtypes=df.dtypes,
                                     index_value=df.index_value, columns_value=df.columns)

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
                }
            else:
                series_chunk = left_chunks[out_idx[0]]
                kw = {
                    'shape': (df_chunk.shape[0], np.nan),
                    'index_value': df_chunk.index_value,
                }
            out_chunk = op.copy().reset_key().new_chunk([series_chunk, df_chunk], index=out_idx, **kw)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, df.shape,
                                     nsplits=tuple(tuple(ns) for ns in nsplits),
                                     chunks=out_chunks, dtypes=df.dtypes,
                                     index_value=df.index_value, columns_value=df.columns)

    @classmethod
    def _tile_scalar(cls, op):
        tileable = op.rhs if np.isscalar(op.lhs) else op.lhs
        df = op.outputs[0]
        out_chunks = []
        for chunk in tileable.chunks:
            out_op = op.copy().reset_key()
            if isinstance(chunk, DATAFRAME_CHUNK_TYPE):
                out_chunk = out_op.new_chunk([chunk], shape=chunk.shape, index=chunk.index, dtypes=chunk.dtypes,
                                            index_value=chunk.index_value, columns_value=getattr(chunk, 'columns'))
            else:
                out_chunk = out_op.new_chunk([chunk], shape=chunk.shape, index=chunk.index, dtype=chunk.dtype,
                                            index_value=chunk.index_value, name=getattr(chunk, 'name'))
            out_chunks.append(out_chunk)

        new_op = op.copy()
        if isinstance(df, SERIES_TYPE):
            return new_op.new_seriess(op.inputs, df.shape, nsplits=tileable.nsplits, dtype=df.dtype,
                                      index_value=df.index_value, name=df.name, chunks=out_chunks)
        else:
            return new_op.new_dataframes(op.inputs, df.shape, nsplits=tileable.nsplits, dtypes=df.dtypes,
                                         index_value=df.index_value, columns_value=df.columns, chunks=out_chunks)

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

    @classmethod
    def execute(cls, ctx, op):
        if len(op.inputs) == 2:
            df, other = ctx[op.inputs[0].key], ctx[op.inputs[1].key]
            if isinstance(op.inputs[0], SERIES_CHUNK_TYPE) and isinstance(op.inputs[1], DATAFRAME_CHUNK_TYPE):
                df, other = other, df
                func_name = getattr(cls, '_rfunc_name')
            else:
                func_name = getattr(cls, '_func_name')
        elif np.isscalar(op.lhs):
            df = ctx[op.rhs.key]
            other = op.lhs
            func_name = getattr(cls, '_rfunc_name')
        else:
            df = ctx[op.lhs.key]
            other = op.rhs
            func_name = getattr(cls, '_func_name')
        if op.object_type == ObjectType.dataframe:
            kw = dict({'axis': op.axis})
        else:
            kw = dict()
        ctx[op.outputs[0].key] = getattr(df, func_name)(other, level=op.level, fill_value=op.fill_value, **kw)

    @classproperty
    def _operator(self):
        raise NotImplementedError

    @classmethod
    def _calc_properties(cls, x1, x2=None, axis='columns'):
        if isinstance(x1, (DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)) and (x2 is None or np.isscalar(x2)):
            # FIXME infer the dtypes of result df properly
            return {'shape': x1.shape, 'dtypes': x1.dtypes,
                    'columns_value': x1.columns, 'index_value': x1.index_value}

        if isinstance(x1, (SERIES_TYPE, SERIES_CHUNK_TYPE)) and (x2 is None or np.isscalar(x2)):
            dtype = find_common_type([x1.dtype, type(x2)])
            return {'shape': x1.shape, 'dtype': dtype, 'index_value': x1.index_value}

        if isinstance(x1, (DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)) and isinstance(x2, (DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)):
            index_shape, column_shape, dtypes, columns, index = np.nan, np.nan, None, None, None

            if x1.columns is not None and x2.columns is not None and \
                    x1.columns.key == x2.columns.key:
                dtypes = x1.dtypes
                columns = copy.copy(x1.columns)
                columns.value.should_be_monotonic = True
                column_shape = len(dtypes)
            elif x1.dtypes is not None and x2.dtypes is not None:
                dtypes = infer_dtypes(x1.dtypes, x2.dtypes, cls._operator)
                columns = parse_index(dtypes.index, store_data=True)
                columns.value.should_be_monotonic = True
                column_shape = len(dtypes)
            if x1.index_value is not None and x2.index_value is not None:
                if x1.index_value.key == x2.index_value.key:
                    index = copy.copy(x1.index_value)
                    index.value.should_be_monotonic = True
                    index_shape = x1.shape[0]
                else:
                    index = infer_index_value(x1.index_value, x2.index_value)
                    index.value.should_be_monotonic = True
                    if index.key == x1.index_value.key == x2.index_value.key and \
                            (not np.isnan(x1.shape[0]) or not np.isnan(x2.shape[0])):
                        index_shape = x1.shape[0] if not np.isnan(x1.shape[0]) else x2.shape[0]

            return {'shape': (index_shape, column_shape), 'dtypes': dtypes,
                    'columns_value': columns, 'index_value': index}

        if isinstance(x1, (DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)) and isinstance(x2, (SERIES_TYPE, SERIES_CHUNK_TYPE)):
            if axis == 'columns' or axis == 1:
                index_shape = x1.shape[0]
                index = x1.index_value
                column_shape, dtypes, columns = np.nan, None, None
                if x1.columns is not None and x1.index_value is not None:
                    if x1.columns.key == x2.index_value.key:
                        dtypes = x1.dtypes
                        columns = copy.copy(x1.columns)
                        columns.value.should_be_monotonic = True
                        column_shape = len(dtypes)
                    else:
                        dtypes = x1.dtypes  # FIXME
                        columns = infer_index_value(x1.columns, x2.index_value)
                        columns.value.should_be_monotonic = True
                        column_shape = np.nan
            else:
                assert axis == 'index' or axis == 0
                column_shape = x1.shape[1]
                columns = x1.columns
                dtypes = x1.dtypes
                index_shape, index = np.nan, None
                if x1.index_value is not None and x1.index_value is not None:
                    if x1.columns.key == x2.index_value.key:
                        index = copy.copy(x1.columns)
                        index.value.should_be_monotonic = True
                        index_shape = x1.shape[0]
                    else:
                        index = infer_index_value(x1.index_value, x2.index_value)
                        index.value.should_be_monotonic = True
                        index_shape = np.nan
            return {'shape': (index_shape, column_shape), 'dtypes': dtypes,
                    'columns_value': columns, 'index_value': index}

        if isinstance(x1, (SERIES_TYPE, SERIES_CHUNK_TYPE)) and isinstance(x2, (SERIES_TYPE, SERIES_CHUNK_TYPE)):
            index_shape, dtype, index = np.nan, None, None

            dtype = find_common_type([x1.dtype, x2.dtype])
            if x1.index_value is not None and x2.index_value is not None:
                if x1.index_value.key == x2.index_value.key:
                    index = copy.copy(x1.index_value)
                    index.value.should_be_monotonic = True
                    index_shape = x1.shape[0]
                else:
                    index = infer_index_value(x1.index_value, x2.index_value)
                    index.value.should_be_monotonic = True
                    if index.key == x1.index_value.key == x2.index_value.key and \
                            (not np.isnan(x1.shape[0]) or not np.isnan(x2.shape[0])):
                        index_shape = x1.shape[0] if not np.isnan(x1.shape[0]) else x2.shape[0]

            return {'shape': (index_shape,), 'dtype': dtype, 'index_value': index}

        raise NotImplementedError('Unknown combination of parameters')

    def _new_chunks(self, inputs, kws=None, **kw):
        if len(inputs) == 1:
            properties = self._calc_properties(*inputs)
        else:
            df1 = inputs[0] if isinstance(inputs[0], DATAFRAME_CHUNK_TYPE) else inputs[1]
            df2 = inputs[1] if isinstance(inputs[0], DATAFRAME_CHUNK_TYPE) else inputs[0]
            properties = self._calc_properties(df1, df2, axis=self.axis)

        shapes = [properties.pop('shape')]
        shapes.extend(kw_item.pop('shape') for kw_item in kws or ())
        if 'shape' in kw:
            shapes.append(kw.pop('shape'))
        shape = self._merge_shape(*shapes)

        for prop, value in properties.items():
            if kw.get(prop, None) is None:
                kw[prop] = value

        return super(DataFrameBinOpMixin, self)._new_chunks(
            inputs, shape=shape, kws=kws, **kw)

    def _call(self, x1, x2):
        if isinstance(x1, DATAFRAME_TYPE) or isinstance(x2, DATAFRAME_TYPE):
            df1 = x1 if isinstance(x1, DATAFRAME_TYPE) else x2
            df2 = x2 if isinstance(x1, DATAFRAME_TYPE) else x1
            setattr(self, '_object_type', ObjectType.dataframe)
            kw = self._calc_properties(df1, df2, axis=self.axis)
            if isinstance(df2, (DATAFRAME_TYPE, SERIES_TYPE)):
                return self.new_dataframe([x1, x2], **kw)
            else:
                return self.new_dataframe([df1], **kw)
        if isinstance(x1, SERIES_TYPE) or isinstance(x2, SERIES_TYPE):
            s1 = x1 if isinstance(x1, SERIES_TYPE) else x2
            s2 = x2 if isinstance(x1, SERIES_TYPE) else x1
            setattr(self, '_object_type', ObjectType.series)
            kw = self._calc_properties(s1, s2)
            if isinstance(s2, SERIES_TYPE):
                return self.new_series([x1, x2], **kw)
            else:
                return self.new_series([s1], **kw)
        raise NotImplementedError('Only support add dataframe or scalar for now')

    def __call__(self, x1, x2):
        if isinstance(x1, SERIES_TYPE) and isinstance(x2, DATAFRAME_TYPE):
            # reject invokeing series's op on dataframe
            raise NotImplementedError
        return self._call(x1, x2)

    def rcall(self, x1, x2):
        if isinstance(x1, SERIES_TYPE) and isinstance(x2, DATAFRAME_TYPE):
            # reject invokeing series's op on dataframe
            raise NotImplementedError
        return self._call(x2, x1)


class DataFrameUnaryOpMixin(DataFrameOperandMixin):
    __slots__ = ()

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        out_chunks = []
        for in_chunk in in_df.chunks:
            out_op = op.copy().reset_key()
            out_chunk = out_op.new_chunk([in_chunk], shape=in_chunk.shape, index=in_chunk.index,
                                         index_value=in_chunk.index_value, columns_value=in_chunk.columns)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, out_df.shape, dtypes=out_df.dtypes,
                                     index_value=out_df.index_value,
                                     columns_value=out_df.columns,
                                     chunks=out_chunks, nsplits=in_df.nsplits)

    @classmethod
    def execute(cls, ctx, op):
        df = ctx[op.inputs[0].key]
        func_name = getattr(cls, '_func_name')
        ctx[op.outputs[0].key] = getattr(df, func_name)()

    def __call__(self, df):
        return self.new_dataframe([df], df.shape, dtypes=df.dtypes,
                                  columns_value=df.columns, index_value=df.index_value)
