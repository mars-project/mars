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

import itertools
from numbers import Integral

import pandas as pd
import numpy as np

from ... import opcodes as OperandDef
from ...config import options
from ...core import Base, Entity
from ...serialize import AnyField, Int32Field, BoolField
from ...tensor.core import TENSOR_TYPE
from ...tensor.datasource import tensor as astensor
from ...utils import check_chunks_unknown_shape
from ...tiles import TilesError
from ..align import align_dataframe_series, align_dataframe_dataframe
from ..core import SERIES_TYPE, SERIES_CHUNK_TYPE, DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE
from ..merge import DataFrameConcat
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import parse_index, in_range_index
from .utils import calc_columns_index


class SeriesIndex(DataFrameOperand, DataFrameOperandMixin):
    _op_module_ = 'series'
    _op_type_ = OperandDef.INDEX

    _labels = AnyField('labels')

    _combine_size = Int32Field('combine_size')
    _is_intermediate = BoolField('is_intermediate')

    def __init__(self, labels=None, combine_size=None, is_intermediate=None, object_type=None, **kw):
        super().__init__(_labels=labels, _combine_size=combine_size, _is_intermediate=is_intermediate,
                         _object_type=object_type, **kw)

    @property
    def labels(self):
        return self._labels

    @property
    def combine_size(self):
        return self._combine_size

    @property
    def is_intermediate(self):
        return self._is_intermediate

    def __call__(self, series, name=None):
        return self.new_tileable([series], dtype=series.dtype, name=name)

    def _new_tileables(self, inputs, kws=None, **kw):
        # Override this method to automatically decide the output type,
        # when `labels` is a list, we will set `object_type` as series,
        # otherwise it will be a scalar.
        object_type = getattr(self, '_object_type', None)
        shape = kw.pop('shape', None)
        is_scalar = not isinstance(self._labels, list)
        if object_type is None:
            object_type = ObjectType.scalar if is_scalar else ObjectType.series
            self._object_type = object_type
        if shape is None:
            shape = () if is_scalar else ((len(self._labels)),)
            kw['shape'] = shape
        if not is_scalar:
            index_value = kw.pop('index_value', None) or parse_index(pd.Index(self._labels))
            kw['index_value'] = index_value
        return super()._new_tileables(inputs, kws=kws, **kw)

    def _new_chunks(self, inputs, kws=None, **kw):
        # Override this method to automatically decide the output type,
        # when `labels` is a list, we will set `object_type` as series,
        # otherwise it will be a scalar.
        object_type = getattr(self, '_object_type', None)
        is_scalar = not isinstance(self._labels, list)
        if object_type is None:
            object_type = ObjectType.scalar if is_scalar else ObjectType.series
            self._object_type = object_type
        if kw.get('shape', None) is None:
            shape = () if is_scalar else ((len(self._labels)),)
            kw['shape'] = shape
        if not is_scalar:
            index_value = kw.pop('index_value', None) or parse_index(pd.Index(self._labels))
            kw['index_value'] = index_value
        else:
            # tensor chunk cannot accept index_value
            kw.pop('index_value', None)
        return super()._new_chunks(inputs, kws=kws, **kw)

    @classmethod
    def _calc_chunk_index(cls, label, chunk_indexes):
        for i, index in enumerate(chunk_indexes):
            if isinstance(index, pd.RangeIndex) and in_range_index(label, index):
                return i
            elif label in index:
                return i
        raise TypeError("label %s doesn't exist" % label)

    @classmethod
    def _tile_one_chunk(cls, op):
        in_series = op.inputs[0]
        out_series = op.outputs[0]

        index_op = SeriesIndex(labels=op.labels)
        kw = {'name': out_series.name} if hasattr(out_series, 'name') else {}
        index_chunk = index_op.new_chunk(in_series.chunks, dtype=out_series.dtype, **kw)
        new_op = op.copy()
        nsplits = ((len(op.labels),),) if isinstance(op.labels, list) else ()
        return new_op.new_tileables(op.inputs, chunks=[index_chunk], nsplits=nsplits,
                                    dtype=out_series.dtype)

    @classmethod
    def _tree_getitem(cls, op):
        """
        DataFrame doesn't store the index value except RangeIndex or specify `store=True` in `parse_index`,
        So we build a tree structure to avoid too much dependence for getitem node.
        """
        out_series = op.outputs[0]
        combine_size = options.combine_size
        chunks = op.inputs[0].chunks
        while len(chunks) > combine_size:
            new_chunks = []
            for i in range(0, len(chunks), combine_size):
                chks = chunks[i: i + combine_size]
                if len(chks) == 1:
                    chk = chks[0]
                else:
                    concat_op = DataFrameConcat(object_type=ObjectType.series)
                    chk = concat_op.new_chunk(chks, dtype=chks[0].dtype)
                chk_op = SeriesIndex(labels=op.labels, is_intermediate=True)
                kw = {'name': out_series.name} if hasattr(out_series, 'name') else {}
                chk = chk_op.new_chunk([chk], shape=(np.nan,), dtype=chk.dtype,
                                       index_value=parse_index(pd.RangeIndex(-1)), **kw)
                new_chunks.append(chk)
            chunks = new_chunks

        concat_op = DataFrameConcat(object_type=ObjectType.series)
        kw = {'name': out_series.name} if hasattr(out_series, 'name') else {}
        kw['index'] = (0,)
        chk = concat_op.new_chunk(chunks, dtype=chunks[0].dtype, **kw)
        index_op = SeriesIndex(labels=op.labels)
        chunk = index_op.new_chunk([chk], dtype=chk.dtype, **kw)
        new_op = op.copy()
        nsplits = ((len(op.labels),),) if isinstance(op.labels, list) else ()
        kw = out_series.params
        kw['nsplits'] = nsplits
        kw['chunks'] = [chunk]
        return new_op.new_tileables(op.inputs, kws=[kw])

    @classmethod
    def tile(cls, op):
        in_series = op.inputs[0]
        out_series = op.outputs[0]

        if len(in_series.chunks) == 1:
            return cls._tile_one_chunk(op)
        if not in_series.index_value.has_value():
            return cls._tree_getitem(op)

        chunk_indexes = [c.index_value.to_pandas() for c in in_series.chunks]
        if not isinstance(op.labels, list):
            selected_chunk = in_series.chunks[cls._calc_chunk_index(op.labels, chunk_indexes)]
            index_op = op.copy().reset_key()
            out_chunk = index_op.new_chunk([selected_chunk], shape=(), dtype=selected_chunk.dtype)
            new_op = op.copy()
            return new_op.new_scalars(op.inputs, dtype=out_series.dtype, chunks=[out_chunk])
        else:
            # When input series's index is RangeIndex(5), chunk_size is 3, and labels is [4, 2, 3, 4],
            # Combine the labels in the same chunk, so the splits will be [[4], [2], [3, 4]],
            # the corresponding chunk index is [1, 0, 1].
            selected_index = [cls._calc_chunk_index(label, chunk_indexes) for label in op.labels]
            condition = np.where(np.diff(selected_index))[0] + 1
            column_splits = np.split(op.labels, condition)
            column_indexes = np.split(selected_index, condition)

            out_chunks = []
            nsplits = []
            for i, (labels, idx) in enumerate(zip(column_splits, column_indexes)):
                index_op = SeriesIndex(labels=list(labels))
                c = in_series.chunks[idx[0]]
                nsplits.append(len(labels))
                index_value = parse_index(pd.Index([], dtype=c.index_value.to_pandas().dtype),
                                          c, labels)
                out_chunks.append(index_op.new_chunk([c], shape=(len(labels),), dtype=c.dtype,
                                                     index_value=index_value,
                                                     name=c.name, index=(i,)))
            new_op = op.copy()
            return new_op.new_seriess(op.inputs, shape=out_series.shape, dtype=out_series.dtype,
                                      index_value=out_series.index_value, nsplits=(tuple(nsplits),),
                                      chunks=out_chunks, name=out_series.name)

    @classmethod
    def execute(cls, ctx, op):
        series = ctx[op.inputs[0].key]
        labels = op.labels
        if op.is_intermediate:
            # for intermediate result, it is always a series even if labels is a scalar.
            labels = labels if isinstance(labels, list) else [labels]
            labels = [label for label in set(labels) if label in series]
        ctx[op.outputs[0].key] = series[labels]


class DataFrameIndex(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.INDEX

    _col_names = AnyField('col_names')
    _mask = AnyField('mask')

    def __init__(self, col_names=None, mask=None, object_type=ObjectType.series, **kw):
        super().__init__(_col_names=col_names, _mask=mask, _object_type=object_type, **kw)

    @property
    def col_names(self):
        return self._col_names

    @property
    def mask(self):
        return self._mask

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if isinstance(self._col_names, (Base, Entity)):
            self._col_names = self._inputs[0]
        if isinstance(self._mask, (Base, Entity)):
            self._mask = self._inputs[-1]

    def __call__(self, df):
        if self.col_names is not None:
            # if col_names is a list, return a DataFrame, else return a Series
            if isinstance(self._col_names, list):
                dtypes = df.dtypes[self._col_names]
                columns = parse_index(pd.Index(self._col_names), store_data=True)
                return self.new_dataframe([df], shape=(df.shape[0], len(self._col_names)), dtypes=dtypes,
                                          index_value=df.index_value, columns_value=columns)
            else:
                dtype = df.dtypes[self._col_names]
                return self.new_series([df], shape=(df.shape[0],), dtype=dtype, index_value=df.index_value,
                                       name=self._col_names)
        else:
            if isinstance(self.mask, (SERIES_TYPE, DATAFRAME_TYPE)):
                index_value = parse_index(pd.Index([], dtype=df.index_value.to_pandas().dtype),
                                          df, self._mask)
                return self.new_dataframe([df, self._mask], shape=(np.nan, df.shape[1]), dtypes=df.dtypes,
                                          index_value=index_value, columns_value=df.columns_value)
            else:
                index_value = parse_index(pd.Index([], dtype=df.index_value.to_pandas().dtype),
                                          df, self._mask)
                return self.new_dataframe([df], shape=(np.nan, df.shape[1]), dtypes=df.dtypes,
                                          index_value=index_value, columns_value=df.columns_value)

    @classmethod
    def tile(cls, op):
        if op.col_names is not None:
            return cls.tile_with_columns(op)
        else:
            return cls.tile_with_mask(op)

    @classmethod
    def tile_with_mask(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        out_chunks = []

        if isinstance(op.mask, (SERIES_TYPE, DATAFRAME_TYPE)):
            mask = op.inputs[1]

            if isinstance(op.mask, SERIES_TYPE):
                nsplits, out_shape, df_chunks, mask_chunks = \
                    align_dataframe_series(in_df, mask, axis='index')
            else:
                nsplits, out_shape, df_chunks, mask_chunks = \
                    align_dataframe_dataframe(in_df, mask)
            out_chunk_indexes = itertools.product(*(range(s) for s in out_shape))

            out_chunks = []
            for i, idx, df_chunk in zip(itertools.count(), out_chunk_indexes, df_chunks):
                if op.mask.ndim == 1:
                    mask_chunk = mask_chunks[df_chunk.index[0]]
                else:
                    mask_chunk = mask_chunks[i]
                index_value = parse_index(out_df.index_value.to_pandas(), df_chunk)
                out_chunk = op.copy().reset_key().new_chunk([df_chunk, mask_chunk], index=idx,
                                                            shape=(np.nan, df_chunk.shape[1]),
                                                            dtypes=df_chunk.dtypes,
                                                            index_value=index_value,
                                                            columns_value=df_chunk.columns_value)
                out_chunks.append(out_chunk)

            nsplits = ((np.nan,) * len(nsplits[0]), nsplits[1])
        else:
            check_chunks_unknown_shape([in_df], TilesError)
            nsplits_acc = np.cumsum((0,) + in_df.nsplits[0])
            for idx in range(in_df.chunk_shape[0]):
                for idxj in range(in_df.chunk_shape[1]):
                    in_chunk = in_df.cix[idx, idxj]
                    chunk_op = op.copy().reset_key()
                    chunk_op._mask = op.mask.iloc[nsplits_acc[idx]:nsplits_acc[idx+1]]
                    out_chunk = chunk_op.new_chunk([in_chunk], index=in_chunk.index,
                                                   shape=(np.nan, in_chunk.shape[1]),
                                                   dtypes=in_chunk.dtypes,
                                                   index_value=in_df.index_value,
                                                   columns_value=in_chunk.columns_value)
                    out_chunks.append(out_chunk)

        nsplits_on_columns = tuple(c.shape[1] for c in out_chunks if c.index[0] == 0)
        row_chunk_num = len([c.shape[0] for c in out_chunks if c.index[1] == 0])
        nsplits = ((np.nan,) * row_chunk_num, nsplits_on_columns)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, shape=out_df.shape, dtypes=out_df.dtypes,
                                     index_value=out_df.index_value, columns_value=out_df.columns_value,
                                     chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def tile_with_columns(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]
        col_names = op.col_names
        if not isinstance(col_names, list):
            column_index = calc_columns_index(col_names, in_df)
            out_chunks = []
            dtype = in_df.dtypes[col_names]
            for i in range(in_df.chunk_shape[0]):
                c = in_df.cix[(i, column_index)]
                op = DataFrameIndex(col_names=col_names)
                out_chunks.append(op.new_chunk([c], shape=(c.shape[0],), index=(i,), dtype=dtype,
                                               index_value=c.index_value, name=col_names))
            new_op = op.copy()
            return new_op.new_seriess(op.inputs, shape=out_df.shape, dtype=out_df.dtype,
                                      index_value=out_df.index_value, name=out_df.name,
                                      nsplits=(in_df.nsplits[0],), chunks=out_chunks)
        else:
            # combine columns into one chunk and keep the columns order at the same time.
            # When chunk columns are ['c1', 'c2', 'c3'], ['c4', 'c5'],
            # selected columns are ['c2', 'c3', 'c4', 'c2'], `column_splits` will be
            # [(['c2', 'c3'], 0), ('c4', 1), ('c2', 0)].
            selected_index = [calc_columns_index(col, in_df) for col in col_names]
            condition = np.where(np.diff(selected_index))[0] + 1
            column_splits = np.split(col_names, condition)
            column_indexes = np.split(selected_index, condition)

            out_chunks = [[] for _ in range(in_df.chunk_shape[0])]
            column_nsplits = []
            for i, (columns, column_idx) in enumerate(zip(column_splits, column_indexes)):
                dtypes = in_df.dtypes[columns]
                column_nsplits.append(len(columns))
                for j in range(in_df.chunk_shape[0]):
                    c = in_df.cix[(j, column_idx[0])]
                    index_op = DataFrameIndex(col_names=list(columns), object_type=ObjectType.dataframe)
                    out_chunk = index_op.new_chunk([c], shape=(c.shape[0], len(columns)), index=(j, i),
                                                   dtypes=dtypes, index_value=c.index_value,
                                                   columns_value=parse_index(pd.Index(columns),
                                                                             store_data=True))
                    out_chunks[j].append(out_chunk)
            out_chunks = [item for cl in out_chunks for item in cl]
            new_op = op.copy()
            nsplits = (in_df.nsplits[0], tuple(column_nsplits))
            return new_op.new_dataframes(op.inputs, shape=out_df.shape, dtypes=out_df.dtypes,
                                         index_value=out_df.index_value,
                                         columns_value=out_df.columns_value,
                                         chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        if op.mask is None:
            df = ctx[op.inputs[0].key]
            ctx[op.outputs[0].key] = df[op.col_names]
        else:
            df = ctx[op.inputs[0].key]
            if isinstance(op.mask, (SERIES_CHUNK_TYPE, DATAFRAME_CHUNK_TYPE)):
                mask = ctx[op.inputs[1].key]
            else:
                mask = op.mask
            mask = mask.reindex_like(df).fillna(False)
            if mask.ndim == 2:
                mask = mask[df.columns.tolist()]
            ctx[op.outputs[0].key] = df[mask]


_list_like_types = (list, np.ndarray, SERIES_TYPE, pd.Series, TENSOR_TYPE)


def dataframe_getitem(df, item):
    columns = df.columns_value.to_pandas()

    if isinstance(item, (np.ndarray, pd.Series)) and item.dtype != np.bool_:
        item = item.tolist()

    if isinstance(item, slice):
        edge = item.start if item.start is not None else item.stop
        if isinstance(edge, Integral):
            return df.iloc[item]
        else:
            return df.loc[item]
    elif isinstance(item, list):
        for col_name in item:
            if col_name not in columns:
                raise KeyError('%s not in columns' % col_name)
        op = DataFrameIndex(col_names=item, object_type=ObjectType.dataframe)
    elif isinstance(item, _list_like_types) or hasattr(item, 'dtypes'):
        # NB: don't enforce the dtype of `item` to be `bool` since it may be unknown
        op = DataFrameIndex(mask=item, object_type=ObjectType.dataframe)
    else:
        if item not in columns:
            raise KeyError('%s not in columns' % item)
        op = DataFrameIndex(col_names=item)
    return op(df)


def series_getitem(series, labels, combine_size=None):
    if isinstance(labels, list) or np.isscalar(labels):
        op = SeriesIndex(labels=labels, combine_size=combine_size)
        return op(series, name=series.name)
    elif isinstance(labels, _list_like_types) and astensor(labels).dtype == np.bool:
        return series.loc[labels]
    elif isinstance(labels, slice):
        edge = labels.start if labels.start is not None else labels.stop
        if isinstance(edge, Integral):
            return series.iloc[labels]
        else:
            return series.loc[labels]
    else:
        raise NotImplementedError('type %s is not support for getitem' % type(labels))
