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

import numpy as np
import pandas as pd
from pandas.core.dtypes.cast import find_common_type
from pandas.core.indexing import IndexingError

from ...compat import six
from ...core import Entity, Base
from ...serialize import AnyField, KeyField, ListField
from ...tensor import asarray
from ...tensor.datasource.empty import empty
from ...tensor.indexing.core import calc_shape
from ... import opcodes as OperandDef
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType, DATAFRAME_TYPE
from ..utils import indexing_index_value
from .index_lib import DataFrameIlocIndexesHandler


_ILOC_ERROR_MSG = 'Location based indexing can only have [integer, ' \
                  'integer slice (START point is INCLUDED, END point is EXCLUDED), ' \
                  'listlike of integers, boolean array] types'


def process_iloc_indexes(inp, indexes):
    ndim = inp.ndim

    if not isinstance(indexes, tuple):
        indexes = (indexes,)
    if len(indexes) < ndim:
        indexes += (slice(None),) * (ndim - len(indexes))
    if len(indexes) > ndim:
        raise IndexingError('Too many indexers')

    new_indexes = []
    # check each index
    for ax, index in enumerate(indexes):
        if isinstance(index, tuple):
            # a tuple should already have been caught by this point
            # so don't treat a tuple as a valid indexer
            raise IndexingError("Too many indexers")
        elif isinstance(index, slice):
            pd_index = (inp.index_value if ax == 0 else inp.columns_value).to_pandas()
            for val in [index.start, index.stop, index.step]:
                if val is not None:
                    try:
                        pd_index[val]  # check on the pandas
                    except IndexError:
                        pass
                    except TypeError:
                        raise TypeError(
                            'cannot do slice indexing on {} '
                            'with these indexers [{}] '
                            'of {}'.format(type(pd_index), val, type(val)))
            new_indexes.append(index)
        elif isinstance(index, (list, np.ndarray, pd.Series, Base, Entity)):
            if not isinstance(index, (Base, Entity)):
                index = np.asarray(index)
            else:
                index = asarray(index)
                if ax == 1:
                    # do not support tensor index on axis 1
                    # because if so, the dtypes and columns_value would be unknown
                    try:
                        index = index.fetch()
                    except (RuntimeError, ValueError):
                        raise NotImplementedError('indexer on axis columns cannot be '
                                                  'non-executed tensor')
            if index.dtype != np.bool_:
                index = index.astype(np.int64)
            if index.ndim != 1:
                raise ValueError('Buffer has wrong number of dimensions '
                                 '(expected 1, got {})'.format(index.ndim))
            new_indexes.append(index)
        elif isinstance(index, Integral):
            shape = inp.shape[ax]
            if not np.isnan(shape):
                if index < -shape or index >= shape:
                    raise IndexError('single positional indexer is out-of-bounds')
            new_indexes.append(index)
        else:
            raise ValueError(_ILOC_ERROR_MSG)

    return new_indexes


class DataFrameIloc(object):
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, indexes):
        if isinstance(self._obj, DATAFRAME_TYPE):
            op = DataFrameIlocGetItem(indexes=process_iloc_indexes(self._obj, indexes))
        else:
            op = SeriesIlocGetItem(indexes=process_iloc_indexes(self._obj, indexes))
        return op(self._obj)

    def __setitem__(self, indexes, value):
        if not np.isscalar(value):
            raise NotImplementedError('Only scalar value is supported to set by iloc')

        if isinstance(self._obj, DATAFRAME_TYPE):
            op = DataFrameIlocSetItem(indexes=process_iloc_indexes(self._obj, indexes), value=value)
        else:
            op = SeriesIlocSetItem(indexes=process_iloc_indexes(self._obj, indexes), value=value)

        ret = op(self._obj)
        self._obj.data = ret.data


class DataFrameIlocGetItem(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_ILOC_GETITEM

    _input = KeyField('input')
    _indexes = ListField('indexes')

    def __init__(self, indexes=None, gpu=False, sparse=False, object_type=ObjectType.dataframe, **kw):
        super(DataFrameIlocGetItem, self).__init__(_indexes=indexes,
                                                   _gpu=gpu, _sparse=sparse,
                                                   _object_type=object_type, **kw)

    @property
    def input(self):
        return self._input

    @property
    def indexes(self):
        return self._indexes

    def _set_inputs(self, inputs):
        super(DataFrameIlocGetItem, self)._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        self._input = next(inputs_iter)
        indexes = []
        for index in self._indexes:
            if isinstance(index, (Entity, Base)):
                indexes.append(next(inputs_iter))
            else:
                indexes.append(index)
        self._indexes = tuple(indexes)

    def __call__(self, df):
        # Note [Fancy Index of Numpy and Pandas]
        #
        # The numpy and pandas.iloc have different semantic when processing fancy index:
        #
        # >>> np.ones((3,3))[[1,2],[1,2]]
        # array([1., 1.])
        #
        # >>> pd.DataFrame(np.ones((3,3))).iloc[[1,2],[1,2]]
        #    1    2
        # 1  1.0  1.0
        # 2  1.0  1.0
        #
        # Thus, we processing the index along two axis of DataFrame separately.
        shape0 = tuple(calc_shape((df.shape[0],), (self.indexes[0],)))
        shape1 = tuple(calc_shape((df.shape[1],), (self.indexes[1],)))

        inputs = [df] + [index for index in self._indexes if isinstance(index, (Base, Entity))]

        # NB: pandas only compresses the result to series when index on one of axis is integral
        if isinstance(self.indexes[1], Integral):
            shape = shape0
            dtype = df.dtypes.iloc[self.indexes[1]]
            index_value = indexing_index_value(df.index_value, self.indexes[0])
            if isinstance(self.indexes[0], Integral):
                # scalar
                self._object_type = ObjectType.scalar
                return self.new_scalar(inputs, dtype=dtype)
            else:
                self._object_type = ObjectType.series
                return self.new_series(inputs, shape=shape, dtype=dtype, index_value=index_value)
        elif isinstance(self.indexes[0], Integral):
            shape = shape1
            dtype = find_common_type(df.dtypes.iloc[self.indexes[1]].values)
            index_value = indexing_index_value(df.columns_value, self.indexes[1])
            self._object_type = ObjectType.series
            return self.new_series(inputs, shape=shape, dtype=dtype, index_value=index_value)
        else:
            return self.new_dataframe(inputs, shape=shape0 + shape1, dtypes=df.dtypes.iloc[self.indexes[1]],
                                      index_value=indexing_index_value(df.index_value, self.indexes[0]),
                                      columns_value=indexing_index_value(df.columns_value, self.indexes[1],
                                                                         store_data=True))

    # FIXME The view behavior of DataFrame.iloc
    #
    # The pandas's iloc has complicated behavior about whether to create a view or not, it depends
    # on the further operation on the view, as illustrated by the following example:
    #
    # >>> df = pd.DataFrame([[1,2], [3,4]])
    # >>> x = df.iloc[:]
    # >>> df
    #    0  1
    # 0  1  2
    # 1  3  4
    # >>> x
    #    0  1
    # 0  1  2
    # 1  3  4
    #
    # >>> x.iloc[:] = 1000
    # >>> x
    #       0     1
    # 0  1000  1000
    # 1  1000  1000
    # df
    #       0     1
    # 0  1000  1000
    # 1  1000  1000
    #
    # >>> x.iloc[:] = 2000.0
    # >>> x
    #         0       1
    # 0  2000.0  2000.0
    # 1  2000.0  2000.0
    # >>> df
    #       0     1
    # 0  1000  1000
    # 1  1000  1000

    @classmethod
    def tile(cls, op):
        handler = DataFrameIlocIndexesHandler()
        return [handler.handle(op)]

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]

        df = ctx[op.input.key]
        if len(op.inputs) > 1:
            indexes = tuple(ctx[index.key] if hasattr(index, 'key') else index
                            for index in op.indexes)
        else:
            indexes = tuple(op.indexes)
        if six.PY2:
            # for python 2, indexes requires to be writeable
            # thus copy them first if have to
            new_indexes = []
            for ind in indexes:
                if hasattr(ind, 'flags') and not ind.flags.writeable:
                    new_indexes.append(ind.copy())
                else:
                    new_indexes.append(ind)
            indexes = tuple(new_indexes)

        r = df.iloc[indexes]
        if isinstance(r, pd.Series) and r.dtype != chunk.dtype:
            r = r.astype(chunk.dtype)
        ctx[chunk.key] = r


class DataFrameIlocSetItem(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_ILOC_SETITEM

    _indexes = ListField('indexes')
    _value = AnyField('value')

    def __init__(self, indexes=None, value=None, gpu=False, sparse=False,
                 object_type=ObjectType.dataframe, **kw):
        super(DataFrameIlocSetItem, self).__init__(_indexes=indexes, _value=value,
                                                   _gpu=gpu, _sparse=sparse,
                                                   _object_type=object_type, **kw)

    @property
    def indexes(self):
        return self._indexes

    @property
    def value(self):
        return self._value

    def __call__(self, df):
        return self.new_dataframe([df], shape=df.shape, dtypes=df.dtypes,
                                  index_value=df.index_value, columns_value=df.columns_value)

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        # See Note [Fancy Index of Numpy and Pandas]
        tensor0 = empty(in_df.shape[0], chunk_size=(in_df.nsplits[0],))[op.indexes[0]].tiles()
        tensor1 = empty(in_df.shape[1], chunk_size=(in_df.nsplits[1],))[op.indexes[1]].tiles()

        chunk_mapping = {c0.inputs[0].index + c1.inputs[0].index: (c0, c1)
                         for c0, c1 in itertools.product(tensor0.chunks, tensor1.chunks)}

        out_chunks = []
        for chunk in in_df.chunks:
            if chunk.index not in chunk_mapping:
                out_chunks.append(chunk)
            else:
                chunk_op = op.copy().reset_key()
                index_chunk, column_chunk = chunk_mapping[chunk.index]
                chunk_op._indexes = (index_chunk.op.indexes[0], column_chunk.op.indexes[0])
                chunk_op._value = op.value
                out_chunk = chunk_op.new_chunk([chunk],
                                               shape=chunk.shape, index=chunk.index, dtypes=chunk.dtypes,
                                               index_value=chunk.index_value, columns_value=chunk.columns_value)
                out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, shape=out_df.shape, dtypes=out_df.dtypes,
                                     index_value=out_df.index_value, columns_value=out_df.columns_value,
                                     chunks=out_chunks, nsplits=in_df.nsplits)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        r = ctx[op.inputs[0].key].copy(deep=True)
        r.iloc[tuple(op.indexes)] = op.value
        ctx[chunk.key] = r


class SeriesIlocGetItem(DataFrameOperand, DataFrameOperandMixin):
    _op_module_ = 'series'
    _op_type_ = OperandDef.DATAFRAME_ILOC_GETITEM

    _input = KeyField('input')
    _indexes = ListField('indexes')

    def __init__(self, indexes=None, gpu=False, sparse=False, object_type=ObjectType.series, **kw):
        super(SeriesIlocGetItem, self).__init__(_indexes=indexes, _gpu=gpu, _sparse=sparse,
                                                _object_type=object_type, **kw)

    @property
    def input(self):
        return self._input

    @property
    def indexes(self):
        return self._indexes

    def _set_inputs(self, inputs):
        super(SeriesIlocGetItem, self)._set_inputs(inputs)

        inputs_iter = iter(self._inputs)
        self._input = next(inputs_iter)

        indexes = []
        for index in self._indexes:
            if isinstance(index, (Entity, Base)):
                indexes.append(next(inputs_iter))
            else:
                indexes.append(index)
        self._indexes = tuple(indexes)

    @classmethod
    def tile(cls, op):
        handler = DataFrameIlocIndexesHandler()
        return [handler.handle(op)]

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        series = ctx[op.input.key]
        if len(op.inputs) > 1:
            indexes = tuple(ctx[index.key] if hasattr(index, 'key') else index
                            for index in op.indexes)
        else:
            indexes = tuple(op.indexes)
        if hasattr(series, 'iloc'):
            ctx[chunk.key] = series.iloc[indexes]
        else:
            # index, only happen for calling from rechunk
            ctx[chunk.key] = series[indexes]

    def __call__(self, series):
        shape = tuple(calc_shape(series.shape, self.indexes))
        index_value = indexing_index_value(series.index_value, self.indexes[0])
        inputs = [series] + [index for index in self._indexes if isinstance(index, (Base, Entity))]
        return self.new_series(inputs, shape=shape, dtype=series.dtype, index_value=index_value)


class SeriesIlocSetItem(DataFrameOperand, DataFrameOperandMixin):
    _op_module_ = 'series'
    _op_type_ = OperandDef.DATAFRAME_ILOC_SETITEM

    _indexes = ListField('indexes')
    _value = AnyField('value')

    def __init__(self, indexes=None, value=None, gpu=False, sparse=False, **kw):
        super(SeriesIlocSetItem, self).__init__(_indexes=indexes, _value=value, _gpu=gpu, _sparse=sparse,
                                                _object_type=ObjectType.series, **kw)

    @property
    def indexes(self):
        return self._indexes

    @property
    def value(self):
        return self._value

    def __call__(self, series):
        return self.new_series([series], shape=series.shape, dtype=series.dtype,
                               index_value=series.index_value, name=series.name)

    @classmethod
    def tile(cls, op):
        in_series = op.inputs[0]
        out = op.outputs[0]

        # Reuse the logic of fancy indexing in tensor module.
        tensor = empty(in_series.shape, chunk_size=in_series.nsplits)[op.indexes[0]].tiles()

        chunk_mapping = dict((c.inputs[0].index, c) for c in tensor.chunks)

        out_chunks = []
        for chunk in in_series.chunks:
            if chunk.index not in chunk_mapping:
                out_chunks.append(chunk)
            else:
                chunk_op = op.copy().reset_key()
                index_chunk = chunk_mapping[chunk.index]
                chunk_op._indexes = index_chunk.op.indexes
                chunk_op._value = op.value
                out_chunk = chunk_op.new_chunk([chunk], shape=chunk.shape, index=chunk.index, dtype=chunk.dtype,
                                               index_value=chunk.index_value, name=chunk.name)
                out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_seriess(op.inputs, shape=out.shape, dtype=out.dtype,
                                  index_value=out.index_value, name=out.name,
                                  chunks=out_chunks, nsplits=in_series.nsplits)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        r = ctx[op.inputs[0].key].copy(deep=True)
        r.iloc[tuple(op.indexes)] = op.value
        ctx[chunk.key] = r


def iloc(a):
    return DataFrameIloc(a)


def head(a, n=5):
    return DataFrameIloc(a)[0:n]


def tail(a, n=5):
    return DataFrameIloc(a)[-n:]
