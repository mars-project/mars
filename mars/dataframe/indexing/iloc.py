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
from numbers import Integral

import numpy as np
import pandas as pd
from pandas.core.dtypes.cast import find_common_type
from pandas.core.indexing import IndexingError

from ... import opcodes as OperandDef
from ...core import ENTITY_TYPE, OutputType, recursive_tile
from ...config import options
from ...serialization.serializables import AnyField, KeyField, ListField
from ...tensor import asarray
from ...tensor.datasource.empty import empty
from ...tensor.indexing.core import calc_shape
from ...utils import ceildiv
from ..operands import DataFrameOperand, DataFrameOperandMixin, DATAFRAME_TYPE
from ..utils import indexing_index_value, is_cudf
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
            if any(v is not None for v in [index.start, index.stop, index.step]):
                pd_index = (inp.index_value if ax == 0 else inp.columns_value).to_pandas()
                for val in [index.start, index.stop, index.step]:
                    if val is not None:
                        try:
                            pd_index[val]  # check on the pandas
                        except IndexError:
                            pass
                        except TypeError:
                            raise TypeError(f'cannot do slice indexing on {type(pd_index)} '
                                            f'with these indexers [{val}] of {type(val)}')
            new_indexes.append(index)
        elif isinstance(index, (list, np.ndarray, pd.Series, ENTITY_TYPE)):
            if not isinstance(index, ENTITY_TYPE):
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
                                 f'(expected 1, got {index.ndim})')
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


class DataFrameIloc:
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


class HeadTailOptimizedOperandMixin(DataFrameOperandMixin):
    __slots__ = ()

    @classmethod
    def _is_head(cls, index0):
        return (index0.start is None or index0.start == 0) and \
               index0.stop is not None and index0.stop > 0

    @classmethod
    def _is_tail(cls, index0):
        return index0.start is not None and index0.start < 0 and \
               index0.stop is None

    @classmethod
    def _is_indexes_head_or_tail(cls, indexes):
        index0 = indexes[0]
        if not isinstance(index0, slice):
            # have to be slice
            return False
        if index0.step is not None and index0.step != 1:
            return False
        if len(indexes) == 2:
            if not isinstance(indexes[1], slice):
                return False
            if indexes[1] != slice(None):
                return False
        if cls._is_tail(index0):
            # tail
            return True
        if cls._is_head(index0):
            # head
            return True
        return False

    @classmethod
    def _need_tile_head_tail(cls, op):
        # first, the input DataFrame should
        # have unknown chunk shapes on the index axis,
        inp = op.input
        if not any(np.isnan(s) for s in inp.nsplits[0]):
            return False

        # if input is a DataFrame,
        # should have 1 chunk on columns axis
        if inp.ndim > 1 and inp.chunk_shape[1] > 1:
            return False

        return cls._is_indexes_head_or_tail(op.indexes)

    @classmethod
    def _tile_head_tail(cls, op):
        from ..merge import DataFrameConcat

        inp = op.input
        out = op.outputs[0]
        combine_size = options.combine_size

        chunks = inp.chunks

        new_chunks = []
        for c in chunks:
            chunk_op = op.copy().reset_key()
            params = out.params
            params['index'] = c.index
            params['shape'] = c.shape if np.isnan(c.shape[0]) else out.shape
            new_chunks.append(chunk_op.new_chunk([c], kws=[params]))
        chunks = new_chunks

        while len(chunks) > 1:
            new_size = ceildiv(len(chunks), combine_size)
            new_chunks = []
            for i in range(new_size):
                in_chunks = chunks[combine_size * i: combine_size * (i + 1)]
                chunk_index = (i, 0) if in_chunks[0].ndim == 2 else (i,)
                if len(inp.shape) == 1:
                    shape = (sum(c.shape[0] for c in in_chunks),)
                else:
                    shape = (sum(c.shape[0] for c in in_chunks), in_chunks[0].shape[1])
                concat_chunk = DataFrameConcat(
                    axis=0, output_types=in_chunks[0].op.output_types).new_chunk(
                    in_chunks, index=chunk_index, shape=shape)
                chunk_op = op.copy().reset_key()
                params = out.params
                params['index'] = chunk_index
                params['shape'] = in_chunks[0].shape if np.isnan(in_chunks[0].shape[0]) else out.shape
                new_chunks.append(chunk_op.new_chunk([concat_chunk], kws=[params]))
            chunks = new_chunks

        new_op = op.copy()
        params = out.params
        params['nsplits'] = tuple((s,) for s in out.shape)
        params['chunks'] = chunks
        return new_op.new_tileables(op.inputs, kws=[params])

    def can_be_optimized(self):
        return self._is_indexes_head_or_tail(self._indexes) and \
               self._is_head(self._indexes[0]) and \
               self._indexes[0].stop <= options.optimize.head_optimize_threshold

    @classmethod
    def tile(cls, op):
        if cls._need_tile_head_tail(op):
            return cls._tile_head_tail(op)


class DataFrameIlocGetItem(DataFrameOperand, HeadTailOptimizedOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_ILOC_GETITEM

    _input = KeyField('input')
    _indexes = ListField('indexes')

    def __init__(self, indexes=None, gpu=False, sparse=False, output_types=None, **kw):
        super().__init__(_indexes=indexes, _gpu=gpu, _sparse=sparse, _output_types=output_types, **kw)
        if not self.output_types:
            self.output_types = [OutputType.dataframe]

    @property
    def input(self):
        return self._input

    @property
    def indexes(self):
        return self._indexes

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        self._input = next(inputs_iter)
        indexes = []
        for index in self._indexes:
            if isinstance(index, ENTITY_TYPE):
                indexes.append(next(inputs_iter))
            else:
                indexes.append(index)
        self._indexes = indexes

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

        inputs = [df] + [index for index in self._indexes if isinstance(index, ENTITY_TYPE)]

        # NB: pandas only compresses the result to series when index on one of axis is integral
        if isinstance(self.indexes[1], Integral):
            shape = shape0
            dtype = df.dtypes.iloc[self.indexes[1]]
            index_value = indexing_index_value(df.index_value, self.indexes[0])
            if isinstance(self.indexes[0], Integral):
                # scalar
                return self.new_scalar(inputs, dtype=dtype)
            else:
                return self.new_series(inputs, shape=shape, dtype=dtype,
                                       index_value=index_value,
                                       name=df.dtypes.index[self.indexes[1]])
        elif isinstance(self.indexes[0], Integral):
            shape = shape1
            dtype = find_common_type(list(df.dtypes.iloc[self.indexes[1]].values))
            index_value = indexing_index_value(df.columns_value, self.indexes[1])
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
        tileds = super().tile(op)
        if tileds is not None:
            return tileds

        handler = DataFrameIlocIndexesHandler()
        return [(yield from handler.handle(op))]

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        df = ctx[op.input.key]
        if len(op.inputs) > 1:
            indexes = tuple(ctx[index.key] if hasattr(index, 'key') else index
                            for index in op.indexes)
        else:
            indexes = tuple(op.indexes)
        r = df.iloc[indexes]
        if isinstance(r, pd.Series) and r.dtype != chunk.dtype:
            r = r.astype(chunk.dtype)
        if is_cudf(r):  # pragma: no cover
            r = r.copy()
        ctx[chunk.key] = r


class DataFrameIlocSetItem(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_ILOC_SETITEM

    _indexes = ListField('indexes')
    _value = AnyField('value')

    def __init__(self, indexes=None, value=None, gpu=False, sparse=False,
                 output_types=None, **kw):
        super().__init__(_indexes=indexes, _value=value, _gpu=gpu, _sparse=sparse,
                         _output_types=output_types, **kw)
        if not self.output_types:
            self.output_types = [OutputType.dataframe]

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
        tensor0 = yield from recursive_tile(
            empty(in_df.shape[0], chunk_size=(in_df.nsplits[0],))[op.indexes[0]])
        tensor1 = yield from recursive_tile(
            empty(in_df.shape[1], chunk_size=(in_df.nsplits[1],))[op.indexes[1]])

        chunk_mapping = {c0.inputs[0].index + c1.inputs[0].index: (c0, c1)
                         for c0, c1 in itertools.product(tensor0.chunks, tensor1.chunks)}

        out_chunks = []
        for chunk in in_df.chunks:
            if chunk.index not in chunk_mapping:
                out_chunks.append(chunk)
            else:
                chunk_op = op.copy().reset_key()
                index_chunk, column_chunk = chunk_mapping[chunk.index]
                chunk_op._indexes = [index_chunk.op.indexes[0], column_chunk.op.indexes[0]]
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


class SeriesIlocGetItem(DataFrameOperand, HeadTailOptimizedOperandMixin):
    _op_module_ = 'series'
    _op_type_ = OperandDef.DATAFRAME_ILOC_GETITEM

    _input = KeyField('input')
    _indexes = ListField('indexes')

    def __init__(self, indexes=None, gpu=False, sparse=False, output_types=None, **kw):
        super().__init__(_indexes=indexes, _gpu=gpu, _sparse=sparse, _output_types=output_types, **kw)
        if not self.output_types:
            self.output_types = [OutputType.series]

    @property
    def input(self):
        return self._input

    @property
    def indexes(self):
        return self._indexes

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)

        inputs_iter = iter(self._inputs)
        self._input = next(inputs_iter)

        indexes = []
        for index in self._indexes:
            if isinstance(index, ENTITY_TYPE):
                indexes.append(next(inputs_iter))
            else:
                indexes.append(index)
        self._indexes = indexes

    @classmethod
    def tile(cls, op):
        tileds = super().tile(op)
        if tileds is not None:
            return tileds

        handler = DataFrameIlocIndexesHandler()
        return [(yield from handler.handle(op))]

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
            ctx[chunk.key] = series[indexes if len(indexes) > 1 else indexes[0]]

    def __call__(self, series):
        if isinstance(self._indexes[0], Integral):
            return self.new_scalar([series], dtype=series.dtype)
        else:
            shape = tuple(calc_shape(series.shape, self.indexes))
            index_value = indexing_index_value(series.index_value, self.indexes[0])
            inputs = [series] + [index for index in self._indexes if isinstance(index, ENTITY_TYPE)]
            return self.new_series(inputs, shape=shape, dtype=series.dtype,
                                   index_value=index_value, name=series.name)


class SeriesIlocSetItem(DataFrameOperand, DataFrameOperandMixin):
    _op_module_ = 'series'
    _op_type_ = OperandDef.DATAFRAME_ILOC_SETITEM

    _indexes = ListField('indexes')
    _value = AnyField('value')

    def __init__(self, indexes=None, value=None, gpu=False, sparse=False, **kw):
        super().__init__(_indexes=indexes, _value=value, _gpu=gpu, _sparse=sparse,
                         _output_types=[OutputType.series], **kw)

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
        tensor = yield from recursive_tile(
            empty(in_series.shape, chunk_size=in_series.nsplits)[op.indexes[0]])

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


class IndexIlocGetItem(DataFrameOperand, DataFrameOperandMixin):
    _op_module_ = 'index'
    _op_type_ = OperandDef.DATAFRAME_ILOC_GETITEM

    _input = KeyField('input')
    _indexes = ListField('indexes')

    def __init__(self, indexes=None, gpu=False, sparse=False, output_types=None, **kw):
        super().__init__(_indexes=indexes, _gpu=gpu, _sparse=sparse, _output_types=output_types, **kw)
        if not self.output_types:
            self.output_types = [OutputType.index]

    @property
    def input(self):
        return self._input

    @property
    def indexes(self):
        return self._indexes

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)

        inputs_iter = iter(self._inputs)
        self._input = next(inputs_iter)

        indexes = []
        for index in self._indexes:
            if isinstance(index, ENTITY_TYPE):
                indexes.append(next(inputs_iter))
            else:
                indexes.append(index)
        self._indexes = indexes

    @classmethod
    def tile(cls, op):
        handler = DataFrameIlocIndexesHandler()
        return [(yield from handler.handle(op))]

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        idx = ctx[op.input.key]
        if len(op.inputs) > 1:
            indexes = tuple(ctx[index.key] if hasattr(index, 'key') else index
                            for index in op.indexes)
        else:
            indexes = tuple(op.indexes)
        ctx[chunk.key] = idx[indexes]

    def __call__(self, idx):
        if isinstance(self._indexes[0], Integral):
            return self.new_scalar([idx], dtype=idx.dtype)
        else:
            shape = tuple(calc_shape(idx.shape, self.indexes))
            index_value = indexing_index_value(idx.index_value, self.indexes[0])
            inputs = [idx] + [index for index in self._indexes if isinstance(index, ENTITY_TYPE)]
            return self.new_index(inputs, shape=shape, dtype=idx.dtype,
                                  index_value=index_value, name=idx.name)


def index_getitem(idx, indexes):
    op = IndexIlocGetItem(indexes=process_iloc_indexes(idx, indexes))
    return op(idx)


def index_setitem(_idx, *_):
    raise TypeError('Index does not support mutable operations')


def iloc(a):
    return DataFrameIloc(a)


def head(a, n=5):
    """
    Return the first `n` rows.

    This function returns the first `n` rows for the object based
    on position. It is useful for quickly testing if your object
    has the right type of data in it.

    For negative values of `n`, this function returns all rows except
    the last `n` rows, equivalent to ``df[:-n]``.

    Parameters
    ----------
    n : int, default 5
        Number of rows to select.

    Returns
    -------
    same type as caller
        The first `n` rows of the caller object.

    See Also
    --------
    DataFrame.tail: Returns the last `n` rows.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'animal': ['alligator', 'bee', 'falcon', 'lion',
    ...                    'monkey', 'parrot', 'shark', 'whale', 'zebra']})
    >>> df.execute()
          animal
    0  alligator
    1        bee
    2     falcon
    3       lion
    4     monkey
    5     parrot
    6      shark
    7      whale
    8      zebra

    Viewing the first 5 lines

    >>> df.head().execute()
          animal
    0  alligator
    1        bee
    2     falcon
    3       lion
    4     monkey

    Viewing the first `n` lines (three in this case)

    >>> df.head(3).execute()
          animal
    0  alligator
    1        bee
    2     falcon

    For negative values of `n`

    >>> df.head(-3).execute()
          animal
    0  alligator
    1        bee
    2     falcon
    3       lion
    4     monkey
    5     parrot
    """
    return DataFrameIloc(a)[0:n]


def tail(a, n=5):
    """
    Return the last `n` rows.

    This function returns last `n` rows from the object based on
    position. It is useful for quickly verifying data, for example,
    after sorting or appending rows.

    For negative values of `n`, this function returns all rows except
    the first `n` rows, equivalent to ``df[n:]``.

    Parameters
    ----------
    n : int, default 5
        Number of rows to select.

    Returns
    -------
    type of caller
        The last `n` rows of the caller object.

    See Also
    --------
    DataFrame.head : The first `n` rows of the caller object.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'animal': ['alligator', 'bee', 'falcon', 'lion',
    ...                    'monkey', 'parrot', 'shark', 'whale', 'zebra']})
    >>> df.execute()
          animal
    0  alligator
    1        bee
    2     falcon
    3       lion
    4     monkey
    5     parrot
    6      shark
    7      whale
    8      zebra

    Viewing the last 5 lines

    >>> df.tail().execute()
       animal
    4  monkey
    5  parrot
    6   shark
    7   whale
    8   zebra

    Viewing the last `n` lines (three in this case)

    >>> df.tail(3).execute()
      animal
    6  shark
    7  whale
    8  zebra

    For negative values of `n`

    >>> df.tail(-3).execute()
       animal
    3    lion
    4  monkey
    5  parrot
    6   shark
    7   whale
    8   zebra
    """
    return DataFrameIloc(a)[-n:]
