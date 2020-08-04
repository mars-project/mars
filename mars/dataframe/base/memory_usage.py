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

import operator
from functools import reduce

import numpy as np
import pandas as pd

from ... import opcodes
from ...config import options
from ...operands import OperandStage
from ...serialize import BoolField, Int64Field
from ...utils import ceildiv, lazy_import
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..core import IndexValue
from ..utils import parse_index

cudf = lazy_import('cudf', globals=globals())


class DataFrameMemoryUsage(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.MEMORY_USAGE

    # raw arguments of memory_usage method
    _index = BoolField('index')
    _deep = BoolField('deep')

    # size of range index, when set, the value will be prepended to the result series
    # if the input is a dataframe, or added to the result when the input is a series
    _range_index_size = Int64Field('range_index_size')

    def __init__(self, index=None, deep=None, range_index_size=None, **kw):
        super().__init__(_index=index, _deep=deep, _range_index_size=range_index_size, **kw)

    @property
    def index(self) -> bool:
        return self._index

    @index.setter
    def index(self, value: bool):
        self._index = value

    @property
    def deep(self) -> bool:
        return self._deep

    @property
    def range_index_size(self) -> int:
        return self._range_index_size

    @range_index_size.setter
    def range_index_size(self, value: int):
        self._range_index_size = value

    def _adapt_index(self, input_index, index=0):
        """
        When ``index=True`` is passed, an extra column will be prepended to the result series
        Thus we need to update the index of initial chunk for returned dataframe chunks
        """
        if not self.index or index != 0:
            return input_index
        idx_data = input_index.to_pandas().insert(0, 'Index')
        return parse_index(idx_data, store_data=True)

    def _adapt_nsplits(self, input_nsplit):
        """
        When ``index=True`` is passed, the size of returned series is one element larger
        than the number of columns, which affects ``nsplits``.
        """
        if not self.index:
            return (input_nsplit[-1],)
        nsplits_list = list(input_nsplit[-1])
        nsplits_list[0] += 1
        return (tuple(nsplits_list),)

    def __call__(self, df_or_series):
        """
        Return output object of memory_usage() call
        """
        if df_or_series.ndim == 1:
            # the input data is a series, a Scalar will be returned
            self._object_type = ObjectType.scalar
            return self.new_scalar([df_or_series], dtype=np.dtype(np.int_))
        else:
            # the input data is a DataFrame, a Scalar will be returned
            # calculate shape of returning series given ``op.index``
            self._object_type = ObjectType.series
            new_shape = (df_or_series.shape[-1] + 1,) if self.index else (df_or_series.shape[-1],)
            return self.new_series(
                [df_or_series], index_value=self._adapt_index(df_or_series.columns_value),
                shape=new_shape, dtype=np.dtype(np.int_))

    @classmethod
    def _tile_single(cls, op: "DataFrameMemoryUsage"):
        """
        Tile when input data has only one chunk on rows
        """
        df_or_series = op.inputs[0]
        output = op.outputs[0]

        chunks = []
        for c in df_or_series.chunks:
            new_op = op.copy().reset_key()
            if c.ndim == 1:
                # Tile for series
                chunks.append(new_op.new_chunk([c], index=c.index, dtype=output.dtype, shape=()))
            else:
                # tile for dataframes
                # only calculate with index=True on the initial chunk
                new_op.index = op.index and c.index[-1] == 0

                # calculate shape of returning chunk given ``op.index``
                new_shape = (c.shape[-1] + 1,) if c.index[-1] == 0 and op.index else (c.shape[-1],)
                chunks.append(new_op.new_chunk(
                    [c], shape=new_shape, dtype=output.dtype, index=(c.index[-1],),
                    index_value=op._adapt_index(c.columns_value, c.index[-1])))

        new_op = op.copy().reset_key()
        # return objects with chunks and nsplits (if needed)
        if df_or_series.ndim == 1:
            return new_op.new_tileables([df_or_series], dtype=output.dtype, chunks=chunks, nsplits=())
        else:
            return new_op.new_tileables([df_or_series], shape=output.shape, dtype=output.dtype,
                                        index_value=output.index_value, chunks=chunks,
                                        nsplits=op._adapt_nsplits(df_or_series.nsplits))

    @classmethod
    def _tile_dataframe(cls, op: "DataFrameMemoryUsage"):
        """
        Tile dataframes using tree reduction
        """
        df = op.inputs[0]
        output = op.outputs[0]
        is_range_index = isinstance(df.index_value.value, IndexValue.RangeIndex)

        # produce map chunks
        # allocate matrix of chunks
        chunks_to_reduce = np.empty(shape=df.chunk_shape, dtype=np.object)
        for c in df.chunks:
            new_op = op.copy().reset_key()
            new_op._stage = OperandStage.map

            if op.index and is_range_index:
                # when the index is ``pd.RangeIndex``, the size should be included
                # after all computations are done
                new_op.index = False
            else:
                # when the chunk is not the first chunk in the row, index size is not needed
                new_op.index = op.index and c.index[-1] == 0

            new_shape = (c.shape[-1] + 1,) if c.index[-1] == 0 and op.index else (c.shape[-1],)

            chunks_to_reduce[c.index] = new_op.new_chunk(
                [c], index=(c.index[-1],), dtype=output.dtype, shape=new_shape,
                index_value=op._adapt_index(c.columns_value, c.index[-1]))

        # reduce chunks using tree reduction
        combine_size = options.combine_size
        while chunks_to_reduce.shape[0] > 1:
            # allocate matrix of chunks
            new_chunks_to_reduce = np.empty((ceildiv(chunks_to_reduce.shape[0], combine_size),
                                             chunks_to_reduce.shape[1]), dtype=np.object)
            for idx in range(0, chunks_to_reduce.shape[0], combine_size):
                for idx2 in range(chunks_to_reduce.shape[1]):
                    new_op = op.copy().reset_key()
                    new_op._stage = OperandStage.reduce
                    chunks = list(chunks_to_reduce[idx:idx + combine_size, idx2])

                    new_chunks_to_reduce[idx // combine_size, idx2] = new_op.new_chunk(
                        chunks, index=(idx2,), dtype=output.dtype, shape=chunks[0].shape,
                        index_value=chunks[0].index_value)

            chunks_to_reduce = new_chunks_to_reduce

        # handle RangeIndex at final outputs
        if op.index and is_range_index:
            chunks_to_reduce[0, 0].op.range_index_size = df.index_value.to_pandas().memory_usage()

        # return series with chunks and nsplits
        new_op = op.copy().reset_key()
        return new_op.new_tileables([df], dtype=output.dtype, shape=output.shape,
                                    index_value=output.index_value,
                                    chunks=list(chunks_to_reduce[0, :]),
                                    nsplits=op._adapt_nsplits(df.nsplits))

    @classmethod
    def _tile_series(cls, op: "DataFrameMemoryUsage"):
        """
        Tile series using tree reduction
        """
        series = op.inputs[0]
        output = op.outputs[0]
        is_range_index = isinstance(series.index_value.value, IndexValue.RangeIndex)

        chunks_to_reduce = []
        for c in series.chunks:
            new_op = op.copy().reset_key()
            new_op._stage = OperandStage.map

            # when the index is ``pd.RangeIndex``, the size should be included
            # after all computations are done
            new_op.index = op.index and not is_range_index

            chunks_to_reduce.append(
                new_op.new_chunk([c], index=c.index, dtype=output.dtype, shape=()))

        # reduce chunks using tree reduction
        combine_size = options.combine_size
        while len(chunks_to_reduce) > 1:
            new_chunks_to_reduce = []
            for idx in range(0, len(chunks_to_reduce), combine_size):
                new_op = op.copy().reset_key()
                new_op._stage = OperandStage.reduce

                new_chunks_to_reduce.append(new_op.new_chunk(
                    chunks_to_reduce[idx:idx + combine_size], shape=(), index=(0,),
                    dtype=output.dtype))

            chunks_to_reduce = new_chunks_to_reduce

        # handle RangeIndex at final outputs
        if op.index and is_range_index:
            chunks_to_reduce[0].op.range_index_size = series.index_value.to_pandas().memory_usage()

        # return series with chunks
        new_op = op.copy().reset_key()
        return new_op.new_tileables([series], dtype=output.dtype, chunks=chunks_to_reduce,
                                    nsplits=())

    @classmethod
    def tile(cls, op: "DataFrameMemoryUsage"):
        df_or_series = op.inputs[0]
        if df_or_series.chunk_shape[0] == 1:  # only one chunk in row, no aggregation needed
            return cls._tile_single(op)
        elif df_or_series.ndim == 1:   # series
            return cls._tile_series(op)
        else:  # dataframe
            return cls._tile_dataframe(op)

    @classmethod
    def execute(cls, ctx, op: "DataFrameMemoryUsage"):
        in_data = ctx[op.inputs[0].key]
        # choose correct dataframe library
        xdf = cudf if op.gpu else pd

        if op.stage == OperandStage.reduce:
            result = reduce(operator.add, (ctx[c.key] for c in op.inputs))
            if op.range_index_size is not None:
                if hasattr(in_data, 'ndim'):
                    # dataframe input: prepend index size column
                    prepend_series = xdf.Series([op.range_index_size], index=['Index'], dtype=result.dtype)
                    result = xdf.concat([prepend_series, result])
                else:
                    # series input: add index size to the output
                    result += op.range_index_size
            ctx[op.outputs[0].key] = result
        elif isinstance(in_data, xdf.Index):
            ctx[op.outputs[0].key] = in_data.memory_usage(deep=op.deep)
        else:
            ctx[op.outputs[0].key] = in_data.memory_usage(index=op.index, deep=op.deep)


def df_memory_usage(df, index=True, deep=False):
    """
    Return the memory usage of each column in bytes.

    The memory usage can optionally include the contribution of
    the index and elements of `object` dtype.

    This value is displayed in `DataFrame.info` by default. This can be
    suppressed by setting ``pandas.options.display.memory_usage`` to False.

    Parameters
    ----------
    index : bool, default True
        Specifies whether to include the memory usage of the DataFrame's
        index in returned Series. If ``index=True``, the memory usage of
        the index is the first item in the output.
    deep : bool, default False
        If True, introspect the data deeply by interrogating
        `object` dtypes for system-level memory consumption, and include
        it in the returned values.

    Returns
    -------
    Series
        A Series whose index is the original column names and whose values
        is the memory usage of each column in bytes.

    See Also
    --------
    numpy.ndarray.nbytes : Total bytes consumed by the elements of an
        ndarray.
    Series.memory_usage : Bytes consumed by a Series.
    Categorical : Memory-efficient array for string values with
        many repeated values.
    DataFrame.info : Concise summary of a DataFrame.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> import mars.dataframe as md
    >>> dtypes = ['int64', 'float64', 'complex128', 'object', 'bool']
    >>> data = dict([(t, mt.ones(shape=5000).astype(t))
    ...              for t in dtypes])
    >>> df = md.DataFrame(data)
    >>> df.head().execute()
       int64  float64            complex128  object  bool
    0      1      1.0    1.000000+0.000000j       1  True
    1      1      1.0    1.000000+0.000000j       1  True
    2      1      1.0    1.000000+0.000000j       1  True
    3      1      1.0    1.000000+0.000000j       1  True
    4      1      1.0    1.000000+0.000000j       1  True

    >>> df.memory_usage().execute()
    Index           128
    int64         40000
    float64       40000
    complex128    80000
    object        40000
    bool           5000
    dtype: int64

    >>> df.memory_usage(index=False).execute()
    int64         40000
    float64       40000
    complex128    80000
    object        40000
    bool           5000
    dtype: int64

    The memory footprint of `object` dtype columns is ignored by default:

    >>> df.memory_usage(deep=True).execute()
    Index            128
    int64          40000
    float64        40000
    complex128     80000
    object        160000
    bool            5000
    dtype: int64

    Use a Categorical for efficient storage of an object-dtype column with
    many repeated values.

    >>> df['object'].astype('category').memory_usage(deep=True).execute()
    5216
    """
    op = DataFrameMemoryUsage(index=index, deep=deep)
    return op(df)


def series_memory_usage(series, index=True, deep=False):
    """
    Return the memory usage of the Series.

    The memory usage can optionally include the contribution of
    the index and of elements of `object` dtype.

    Parameters
    ----------
    index : bool, default True
        Specifies whether to include the memory usage of the Series index.
    deep : bool, default False
        If True, introspect the data deeply by interrogating
        `object` dtypes for system-level memory consumption, and include
        it in the returned value.

    Returns
    -------
    int
        Bytes of memory consumed.

    See Also
    --------
    numpy.ndarray.nbytes : Total bytes consumed by the elements of the
        array.
    DataFrame.memory_usage : Bytes consumed by a DataFrame.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> s = md.Series(range(3))
    >>> s.memory_usage().execute()
    152

    Not including the index gives the size of the rest of the data, which
    is necessarily smaller:

    >>> s.memory_usage(index=False).execute()
    24

    The memory footprint of `object` values is ignored by default:

    >>> s = md.Series(["a", "b"])
    >>> s.values.execute()
    array(['a', 'b'], dtype=object)

    >>> s.memory_usage().execute()
    144

    >>> s.memory_usage(deep=True).execute()
    260
    """
    op = DataFrameMemoryUsage(index=index, deep=deep)
    return op(series)


def index_memory_usage(index, deep=False):
    """
    Memory usage of the values.

    Parameters
    ----------
    deep : bool
        Introspect the data deeply, interrogate
        `object` dtypes for system-level memory consumption.

    Returns
    -------
    bytes used

    See Also
    --------
    numpy.ndarray.nbytes

    Notes
    -----
    Memory usage does not include memory consumed by elements that
    are not components of the array if deep=False
    """
    op = DataFrameMemoryUsage(index=False, deep=deep)
    return op(index)
