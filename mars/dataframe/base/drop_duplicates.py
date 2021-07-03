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

import numpy as np
import pandas as pd

from ... import opcodes
from ...core.operand import OperandStage
from ...serialization.serializables import BoolField
from ...utils import lazy_import
from ..operands import OutputType
from ..utils import parse_index, hash_dataframe_on, gen_unknown_index_value, standardize_range_index
from ._duplicate import DuplicateOperand, validate_subset

cudf = lazy_import('cudf', globals=globals())


class DataFrameDropDuplicates(DuplicateOperand):
    _op_type_ = opcodes.DROP_DUPLICATES

    _ignore_index = BoolField('ignore_index')

    def __init__(self, subset=None, keep=None, ignore_index=None,
                 output_types=None, method=None, subset_chunk=None,
                 shuffle_size=None, **kw):
        super().__init__(_subset=subset, _keep=keep, _ignore_index=ignore_index,
                         _output_types=output_types, _method=method,
                         _subset_chunk=subset_chunk,
                         _shuffle_size=shuffle_size, **kw)

    @property
    def ignore_index(self):
        return self._ignore_index

    @classmethod
    def _get_shape(cls, input_shape, op):
        shape = (np.nan,) + input_shape[1:]
        if op.output_types[0] == OutputType.dataframe and len(shape) == 1:
            shape += (3,)
        return shape

    @classmethod
    def _gen_tileable_params(cls, op: "DataFrameDropDuplicates", input_params):
        params = input_params.copy()
        if op.ignore_index:
            params['index_value'] = parse_index(pd.RangeIndex(-1))
        else:
            params['index_value'] = gen_unknown_index_value(
                input_params['index_value'], op.keep, op.subset, type(op).__name__)
        params['shape'] = cls._get_shape(input_params['shape'], op)
        return params

    def __call__(self, inp, inplace=False):
        self._output_types = inp.op.output_types
        params = self._gen_tileable_params(self, inp.params)

        ret = self.new_tileable([inp], kws=[params])
        if inplace:
            inp.data = ret.data
        return ret

    @classmethod
    def _gen_chunk_params(cls, op: "DataFrameDropDuplicates", input_chunk):
        input_params = input_chunk.params
        inp = op.inputs[0]
        chunk_params = input_params.copy()
        chunk_params['index'] = input_chunk.index[:1] + (0,) * (inp.ndim - 1)
        chunk_params['shape'] = cls._get_shape(input_params['shape'], op)
        chunk_params['index_value'] = gen_unknown_index_value(
            input_params['index_value'], input_chunk)
        if inp.ndim == 2:
            chunk_params['columns_value'] = inp.columns_value
            chunk_params['dtypes'] = inp.dtypes
        else:
            chunk_params['name'] = inp.name
            chunk_params['dtype'] = inp.dtype
        return chunk_params

    @classmethod
    def _get_map_output_types(cls, input_chunk, method: str):
        if method == 'subset_tree':
            return [OutputType.dataframe]
        else:
            return input_chunk.op.output_types

    @classmethod
    def _tile_shuffle(cls, op: "DataFrameDropDuplicates", inp):
        tiled = super()._tile_shuffle(op, inp)[0]
        put_back_chunks = tiled.chunks
        if op.ignore_index:
            put_back_chunks = standardize_range_index(put_back_chunks)
        new_op = op.copy()
        params = tiled.params
        params['nsplits'] = tiled.nsplits
        params['chunks'] = put_back_chunks
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def _execute_chunk(cls, ctx, op):
        inp = ctx[op.input.key]
        ctx[op.outputs[0].key] = cls._drop_duplicates(inp, op)

    @classmethod
    def _execute_subset_tree_post(cls, ctx, op):
        inp = ctx[op.input.key]
        out = op.outputs[0]
        idx = op.outputs[0].index[0]
        subset = ctx[op.subset_chunk.key]
        selected = subset[subset['_chunk_index_'] == idx]['_i_']
        ret = inp.iloc[selected]
        if op.ignore_index:
            prev_size = (subset['_chunk_index_'] < out.index[0]).sum()
            ret.index = pd.RangeIndex(prev_size, prev_size + len(ret))
        ctx[op.outputs[0].key] = ret

    @classmethod
    def _execute_shuffle_map(cls, ctx, op):
        out = op.outputs[0]
        shuffle_size = op.shuffle_size
        subset = op.subset

        inp = ctx[op.input.key]
        dropped = cls._drop_duplicates(inp, op)
        if dropped.ndim == 1:
            dropped = dropped.to_frame()
            subset = dropped.columns.tolist()
        else:
            if subset is None:
                subset = dropped.columns.tolist()
        dropped['_chunk_index_'] = out.index[0]
        dropped['_i_'] = np.arange(dropped.shape[0])
        hashed = hash_dataframe_on(dropped, subset, shuffle_size)
        for i, data in enumerate(hashed):
            reducer_idx = (i,) + out.index[1:]
            ctx[out.key, reducer_idx] = dropped.iloc[data]

    @classmethod
    def _execute_shuffle_reduce(cls, ctx, op: "DataFrameDropDuplicates"):
        out = op.outputs[0]
        inputs = list(op.iter_mapper_data(ctx))

        xdf = cls._get_xdf(inputs[0])
        inp = xdf.concat(inputs)
        dropped = cls._drop_duplicates(inp, op,
                                       subset=[c for c in inp.columns
                                               if c not in ('_chunk_index_', '_i_')],
                                       keep=op.keep, ignore_index=op.ignore_index)
        for i in range(op.shuffle_size):
            filtered = dropped[dropped['_chunk_index_'] == i]
            del filtered['_chunk_index_']
            ctx[out.key, (i,)] = filtered

    @classmethod
    def _execute_shuffle_put_back(cls, ctx, op: "DataFrameDropDuplicates"):
        out = op.outputs[0]
        inputs = list(op.iter_mapper_data(ctx))

        xdf = cls._get_xdf(inputs[0])
        inp = xdf.concat(inputs)
        inp.sort_values('_i_', inplace=True)
        del inp['_i_']

        if out.op.output_types[0] == OutputType.index:
            assert inp.shape[1] == 1
            ret = xdf.Index(inp.iloc[:, 0])
        elif out.op.output_types[0] == OutputType.series:
            assert inp.shape[1] == 1
            ret = inp.iloc[:, 0]
        else:
            ret = inp

        if op.ignore_index:
            ret.reset_index(drop=True, inplace=True)
        ctx[out.key] = ret

    @classmethod
    def execute(cls, ctx, op):
        if op.method is None:
            # one chunk
            cls._execute_chunk(ctx, op)
        elif op.method == 'tree':
            # tree
            cls._execute_chunk(ctx, op)
        elif op.method == 'subset_tree':
            # subset tree
            if op.stage == OperandStage.map:
                cls._execute_subset_tree_map(ctx, op)
            elif op.stage == OperandStage.combine:
                cls._execute_subset_tree_combine(ctx, op)
            elif op.stage == OperandStage.agg:
                cls._execute_subset_tree_agg(ctx, op)
            else:
                # post
                cls._execute_subset_tree_post(ctx, op)
        else:
            assert op.method == 'shuffle'
            if op.stage == OperandStage.map:
                cls._execute_shuffle_map(ctx, op)
            elif op.reducer_phase == 'drop_duplicates':
                cls._execute_shuffle_reduce(ctx, op)
            else:
                assert op.reducer_phase == 'put_back'
                cls._execute_shuffle_put_back(ctx, op)


def df_drop_duplicates(df, subset=None, keep='first',
                       inplace=False, ignore_index=False, method='auto'):
    """
    Return DataFrame with duplicate rows removed.

    Considering certain columns is optional. Indexes, including time indexes
    are ignored.

    Parameters
    ----------
    subset : column label or sequence of labels, optional
        Only consider certain columns for identifying duplicates, by
        default use all of the columns.
    keep : {'first', 'last', False}, default 'first'
        Determines which duplicates (if any) to keep.
        - ``first`` : Drop duplicates except for the first occurrence.
        - ``last`` : Drop duplicates except for the last occurrence.
        - False : Drop all duplicates.
    inplace : bool, default False
        Whether to drop duplicates in place or to return a copy.
    ignore_index : bool, default False
        If True, the resulting axis will be labeled 0, 1, …, n - 1.

    Returns
    -------
    DataFrame
        DataFrame with duplicates removed or None if ``inplace=True``.
    """
    if method not in ('auto', 'tree', 'subset_tree', 'shuffle', None):
        raise ValueError("method could only be one of "
                         "'auto', 'tree', 'subset_tree', 'shuffle' or None")
    subset = validate_subset(df, subset)
    op = DataFrameDropDuplicates(subset=subset, keep=keep,
                                 ignore_index=ignore_index,
                                 method=method)
    return op(df, inplace=inplace)


def series_drop_duplicates(series, keep='first', inplace=False, method='auto'):
    """
    Return Series with duplicate values removed.

    Parameters
    ----------
    keep : {'first', 'last', ``False``}, default 'first'
        Method to handle dropping duplicates:

        - 'first' : Drop duplicates except for the first occurrence.
        - 'last' : Drop duplicates except for the last occurrence.
        - ``False`` : Drop all duplicates.

    inplace : bool, default ``False``
        If ``True``, performs operation inplace and returns None.

    Returns
    -------
    Series
        Series with duplicates dropped.

    See Also
    --------
    Index.drop_duplicates : Equivalent method on Index.
    DataFrame.drop_duplicates : Equivalent method on DataFrame.
    Series.duplicated : Related method on Series, indicating duplicate
        Series values.

    Examples
    --------
    Generate a Series with duplicated entries.

    >>> import mars.dataframe as md
    >>> s = md.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'],
    ...               name='animal')
    >>> s.execute()
    0      lama
    1       cow
    2      lama
    3    beetle
    4      lama
    5     hippo
    Name: animal, dtype: object

    With the 'keep' parameter, the selection behaviour of duplicated values
    can be changed. The value 'first' keeps the first occurrence for each
    set of duplicated entries. The default value of keep is 'first'.

    >>> s.drop_duplicates().execute()
    0      lama
    1       cow
    3    beetle
    5     hippo
    Name: animal, dtype: object

    The value 'last' for parameter 'keep' keeps the last occurrence for
    each set of duplicated entries.

    >>> s.drop_duplicates(keep='last').execute()
    1       cow
    3    beetle
    4      lama
    5     hippo
    Name: animal, dtype: object

    The value ``False`` for parameter 'keep' discards all sets of
    duplicated entries. Setting the value of 'inplace' to ``True`` performs
    the operation inplace and returns ``None``.

    >>> s.drop_duplicates(keep=False, inplace=True)
    >>> s.execute()
    1       cow
    3    beetle
    5     hippo
    Name: animal, dtype: object
    """
    if method not in ('auto', 'tree', 'shuffle', None):
        raise ValueError("method could only be one of "
                         "'auto', 'tree', 'shuffle' or None")
    op = DataFrameDropDuplicates(keep=keep, method=method)
    return op(series, inplace=inplace)


def index_drop_duplicates(index, keep='first', method='auto'):
    """
    Return Index with duplicate values removed.

    Parameters
    ----------
    keep : {'first', 'last', ``False``}, default 'first'
        - 'first' : Drop duplicates except for the first occurrence.
        - 'last' : Drop duplicates except for the last occurrence.
        - ``False`` : Drop all duplicates.

    Returns
    -------
    deduplicated : Index

    See Also
    --------
    Series.drop_duplicates : Equivalent method on Series.
    DataFrame.drop_duplicates : Equivalent method on DataFrame.
    Index.duplicated : Related method on Index, indicating duplicate
        Index values.

    Examples
    --------
    Generate an pandas.Index with duplicate values.

    >>> import mars.dataframe as md

    >>> idx = md.Index(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'])

    The `keep` parameter controls  which duplicate values are removed.
    The value 'first' keeps the first occurrence for each
    set of duplicated entries. The default value of keep is 'first'.

    >>> idx.drop_duplicates(keep='first').execute()
    Index(['lama', 'cow', 'beetle', 'hippo'], dtype='object')

    The value 'last' keeps the last occurrence for each set of duplicated
    entries.

    >>> idx.drop_duplicates(keep='last').execute()
    Index(['cow', 'beetle', 'lama', 'hippo'], dtype='object')

    The value ``False`` discards all sets of duplicated entries.

    >>> idx.drop_duplicates(keep=False).execute()
    Index(['cow', 'beetle', 'hippo'], dtype='object')
    """
    if method not in ('auto', 'tree', 'shuffle', None):
        raise ValueError("method could only be one of "
                         "'auto', 'tree', 'shuffle' or None")
    op = DataFrameDropDuplicates(keep=keep, method=method)
    return op(index)
