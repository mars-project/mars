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

from ... import opcodes
from ...core import OutputType
from ...core.operand import OperandStage
from ..utils import gen_unknown_index_value, hash_dataframe_on
from ._duplicate import DuplicateOperand, validate_subset


class DataFrameDuplicated(DuplicateOperand):
    _op_type_ = opcodes.DUPLICATED

    def __init__(self, subset=None, keep=None, output_types=None,
                 method=None, subset_chunk=None, shuffle_size=None, **kw):
        super().__init__(_subset=subset, _keep=keep,
                         _output_types=output_types, _method=method,
                         _subset_chunk=subset_chunk,
                         _shuffle_size=shuffle_size, **kw)

    @classmethod
    def _get_shape(cls, input_shape, op):
        return (input_shape[0],)

    @classmethod
    def _gen_tileable_params(cls, op: "DataFrameDuplicated", input_params):
        # duplicated() always returns a Series
        return {
            'shape': cls._get_shape(input_params['shape'], op),
            'index_value': input_params['index_value'],
            'dtype': np.dtype(bool),
            'name': input_params.get('name')
        }

    def __call__(self, inp, inplace=False):
        self._output_types = [OutputType.series]
        params = self._gen_tileable_params(self, inp.params)

        return self.new_tileable([inp], kws=[params])

    @classmethod
    def _get_map_output_types(cls, input_chunk, method: str):
        if method in ('tree', 'subset_tree'):
            return [OutputType.dataframe]
        else:
            return input_chunk.op.output_types

    @classmethod
    def _gen_chunk_params_default(cls, op: "DataFrameDuplicated", input_chunk):
        return {
            'shape': cls._get_shape(input_chunk.shape, op),
            'index_value': input_chunk.index_value,
            'dtype': np.dtype(bool),
            'name': input_chunk.name if input_chunk.ndim == 1 else None,
            'index': (input_chunk.index[0],)
        }

    @classmethod
    def _get_intermediate_shape(cls, input_shape):
        if len(input_shape) > 1:
            s = input_shape[1:]
        else:
            s = (2,)
        return (np.nan,) + s

    @classmethod
    def _gen_intermediate_chunk_params(cls, op: "DataFrameDuplicated", input_chunk):
        inp = op.input
        chunk_params = dict()
        chunk_params['index'] = input_chunk.index[:1] + (0,) * (inp.ndim - 1)
        chunk_params['shape'] = shape = cls._get_intermediate_shape(input_chunk.shape)
        chunk_params['index_value'] = gen_unknown_index_value(
            input_chunk.index_value, input_chunk)
        if inp.ndim == 2 and len(shape) == 2:
            chunk_params['columns_value'] = input_chunk.columns_value
            chunk_params['dtypes'] = input_chunk.dtypes
        return chunk_params

    @classmethod
    def _gen_chunk_params(cls, op: "DataFrameDuplicated", input_chunk):
        is_terminal_chunk = False
        if op.method is None:
            # one chunk
            is_terminal_chunk = True
        elif op.method == 'subset_tree' and op.stage is None:
            is_terminal_chunk = True
        elif op.method == 'tree' and op.stage == OperandStage.agg:
            is_terminal_chunk = True
        elif op.method == 'shuffle' and op.reducer_phase == 'put_back':
            is_terminal_chunk = True

        if is_terminal_chunk:
            return cls._gen_chunk_params_default(op, input_chunk)
        else:
            return cls._gen_intermediate_chunk_params(op, input_chunk)

    @classmethod
    def _duplicated(cls, inp, op, subset=None, keep=None):
        if keep is None:
            keep = op.keep
        if inp.ndim == 2:
            if subset is None:
                subset = op.subset
            return inp.duplicated(subset=subset, keep=keep)
        else:
            return inp.duplicated(keep=keep)

    @classmethod
    def _execute_chunk(cls, ctx, op):
        inp = ctx[op.input.key]
        ctx[op.outputs[0].key] = cls._duplicated(inp, op)

    @classmethod
    def _execute_tree_map(cls, ctx, op):
        inp = ctx[op.input.key]
        xdf = cls._get_xdf(inp)
        if op.subset is not None:
            result = inp[op.subset].copy()
        else:
            result = inp.copy()
        duplicated = cls._duplicated(inp, op)
        if not duplicated.name:
            duplicated.name = '_duplicated_'
        result.iloc[duplicated] = None
        result = xdf.concat([result, duplicated], axis=1)
        ctx[op.outputs[0].key] = result

    @classmethod
    def _execute_tree_combine(cls, ctx, op):
        inp = ctx[op.input.key]
        result = inp.copy()
        duplicates = inp[~inp.iloc[:, -1]]
        dup_on_duplicated = cls._duplicated(duplicates, op)
        result.iloc[~inp.iloc[:, -1], -1] = dup_on_duplicated
        duplicated = result.iloc[:, -1]
        result.iloc[duplicated, :-1] = None
        ctx[op.outputs[0].key] = result

    @classmethod
    def _execute_tree_agg(cls, ctx, op):
        inp = ctx[op.input.key]
        result = inp.iloc[:, -1].copy()
        duplicates = inp[~inp.iloc[:, -1]]
        dup_on_duplicated = cls._duplicated(duplicates, op)
        result[~inp.iloc[:, -1]] = dup_on_duplicated
        expect_name = op.outputs[0].name
        if result.name != expect_name:
            result.name = expect_name
        result = result.astype(bool)
        ctx[op.outputs[0].key] = result

    @classmethod
    def _execute_subset_tree_post(cls, ctx, op):
        inp = ctx[op.input.key]
        idx = op.outputs[0].index[0]
        subset = ctx[op.subset_chunk.key]
        selected = subset[subset['_chunk_index_'] == idx]['_i_']

        xdf = cls._get_xdf(inp)
        duplicated = np.ones(len(inp), dtype=bool)
        duplicated[selected] = False

        ctx[op.outputs[0].key] = xdf.Series(duplicated, index=inp.index)

    @classmethod
    def _execute_shuffle_map(cls, ctx, op):
        out = op.outputs[0]
        shuffle_size = op.shuffle_size
        subset = op.subset

        inp = ctx[op.input.key]
        if subset is not None:
            result = inp[subset].copy()
        else:
            result = inp.copy()
        if result.ndim == 1:
            name = result.name
            result = result.to_frame()
            if name is None:
                result.columns = ['_duplicated_']
            subset = result.columns.tolist()
        else:
            if subset is None:
                subset = result.columns.tolist()
            if len(subset) == 1:
                result.columns = subset = ['_duplicated_']
        result['_chunk_index_'] = out.index[0]
        result['_i_'] = np.arange(result.shape[0])
        hashed = hash_dataframe_on(result, subset, shuffle_size)
        for i, data in enumerate(hashed):
            reducer_idx = (i,) + out.index[1:]
            ctx[out.key, reducer_idx] = result.iloc[data]

    @classmethod
    def _execute_shuffle_reduce(cls, ctx, op: "DataFrameDuplicated"):
        out = op.outputs[0]
        inputs = list(op.iter_mapper_data(ctx))

        xdf = cls._get_xdf(inputs[0])
        inp = xdf.concat(inputs)
        subset = [c for c in inp.columns
                  if c not in ('_chunk_index_', '_i_')]
        duplicated = cls._duplicated(inp, op, subset=subset)
        result = xdf.concat([duplicated, inp[['_chunk_index_', '_i_']]], axis=1)
        for i in range(op.shuffle_size):
            filtered = result[result['_chunk_index_'] == i]
            del filtered['_chunk_index_']
            if len(subset) > 1 or subset[0] == '_duplicated_':
                filtered.columns = ['_duplicated_'] + filtered.columns[1:].tolist()
            else:
                filtered.columns = [subset[0]] + filtered.columns[1:].tolist()
            ctx[out.key, (i,)] = filtered

    @classmethod
    def _execute_shuffle_put_back(cls, ctx, op: "DataFrameDuplicated"):
        inputs = list(op.iter_mapper_data(ctx))

        xdf = cls._get_xdf(inputs[0])
        inp = xdf.concat(inputs)
        inp.sort_values('_i_', inplace=True)
        del inp['_i_']
        duplicated = inp.iloc[:, 0]
        if duplicated.name == '_duplicated_':
            duplicated.name = None
        ctx[op.outputs[0].key] = duplicated

    @classmethod
    def execute(cls, ctx, op: "DataFrameDuplicated"):
        if op.method is None:
            # one chunk
            cls._execute_chunk(ctx, op)
        elif op.method == 'tree':
            # tree
            if op.stage == OperandStage.map:
                cls._execute_tree_map(ctx, op)
            elif op.stage == OperandStage.combine:
                cls._execute_tree_combine(ctx, op)
            else:
                assert op.stage == OperandStage.agg
                cls._execute_tree_agg(ctx, op)
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


def df_duplicated(df, subset=None, keep='first', method='auto'):
    """
    Return boolean Series denoting duplicate rows.

    Considering certain columns is optional.

    Parameters
    ----------
    subset : column label or sequence of labels, optional
        Only consider certain columns for identifying duplicates, by
        default use all of the columns.
    keep : {'first', 'last', False}, default 'first'
        Determines which duplicates (if any) to mark.

        - ``first`` : Mark duplicates as ``True`` except for the first occurrence.
        - ``last`` : Mark duplicates as ``True`` except for the last occurrence.
        - False : Mark all duplicates as ``True``.

    Returns
    -------
    Series
        Boolean series for each duplicated rows.

    See Also
    --------
    Index.duplicated : Equivalent method on index.
    Series.duplicated : Equivalent method on Series.
    Series.drop_duplicates : Remove duplicate values from Series.
    DataFrame.drop_duplicates : Remove duplicate values from DataFrame.

    Examples
    --------
    Consider dataset containing ramen rating.

    >>> import mars.dataframe as md

    >>> df = md.DataFrame({
    ...     'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
    ...     'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
    ...     'rating': [4, 4, 3.5, 15, 5]
    ... })
    >>> df.execute()
        brand style  rating
    0  Yum Yum   cup     4.0
    1  Yum Yum   cup     4.0
    2  Indomie   cup     3.5
    3  Indomie  pack    15.0
    4  Indomie  pack     5.0

    By default, for each set of duplicated values, the first occurrence
    is set on False and all others on True.

    >>> df.duplicated().execute()
    0    False
    1     True
    2    False
    3    False
    4    False
    dtype: bool

    By using 'last', the last occurrence of each set of duplicated values
    is set on False and all others on True.

    >>> df.duplicated(keep='last').execute()
    0     True
    1    False
    2    False
    3    False
    4    False
    dtype: bool

    By setting ``keep`` on False, all duplicates are True.

    >>> df.duplicated(keep=False).execute()
    0     True
    1     True
    2    False
    3    False
    4    False
    dtype: bool

    To find duplicates on specific column(s), use ``subset``.

    >>> df.duplicated(subset=['brand']).execute()
    0    False
    1     True
    2    False
    3     True
    4     True
    dtype: bool
    """

    if method not in ('auto', 'tree', 'subset_tree', 'shuffle', None):
        raise ValueError("method could only be one of "
                         "'auto', 'tree', 'subset_tree', 'shuffle' or None")
    subset = validate_subset(df, subset)
    op = DataFrameDuplicated(subset=subset, keep=keep, method=method)
    return op(df)


def series_duplicated(series, keep='first', method='auto'):
    """
    Indicate duplicate Series values.

    Duplicated values are indicated as ``True`` values in the resulting
    Series. Either all duplicates, all except the first or all except the
    last occurrence of duplicates can be indicated.

    Parameters
    ----------
    keep : {'first', 'last', False}, default 'first'
        Method to handle dropping duplicates:

        - 'first' : Mark duplicates as ``True`` except for the first
          occurrence.
        - 'last' : Mark duplicates as ``True`` except for the last
          occurrence.
        - ``False`` : Mark all duplicates as ``True``.

    Returns
    -------
    Series
        Series indicating whether each value has occurred in the
        preceding values.

    See Also
    --------
    Index.duplicated : Equivalent method on pandas.Index.
    DataFrame.duplicated : Equivalent method on pandas.DataFrame.
    Series.drop_duplicates : Remove duplicate values from Series.

    Examples
    --------
    By default, for each set of duplicated values, the first occurrence is
    set on False and all others on True:

    >>> import mars.dataframe as md

    >>> animals = md.Series(['lama', 'cow', 'lama', 'beetle', 'lama'])
    >>> animals.duplicated().execute()
    0    False
    1    False
    2     True
    3    False
    4     True
    dtype: bool

    which is equivalent to

    >>> animals.duplicated(keep='first').execute()
    0    False
    1    False
    2     True
    3    False
    4     True
    dtype: bool

    By using 'last', the last occurrence of each set of duplicated values
    is set on False and all others on True:

    >>> animals.duplicated(keep='last').execute()
    0     True
    1    False
    2     True
    3    False
    4    False
    dtype: bool

    By setting keep on ``False``, all duplicates are True:

    >>> animals.duplicated(keep=False).execute()
    0     True
    1    False
    2     True
    3    False
    4     True
    dtype: bool
    """
    if method not in ('auto', 'tree', 'shuffle', None):
        raise ValueError("method could only be one of "
                         "'auto', 'tree', 'shuffle' or None")
    op = DataFrameDuplicated(keep=keep, method=method)
    return op(series)


def index_duplicated(index, keep='first'):
    """
    Indicate duplicate index values.

    Duplicated values are indicated as ``True`` values in the resulting
    array. Either all duplicates, all except the first, or all except the
    last occurrence of duplicates can be indicated.

    Parameters
    ----------
    keep : {'first', 'last', False}, default 'first'
        The value or values in a set of duplicates to mark as missing.
        - 'first' : Mark duplicates as ``True`` except for the first
          occurrence.
        - 'last' : Mark duplicates as ``True`` except for the last
          occurrence.
        - ``False`` : Mark all duplicates as ``True``.

    Returns
    -------
    Tensor

    See Also
    --------
    Series.duplicated : Equivalent method on pandas.Series.
    DataFrame.duplicated : Equivalent method on pandas.DataFrame.
    Index.drop_duplicates : Remove duplicate values from Index.

    Examples
    --------
    By default, for each set of duplicated values, the first occurrence is
    set to False and all others to True:

    >>> import mars.dataframe as md

    >>> idx = md.Index(['lama', 'cow', 'lama', 'beetle', 'lama'])
    >>> idx.duplicated().execute()
    array([False, False,  True, False,  True])

    which is equivalent to

    >>> idx.duplicated(keep='first').execute()
    array([False, False,  True, False,  True])

    By using 'last', the last occurrence of each set of duplicated values
    is set on False and all others on True:

    >>> idx.duplicated(keep='last').execute()
    array([ True, False,  True, False, False])

    By setting keep on ``False``, all duplicates are True:

    >>> idx.duplicated(keep=False).execute()
    array([ True, False,  True, False,  True])
    """
    return index.to_series().duplicated(keep=keep).to_tensor()
