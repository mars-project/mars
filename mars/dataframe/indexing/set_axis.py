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
from ...core import ENTITY_TYPE, get_output_types, recursive_tile
from ...serialization.serializables import AnyField, Int8Field, KeyField
from ...utils import has_unknown_shape
from ..operands import DataFrameOperandMixin, DataFrameOperand
from ..utils import parse_index, validate_axis


class DataFrameSetAxis(DataFrameOperand, DataFrameOperandMixin):
    _op_code_ = opcodes.DATAFRAME_SET_AXIS

    _input = KeyField('input')
    _axis = Int8Field('axis')
    _value = AnyField('value')

    def __init__(self, value=None, axis=None, **kw):
        super().__init__(_value=value, _axis=axis, **kw)

    @property
    def input(self):
        return self._input

    @property
    def value(self):
        return self._value

    @property
    def axis(self):
        return self._axis

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = inputs[0]
        if isinstance(self.value, ENTITY_TYPE):
            self._value = inputs[-1]

    def __call__(self, df_or_series):
        new_size = self.value.shape[0]
        expect_size = df_or_series.axes[self.axis].shape[0]
        if not np.isnan(new_size) and not np.isnan(expect_size) \
                and new_size != expect_size:
            raise ValueError(
                f'Length mismatch: Expected axis has {expect_size} elements, '
                f'new values have {new_size} elements'
            )

        params = df_or_series.params
        if self.axis == 0:
            params['index_value'] = parse_index(self.value) \
                if isinstance(self.value, pd.Index) else self.value.index_value
        else:
            params['columns_value'] = parse_index(self.value, store_data=True) \
                if isinstance(self.value, pd.Index) else self.value.index_value
            pd_columns = self.value.index_value.to_pandas() \
                if isinstance(self.value, ENTITY_TYPE) else self.value
            params['dtypes'] = params['dtypes'].set_axis(pd_columns)

        self._output_types = get_output_types(df_or_series)
        inputs = [df_or_series]
        if isinstance(self.value, ENTITY_TYPE):
            inputs += [self.value]
        return self.new_tileable(inputs, **params)

    @classmethod
    def tile(cls, op: 'DataFrameSetAxis'):
        output = op.outputs[0]
        input_tileables = [op.input]

        value = op.value
        if isinstance(value, ENTITY_TYPE):
            input_tileables.append(value)
            if has_unknown_shape(value):
                yield

        if any(np.isnan(s) for s in op.input.nsplits[op.axis]):
            yield

        if op.input.shape[op.axis] != value.shape[0]:
            raise ValueError(
                f'Length mismatch: Expected axis has {value.shape[0]} elements, '
                f'new values have {op.input.shape[op.axis]} elements'
            )

        if isinstance(value, ENTITY_TYPE):
            value = yield from recursive_tile(
                value.rechunk({0: op.input.nsplits[op.axis]}))
            input_tileables[-1] = value

        slices = np.array((0,) + op.input.nsplits[op.axis]).cumsum()
        slice_left = slices[:-1]
        slice_right = slices[1:]

        chunks = []
        param_cache = [None] * len(op.input.nsplits[op.axis])
        for inp_chunk in op.input.chunks:
            input_chunks = [inp_chunk]
            value_index = inp_chunk.index[op.axis]
            params = inp_chunk.params

            if isinstance(value, ENTITY_TYPE):
                value_data = value.chunks[value_index]
                input_chunks.append(value_data)
            else:
                value_data = value[slice_left[value_index]:slice_right[value_index]]

            if param_cache[value_index] is None:
                cached_params = param_cache[value_index] = dict()
                if isinstance(value, ENTITY_TYPE):
                    if op.axis == 0:
                        cached_params['index_value'] = value_data.index_value
                    else:
                        cached_params['columns_value'] = value_data.index_value
                        cached_params['dtypes'] = output.dtypes.iloc[
                            slice_left[value_index]:slice_right[value_index]
                        ]
                else:
                    if op.axis == 0:
                        cached_params['index_value'] = parse_index(value_data)
                    else:
                        cached_params['columns_value'] = parse_index(value_data, store_data=True)
                        cached_params['dtypes'] = params['dtypes'].set_axis(value_data)

            params.update(param_cache[value_index])

            new_op = op.copy().reset_key()
            new_op._value = value_data
            chunks.append(new_op.new_chunk(input_chunks, **params))

        params = op.outputs[0].params
        params['chunks'] = chunks
        params['nsplits'] = op.input.nsplits
        new_op = op.copy().reset_key()
        return new_op.new_tileables(input_tileables, **params)

    @classmethod
    def execute(cls, ctx, op: 'DataFrameSetAxis'):
        in_data = ctx[op.input.key]
        value = op.value
        if isinstance(value, ENTITY_TYPE):
            value = ctx[value.key]
        ctx[op.outputs[0].key] = in_data.set_axis(value, axis=op.axis)


def _set_axis(df_or_axis, labels, axis=0, inplace=False):
    axis = validate_axis(axis, df_or_axis)
    if not isinstance(labels, ENTITY_TYPE) and not isinstance(labels, pd.Index):
        labels = pd.Index(labels)

    op = DataFrameSetAxis(value=labels, axis=axis)
    result = op(df_or_axis)
    if inplace:
        df_or_axis.data = result.data
    else:
        return result


def df_set_axis(df, labels, axis=0, inplace=False):
    """
    Assign desired index to given axis.

    Indexes for column or row labels can be changed by assigning
    a list-like or Index.

    Parameters
    ----------
    labels : list-like, Index
        The values for the new index.

    axis : {0 or 'index', 1 or 'columns'}, default 0
        The axis to update. The value 0 identifies the rows, and 1 identifies the columns.

    inplace : bool, default False
        Whether to return a new DataFrame instance.

    Returns
    -------
    renamed : DataFrame or None
        An object of type DataFrame or None if ``inplace=True``.

    See Also
    --------
    DataFrame.rename_axis : Alter the name of the index or columns.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    Change the row labels.

    >>> df.set_axis(['a', 'b', 'c'], axis='index').execute()
       A  B
    a  1  4
    b  2  5
    c  3  6

    Change the column labels.

    >>> df.set_axis(['I', 'II'], axis='columns').execute()
       I  II
    0  1   4
    1  2   5
    2  3   6

    Now, update the labels inplace.

    >>> df.set_axis(['i', 'ii'], axis='columns', inplace=True)
    >>> df.execute()
       i  ii
    0  1   4
    1  2   5
    2  3   6
    """
    return _set_axis(df, labels, axis=axis, inplace=inplace)


def series_set_axis(series, labels, axis=0, inplace=False):
    """
    Assign desired index to given axis.

    Indexes for row labels can be changed by assigning
    a list-like or Index.

    Parameters
    ----------
    labels : list-like, Index
        The values for the new index.

    axis : {0 or 'index'}, default 0
        The axis to update. The value 0 identifies the rows.

    inplace : bool, default False
        Whether to return a new Series instance.

    Returns
    -------
    renamed : Series or None
        An object of type Series or None if ``inplace=True``.

    See Also
    --------
    Series.rename_axis : Alter the name of the index.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> s = md.Series([1, 2, 3])
    >>> s.execute()
    0    1
    1    2
    2    3
    dtype: int64

    >>> s.set_axis(['a', 'b', 'c'], axis=0).execute()
    a    1
    b    2
    c    3
    dtype: int64
    """
    return _set_axis(series, labels, axis=axis, inplace=inplace)
