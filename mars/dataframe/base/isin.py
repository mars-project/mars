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
from pandas.api.types import is_list_like

from ... import opcodes as OperandDef
from ...core import ENTITY_TYPE, recursive_tile
from ...serialization.serializables import KeyField, AnyField
from ...tensor.core import TENSOR_TYPE
from ...utils import has_unknown_shape
from ..core import DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin


class DataFrameIsin(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.ISIN

    _input = KeyField('input')
    _values = AnyField('values')

    def __init__(self, values=None, output_types=None, **kw):
        super().__init__(_values=values, _output_types=output_types, **kw)

    @property
    def input(self):
        return self._input

    @property
    def values(self):
        return self._values

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        self._input = next(inputs_iter)
        if len(self._inputs) > 1:
            if isinstance(self._values, dict):
                new_values = dict()
                for k, v in self._values.items():
                    if isinstance(v, ENTITY_TYPE):
                        new_values[k] = next(inputs_iter)
                    else:
                        new_values[k] = v
                self._values = new_values
            else:
                self._values = self._inputs[1]

    def __call__(self, elements):
        inputs = [elements]
        if isinstance(self._values, ENTITY_TYPE):
            inputs.append(self._values)
        elif isinstance(self._values, dict):
            for v in self._values.values():
                if isinstance(v, ENTITY_TYPE):
                    inputs.append(v)

        if elements.ndim == 1:
            return self.new_series(inputs, shape=elements.shape,
                                   dtype=np.dtype('bool'),
                                   index_value=elements.index_value,
                                   name=elements.name)
        else:
            dtypes = pd.Series([np.dtype(bool) for _ in elements.dtypes],
                               index=elements.dtypes.index)
            return self.new_dataframe(inputs, shape=elements.shape,
                                      index_value=elements.index_value,
                                      columns_value=elements.columns_value,
                                      dtypes=dtypes)

    @classmethod
    def tile(cls, op):
        in_elements = op.input
        out_elements = op.outputs[0]

        values_inputs = []
        if len(op.inputs) > 1:
            for value in op.inputs[1:]:
                # make sure arg has known shape when it's a md.Series
                if has_unknown_shape(value):
                    yield
                value = yield from recursive_tile(value.rechunk(value.shape))
                values_inputs.append(value)

        out_chunks = []
        for chunk in in_elements.chunks:
            chunk_op = op.copy().reset_key()
            chunk_inputs = [chunk]
            if len(op.inputs) > 1:
                chunk_inputs.extend(v.chunks[0] for v in values_inputs)
            if out_elements.ndim == 1:
                out_chunk = chunk_op.new_chunk(chunk_inputs, shape=chunk.shape,
                                               dtype=out_elements.dtype,
                                               index_value=chunk.index_value,
                                               name=out_elements.name,
                                               index=chunk.index)
            else:
                chunk_dtypes = pd.Series([np.dtype(bool) for _ in chunk.dtypes],
                                         index=chunk.dtypes.index)
                out_chunk = chunk_op.new_chunk(chunk_inputs, shape=chunk.shape,
                                               index_value=chunk.index_value,
                                               columns_value=chunk.columns_value,
                                               dtypes=chunk_dtypes,
                                               index=chunk.index)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        params = out_elements.params
        params['nsplits'] = in_elements.nsplits
        params['chunks'] = out_chunks
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        inputs_iter = iter(op.inputs)
        elements = ctx[next(inputs_iter).key]

        if isinstance(op.values, dict):
            values = dict()
            for k, v in op.values.items():
                if isinstance(v, ENTITY_TYPE):
                    values[k] = ctx[next(inputs_iter).key]
                else:
                    values[k] = v
        else:
            if isinstance(op.values, ENTITY_TYPE):
                values = ctx[next(inputs_iter).key]
            else:
                values = op.values

        try:
            ctx[op.outputs[0].key] = elements.isin(values)
        except ValueError:
            # buffer read-only
            ctx[op.outputs[0].key] = elements.copy().isin(values.copy())


def series_isin(elements, values):
    """
    Whether elements in Series are contained in `values`.

    Return a boolean Series showing whether each element in the Series
    matches an element in the passed sequence of `values` exactly.

    Parameters
    ----------
    values : set or list-like
        The sequence of values to test. Passing in a single string will
        raise a ``TypeError``. Instead, turn a single string into a
        list of one element.

    Returns
    -------
    Series
        Series of booleans indicating if each element is in values.

    Raises
    ------
    TypeError
      * If `values` is a string

    See Also
    --------
    DataFrame.isin : Equivalent method on DataFrame.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> s = md.Series(['lama', 'cow', 'lama', 'beetle', 'lama',
    ...                'hippo'], name='animal')
    >>> s.isin(['cow', 'lama']).execute()
    0     True
    1     True
    2     True
    3    False
    4     True
    5    False
    Name: animal, dtype: bool

    Passing a single string as ``s.isin('lama')`` will raise an error. Use
    a list of one element instead:

    >>> s.isin(['lama']).execute()
    0     True
    1    False
    2     True
    3    False
    4     True
    5    False
    Name: animal, dtype: bool
    """
    if is_list_like(values):
        values = list(values)
    elif not isinstance(values, (SERIES_TYPE, TENSOR_TYPE, INDEX_TYPE)):
        raise TypeError('only list-like objects are allowed to be passed to isin(), '
                        f'you passed a [{type(values)}]')
    op = DataFrameIsin(values)
    return op(elements)


def df_isin(df, values):
    """
    Whether each element in the DataFrame is contained in values.

    Parameters
    ----------
    values : iterable, Series, DataFrame or dict
        The result will only be true at a location if all the
        labels match. If `values` is a Series, that's the index. If
        `values` is a dict, the keys must be the column names,
        which must match. If `values` is a DataFrame,
        then both the index and column labels must match.

    Returns
    -------
    DataFrame
        DataFrame of booleans showing whether each element in the DataFrame
        is contained in values.

    See Also
    --------
    DataFrame.eq: Equality test for DataFrame.
    Series.isin: Equivalent method on Series.
    Series.str.contains: Test if pattern or regex is contained within a
        string of a Series or Index.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'num_legs': [2, 4], 'num_wings': [2, 0]},
    ...                   index=['falcon', 'dog'])
    >>> df.execute()
            num_legs  num_wings
    falcon         2          2
    dog            4          0

    When ``values`` is a list check whether every value in the DataFrame
    is present in the list (which animals have 0 or 2 legs or wings)

    >>> df.isin([0, 2]).execute()
            num_legs  num_wings
    falcon      True       True
    dog        False       True

    When ``values`` is a dict, we can pass values to check for each
    column separately:

    >>> df.isin({'num_wings': [0, 3]}).execute()
            num_legs  num_wings
    falcon     False      False
    dog        False       True

    When ``values`` is a Series or DataFrame the index and column must
    match. Note that 'falcon' does not match based on the number of legs
    in df2.

    >>> other = md.DataFrame({'num_legs': [8, 2], 'num_wings': [0, 2]},
    ...                      index=['spider', 'falcon'])
    >>> df.isin(other).execute()
            num_legs  num_wings
    falcon      True       True
    dog        False      False
    """
    if is_list_like(values) and not isinstance(values, dict):
        values = list(values)
    elif not isinstance(values, (SERIES_TYPE, DATAFRAME_TYPE, TENSOR_TYPE, INDEX_TYPE, dict)):
        raise TypeError('only list-like objects or dict are allowed to be passed to isin(), '
                        f'you passed a [{type(values)}]')
    op = DataFrameIsin(values)
    return op(df)
