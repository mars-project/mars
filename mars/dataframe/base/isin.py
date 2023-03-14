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

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

from ... import opcodes as OperandDef
from ...core import ENTITY_TYPE
from ...serialization.serializables import KeyField, AnyField
from ...tensor.core import TENSOR_TYPE
from ..core import DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE, OutputType
from ..operands import DataFrameOperand, DataFrameOperandMixin
from .drop_duplicates import DataFrameDropDuplicates


class DataFrameIsin(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.ISIN

    input = KeyField("input")
    values = AnyField("values")

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        self.input = next(inputs_iter)
        if len(self._inputs) > 1:
            if isinstance(self.values, dict):
                new_values = dict()
                for k, v in self.values.items():
                    if isinstance(v, ENTITY_TYPE):
                        new_values[k] = next(inputs_iter)
                    else:
                        new_values[k] = v
                self.values = new_values
            else:
                self.values = self._inputs[1]

    def __call__(self, elements):
        inputs = [elements]
        if isinstance(self.values, ENTITY_TYPE):
            inputs.append(self.values)
        elif isinstance(self.values, dict):
            for v in self.values.values():
                if isinstance(v, ENTITY_TYPE):
                    inputs.append(v)

        if elements.ndim == 1:
            return self.new_series(
                inputs,
                shape=elements.shape,
                dtype=np.dtype("bool"),
                index_value=elements.index_value,
                name=elements.name,
            )
        else:
            dtypes = pd.Series(
                [np.dtype(bool) for _ in elements.dtypes], index=elements.dtypes.index
            )
            return self.new_dataframe(
                inputs,
                shape=elements.shape,
                index_value=elements.index_value,
                columns_value=elements.columns_value,
                dtypes=dtypes,
            )

    @classmethod
    def _tile_entity_values(cls, op):
        from ...core.context import get_context
        from ...tensor.base.unique import TensorUnique
        from ..utils import auto_merge_chunks
        from ..arithmetic.bitwise_or import tree_dataframe_or

        in_elements = op.input
        out_elements = op.outputs[0]
        # values contains mars objects
        chunks_list = []
        in_chunks = in_elements.chunks
        if any(len(t.chunks) > 4 for t in op.inputs):
            # yield and merge value chunks to reduce graph nodes
            yield_chunks = [c for c in in_chunks]
            unique_values = []
            for value in op.inputs[1:]:
                if len(value.chunks) >= len(in_chunks) * 2:
                    # when value chunks is much more than in_chunks,
                    # we call drop_duplicates to reduce the amount of data.
                    if isinstance(value, TENSOR_TYPE):
                        chunks = [
                            TensorUnique(
                                return_index=False,
                                return_inverse=False,
                                return_counts=False,
                            ).new_chunk(
                                [c], index=c.index, shape=(np.nan,), dtype=c.dtype
                            )
                            for c in value.chunks
                        ]
                        unique_values.append(
                            TensorUnique(
                                return_index=False,
                                return_inverse=False,
                                return_counts=False,
                            ).new_tensor(
                                [value],
                                chunks=chunks,
                                nsplits=((np.nan,) * len(chunks),),
                                shape=(np.nan,),
                                dtype=value.dtype,
                            )
                        )
                        yield_chunks += chunks
                    else:
                        # is series
                        chunks = [
                            DataFrameDropDuplicates(
                                keep="first",
                                ignore_index=False,
                                method="tree",
                                output_types=[OutputType.series],
                            ).new_chunk(
                                [c],
                                index=c.index,
                                index_value=c.index_value,
                                name=c.name,
                                dtype=c.dtype,
                                shape=(np.nan,),
                            )
                            for c in value.chunks
                        ]
                        unique_values.append(
                            DataFrameDropDuplicates(
                                keep="first",
                                ignore_index=False,
                                method="tree",
                                output_types=[OutputType.series],
                            ).new_series(
                                [value],
                                chunks=chunks,
                                nsplits=((np.nan,) * len(chunks),),
                                index_value=value.index_value,
                                dtype=value.dtype,
                                shape=(np.nan,),
                            )
                        )
                        yield_chunks += chunks
                else:
                    yield_chunks += value.chunks
                    unique_values.append(value)
            yield yield_chunks
            in_elements = auto_merge_chunks(get_context(), op.input)
            in_chunks = in_elements.chunks
            for value in unique_values:
                if isinstance(value, SERIES_TYPE):
                    merged = auto_merge_chunks(get_context(), value)
                    chunks_list.append(merged.chunks)
                elif isinstance(value, ENTITY_TYPE):
                    chunks_list.append(value.chunks)
        else:
            for value in op.inputs[1:]:
                if isinstance(value, ENTITY_TYPE):
                    chunks_list.append(value.chunks)

        out_chunks = []
        for in_chunk in in_chunks:
            isin_chunks = []
            for value_chunks in itertools.product(*chunks_list):
                input_chunks = [in_chunk] + list(value_chunks)
                isin_chunks.append(cls._new_chunk(op, in_chunk, input_chunks))
            out_chunk = tree_dataframe_or(*isin_chunks, index=in_chunk.index)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        params = out_elements.params
        params["nsplits"] = in_elements.nsplits
        params["chunks"] = out_chunks
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def tile(cls, op):
        in_elements = op.input
        out_elements = op.outputs[0]

        if len(op.inputs) > 1:
            return (yield from cls._tile_entity_values(op))

        out_chunks = []
        for chunk in in_elements.chunks:
            out_chunk = cls._new_chunk(op, chunk, [chunk])
            out_chunks.append(out_chunk)

        new_op = op.copy()
        params = out_elements.params
        params["nsplits"] = in_elements.nsplits
        params["chunks"] = out_chunks
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def _new_chunk(cls, op, chunk, input_chunks):
        out_elements = op.outputs[0]
        chunk_op = op.copy().reset_key()
        if out_elements.ndim == 1:
            out_chunk = chunk_op.new_chunk(
                input_chunks,
                shape=chunk.shape,
                dtype=out_elements.dtype,
                index_value=chunk.index_value,
                name=out_elements.name,
                index=chunk.index,
            )
        else:
            chunk_dtypes = pd.Series(
                [np.dtype(bool) for _ in chunk.dtypes], index=chunk.dtypes.index
            )
            out_chunk = chunk_op.new_chunk(
                input_chunks,
                shape=chunk.shape,
                index_value=chunk.index_value,
                columns_value=chunk.columns_value,
                dtypes=chunk_dtypes,
                index=chunk.index,
            )
        return out_chunk

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
        raise TypeError(
            "only list-like objects are allowed to be passed to isin(), "
            f"you passed a [{type(values)}]"
        )
    op = DataFrameIsin(values=values)
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
    elif not isinstance(
        values, (SERIES_TYPE, DATAFRAME_TYPE, TENSOR_TYPE, INDEX_TYPE, dict)
    ):
        raise TypeError(
            "only list-like objects or dict are allowed to be passed to isin(), "
            f"you passed a [{type(values)}]"
        )
    op = DataFrameIsin(values=values)
    return op(df)
