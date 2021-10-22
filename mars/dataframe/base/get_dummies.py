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

from ...core import OutputType, recursive_tile
from ...serialization.serializables import StringField, BoolField, AnyField, ListField
from ..core import SERIES_TYPE
from ..datasource.dataframe import from_pandas as from_pandas_df
from ..datasource.series import from_pandas as from_pandas_series
from ..initializer import Series as asseries
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..reduction.unique import unique
from ..utils import gen_unknown_index_value

_encoding_dtype_kind = ["O", "S", "U"]


class DataFrameGetDummies(DataFrameOperand, DataFrameOperandMixin):
    prefix = AnyField("prefix")
    prefix_sep = StringField("prefix_sep")
    dummy_na = BoolField("dummy_na")
    columns = ListField("columns")
    sparse = BoolField("sparse")
    drop_first = BoolField("drop_first")
    dtype = AnyField("dtype")

    def __init__(
        self,
        prefix=None,
        prefix_sep=None,
        dummy_na=None,
        columns=None,
        sparse=None,
        drop_first=None,
        dtype=None,
        **kws,
    ):
        super().__init__(
            prefix=prefix,
            prefix_sep=prefix_sep,
            dummy_na=dummy_na,
            columns=columns,
            sparse=sparse,
            drop_first=drop_first,
            dtype=dtype,
            **kws,
        )
        self.output_types = [OutputType.dataframe]

    @classmethod
    def tile(cls, op):
        inp = op.inputs[0]
        out = op.outputs[0]
        if len(inp.chunks) == 1:
            chunk_op = op.copy().reset_key()
            chunk_param = out.params
            chunk_param["index"] = (0, 0)
            chunk = chunk_op.new_chunk(inp.chunks, kws=[chunk_param])
            new_op = op.copy().reset_key()
            param = out.params
            param["chunks"] = [chunk]
            param["nsplits"] = ((np.nan,), (np.nan,))
            return new_op.new_dataframe(op.inputs, kws=[param])
        elif isinstance(inp, SERIES_TYPE):
            unique_inp = yield from recursive_tile(unique(inp))
            chunks = []
            for c in inp.chunks:
                chunk_op = op.copy().reset_key()
                chunk_param = out.params
                chunk_param["index_value"] = gen_unknown_index_value(c.index_value)
                chunk_param["index"] = (c.index[0], 0)
                chunk = chunk_op.new_chunk([c] + unique_inp.chunks, kws=[chunk_param])
                chunks.append(chunk)

            new_op = op.copy().reset_key()
            param = out.params
            param["chunks"] = chunks
            param["nsplits"] = (tuple([np.nan] * inp.chunk_shape[0]), (np.nan,))
            return new_op.new_dataframe(op.inputs, kws=[param])
        else:
            if op.columns:
                encoding_columns = op.columns
            else:
                encoding_columns = []
                for idx, dtype in enumerate(inp.dtypes.values):
                    if dtype.kind in _encoding_dtype_kind:
                        column = inp.dtypes.index[idx]
                        encoding_columns.append(column)
            # reindex, make encoding columns in the end of dataframe, to keep pace with pandas.get_dummies
            total_columns = list(inp.columns.to_pandas().array)
            for col in encoding_columns:
                total_columns.remove(col)
            total_columns.extend(encoding_columns)
            inp = yield from recursive_tile(inp[total_columns])

            unique_chunks = dict()
            for col in encoding_columns:
                unique_chunks[col] = yield from recursive_tile(unique(inp[col]))

            chunks = []
            prefix = op.prefix
            column_to_prefix = dict()
            for c in inp.chunks:
                chunk_op = op.copy().reset_key()
                chunk_op.columns = []
                if isinstance(chunk_op.prefix, list):
                    chunk_op.prefix = []
                chunk_param = c.params
                chunk_param["shape"] = (np.nan, np.nan)
                chunk_columns = c.dtypes.index
                inp_chunk = [c]
                for chunk_column in chunk_columns:
                    if chunk_column in encoding_columns:
                        chunk_op.columns.append(chunk_column)
                        inp_chunk.extend(unique_chunks[chunk_column].chunks)
                        if isinstance(prefix, list):
                            if chunk_column in column_to_prefix.keys():
                                chunk_op.prefix.append(column_to_prefix[chunk_column])
                            else:
                                column_to_prefix[chunk_column] = prefix[0]
                                chunk_op.prefix.append(prefix[0])
                                prefix = prefix[1:]
                chunk = chunk_op.new_chunk(inp_chunk, kws=[chunk_param])
                chunks.append(chunk)

            new_op = op.copy()
            kw = out.params.copy()
            kw["chunks"] = chunks
            kw["nsplits"] = (
                tuple([np.nan] * inp.chunk_shape[0]),
                tuple([np.nan] * inp.chunk_shape[1]),
            )
            return new_op.new_dataframe(op.inputs, kws=[kw])

    @classmethod
    def execute(cls, ctx, op):
        inp = ctx[op.inputs[0].key]
        result_length = inp.shape[0]
        unique_inputs = []
        for unique_input in op.inputs[1:]:
            unique_inputs.append(ctx[unique_input.key].tolist())

        if unique_inputs:
            if isinstance(inp, pd.Series):
                extra_series = pd.Series(unique_inputs[0])
                inp = pd.concat([inp, extra_series])
            else:
                # make all unique_input's length the same, then get a dataframe
                max_length = len(max(unique_inputs, key=len))
                unique_inputs = [
                    unique_list + [unique_list[0]] * (max_length - len(unique_list))
                    for unique_list in unique_inputs
                ]
                extra_dataframe = pd.DataFrame(dict(zip(op.columns, unique_inputs)))

                # add the columns that need not to encode, to concat extra_dataframe and inp
                total_columns = list(inp.columns.array)
                for col in op.columns:
                    total_columns.remove(col)
                remain_columns = total_columns
                not_encode_columns = []
                if len(remain_columns) > 0:
                    for col in remain_columns:
                        not_encode_columns.append([inp[col].iloc[0]] * max_length)
                not_encode_dataframe = pd.DataFrame(
                    dict(zip(remain_columns, not_encode_columns))
                )

                extra_dataframe = pd.concat(
                    [not_encode_dataframe, extra_dataframe], axis=1
                )
                inp = pd.concat([inp, extra_dataframe], axis=0)

        result = pd.get_dummies(
            inp,
            op.prefix,
            op.prefix_sep,
            op.dummy_na,
            op.columns,
            op.sparse,
            op.drop_first,
            op.dtype,
        )
        ctx[op.outputs[0].key] = result.iloc[:result_length]

    def __call__(self, data):
        if isinstance(data, (list, tuple)):
            data = asseries(data)
        elif isinstance(data, pd.Series):
            data = from_pandas_series(data)
        elif isinstance(data, pd.DataFrame):
            data = from_pandas_df(data)

        if self.prefix is not None:
            if isinstance(self.prefix, list):
                if self.columns is not None:
                    encoding_col_num = len(self.columns)
                else:
                    encoding_col_num = 0
                    for dtype in data.dtypes.values:
                        if dtype.kind in _encoding_dtype_kind:
                            encoding_col_num += 1
                prefix_num = len(self.prefix)
                if prefix_num != encoding_col_num:
                    raise ValueError(
                        f"Length of 'prefix' ({prefix_num}) did not match "
                        + f"the length of the columns being encoded ({encoding_col_num})"
                    )
            elif isinstance(self.prefix, dict):
                if self.columns is not None:
                    encoding_col_num = len(self.columns)
                    prefix_num = len(self.prefix)
                    if prefix_num != encoding_col_num:
                        raise ValueError(
                            f"Length of 'prefix' ({prefix_num}) did not match "
                            + f"the length of the columns being encoded ({encoding_col_num})"
                        )
                    columns = self.prefix.keys()
                    for columns_columnname, prefix_columnname in zip(
                        columns, list(self.columns)
                    ):
                        if columns_columnname != prefix_columnname:
                            raise KeyError(f"{columns_columnname}")
                else:
                    self.columns = list(self.prefix.keys())
                # Convert prefix from dict to list, to simplify tile work
                self.prefix = list(self.prefix.values())

        return self.new_dataframe(
            [data],
            shape=(np.nan, np.nan),
            dtypes=None,
            index_value=data.index_value,
            columns_value=None,
        )


def get_dummies(
    data,
    prefix=None,
    prefix_sep="_",
    dummy_na=False,
    columns=None,
    sparse=False,
    drop_first=False,
    dtype=None,
):
    """
    Convert categorical variable into dummy/indicator variables.

    Parameters
    ----------
    data : array-like, Series, or DataFrame
        Data of which to get dummy indicators.
    prefix : str, list of str, or dict of str, default None
        String to append DataFrame column names.
        Pass a list with length equal to the number of columns
        when calling get_dummies on a DataFrame. Alternatively, `prefix`
        can be a dictionary mapping column names to prefixes.
    prefix_sep : str, default '_'
        If appending prefix, separator/delimiter to use. Or pass a
        list or dictionary as with `prefix`.
    dummy_na : bool, default False
        Add a column to indicate NaNs, if False NaNs are ignored.
    columns : list-like, default None
        Column names in the DataFrame to be encoded.
        If `columns` is None then all the columns with
        `object` or `category` dtype will be converted.
    sparse : bool, default False
        Whether the dummy-encoded columns should be backed by
        a :class:`SparseArray` (True) or a regular NumPy array (False).
    drop_first : bool, default False
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level.
    dtype : dtype, default np.uint8
        Data type for new columns. Only a single dtype is allowed.

    Returns
    -------
    DataFrame
        Dummy-coded data.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> s = md.Series(list('abca'))

    >>> md.get_dummies(s).execute()
       a  b  c
    0  1  0  0
    1  0  1  0
    2  0  0  1
    3  1  0  0

    >>> s1 = ['a', 'b', np.nan]

    >>> md.get_dummies(s1).execute()
       a  b
    0  1  0
    1  0  1
    2  0  0

    >>> md.get_dummies(s1, dummy_na=True).execute()
       a  b  NaN
    0  1  0    0
    1  0  1    0
    2  0  0    1

    >>> df = md.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
    ...                    'C': [1, 2, 3]})

    >>> md.get_dummies(df, prefix=['col1', 'col2']).execute()
       C  col1_a  col1_b  col2_a  col2_b  col2_c
    0  1       1       0       0       1       0
    1  2       0       1       1       0       0
    2  3       1       0       0       0       1

    >>> md.get_dummies(pd.Series(list('abcaa'))).execute()
       a  b  c
    0  1  0  0
    1  0  1  0
    2  0  0  1
    3  1  0  0
    4  1  0  0

    >>> md.get_dummies(pd.Series(list('abcaa')), drop_first=True).execute()
       b  c
    0  0  0
    1  1  0
    2  0  1
    3  0  0
    4  0  0

    >>> md.get_dummies(pd.Series(list('abc')), dtype=float).execute()
         a    b    c
    0  1.0  0.0  0.0
    1  0.0  1.0  0.0
    2  0.0  0.0  1.0
    """
    if columns is not None and not isinstance(columns, list):
        raise TypeError("Input must be a list-like for parameter `columns`")

    op = DataFrameGetDummies(
        prefix, prefix_sep, dummy_na, columns, sparse, drop_first, dtype
    )

    return op(data)
