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
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index


class DataFrameTranspose(DataFrameOperand, DataFrameOperandMixin):
    _op_code_ = opcodes.TRANSPOSE

    def __init__(self, **kw):
        super().__init__(**kw)
        self.output_types = [OutputType.dataframe]

    def __call__(self, args):
        arg = args[0]
        new_shape = arg.shape[::-1]
        columns_value = arg.index_value
        index_value = parse_index(arg.dtypes.index)
        return self.new_dataframe(
            [arg],
            shape=new_shape,
            dtypes=None,
            columns_value=columns_value,
            index_value=index_value,
        )

    @classmethod
    def tile(cls, op):
        out_chunks = []
        for c in op.inputs[0].chunks:
            chunk_op = op.copy().reset_key()
            chunk_shape = tuple(s if np.isnan(s) else int(s) for s in c.shape[::-1])
            chunk_idx = c.index[::-1]
            index_value = parse_index(c.dtypes.index)
            columns_value = c.index_value
            out_chunk = chunk_op.new_chunk(
                [c],
                shape=chunk_shape,
                index=chunk_idx,
                index_value=index_value,
                columns_value=columns_value,
                dtypes=None,
            )
            out_chunks.append(out_chunk)

        new_op = op.copy()
        nsplits = op.inputs[0].nsplits[::-1]
        params = op.outputs[0].params
        return new_op.new_dataframe(
            op.inputs, chunks=out_chunks, nsplits=nsplits, **params
        )

    @classmethod
    def execute(cls, ctx, op):
        inp = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = inp.transpose()


def transpose(*args):
    """
    Transpose index and columns.

    Reflect the DataFrame over its main diagonal by writing rows as columns
    and vice-versa. The property :attr:`.T` is an accessor to the method
    :meth:`transpose`.

    Parameters
    ----------
    *args : tuple, optional
            Accepted for compatibility with NumPy.

    Returns
    -------
    DataFrame
        The transposed DataFrame.

    See Also
    --------
    numpy.transpose : Permute the dimensions of a given array.

    Notes
    -----
    Transposing a DataFrame with mixed dtypes will result in a homogeneous
    DataFrame with the `object` dtype.

    Examples
    --------
    **Square DataFrame with homogeneous dtype**

    >>> import mars.dataframe as md
    >>> d1 = {'col1': [1, 2], 'col2': [3, 4]}
    >>> df1 = md.DataFrame(data=d1).execute()
    >>> df1
        col1  col2
    0     1     3
    1     2     4

    >>> df1_transposed = df1.T.execute() # or df1.transpose().execute()
    >>> df1_transposed
          0  1
    col1  1  2
    col2  3  4

    When the dtype is homogeneous in the original DataFrame, we get a
    transposed DataFrame with the same dtype:

    >>> df1.dtypes
    col1    int64
    col2    int64
    dtype: object

    >>> df1_transposed.dtypes
    0    int64
    1    int64
    dtype: object

    **Non-square DataFrame with mixed dtypes**

    >>> d2 = {'name': ['Alice', 'Bob'],
    ...       'score': [9.5, 8],
    ...       'employed': [False, True],
    ...       'kids': [0, 0]}
    >>> df2 = md.DataFrame(data=d2).execute()
    >>> df2
        name  score  employed  kids
    0  Alice    9.5     False     0
    1    Bob    8.0      True     0

    >>> df2_transposed = df2.T.execute() # or df2.transpose().execute()
    >>> df2_transposed
                  0     1
    name      Alice   Bob
    score       9.5   8.0
    employed  False  True
    kids          0     0

    When the DataFrame has mixed dtypes, we get a transposed DataFrame with
    the `object` dtype:

    >>> df2.dtypes
    name         object
    score       float64
    employed       bool
    kids          int64
    dtype: object

    >>> df2_transposed.dtypes
    0    object
    1    object
    dtype: object
    """
    op = DataFrameTranspose()
    return op(args)
