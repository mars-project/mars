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
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ...core import OutputType
from ..utils import parse_index


def reverse(x):
    if x is None:
        return
    return x[::-1]


class DataFrameTranspose(DataFrameOperand, DataFrameOperandMixin):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.output_types = [OutputType.dataframe]

    def __call__(self, arg):
        new_shape = arg.shape[::-1]
        dtypes = arg.dtypes
        columns_value = arg.index_value
        index_value = parse_index(dtypes.index, store_data=True)
        return self.new_dataframe([arg], shape=new_shape, dtypes=dtypes,
                                  columns_value=columns_value, index_value=index_value)

    @classmethod
    def tile(cls, op):
        out_chunks = []
        for c in op.inputs[0].chunks:
            chunk_op = op.copy().reset_key()
            chunk_shape = tuple(s if np.isnan(s) else int(s)
                                for s in reverse(c.shape))
            chunk_idx = reverse(c.index)
            index_value = parse_index(c.dtypes.index)
            columns_value = c.index_value
            out_chunk = chunk_op.new_chunk([c], shape=chunk_shape,
                                           index=chunk_idx,
                                           index_value=index_value,
                                           columns_value=columns_value,
                                           dtypes=None)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        nsplits = reverse(op.inputs[0].nsplits)
        return new_op.new_dataframe(op.inputs, op.outputs[0].shape, dtypes=op.outputs[0].dtypes,
                                    chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        inp = ctx[op.inputs[0].key]
        out = inp.transpose()
        ctx[op.outputs[0].key] = out


def transpose(arg):
    """
            Transpose index and columns.

            Reflect the DataFrame over its main diagonal by writing rows as columns
            and vice-versa. The property :attr:`.T` is an accessor to the method
            :meth:`transpose`.

            Parameters
            ----------
            *args : tuple, optional
                Accepted for compatibility with NumPy.
            copy : bool, default False
                Whether to copy the data after transposing, even for DataFrames
                with a single dtype.

                Note that a copy is always required for mixed dtype DataFrames,
                or for DataFrames with any extension types.

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
            DataFrame with the `object` dtype. In such a case, a copy of the data
            is always made.

            Examples
            --------
            **Square DataFrame with homogeneous dtype**

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
    return op(arg)
