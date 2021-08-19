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
from ...serialization.serializables import StringField
from ..initializer import Series as asseries
from ...tensor import tensor as astensor
from ..core import SERIES_TYPE
from ...tensor.core import TENSOR_TYPE, TensorOrder
from ...core import ENTITY_TYPE, OutputType


class DataFrameToNumeric(DataFrameOperand, DataFrameOperandMixin):

    errors = StringField("errors")
    downcast = StringField("downcast")

    def __init__(self, errors="raise", downcast=None, **kw):
        super().__init__(errors=errors, downcast=downcast, **kw)

    def __call__(self, arg):
        if isinstance(arg, pd.Series):
            arg = asseries(arg)
        elif not isinstance(arg, ENTITY_TYPE):
            arg = astensor(arg)
        if arg.ndim != 1:
            raise ValueError('Input array must be 1 dimensional')
        if arg.size == 0:
            raise ValueError('Input array can not be empty')

        if isinstance(arg, asseries):
            series = arg
            self.output_types = [OutputType.series]
            return self.new_series([series], shape=series.shape, name=series.name,
                                   index_value=series.index_value,
                                   dtype=series.dtype)
        else:
            tensor = arg
            self.output_types = [OutputType.tensor]
            dtype = tensor.dtype
            if dtype.kind == "U":
                dtype = np.dtype(np.object_)
            return self.new_tileables([tensor], shape=tensor.shape, dtype=dtype)[0]

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        out_chunks = []
        for in_chunk in in_df.chunks:
            out_op = op.copy().reset_key()
            chunk_kws = []
            if isinstance(out_df, SERIES_TYPE):
                chunk_kws.append({
                    "dtype": out_df.dtype,
                    "shape": in_chunk.shape,
                    "index": in_chunk.index,
                    "index_value": in_chunk.index_value,
                    "name": in_chunk.name
                })
            elif isinstance(out_df, TENSOR_TYPE):
                chunk_kws.append({
                    'dtype': out_df.dtype,
                    'shape': in_chunk.shape,
                    'order': TensorOrder.C_ORDER,
                    'index': in_chunk.index,
                })
            out_chunks.append(out_op.new_chunk([in_chunk], kws=chunk_kws))

        new_op = op.copy()
        kw = out_df.params
        kw['nsplits'] = in_df.nsplits
        kw['chunks'] = out_chunks
        return new_op.new_tileables(op.inputs, kws=[kw])

    @classmethod
    def execute(cls, ctx, op):
        input_data = ctx[op.inputs[0].key]
        errors_ = op.errors
        downcast_ = op.downcast
        ctx[op.outputs[0].key] = pd.to_numeric(input_data, errors=errors_, downcast=downcast_)


def to_numeric(arg, errors="raise", downcast=None):
    """
        Convert argument to a numeric type.

        The default return dtype is `float64` or `int64`
        depending on the data supplied. Use the `downcast` parameter
        to obtain other dtypes.

        Please note that precision loss may occur if really large numbers
        are passed in. Due to the internal limitations of `ndarray`, if
        numbers smaller than `-9223372036854775808` (np.iinfo(np.int64).min)
        or larger than `18446744073709551615` (np.iinfo(np.uint64).max) are
        passed in, it is very likely they will be converted to float so that
        they can stored in an `ndarray`. These warnings apply similarly to
        `Series` since it internally leverages `ndarray`.

        Parameters
        ----------
        arg : scalar, list, tuple, 1-d array, or Series
            Argument to be converted.
        errors : {'ignore', 'raise', 'coerce'}, default 'raise'
            - If 'raise', then invalid parsing will raise an exception.
            - If 'coerce', then invalid parsing will be set as NaN.
            - If 'ignore', then invalid parsing will return the input.
        downcast : {'integer', 'signed', 'unsigned', 'float'}, default None
            If not None, and if the data has been successfully cast to a
            numerical dtype (or if the data was numeric to begin with),
            downcast that resulting data to the smallest numerical dtype
            possible according to the following rules:

            - 'integer' or 'signed': smallest signed int dtype (min.: np.int8)
            - 'unsigned': smallest unsigned int dtype (min.: np.uint8)
            - 'float': smallest float dtype (min.: np.float32)

            As this behaviour is separate from the core conversion to
            numeric values, any errors raised during the downcasting
            will be surfaced regardless of the value of the 'errors' input.

            In addition, downcasting will only occur if the size
            of the resulting data's dtype is strictly larger than
            the dtype it is to be cast to, so if none of the dtypes
            checked satisfy that specification, no downcasting will be
            performed on the data.

        Returns
        -------
        ret
            Numeric if parsing succeeded.
            Return type depends on input.  Series if Series, otherwise Tensor.

        See Also
        --------
        DataFrame.astype : Cast argument to a specified dtype.
        to_datetime : Convert argument to datetime.
        to_timedelta : Convert argument to timedelta.
        numpy.ndarray.astype : Cast a numpy array to a specified type.
        DataFrame.convert_dtypes : Convert dtypes.

        Examples
        --------
        Take separate series and convert to numeric, coercing when told to

        >>> s = md.Series(['1.0', '2', -3])
        >>> md.to_numeric(s).execute()
        0    1.0
        1    2.0
        2   -3.0
        dtype: float64
        >>> md.to_numeric(s, downcast='float').execute()
        0    1.0
        1    2.0
        2   -3.0
        dtype: float32
        >>> md.to_numeric(s, downcast='signed').execute()
        0    1
        1    2
        2   -3
        dtype: int8
        >>> s = md.Series(['apple', '1.0', '2', -3])
        >>> md.to_numeric(s, errors='ignore').execute()
        0    apple
        1      1.0
        2        2
        3       -3
        dtype: object
        >>> md.to_numeric(s, errors='coerce').execute()
        0    NaN
        1    1.0
        2    2.0
        3   -3.0
        dtype: float64

        Downcasting of nullable integer and floating dtypes is supported:

        >>> s = md.Series([1, 2, 3], dtype="int64")
        >>> md.to_numeric(s, downcast="integer").execute()
        0    1
        1    2
        2    3
        dtype: int8
        >>> s = md.Series([1.0, 2.1, 3.0], dtype="float64")
        >>> md.to_numeric(s, downcast="float").execute()
        0    1.0
        1    2.1
        2    3.0
        dtype: float32
        """
    if errors not in ("ignore", "raise", "coerce"):
        raise ValueError("invalid error value specified")
    if downcast not in (None, "integer", "signed", "unsigned", "float"):
        raise ValueError("invalid downcasting method provided")

    op = DataFrameToNumeric(errors=errors, downcast=downcast)
    return op(arg)
