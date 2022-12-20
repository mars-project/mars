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

from ... import opcodes as OperandDef
from ...config import options
from ...core import OutputType
from .core import (
    DataFrameReductionOperand,
    DataFrameReductionMixin,
    recursive_tile,
    DATAFRAME_TYPE,
)


class DataFrameAny(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = OperandDef.ANY
    _func_name = "any"

    @property
    def is_atomic(self):
        return True

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]
        if op.axis is None and isinstance(in_df, DATAFRAME_TYPE):
            dtypes = pd.Series([out_df.dtype])
            index = in_df.dtypes.index
            out_df = yield from recursive_tile(
                in_df.agg(
                    cls.get_reduction_callable(op),
                    axis=0,
                    _numeric_only=op.numeric_only,
                    _bool_only=op.bool_only,
                    _combine_size=op.combine_size,
                    _output_type=OutputType.series,
                    _dtypes=dtypes,
                    _index=index,
                )
            )
            out_df = yield from recursive_tile(
                out_df.agg(
                    cls.get_reduction_callable(op),
                    axis=0,
                    _numeric_only=op.numeric_only,
                    _bool_only=op.bool_only,
                    _combine_size=op.combine_size,
                    _output_type=OutputType.scalar,
                    _dtypes=out_df.dtype,
                    _index=None,
                )
            )
            return [out_df]
        else:
            return (yield from super().tile(op))

    def __call__(self, df):
        if self.axis is None and isinstance(df, DATAFRAME_TYPE):
            return self.new_scalar([df], np.dtype("bool"))
        else:
            return super().__call__(df)


def any_series(
    series,
    axis=0,
    bool_only=None,
    skipna=True,
    level=None,
    combine_size=None,
    method=None,
):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameAny(
        axis=axis,
        skipna=skipna,
        level=level,
        bool_only=bool_only,
        combine_size=combine_size,
        output_types=[OutputType.scalar],
        use_inf_as_na=use_inf_as_na,
        method=method,
    )
    return op(series)


def any_dataframe(
    df,
    axis=0,
    bool_only=None,
    skipna=True,
    level=None,
    combine_size=None,
    method=None,
):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    output_types = [OutputType.series] if axis is not None else [OutputType.scalar]
    op = DataFrameAny(
        axis=axis,
        skipna=skipna,
        level=level,
        bool_only=bool_only,
        combine_size=combine_size,
        output_types=output_types,
        use_inf_as_na=use_inf_as_na,
        method=method,
    )
    return op(df)


def any_index(index):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameAny(output_types=[OutputType.scalar], use_inf_as_na=use_inf_as_na)
    return op(index)
