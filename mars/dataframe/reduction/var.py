# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
from ...serialize import Int32Field
from ...utils import lazy_import
from .core import DataFrameReductionOperand, DataFrameReductionMixin, ObjectType

cudf = lazy_import('cudf', globals=globals())


class DataFrameVar(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = OperandDef.VAR
    _func_name = 'var'

    _ddof = Int32Field('ddof')

    def __init__(self, ddof=None, **kw):
        super().__init__(_ddof=ddof, **kw)

    @property
    def ddof(self):
        return self._ddof

    @classmethod
    def _keep_dim(cls, df, op):
        xdf = cudf if op.gpu else pd
        if np.isscalar(df):
            return df
        if op.axis == 1:
            return xdf.DataFrame(df)
        else:
            return xdf.DataFrame(df).transpose()

    @classmethod
    def _execute_map(cls, ctx, op):
        xdf = cudf if op.gpu else pd
        in_data = ctx[op.inputs[0].key]
        if isinstance(in_data, pd.Series):
            count = in_data.count()
        else:
            count = in_data.count(axis=op.axis, numeric_only=op.numeric_only)
        r = cls._execute_reduction(in_data, op, reduction_func='sum')
        avg = cls._keep_dim(r / count, op)

        kwargs = dict(axis=op.axis, skipna=op.skipna)
        if op.numeric_only:
            in_data = in_data[avg.columns]
        avg = avg if np.isscalar(avg) else np.array(avg)
        var_square = ((in_data.subtract(avg)) ** 2).sum(**kwargs)

        if isinstance(in_data, xdf.Series):
            ctx[op.outputs[0].key] = (r, count, var_square)
        else:
            ctx[op.outputs[0].key] = tuple(cls._keep_dim(df, op) for df in [r, count, var_square])

    @classmethod
    def _execute_combine(cls, ctx, op):
        data, concat_count, var_square = ctx[op.inputs[0].key]
        xdf = cudf if op.gpu else pd

        count = concat_count.sum(axis=op.axis)
        r = cls._execute_reduction(data, op, reduction_func='sum')
        avg = cls._keep_dim(r / count, op)
        avg_diff = data / concat_count - avg

        kwargs = dict(axis=op.axis, skipna=op.skipna)
        reduced_var_square = var_square.sum(**kwargs) + (concat_count * avg_diff ** 2).sum(**kwargs)
        if isinstance(data, xdf.Series):
            ctx[op.outputs[0].key] = (r, count, reduced_var_square)
        else:
            ctx[op.outputs[0].key] = tuple(cls._keep_dim(df, op) for df in [r, count, reduced_var_square])

    @classmethod
    def _execute_agg(cls, ctx, op):
        data, concat_count, var_square = ctx[op.inputs[0].key]

        count = concat_count.sum(axis=op.axis)
        r = cls._execute_reduction(data, op, reduction_func='sum')
        avg = cls._keep_dim(r / count, op)
        avg_diff = (data / concat_count).subtract(avg, axis=op.axis)

        kwargs = dict(axis=op.axis, skipna=op.skipna)
        reduced_var_square = var_square.sum(**kwargs) + (concat_count * avg_diff ** 2).sum(**kwargs)
        var = reduced_var_square / (count - op.ddof)
        ctx[op.outputs[0].key] = var


def var_series(series, axis=None, skipna=None, level=None, ddof=1, combine_size=None):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameVar(axis=axis, skipna=skipna, level=level, ddof=ddof,
                      combine_size=combine_size, object_type=ObjectType.scalar,
                      use_inf_as_na=use_inf_as_na)
    return op(series)


def var_dataframe(df, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, combine_size=None):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameVar(axis=axis, skipna=skipna, level=level, ddof=ddof,
                      numeric_only=numeric_only, combine_size=combine_size,
                      object_type=ObjectType.series, use_inf_as_na=use_inf_as_na)
    return op(df)
