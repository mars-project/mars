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
from ...utils import classproperty
from .core import DataFrameUnaryUfunc


class DataFrameIsUFuncMixin:
    @classmethod
    def _get_output_dtype(cls, df):
        if df.ndim == 2:
            return pd.Series(np.dtype(bool), index=df.dtypes.index)
        else:
            return np.dtype(bool)


class DataFrameIsNan(DataFrameIsUFuncMixin, DataFrameUnaryUfunc):
    _op_type_ = OperandDef.ISNAN
    _func_name = 'isnan'

    @classproperty
    def tensor_op_type(self):
        from ...tensor.arithmetic import TensorIsNan
        return TensorIsNan


class DataFrameIsInf(DataFrameIsUFuncMixin, DataFrameUnaryUfunc):
    _op_type_ = OperandDef.ISINF
    _func_name = 'isinf'

    @classproperty
    def tensor_op_type(self):
        from ...tensor.arithmetic import TensorIsInf
        return TensorIsInf


class DataFrameIsFinite(DataFrameIsUFuncMixin, DataFrameUnaryUfunc):
    _op_type_ = OperandDef.ISFINITE
    _func_name = 'isfinite'

    @classproperty
    def tensor_op_type(self):
        from ...tensor.arithmetic import TensorIsFinite
        return TensorIsFinite
