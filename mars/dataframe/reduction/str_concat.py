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

from ... import opcodes as OperandDef
from ...core import OutputType
from ...serialization.serializables import StringField
from .core import DataFrameReductionOperand, DataFrameReductionMixin


class DataFrameStrConcat(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = OperandDef.STR_CONCAT
    _func_name = 'str_concat'

    _sep = StringField('sep')
    _na_rep = StringField('na_rep')

    def __init__(self, sep=None, na_rep=None, **kw):
        super().__init__(_sep=sep, _na_rep=na_rep, **kw)

    @property
    def sep(self):
        return self._sep

    @property
    def na_rep(self):
        return self._na_rep

    def get_reduction_args(self, axis=None):
        return dict(sep=self._sep, na_rep=self._na_rep)

    @property
    def is_atomic(self):
        return True

    @classmethod
    def get_reduction_callable(cls, op):
        sep, na_rep = op.sep, op.na_rep

        def str_concat(obj):
            return build_str_concat_object(obj, sep=sep, na_rep=na_rep)

        return str_concat


def build_str_concat_object(df, sep=None, na_rep=None):
    output_type = OutputType.series if df.ndim == 2 else OutputType.scalar
    op = DataFrameStrConcat(sep=sep, na_rep=na_rep, output_types=[output_type])
    return op(df)
