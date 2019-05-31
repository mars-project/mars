# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import operator

from .... import opcodes as OperandDef
from ....utils import classproperty
from ..core import DataFrameOperand
from .core import DataFrameUnaryOpMixin

class DataFrameAbs(DataFrameOperand, DataFrameUnaryOpMixin):
    _op_type_ = OperandDef.ABS

    def __init__(self, **kw):
        super(DataFrameAbs, self).__init__(**kw)

    @classproperty
    def _operator(self):
        return operator.abs

def abs(df):
    op = DataFrameAbs()
    return op(df)
