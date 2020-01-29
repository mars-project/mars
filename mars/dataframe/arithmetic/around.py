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

from ... import opcodes as OperandDef
from ...serialize import Int32Field
from ...utils import classproperty
from .core import DataFrameUnaryUfunc


class DataFrameAround(DataFrameUnaryUfunc):
    _op_type_ = OperandDef.AROUND
    _func_name = 'around'

    _decimals = Int32Field('decimals')

    def __init__(self, decimals=None, object_type=None, **kw):
        super().__init__(_decimals=decimals, object_type=object_type, **kw)

    @classproperty
    def tensor_op_type(self):
        from ...tensor.arithmetic import TensorAround
        return TensorAround

    @property
    def decimals(self):
        return self._decimals

    @classmethod
    def execute(cls, ctx, op):
        df = ctx[op.inputs[0].key]
        func_name = getattr(cls, '_func_name')
        if hasattr(df, func_name):
            ctx[op.outputs[0].key] = getattr(df, func_name)(decimals=op.decimals)
        else:
            ctx[op.outputs[0].key] = getattr(np, func_name)(df, decimals=op.decimals)


def around(df, decimals=0, *args, **kwargs):
    if len(args) > 0:
        raise TypeError('round() takes 0 positional arguments '
                        'but {} was given'.format(len(args)))
    op = DataFrameAround(decimals=decimals, **kwargs)
    return op(df)
