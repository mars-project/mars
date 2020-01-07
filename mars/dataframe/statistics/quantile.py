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
from ...core import Base, Entity
from ...serialize import KeyField, AnyField, StringField, DataTypeField
from ...tensor.core import TENSOR_TYPE
from ...tensor.datasource import tensor as astensor, from_series as tensor_from_series
from ...tensor.statistics.quantile import quantile as tensor_quantile
from ...tensor.utils import recursive_tile
from ...utils import tokenize
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..core import DATAFRAME_TYPE
from ..datasource.from_tensor import series_from_tensor
from ..utils import parse_index


class DataFrameQuantile(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.QUANTILE

    _input = KeyField('input')
    _q = AnyField('q')
    _interpolation = StringField('interpolation')

    _dtype = DataTypeField('dtype')

    def __init__(self, q=None, interpolation=None, dtype=None, gpu=None,
                 object_type=None, **kw):
        super().__init__(_q=q, _interpolation=interpolation,
                         _dtype=dtype, _gpu=gpu, _object_type=object_type, **kw)

    @property
    def input(self):
        return self._input

    @property
    def q(self):
        return self._q

    @property
    def interpolation(self):
        return self._interpolation

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if isinstance(self._q, TENSOR_TYPE):
            self._q = self._inputs[-1]

    @classmethod
    def _tile_series(cls, op):
        a = tensor_from_series(op.input)
        t = tensor_quantile(a, op.q, interpolation=op.interpolation)
        if op.object_type == ObjectType.scalar:
            r = t
        else:
            r = series_from_tensor(t, index=op.q, name=op.outputs[0].name)
        return [recursive_tile(r)]

    @classmethod
    def tile(cls, op):
        if op.object_type == ObjectType.dataframe:
            raise NotImplementedError
        else:
            return cls._tile_series(op)

    def __call__(self, a, q_input=None):
        inputs = [a]
        if q_input is not None:
            inputs.append(q_input)
        if isinstance(a, DATAFRAME_TYPE):
            raise NotImplementedError
        else:
            if isinstance(self._q, TENSOR_TYPE):
                q_val = self._q
                index_val = pd.Index([], dtype=q_val.dtype)
            else:
                q_val = np.asanyarray(self._q)
                index_val = pd.Index(q_val)
            index_key = tokenize(a, q_val, self._interpolation,
                                 type(self).__name__)

            # get dtype by tensor
            a_t = astensor(a)
            self._dtype = dtype = tensor_quantile(
                a_t, self._q, interpolation=self._interpolation).dtype

            if q_val.ndim == 0:
                self._object_type = ObjectType.scalar
                return self.new_scalar(inputs, dtype=dtype)
            else:
                self._object_type = ObjectType.series
                return self.new_series(
                    inputs, shape=q_val.shape, dtype=dtype,
                    index_value=parse_index(index_val, key=index_key),
                    name=a.name)


def quantile_series(series, q=0.5, interpolation='linear'):
    if isinstance(q, (Base, Entity)):
        q = astensor(q)
        q_input = q
    else:
        q_input = None

    op = DataFrameQuantile(q=q, interpolation=interpolation,
                           gpu=series.op.gpu)
    return op(series, q_input=q_input)
