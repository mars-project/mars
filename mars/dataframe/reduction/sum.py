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

from ... import opcodes as OperandDef
from ...serialize import BoolField, AnyField, DataTypeField, Int32Field
from ..operands import DataFrameOperand, ObjectType
from .core import SeriesReductionMixin


class SeriesSum(DataFrameOperand, SeriesReductionMixin):
    _op_type_ = OperandDef.SUM

    _axis = AnyField('axis')
    _skipna = BoolField('skipna')
    _level = AnyField('level')
    _dtype = DataTypeField('dtype')
    _combine_size = Int32Field('combine_size')

    _func_name = 'sum'

    def __init__(self, axis=None, skipna=None, level=None, dtype=None, gpu=None, sparse=None, **kw):
        super(SeriesSum, self).__init__(_axis=axis, _skipna=skipna, _level=level, _dtype=dtype,
                                        _gpu=gpu, _sparse=sparse, _object_type=ObjectType.series, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def skipna(self):
        return self._skipna

    @property
    def level(self):
        return self._level

    @property
    def dtype(self):
        return self._dtype

    @property
    def combine_size(self):
        return self._combine_size


def sum(df, axis=None, skipna=None, level=None, numeric_only=None, combine_size=None):
    # TODO: enable specify level if we support groupby
    if level is not None:
        raise NotImplementedError('Not support specify level now')
    if axis is not None:
        assert axis == 0 or axis == 'index'
    op = SeriesSum(axis=axis, skipna=skipna, level=level, numeric_only=numeric_only,
                   combine_size=combine_size)
    return op(df)
