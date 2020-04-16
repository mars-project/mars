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

from collections import OrderedDict

from ....serialize import Int64Field, BoolField, Int32Field
from ..core import Window


class Expanding(Window):
    _min_periods = Int64Field('min_periods')
    _axis = Int32Field('axis')
    _center = BoolField('center')

    def __init__(self, min_periods=None, axis=None, center=None, **kw):
        super().__init__(_min_periods=min_periods, _axis=axis, _center=center, **kw)

    @property
    def min_periods(self):
        return self._min_periods

    @property
    def axis(self):
        return self._axis

    @property
    def center(self):
        return self._center

    def __call__(self, df):
        return df.expanding(**self.params)

    @property
    def params(self):
        p = OrderedDict()
        for k in ['min_periods', 'center', 'axis']:
            p[k] = getattr(self, k)
        return p

    def aggregate(self, func):
        from .aggregation import DataFrameExpandingAgg

        op = DataFrameExpandingAgg(func=func, **self.params)
        return op(self)

    agg = aggregate

    def sum(self):
        return self.aggregate('sum')

    def count(self):
        return self.aggregate('count')

    def min(self):
        return self.aggregate('min')

    def max(self):
        return self.aggregate('max')

    def mean(self):
        return self.aggregate('mean')

    def var(self):
        return self.aggregate('var')

    def std(self):
        return self.aggregate('std')


def expanding(obj, min_periods=1, center=False, axis=0):
    if center:
        raise NotImplementedError('center == True is not supported')
    if axis == 1:
        raise NotImplementedError('axis other than 0 is not supported')

    return Expanding(input=obj, min_periods=min_periods, center=center, axis=axis)
