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
from distutils.version import LooseVersion

import pandas as pd

from ....serialize import Int64Field, BoolField, Int32Field
from ...utils import validate_axis
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

    def aggregate(self, func, **kwargs):
        from .aggregation import DataFrameExpandingAgg

        count_always_valid = kwargs.pop('_count_always_valid', LooseVersion(pd.__version__) < '1.0.0')

        op = DataFrameExpandingAgg(func=func, count_always_valid=count_always_valid, **self.params)
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
    """
    Provide expanding transformations.

    Parameters
    ----------
    min_periods : int, default 1
    Minimum number of observations in window required to have a value
    (otherwise result is NA).
    center : bool, default False
    Set the labels at the center of the window.
    axis : int or str, default 0

    Returns
    -------
    a Window sub-classed for the particular operation

    See Also
    --------
    rolling : Provides rolling window calculations.
    ewm : Provides exponential weighted functions.

    Notes
    -----
    By default, the result is set to the right edge of the window. This can be
    changed to the center of the window by setting ``center=True``.

    Examples
    --------
    >>> import numpy as np
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'B': [0, 1, 2, np.nan, 4]})
    >>> df.execute()
         B
    0  0.0
    1  1.0
    2  2.0
    3  NaN
    4  4.0
    >>> df.expanding(2).sum().execute()
         B
    0  NaN
    1  1.0
    2  3.0
    3  3.0
    4  7.0
    """
    axis = validate_axis(axis, obj)

    if center:
        raise NotImplementedError('center == True is not supported')
    if axis == 1:
        raise NotImplementedError('axis other than 0 is not supported')

    return Expanding(input=obj, min_periods=min_periods, center=center, axis=axis)
