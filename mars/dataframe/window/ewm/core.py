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

import math
from collections import OrderedDict

import pandas as pd

from ....lib.version import parse as parse_version
from ....serialization.serializables import Int64Field, BoolField, Int32Field, Float64Field
from ...utils import validate_axis
from ..core import Window

_default_min_period_1 = parse_version(pd.__version__) >= parse_version('1.1.0')
_pd_1_3_repr = parse_version(pd.__version__) >= parse_version('1.3.0')


class EWM(Window):
    _alpha = Float64Field('alpha')
    _min_periods = Int64Field('min_periods')
    _adjust = BoolField('adjust')
    _ignore_na = BoolField('ignore_na')
    _axis = Int32Field('axis')

    def __init__(self, alpha=None, min_periods=None, adjust=None, ignore_na=None, axis=None, **kw):
        super().__init__(_alpha=alpha, _min_periods=min_periods, _adjust=adjust, _ignore_na=ignore_na,
                         _axis=axis, **kw)

    @property
    def alpha(self):
        return self._alpha

    @property
    def min_periods(self):
        return self._min_periods

    @property
    def adjust(self):
        return self._adjust

    @property
    def ignore_na(self):
        return self._ignore_na

    @property
    def axis(self):
        return self._axis

    @property
    def params(self):
        p = OrderedDict()
        for k in ['alpha', 'min_periods', "adjust", "ignore_na", "axis"]:
            p[k] = getattr(self, k)
        return p

    def __call__(self, df):
        return df.ewm(**self.params)

    def _repr(self, params):
        com = 1.0 / params.pop('alpha') - 1
        params['com'] = int(com) if _pd_1_3_repr and com == math.floor(com) else com
        try:
            params.move_to_end('com', last=False)
        except AttributeError:  # pragma: no cover
            pass
        return super()._repr(params)

    def _repr_name(self):
        try:
            from pandas.core.window import ExponentialMovingWindow  # noqa: F401
            return 'ExponentialMovingWindow'
        except ImportError:  # pragma: no cover
            return 'EWM'

    def aggregate(self, func):
        from .aggregation import DataFrameEwmAgg

        params = self.params
        params['alpha_ignore_na'] = params.pop('ignore_na', False)
        params['validate_columns'] = False
        op = DataFrameEwmAgg(func=func, **params)
        return op(self)

    agg = aggregate

    def mean(self):
        return self.aggregate('mean')

    def var(self):
        return self.aggregate('var')

    def std(self):
        return self.aggregate('std')


def ewm(obj, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True,
        ignore_na=False, axis=0):
    r"""
    Provide exponential weighted functions.

    Parameters
    ----------
    com : float, optional
        Specify decay in terms of center of mass,
        :math:`\alpha = 1 / (1 + com),\text{ for } com \geq 0`.
    span : float, optional
        Specify decay in terms of span,
        :math:`\alpha = 2 / (span + 1),\text{ for } span \geq 1`.
    halflife : float, optional
        Specify decay in terms of half-life,
        :math:`\alpha = 1 - exp(log(0.5) / halflife),\text{for} halflife > 0`.
    alpha : float, optional
        Specify smoothing factor :math:`\alpha` directly,
        :math:`0 < \alpha \leq 1`.
    min_periods : int, default 0
        Minimum number of observations in window required to have a value
        (otherwise result is NA).
    adjust : bool, default True
        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings
        (viewing EWMA as a moving average).
    ignore_na : bool, default False
        Ignore missing values when calculating weights;
        specify True to reproduce pre-0.15.0 behavior.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        The axis to use. The value 0 identifies the rows, and 1
        identifies the columns.

    Returns
    -------
    DataFrame
        A Window sub-classed for the particular operation.

    See Also
    --------
    rolling : Provides rolling window calculations.
    expanding : Provides expanding transformations.

    Notes
    -----
    Exactly one of center of mass, span, half-life, and alpha must be provided.

    Allowed values and relationship between the parameters are specified in the
    parameter descriptions above; see the link at the end of this section for
    a detailed explanation.

    When adjust is True (default), weighted averages are calculated using
    weights (1-alpha)**(n-1), (1-alpha)**(n-2), ..., 1-alpha, 1.

    When adjust is False, weighted averages are calculated recursively as:

       weighted_average[0] = arg[0];
       weighted_average[i] = (1-alpha)*weighted_average[i-1] + alpha*arg[i].

    When ignore_na is False (default), weights are based on absolute positions.
    For example, the weights of x and y used in calculating the final weighted
    average of [x, None, y] are (1-alpha)**2 and 1 (if adjust is True), and
    (1-alpha)**2 and alpha (if adjust is False).

    When ignore_na is True (reproducing pre-0.15.0 behavior), weights are based
    on relative positions. For example, the weights of x and y used in
    calculating the final weighted average of [x, None, y] are 1-alpha and 1
    (if adjust is True), and 1-alpha and alpha (if adjust is False).

    More details can be found at
    https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#exponentially-weighted-windows

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
    >>> df.ewm(com=0.5).mean().execute()
              B
    0  0.000000
    1  0.750000
    2  1.615385
    3  1.615385
    4  3.670213
    """
    axis = validate_axis(axis, obj)

    decay_count = 0
    for arg in (com, span, halflife, alpha):
        if arg is not None:
            decay_count += 1

    if decay_count == 0:
        raise ValueError('Must pass one of comass, span, halflife, or alpha')
    if decay_count > 1:
        raise ValueError('comass, span, halflife, and alpha are mutually exclusive')

    if com is not None:
        if com < 0:
            raise ValueError('comass must satisfy: comass >= 0')
        alpha = 1.0 / (1 + com)
    elif span is not None:
        if span < 1:
            raise ValueError('span must satisfy: span >= 1')
        alpha = 2.0 / (1 + span)
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError('halflife must satisfy: halflife > 0')
        alpha = 1.0 - math.exp(math.log(0.5) / halflife)
    if alpha <= 0 or alpha > 1:
        raise ValueError('alpha must satisfy: 0 < alpha <= 1')

    if not adjust and not ignore_na:
        raise NotImplementedError('adjust == False when ignore_na == False not implemented')
    if axis == 1:
        raise NotImplementedError('axis other than 0 is not supported')

    if alpha == 1:
        return obj.expanding(min_periods=min_periods, axis=axis)

    if _default_min_period_1:
        min_periods = min_periods or 1

    return EWM(input=obj, alpha=alpha, min_periods=min_periods, adjust=adjust,
               ignore_na=ignore_na, axis=axis)
