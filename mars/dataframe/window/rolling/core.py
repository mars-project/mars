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

from ....serialize import AnyField, Int64Field, BoolField, StringField, Int32Field
from ...core import DATAFRAME_TYPE
from ...utils import build_empty_df, build_empty_series, validate_axis
from ..core import Window


class Rolling(Window):
    _window = AnyField('window')
    _min_periods = Int64Field('min_periods')
    _center = BoolField('center')
    _win_type = StringField('win_type')
    _on = StringField('on')
    _axis = Int32Field('axis')
    _closed = StringField('closed')

    def __init__(self, window=None, min_periods=None, center=None, win_type=None, on=None,
                 axis=None, closed=None, **kw):
        super().__init__(_window=window, _min_periods=min_periods, _center=center,
                         _win_type=win_type, _on=on, _axis=axis, _closed=closed, **kw)

    @property
    def window(self):
        return self._window

    @property
    def min_periods(self):
        return self._min_periods

    @property
    def center(self):
        return self._center

    @property
    def win_type(self):
        return self._win_type

    @property
    def on(self):
        return self._on

    @property
    def axis(self):
        return self._axis

    @property
    def closed(self):
        return self._closed

    @property
    def params(self):
        p = OrderedDict()
        for attr in ['window', 'min_periods', 'center',
                     'win_type', 'axis', 'on', 'closed']:
            p[attr] = getattr(self, attr)
        return p

    def _repr_name(self):
        return 'Rolling' if self.win_type is None else 'Window'

    def validate(self):
        # leverage pandas itself to do validation
        pd_index = self._input.index_value.to_pandas()
        if isinstance(self._input, DATAFRAME_TYPE):
            empty_obj = build_empty_df(self._input.dtypes, index=pd_index[:0])
        else:
            empty_obj = build_empty_series(self._input.dtype, index=pd_index[:0],
                                           name=self._input.name)
        pd_rolling = empty_obj.rolling(**self.params)
        for k in self.params:
            # update value according to pandas rolling
            setattr(self, '_' + k, getattr(pd_rolling, k))

    def aggregate(self, func, *args, **kwargs):
        from .aggregation import DataFrameRollingAgg

        op = DataFrameRollingAgg(func=func, func_args=args,
                                 func_kwargs=kwargs, **self.params)
        return op(self)

    def agg(self, func, *args, **kwargs):
        return self.aggregate(func, *args, **kwargs)

    def count(self):
        return self.aggregate('count')

    def sum(self, *args, **kwargs):
        return self.aggregate('sum', *args, **kwargs)

    def mean(self, *args, **kwargs):
        return self.aggregate('mean', *args, **kwargs)

    def median(self, **kwargs):
        return self.aggregate('median', **kwargs)

    def var(self, ddof=1, *args, **kwargs):
        return self.aggregate('var', ddof=ddof, *args, **kwargs)

    def std(self, ddof=1, *args, **kwargs):
        return self.aggregate('std', ddof=ddof, *args, **kwargs)

    def min(self, *args, **kwargs):
        return self.aggregate('min', *args, **kwargs)

    def max(self, *args, **kwargs):
        return self.aggregate('max', *args, **kwargs)

    def skew(self, **kwargs):
        return self.aggregate('skew', **kwargs)

    def kurt(self, **kwargs):
        return self.aggregate('kurt', **kwargs)


def rolling(obj, window, min_periods=None, center=False, win_type=None, on=None,
            axis=0, closed=None):
    """
    Provide rolling window calculations.

    Parameters
    ----------
    window : int, or offset
        Size of the moving window. This is the number of observations used for
        calculating the statistic. Each window will be a fixed size.
        If its an offset then this will be the time period of each window. Each
        window will be a variable sized based on the observations included in
        the time-period. This is only valid for datetimelike indexes. This is
        new in 0.19.0
    min_periods : int, default None
        Minimum number of observations in window required to have a value
        (otherwise result is NA). For a window that is specified by an offset,
        `min_periods` will default to 1. Otherwise, `min_periods` will default
        to the size of the window.
    center : bool, default False
        Set the labels at the center of the window.
    win_type : str, default None
        Provide a window type. If ``None``, all points are evenly weighted.
        See the notes below for further information.
    on : str, optional
        For a DataFrame, a datetime-like column on which to calculate the rolling
        window, rather than the DataFrame's index. Provided integer column is
        ignored and excluded from result since an integer index is not used to
        calculate the rolling window.
    axis : int or str, default 0
    closed : str, default None
        Make the interval closed on the 'right', 'left', 'both' or
        'neither' endpoints.
        For offset-based windows, it defaults to 'right'.
        For fixed windows, defaults to 'both'. Remaining cases not implemented
        for fixed windows.

    Returns
    -------
    a Window or Rolling sub-classed for the particular operation

    See Also
    --------
    expanding : Provides expanding transformations.
    ewm : Provides exponential weighted functions.

    Notes
    -----
    By default, the result is set to the right edge of the window. This can be
    changed to the center of the window by setting ``center=True``.
    To learn more about the offsets & frequency strings, please see `this link
    <http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    The recognized win_types are:
    * ``boxcar``
    * ``triang``
    * ``blackman``
    * ``hamming``
    * ``bartlett``
    * ``parzen``
    * ``bohman``
    * ``blackmanharris``
    * ``nuttall``
    * ``barthann``
    * ``kaiser`` (needs beta)
    * ``gaussian`` (needs std)
    * ``general_gaussian`` (needs power, width)
    * ``slepian`` (needs width)
    * ``exponential`` (needs tau), center is set to None.

    If ``win_type=None`` all points are evenly weighted. To learn more about
    different window types see `scipy.signal window functions
    <https://docs.scipy.org/doc/scipy/reference/signal.html#window-functions>`__.

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

    Rolling sum with a window length of 2, using the 'triang'
    window type.

    >>> df.rolling(2, win_type='triang').sum().execute()
         B
    0  NaN
    1  0.5
    2  1.5
    3  NaN
    4  NaN

    Rolling sum with a window length of 2, min_periods defaults
    to the window length.

    >>> df.rolling(2).sum().execute()
         B
    0  NaN
    1  1.0
    2  3.0
    3  NaN
    4  NaN

    Same as above, but explicitly set the min_periods

    >>> df.rolling(2, min_periods=1).sum().execute()
         B
    0  0.0
    1  1.0
    2  3.0
    3  2.0
    4  4.0

    A ragged (meaning not-a-regular frequency), time-indexed DataFrame

    >>> df = md.DataFrame({'B': [0, 1, 2, np.nan, 4]},
    >>>                   index = [md.Timestamp('20130101 09:00:00'),
    >>>                            md.Timestamp('20130101 09:00:02'),
    >>>                            md.Timestamp('20130101 09:00:03'),
    >>>                            md.Timestamp('20130101 09:00:05'),
    >>>                            md.Timestamp('20130101 09:00:06')])
    >>> df.execute()
                           B
    2013-01-01 09:00:00  0.0
    2013-01-01 09:00:02  1.0
    2013-01-01 09:00:03  2.0
    2013-01-01 09:00:05  NaN
    2013-01-01 09:00:06  4.0

    Contrasting to an integer rolling window, this will roll a variable
    length window corresponding to the time period.
    The default for min_periods is 1.

    >>> df.rolling('2s').sum().execute()
                           B
    2013-01-01 09:00:00  0.0
    2013-01-01 09:00:02  1.0
    2013-01-01 09:00:03  3.0
    2013-01-01 09:00:05  NaN
    2013-01-01 09:00:06  4.0
    """
    axis = validate_axis(axis, obj)
    r = Rolling(input=obj, window=window, min_periods=min_periods, center=center,
                win_type=win_type, on=on, axis=axis, closed=closed)
    r.validate()
    return r
