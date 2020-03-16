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

from ....serialize import Serializable, AnyField, Int64Field, \
    BoolField, StringField, Int32Field, KeyField
from ...core import DATAFRAME_TYPE
from ...utils import build_empty_df, build_empty_series


class Rolling(Serializable):
    _input = KeyField('input')
    _window = AnyField('window')
    _min_periods = Int64Field('min_periods')
    _center = BoolField('center')
    _win_type = StringField('win_type')
    _on = StringField('on')
    _axis = Int32Field('axis')
    _closed = StringField('closed')

    def __init__(self, input=None, window=None, min_periods=None, center=None,  # pylint: disable=redefined-builtin
                 win_type=None, on=None, axis=None, closed=None, **kw):
        super(Rolling, self).__init__(
            _input=input, _window=window, _min_periods=min_periods, _center=center,
            _win_type=win_type, _on=on, _axis=axis, _closed=closed, **kw)

    @property
    def input(self):
        return self._input

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

    def __repr__(self):
        kvs = ['{}={}'.format(k, v) for k, v in self.params.items()
               if v is not None]
        return 'Window [{}]'.format(','.join(kvs))

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

    def __getitem__(self, item):
        columns = self._input.dtypes.index
        if isinstance(item, (list, tuple)):
            item = list(item)
            for col in item:
                if col not in columns:
                    raise KeyError('Column not found: {}'.format(col))
        else:
            if item not in columns:
                raise KeyError('Column not found: {}'.format(item))

        return Rolling(input=self._input[item], **self.params)

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
    r = Rolling(input=obj, window=window, min_periods=min_periods, center=center,
                win_type=win_type, on=on, axis=axis, closed=closed)
    r.validate()
    return r
