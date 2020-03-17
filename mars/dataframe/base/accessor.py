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

from functools import wraps

import pandas as pd

from .string_ import _string_method_to_handlers, SeriesStringMethod
from .datetimes import _datetime_method_to_handlers, SeriesDatetimeMethod


class StringAccessor(object):
    def __init__(self, series):
        self._series = series

    def _gen_func(self, method):
        @wraps(getattr(pd.Series.str, method))
        def _inner(*args, **kwargs):
            op = SeriesStringMethod(method=method, method_args=args,
                                    method_kwargs=kwargs)
            return op(self._series)
        return _inner

    def __getitem__(self, item):
        return self._gen_func('__getitem__')(item)

    def __getattr__(self, item):
        if item in _string_method_to_handlers:
            return self._gen_func(item)
        return super(StringAccessor, self).__getattribute__(item)

    def __dir__(self):
        s = set(self.__dict__)
        s.update(_string_method_to_handlers.keys())
        return list(s)

    def split(self, pat=None, n=-1, expand=False):
        return self._gen_func('split')(pat=pat, n=n, expand=expand)

    def rsplit(self, pat=None, n=-1, expand=False):
        return self._gen_func('rsplit')(pat=pat, n=n, expand=expand)

    def cat(self, others=None, sep=None, na_rep=None, join='left'):
        return self._gen_func('cat')(others=others, sep=sep,
                                     na_rep=na_rep, join=join)


class DatetimeAccessor(object):
    def __init__(self, series):
        self._series = series

    def _gen_func(self, method, is_property):
        if is_property:
            op = SeriesDatetimeMethod(method=method,
                                      is_property=is_property)
            return op(self._series)
        else:
            @wraps(getattr(pd.Series.dt, method))
            def _inner(*args, **kwargs):
                op = SeriesDatetimeMethod(method=method,
                                          method_args=args,
                                          method_kwargs=kwargs)
                return op(self._series)
            return _inner

    def __getattr__(self, item):
        if item in _datetime_method_to_handlers:
            is_property = not callable(getattr(pd.Series.dt, item))
            return self._gen_func(item, is_property)
        return super(DatetimeAccessor, self).__getattribute__(item)

    def __dir__(self):
        s = set(self.__dict__)
        s.update(_datetime_method_to_handlers.keys())
        return list(s)
