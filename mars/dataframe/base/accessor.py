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
from typing import Iterable

import pandas as pd

from .string_ import _string_method_to_handlers, SeriesStringMethod
from .datetimes import _datetime_method_to_handlers, SeriesDatetimeMethod


class StringAccessor:
    def __init__(self, series):
        self._series = series

    @classmethod
    def _gen_func(cls, method):
        @wraps(getattr(pd.Series.str, method))
        def _inner(self, *args, **kwargs):
            op = SeriesStringMethod(method=method, method_args=args,
                                    method_kwargs=kwargs)
            return op(self._series)
        return _inner

    def __getitem__(self, item):
        return self._gen_func('__getitem__')(self, item)

    def __dir__(self) -> Iterable[str]:
        s = set(super().__dir__())
        s.update(_string_method_to_handlers.keys())
        return list(s)

    @classmethod
    def _register(cls, method):
        setattr(cls, method, cls._gen_func(method))

    def split(self, pat=None, n=-1, expand=False):
        return self._gen_func('split')(self, pat=pat, n=n, expand=expand)

    def rsplit(self, pat=None, n=-1, expand=False):
        return self._gen_func('rsplit')(self, pat=pat, n=n, expand=expand)

    def cat(self, others=None, sep=None, na_rep=None, join='left'):
        return self._gen_func('cat')(self, others=others, sep=sep,
                                     na_rep=na_rep, join=join)


class DatetimeAccessor:
    def __init__(self, series):
        self._series = series

    @classmethod
    def _gen_func(cls, method, is_property):
        @wraps(getattr(pd.Series.dt, method))
        def _inner(self, *args, **kwargs):
            op = SeriesDatetimeMethod(method=method,
                                      is_property=is_property,
                                      method_args=args,
                                      method_kwargs=kwargs)
            return op(self._series)
        return _inner

    @classmethod
    def _register(cls, method):
        is_property = not callable(getattr(pd.Series.dt, method))
        func = cls._gen_func(method, is_property)
        if is_property:
            func = property(func)
        setattr(cls, method, func)

    def __dir__(self) -> Iterable[str]:
        s = set(super().__dir__())
        s.update(_datetime_method_to_handlers.keys())
        return list(s)


class CachedAccessor:
    def __init__(self, name: str, accessor) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            # we're accessing the attribute of the class, i.e., Dataset.geo
            return self._accessor
        if self._name not in obj._accessors:
            obj._accessors[self._name] = self._accessor(obj)
        return obj._accessors[self._name]
