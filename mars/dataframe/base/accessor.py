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

from functools import wraps
from typing import Iterable

import pandas as pd

from ...utils import adapt_mars_docstring
from .string_ import _string_method_to_handlers, SeriesStringMethod
from .datetimes import _datetime_method_to_handlers, SeriesDatetimeMethod


class StringAccessor:
    """
    Vectorized string functions for Series and Index.
    NAs stay NA unless handled otherwise by a particular method.
    Patterned after Python's string methods, with some inspiration from
    R's stringr package.
    Examples
    --------
    >>> import mars.dataframe as md
    >>> s = md.Series(["A_Str_Series"])
    >>> s.execute()
    0    A_Str_Series
    dtype: object
    >>> s.str.split("_").execute()
    0    [A, Str, Series]
    dtype: object
    >>> s.str.replace("_", "").execute()
    0    AStrSeries
    dtype: object
    """
    def __init__(self, series):
        self._series = series

    @classmethod
    def _gen_func(cls, method):
        @wraps(getattr(pd.Series.str, method))
        def _inner(self, *args, **kwargs):
            op = SeriesStringMethod(method=method, method_args=args,
                                    method_kwargs=kwargs)
            return op(self._series)

        _inner.__doc__ = adapt_mars_docstring(getattr(pd.Series.str, method).__doc__)
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
        r"""
        Split strings around given separator/delimiter.

        Splits the string in the Series/Index from the beginning,
        at the specified delimiter string. Equivalent to :meth:`str.split`.

        Parameters
        ----------
        pat : str, optional
            String or regular expression to split on.
            If not specified, split on whitespace.
        n : int, default -1 (all)
            Limit number of splits in output.
            ``None``, 0 and -1 will be interpreted as return all splits.
        expand : bool, default False
            Expand the splitted strings into separate columns.

            * If ``True``, return DataFrame/MultiIndex expanding dimensionality.
            * If ``False``, return Series/Index, containing lists of strings.

        Returns
        -------
        Series, Index, DataFrame or MultiIndex
            Type matches caller unless ``expand=True`` (see Notes).

        See Also
        --------
        Series.str.split : Split strings around given separator/delimiter.
        Series.str.rsplit : Splits string around given separator/delimiter,
            starting from the right.
        Series.str.join : Join lists contained as elements in the Series/Index
            with passed delimiter.
        str.split : Standard library version for split.
        str.rsplit : Standard library version for rsplit.

        Notes
        -----
        The handling of the `n` keyword depends on the number of found splits:

        - If found splits > `n`,  make first `n` splits only
        - If found splits <= `n`, make all splits
        - If for a certain row the number of found splits < `n`,
          append `None` for padding up to `n` if ``expand=True``

        If using ``expand=True``, Series and Index callers return DataFrame and
        MultiIndex objects, respectively.

        Examples
        --------
        >>> import numpy as np
        >>> import mars.dataframe as md
        >>> s = md.Series(["this is a regular sentence",
        >>>                "https://docs.python.org/3/tutorial/index.html",
        >>>                np.nan])
        >>> s.execute()
        0                       this is a regular sentence
        1    https://docs.python.org/3/tutorial/index.html
        2                                              NaN
        dtype: object

        In the default setting, the string is split by whitespace.

        >>> s.str.split().execute()
        0                   [this, is, a, regular, sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                                NaN
        dtype: object

        Without the `n` parameter, the outputs of `rsplit` and `split`
        are identical.

        >>> s.str.rsplit().execute()
        0                   [this, is, a, regular, sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                                NaN
        dtype: object

        The `n` parameter can be used to limit the number of splits on the
        delimiter. The outputs of `split` and `rsplit` are different.

        >>> s.str.split(n=2).execute()
        0                     [this, is, a regular sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                                NaN
        dtype: object

        >>> s.str.rsplit(n=2).execute()
        0                     [this is a, regular, sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                                NaN
        dtype: object

        The `pat` parameter can be used to split by other characters.

        >>> s.str.split(pat = "/").execute()
        0                         [this is a regular sentence]
        1    [https:, , docs.python.org, 3, tutorial, index...
        2                                                  NaN
        dtype: object

        When using ``expand=True``, the split elements will expand out into
        separate columns. If NaN is present, it is propagated throughout
        the columns during the split.

        >>> s.str.split(expand=True).execute()
                                                       0     1     2        3
        0                                           this    is     a  regular
        1  https://docs.python.org/3/tutorial/index.html  None  None     None
        2                                            NaN   NaN   NaN      NaN \
                     4
        0     sentence
        1         None
        2          NaN

        For slightly more complex use cases like splitting the html document name
        from a url, a combination of parameter settings can be used.

        >>> s.str.rsplit("/", n=1, expand=True).execute()
                                            0           1
        0          this is a regular sentence        None
        1  https://docs.python.org/3/tutorial  index.html
        2                                 NaN         NaN

        Remember to escape special characters when explicitly using regular
        expressions.

        >>> s = pd.Series(["1+1=2"])
        >>> s.str.split(r"\+|=", expand=True).execute()
             0    1    2
        0    1    1    2
        """
        return self._gen_func('split')(self, pat=pat, n=n, expand=expand)

    def rsplit(self, pat=None, n=-1, expand=False):
        return self._gen_func('rsplit')(self, pat=pat, n=n, expand=expand)

    def cat(self, others=None, sep=None, na_rep=None, join='left'):
        return self._gen_func('cat')(self, others=others, sep=sep,
                                     na_rep=na_rep, join=join)

    rsplit.__doc__ = adapt_mars_docstring(pd.Series.str.rsplit.__doc__)
    cat.__doc__ = adapt_mars_docstring(pd.Series.str.cat.__doc__)


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

        _inner.__doc__ = adapt_mars_docstring(getattr(pd.Series.dt, method).__doc__)
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
