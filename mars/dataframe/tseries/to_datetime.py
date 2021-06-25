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

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar, is_dict_like

from ... import opcodes
from ...core import recursive_tile
from ...serialization.serializables import KeyField, StringField, BoolField, AnyField
from ...tensor import tensor as astensor
from ...tensor.core import TENSOR_CHUNK_TYPE
from ..core import DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE, INDEX_CHUNK_TYPE
from ..initializer import DataFrame as asdataframe, Series as asseries, Index as asindex
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index


class DataFrameToDatetime(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.TO_DATETIME

    _arg = KeyField('arg')
    _errors = StringField('errors')
    _dayfirst = BoolField('dayfirst')
    _yearfirst = BoolField('yearfirst')
    _utc = BoolField('utc')
    _format = StringField('format')
    _exact = BoolField('exact')
    _unit = StringField('unit')
    _infer_datetime_format = BoolField('infer_datetime_format')
    _origin = AnyField('origin')
    _cache = BoolField('cache')

    def __init__(self, errors=None, dayfirst=None, yearfirst=None, utc=None,
                 format=None, exact=None, unit=None, infer_datetime_format=None,
                 origin=None, cache=None, **kw):
        super().__init__(_errors=errors, _dayfirst=dayfirst, _yearfirst=yearfirst,
                         _utc=utc, _format=format, _exact=exact, _unit=unit,
                         _infer_datetime_format=infer_datetime_format, _origin=origin,
                         _cache=cache, **kw)

    @property
    def arg(self):
        return self._arg

    @property
    def errors(self):
        return self._errors

    @property
    def dayfirst(self):
        return self._dayfirst

    @property
    def yearfirst(self):
        return self._yearfirst

    @property
    def utc(self):
        return self._utc

    @property
    def format(self):
        return self._format

    @property
    def exact(self):
        return self._exact

    @property
    def unit(self):
        return self._unit

    @property
    def infer_datetime_format(self):
        return self._infer_datetime_format

    @property
    def origin(self):
        return self._origin

    @property
    def cache(self):
        return self._cache

    @property
    def _params(self):
        return tuple(getattr(self, k) for k in self._keys_
                     if k not in self._no_copy_attrs_ and
                     k != '_arg' and hasattr(self, k))

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._arg = self._inputs[0]

    def __call__(self, arg):
        if is_scalar(arg):
            ret = pd.to_datetime(arg, errors=self._errors, dayfirst=self._dayfirst,
                                 yearfirst=self._yearfirst, utc=self._utc,
                                 format=self._format, exact=self._exact,
                                 unit=self._unit, infer_datetime_format=self._infer_datetime_format,
                                 origin=self._origin, cache=self._cache)
            return astensor(ret)

        dtype = np.datetime64(1, 'ns').dtype
        if isinstance(arg, (pd.Series, SERIES_TYPE)):
            arg = asseries(arg)
            return self.new_series([arg], shape=arg.shape,
                                   dtype=dtype, index_value=arg.index_value,
                                   name=arg.name)
        if is_dict_like(arg) or isinstance(arg, DATAFRAME_TYPE):
            arg = asdataframe(arg)
            columns = arg.columns_value.to_pandas().tolist()
            if sorted(columns) != sorted(['year', 'month', 'day']):
                missing = ','.join(c for c in ['day', 'month', 'year'] if c not in columns)
                raise ValueError('to assemble mappings requires at least '
                                 f'that [year, month, day] be specified: [{missing}] is missing')
            return self.new_series([arg], shape=(arg.shape[0],),
                                   dtype=dtype, index_value=arg.index_value)
        elif isinstance(arg, (pd.Index, INDEX_TYPE)):
            arg = asindex(arg)
            return self.new_index([arg], shape=arg.shape,
                                  dtype=dtype,
                                  index_value=parse_index(pd.Index([], dtype=dtype),
                                                          self._params, arg),
                                  name=arg.name)
        else:
            arg = astensor(arg)
            if arg.ndim != 1:
                raise TypeError('arg must be a string, datetime, '
                                'list, tuple, 1-d tensor, or Series')
            return self.new_index([arg], shape=arg.shape,
                                  dtype=dtype,
                                  index_value=parse_index(pd.Index([], dtype=dtype),
                                                          self._params, arg))

    @classmethod
    def tile(cls, op: "DataFrameToDatetime"):
        out = op.outputs[0]
        arg = op.arg

        if isinstance(arg, DATAFRAME_TYPE):
            if np.isnan(arg.shape[0]) or \
                    any(np.isnan(s) for s in arg.nsplits[1]):  # pragma: no cover
                yield

            arg = yield from recursive_tile(arg.rechunk({1: arg.shape[1]}))

        out_chunks = []
        for chunk in arg.chunks:
            chunk_op = op.copy().reset_key()
            if isinstance(chunk, (TENSOR_CHUNK_TYPE, INDEX_CHUNK_TYPE)):
                chunk_index_value = parse_index(
                    pd.Index([], dtype=out.dtype), op._params, chunk)
            else:
                chunk_index_value = chunk.index_value

            out_chunk = chunk_op.new_chunk([chunk], shape=(chunk.shape[0],),
                                           dtype=out.dtype,
                                           index_value=chunk_index_value,
                                           name=out.name,
                                           index=(chunk.index[0],))
            out_chunks.append(out_chunk)

        params = out.params
        params['nsplits'] = (arg.nsplits[0],)
        params['chunks'] = out_chunks
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx, op: "DataFrameToDatetime"):
        arg = ctx[op.arg.key]

        call = partial(pd.to_datetime,
                       errors=op.errors, dayfirst=op.dayfirst,
                       yearfirst=op.yearfirst, utc=op.utc,
                       format=op.format, exact=op.exact, unit=op.unit,
                       infer_datetime_format=op.infer_datetime_format,
                       origin=op.origin, cache=op.cache)

        try:
            ctx[op.outputs[0].key] = call(arg)
        except ValueError:  # pragma: no cover
            ctx[op.outputs[0].key] = call(arg.copy())


def to_datetime(arg,
                errors: str = 'raise',
                dayfirst: bool = False,
                yearfirst: bool = False,
                utc: bool = None,
                format: str = None,
                exact: bool = True,
                unit: str = None,
                infer_datetime_format: bool = False,
                origin: Any = 'unix',
                cache: bool = True):
    """
    Convert argument to datetime.

    Parameters
    ----------
    arg : int, float, str, datetime, list, tuple, 1-d array, Series DataFrame/dict-like
        The object to convert to a datetime.
    errors : {'ignore', 'raise', 'coerce'}, default 'raise'
        - If 'raise', then invalid parsing will raise an exception.
        - If 'coerce', then invalid parsing will be set as NaT.
        - If 'ignore', then invalid parsing will return the input.
    dayfirst : bool, default False
        Specify a date parse order if `arg` is str or its list-likes.
        If True, parses dates with the day first, eg 10/11/12 is parsed as
        2012-11-10.
        Warning: dayfirst=True is not strict, but will prefer to parse
        with day first (this is a known bug, based on dateutil behavior).
    yearfirst : bool, default False
        Specify a date parse order if `arg` is str or its list-likes.

        - If True parses dates with the year first, eg 10/11/12 is parsed as
          2010-11-12.
        - If both dayfirst and yearfirst are True, yearfirst is preceded (same
          as dateutil).

        Warning: yearfirst=True is not strict, but will prefer to parse
        with year first (this is a known bug, based on dateutil behavior).
    utc : bool, default None
        Return UTC DatetimeIndex if True (converting any tz-aware
        datetime.datetime objects as well).
    format : str, default None
        The strftime to parse time, eg "%d/%m/%Y", note that "%f" will parse
        all the way up to nanoseconds.
        See strftime documentation for more information on choices:
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
    exact : bool, True by default
        Behaves as:
        - If True, require an exact format match.
        - If False, allow the format to match anywhere in the target string.

    unit : str, default 'ns'
        The unit of the arg (D,s,ms,us,ns) denote the unit, which is an
        integer or float number. This will be based off the origin.
        Example, with unit='ms' and origin='unix' (the default), this
        would calculate the number of milliseconds to the unix epoch start.
    infer_datetime_format : bool, default False
        If True and no `format` is given, attempt to infer the format of the
        datetime strings, and if it can be inferred, switch to a faster
        method of parsing them. In some cases this can increase the parsing
        speed by ~5-10x.
    origin : scalar, default 'unix'
        Define the reference date. The numeric values would be parsed as number
        of units (defined by `unit`) since this reference date.

        - If 'unix' (or POSIX) time; origin is set to 1970-01-01.
        - If 'julian', unit must be 'D', and origin is set to beginning of
          Julian Calendar. Julian day number 0 is assigned to the day starting
          at noon on January 1, 4713 BC.
        - If Timestamp convertible, origin is set to Timestamp identified by
          origin.
    cache : bool, default True
        If True, use a cache of unique, converted dates to apply the datetime
        conversion. May produce significant speed-up when parsing duplicate
        date strings, especially ones with timezone offsets. The cache is only
        used when there are at least 50 values. The presence of out-of-bounds
        values will render the cache unusable and may slow down parsing.

    Returns
    -------
    datetime
        If parsing succeeded.
        Return type depends on input:

        - list-like: DatetimeIndex
        - Series: Series of datetime64 dtype
        - scalar: Timestamp

        In case when it is not possible to return designated types (e.g. when
        any element of input is before Timestamp.min or after Timestamp.max)
        return will have datetime.datetime type (or corresponding
        array/Series).

    See Also
    --------
    DataFrame.astype : Cast argument to a specified dtype.
    to_timedelta : Convert argument to timedelta.
    convert_dtypes : Convert dtypes.

    Examples
    --------
    Assembling a datetime from multiple columns of a DataFrame. The keys can be
    common abbreviations like ['year', 'month', 'day', 'minute', 'second',
    'ms', 'us', 'ns']) or plurals of the same

    >>> import mars.dataframe as md

    >>> df = md.DataFrame({'year': [2015, 2016],
    ...                    'month': [2, 3],
    ...                    'day': [4, 5]})
    >>> md.to_datetime(df).execute()
    0   2015-02-04
    1   2016-03-05
    dtype: datetime64[ns]

    If a date does not meet the `timestamp limitations
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
    #timeseries-timestamp-limits>`_, passing errors='ignore'
    will return the original input instead of raising any exception.

    Passing errors='coerce' will force an out-of-bounds date to NaT,
    in addition to forcing non-dates (or non-parseable dates) to NaT.

    >>> md.to_datetime('13000101', format='%Y%m%d', errors='ignore').execute()
    datetime.datetime(1300, 1, 1, 0, 0)
    >>> md.to_datetime('13000101', format='%Y%m%d', errors='coerce').execute()
    NaT

    Passing infer_datetime_format=True can often-times speedup a parsing
    if its not an ISO8601 format exactly, but in a regular format.

    >>> s = md.Series(['3/11/2000', '3/12/2000', '3/13/2000'] * 1000)
    >>> s.head().execute()
    0    3/11/2000
    1    3/12/2000
    2    3/13/2000
    3    3/11/2000
    4    3/12/2000
    dtype: object

    Using a unix epoch time

    >>> md.to_datetime(1490195805, unit='s').execute()
    Timestamp('2017-03-22 15:16:45')
    >>> md.to_datetime(1490195805433502912, unit='ns').execute()
    Timestamp('2017-03-22 15:16:45.433502912')

    .. warning:: For float arg, precision rounding might happen. To prevent
        unexpected behavior use a fixed-width exact type.

    Using a non-unix epoch origin

    >>> md.to_datetime([1, 2, 3], unit='D',
    ...                origin=md.Timestamp('1960-01-01')).execute()
    DatetimeIndex(['1960-01-02', '1960-01-03', '1960-01-04'], \
dtype='datetime64[ns]', freq=None)
    """
    op = DataFrameToDatetime(errors=errors, dayfirst=dayfirst,
                             yearfirst=yearfirst, utc=utc, format=format,
                             exact=exact, unit=unit,
                             infer_datetime_format=infer_datetime_format,
                             origin=origin, cache=cache)
    return op(arg)
