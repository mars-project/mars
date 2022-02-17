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

import warnings
from datetime import datetime, date, time

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick
from pandas._libs.tslibs import timezones

from ... import opcodes as OperandDef
from ...config import options
from ...core import OutputType
from ...serialization.serializables import AnyField, Int64Field, BoolField, StringField
from ...tensor.utils import decide_chunk_sizes
from ...utils import pd_release_version, NoDefault
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index

try:
    from pandas._libs.tslib import normalize_date
except ImportError:  # pragma: no cover

    def normalize_date(dt):  # from pandas/_libs/tslibs/conversion.pyx
        if isinstance(dt, datetime):
            if isinstance(dt, pd.Timestamp):
                return dt.replace(
                    hour=0, minute=0, second=0, microsecond=0, nanosecond=0
                )
            else:
                return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif isinstance(dt, date):
            return datetime(dt.year, dt.month, dt.day)
        else:
            raise TypeError(f"Unrecognized type: {type(dt)}")


_date_range_use_inclusive = pd_release_version[:2] >= (1, 4)


class DataFrameDateRange(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATE_RANGE

    start = AnyField("start")
    end = AnyField("end")
    periods = Int64Field("periods")
    freq = AnyField("freq")
    tz = AnyField("tz")
    normalize = BoolField("normalize")
    name = StringField("name")
    inclusive = StringField("closed")

    def __init__(
        self,
        output_types=None,
        **kw,
    ):
        super().__init__(_output_types=output_types, **kw)
        if self.output_types is None:
            self.output_types = [OutputType.index]
        if getattr(self, "inclusive", None) is None:
            self.inclusive = "both"

    def __call__(self, shape, chunk_size=None):
        dtype = pd.Index([self.start]).dtype
        index_value = parse_index(
            pd.Index([], dtype=dtype), self.start, self.end, self.periods, self.tz
        )
        # gen index value info
        index_value.value._min_val = self.start
        index_value.value._min_val_close = True
        index_value.value._max_val = self.end
        index_value.value._max_val_close = True
        index_value.value._is_unique = True
        index_value.value._is_monotonic_increasing = True
        index_value.value._freq = self.freq
        return self.new_index(
            None,
            shape=shape,
            dtype=dtype,
            index_value=index_value,
            name=self.name,
            raw_chunk_size=chunk_size,
            freq=self.freq,
        )

    @classmethod
    def tile(cls, op: "DataFrameDateRange"):
        out = op.outputs[0]
        start = op.start
        end = op.end
        freq = op.freq
        periods = op.periods
        inclusive = op.inclusive

        chunk_length = out.extra_params.raw_chunk_size or options.chunk_size
        chunk_length = decide_chunk_sizes(out.shape, chunk_length, out.dtype.itemsize)[
            0
        ]

        if inclusive in ("neither", "right"):
            # if left not close, add one more for the first chunk
            chunk_length = (chunk_length[0] + 1,) + chunk_length[1:]

        if freq is None:
            if periods > 1:
                freq = (end - op.start) / (periods - 1)
            else:
                freq = end - start

        out_chunks = []
        cum_nsplit = [0] + np.cumsum(chunk_length).tolist()
        for i, chunk_size in enumerate(chunk_length):
            chunk_op = op.copy().reset_key()
            chunk_op.periods = chunk_size

            if i > 0 or inclusive not in ("neither", "right"):
                # for chunks in the middle, all sides are inclusive
                chunk_op.inclusive = "both"
            elif 0 == i and inclusive == "neither":
                chunk_op.inclusive = "right"

            chunk_i_start = cum_nsplit[i]
            if chunk_i_start > 0:
                chunk_start = chunk_op.start = start + freq * chunk_i_start
            else:
                chunk_start = chunk_op.start = start
            chunk_end = chunk_op.end = chunk_start + (chunk_size - 1) * freq

            # gen chunk index_value
            chunk_index_value = parse_index(out.index_value.to_pandas(), i, out)
            chunk_index_value.value._min_val = chunk_start
            chunk_index_value.value._min_val_close = True
            chunk_index_value.value._max_val = chunk_end
            chunk_index_value.value._max_val_close = True
            chunk_index_value.value._is_unique = True
            chunk_index_value.value._is_monotonic_increasing = True

            size = (
                chunk_size - 1
                if i == 0 and inclusive in ("neither", "right")
                else chunk_size
            )
            out_chunk = chunk_op.new_chunk(
                None,
                shape=(size,),
                index=(i,),
                dtype=out.dtype,
                index_value=chunk_index_value,
                name=out.name,
            )
            out_chunks.append(out_chunk)

        new_op = op.copy()
        params = out.params
        params["chunks"] = out_chunks
        params["nsplits"] = (tuple(c.shape[0] for c in out_chunks),)
        return new_op.new_indexes(None, kws=[params])

    @classmethod
    def execute(cls, ctx, op: "DataFrameDateRange"):
        start, end, periods = op.start, op.end, op.periods
        freq = op.freq
        if freq is not None:
            end = None
        kw = dict(
            start=start,
            end=end,
            periods=periods,
            freq=freq,
            tz=op.tz,
            normalize=op.normalize,
            name=op.name,
            inclusive=op.inclusive,
        )
        if not _date_range_use_inclusive:
            closed = kw.pop("inclusive")
            assert closed != "neither"
            kw["closed"] = None if closed == "both" else closed
        ctx[op.outputs[0].key] = pd.date_range(**kw)


_midnight = time(0, 0)


def _maybe_normalize_endpoints(start, end, normalize):  # pragma: no cover
    _normalized = True

    if start is not None:
        if normalize:
            start = normalize_date(start)
            _normalized = True
        else:
            _normalized = _normalized and start.time() == _midnight

    if end is not None:
        if normalize:
            end = normalize_date(end)
            _normalized = True
        else:
            _normalized = _normalized and end.time() == _midnight

    return start, end, _normalized


def _infer_tz_from_endpoints(start, end, tz):  # pragma: no cover
    """
    If a timezone is not explicitly given via `tz`, see if one can
    be inferred from the `start` and `end` endpoints.  If more than one
    of these inputs provides a timezone, require that they all agree.

    Parameters
    ----------
    start : Timestamp
    end : Timestamp
    tz : tzinfo or None

    Returns
    -------
    tz : tzinfo or None

    Raises
    ------
    TypeError : if start and end timezones do not agree
    """
    try:
        inferred_tz = timezones.infer_tzinfo(start, end)
    except AssertionError:
        # infer_tzinfo raises AssertionError if passed mismatched timezones
        raise TypeError(
            "Start and end cannot both be tz-aware with different timezones"
        )

    inferred_tz = timezones.maybe_get_tz(inferred_tz)
    tz = timezones.maybe_get_tz(tz)

    if tz is not None and inferred_tz is not None:
        if not timezones.tz_compare(inferred_tz, tz):
            raise AssertionError("Inferred time zone not equal to passed time zone")

    elif inferred_tz is not None:
        tz = inferred_tz

    return tz


def _maybe_localize_point(
    ts, is_none, is_not_none, freq, tz, ambiguous, nonexistent
):  # pragma: no cover
    """
    Localize a start or end Timestamp to the timezone of the corresponding
    start or end Timestamp

    Parameters
    ----------
    ts : start or end Timestamp to potentially localize
    is_none : argument that should be None
    is_not_none : argument that should not be None
    freq : Tick, DateOffset, or None
    tz : str, timezone object or None
    ambiguous: str, localization behavior for ambiguous times
    nonexistent: str, localization behavior for nonexistent times

    Returns
    -------
    ts : Timestamp
    """
    # Make sure start and end are timezone localized if:
    # 1) freq = a Timedelta-like frequency (Tick)
    # 2) freq = None i.e. generating a linspaced range
    if is_none is None and is_not_none is not None:
        # Note: We can't ambiguous='infer' a singular ambiguous time; however,
        # we have historically defaulted ambiguous=False
        ambiguous = ambiguous if ambiguous != "infer" else False
        localize_args = {"ambiguous": ambiguous, "nonexistent": nonexistent, "tz": None}
        if isinstance(freq, Tick) or freq is None:
            localize_args["tz"] = tz
        ts = ts.tz_localize(**localize_args)
    return ts


def date_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    tz=None,
    normalize=False,
    name=None,
    closed=NoDefault.no_default,
    inclusive=None,
    chunk_size=None,
    **kwargs,
):
    """
    Return a fixed frequency DatetimeIndex.

    Parameters
    ----------
    start : str or datetime-like, optional
        Left bound for generating dates.
    end : str or datetime-like, optional
        Right bound for generating dates.
    periods : int, optional
        Number of periods to generate.
    freq : str or DateOffset, default 'D'
        Frequency strings can have multiples, e.g. '5H'. See
        :ref:`here <timeseries.offset_aliases>` for a list of
        frequency aliases.
    tz : str or tzinfo, optional
        Time zone name for returning localized DatetimeIndex, for example
        'Asia/Hong_Kong'. By default, the resulting DatetimeIndex is
        timezone-naive.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    name : str, default None
        Name of the resulting DatetimeIndex.
    inclusive : {“both”, “neither”, “left”, “right”}, default “both”
        Include boundaries; Whether to set each bound as closed or open.
    **kwargs
        For compatibility. Has no effect on the result.

    Returns
    -------
    rng : DatetimeIndex

    See Also
    --------
    DatetimeIndex : An immutable container for datetimes.
    timedelta_range : Return a fixed frequency TimedeltaIndex.
    period_range : Return a fixed frequency PeriodIndex.
    interval_range : Return a fixed frequency IntervalIndex.

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``DatetimeIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end`` (closed on both sides).

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    **Specifying the values**

    The next four examples generate the same `DatetimeIndex`, but vary
    the combination of `start`, `end` and `periods`.

    Specify `start` and `end`, with the default daily frequency.
    >>> import mars.dataframe as md

    >>> md.date_range(start='1/1/2018', end='1/08/2018').execute()
    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
                   '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
                  dtype='datetime64[ns]', freq='D')

    Specify `start` and `periods`, the number of periods (days).

    >>> md.date_range(start='1/1/2018', periods=8).execute()
    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
                   '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
                  dtype='datetime64[ns]', freq='D')

    Specify `end` and `periods`, the number of periods (days).

    >>> md.date_range(end='1/1/2018', periods=8).execute()
    DatetimeIndex(['2017-12-25', '2017-12-26', '2017-12-27', '2017-12-28',
                   '2017-12-29', '2017-12-30', '2017-12-31', '2018-01-01'],
                  dtype='datetime64[ns]', freq='D')

    Specify `start`, `end`, and `periods`; the frequency is generated
    automatically (linearly spaced).

    >>> md.date_range(start='2018-04-24', end='2018-04-27', periods=3).execute()
    DatetimeIndex(['2018-04-24 00:00:00', '2018-04-25 12:00:00',
                   '2018-04-27 00:00:00'],
                  dtype='datetime64[ns]', freq=None)

    **Other Parameters**

    Changed the `freq` (frequency) to ``'M'`` (month end frequency).

    >>> md.date_range(start='1/1/2018', periods=5, freq='M').execute()
    DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31', '2018-04-30',
                   '2018-05-31'],
                  dtype='datetime64[ns]', freq='M')

    Multiples are allowed

    >>> md.date_range(start='1/1/2018', periods=5, freq='3M').execute()
    DatetimeIndex(['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31',
                   '2019-01-31'],
                  dtype='datetime64[ns]', freq='3M')

    `freq` can also be specified as an Offset object.

    >>> md.date_range(start='1/1/2018', periods=5, freq=md.offsets.MonthEnd(3)).execute()
    DatetimeIndex(['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31',
                   '2019-01-31'],
                  dtype='datetime64[ns]', freq='3M')

    Specify `tz` to set the timezone.

    >>> md.date_range(start='1/1/2018', periods=5, tz='Asia/Tokyo').execute()
    DatetimeIndex(['2018-01-01 00:00:00+09:00', '2018-01-02 00:00:00+09:00',
                   '2018-01-03 00:00:00+09:00', '2018-01-04 00:00:00+09:00',
                   '2018-01-05 00:00:00+09:00'],
                  dtype='datetime64[ns, Asia/Tokyo]', freq='D')

    `inclusive` controls whether to include `start` and `end` that are on the
    boundary. The default, "both", includes boundary points on either end.

    >>> md.date_range(start='2017-01-01', end='2017-01-04', inclusive='both').execute()
    DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04'],
                  dtype='datetime64[ns]', freq='D')

    Use ``inclusive='left'`` to exclude `end` if it falls on the boundary.

    >>> md.date_range(start='2017-01-01', end='2017-01-04', closed='left').execute()
    DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03'],
                  dtype='datetime64[ns]', freq='D')

    Use ``inclusive='right'`` to exclude `start` if it falls on the boundary,
    and similarly inclusive='neither' will exclude both `start` and `end`.

    >>> md.date_range(start='2017-01-01', end='2017-01-04', closed='right').execute()
    DatetimeIndex(['2017-01-02', '2017-01-03', '2017-01-04'],
                  dtype='datetime64[ns]', freq='D')

    .. note::
        Pandas 1.4.0 or later is required to use ``inclusive='neither'``.
        Otherwise an error may be raised.
    """
    # validate periods
    if isinstance(periods, (float, np.floating)):
        periods = int(periods)
    if periods is not None and not isinstance(periods, (int, np.integer)):
        raise TypeError(f"periods must be a number, got {periods}")

    if freq is None and any(arg is None for arg in [periods, start, end]):
        freq = "D"
    if sum(arg is not None for arg in [start, end, periods, freq]) != 3:
        raise ValueError(
            "Of the four parameters: start, end, periods, "
            "and freq, exactly three must be specified"
        )
    freq = to_offset(freq)

    if _date_range_use_inclusive and closed is not NoDefault.no_default:
        warnings.warn(
            "Argument `closed` is deprecated in favor of `inclusive`.", FutureWarning
        )
    elif closed is NoDefault.no_default:
        closed = None

    if inclusive is None and closed is not NoDefault.no_default:
        inclusive = closed

    if start is not None:
        start = pd.Timestamp(start)

    if end is not None:
        end = pd.Timestamp(end)

    if start is pd.NaT or end is pd.NaT:
        raise ValueError("Neither `start` nor `end` can be NaT")

    start, end, _ = _maybe_normalize_endpoints(start, end, normalize)
    tz = _infer_tz_from_endpoints(start, end, tz)

    if start is None and end is not None:
        # start is None and end is not None
        # adjust end first
        end = pd.date_range(end=end, periods=1, freq=freq)[0]
        if inclusive == "neither":
            end -= freq
        size = periods
        start = end - (periods - 1) * freq
        if inclusive in ("neither", "left"):
            size -= 1
        elif inclusive == "right":
            # when start is None, closed == 'left' would not take effect
            # thus just ignore
            inclusive = "both"
    elif end is None:
        # end is None
        # adjust start first
        start = pd.date_range(start=start, periods=1, freq=freq)[0]
        size = periods
        end = start + (periods - 1) * freq
        if inclusive in ("neither", "right"):
            size -= 1
        elif inclusive == "left":
            # when end is None, closed == 'left' would not take effect
            # thus just ignore
            inclusive = "both"
    else:
        if periods is None:
            periods = size = int((end - start) / freq + 1)
        else:
            size = periods
        if inclusive in ("left", "right"):
            size -= 1
        elif inclusive == "neither":
            size -= 2

    shape = (size,)
    op = DataFrameDateRange(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        tz=tz,
        normalize=normalize,
        name=name,
        inclusive=inclusive,
        **kwargs,
    )
    return op(shape, chunk_size=chunk_size)
