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

import numpy as np

from ... import opcodes as OperandDef
from ...serialize import KeyField, AnyField, Int8Field, Int64Field
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import parse_index, build_df, build_series, validate_axis


class DataFrameShift(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.SHIFT

    _input = KeyField('input')
    _periods = Int64Field('periods')
    _freq = AnyField('freq')
    _axis = Int8Field('axis')
    _fill_value = AnyField('fill_value')

    def __init__(self, periods=None, freq=None, axis=None, fill_value=None, **kw):
        super().__init__(_periods=periods, _freq=freq, _axis=axis,
                         _fill_value=fill_value, **kw)

    @property
    def input(self):
        return self._input

    @property
    def periods(self):
        return self._periods

    @property
    def freq(self):
        return self._freq

    @property
    def axis(self):
        return self._axis

    @property
    def fill_value(self):
        return self._fill_value

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def _call_dataframe(self, df):
        test_df = build_df(df)
        result_df = test_df.shift(periods=self._periods, freq=self._freq,
                                  axis=self._axis, fill_value=self._fill_value)

        if self._freq is None:
            # shift data
            index_value = df.index_value
            columns_value = df.columns_value
        else:
            # shift index
            if self._axis == 0:
                index_value = parse_index(df.index_value.to_pandas(),
                                          self._periods, self._freq)
                columns_value = df.columns_value
            else:
                columns_value = parse_index(result_df.dtypes.index, store_data=True)
                index_value = df.index_value

        return self.new_dataframe([df], shape=df.shape,
                                  dtypes=result_df.dtypes, index_value=index_value,
                                  columns_value=columns_value)

    def _call_series(self, series):
        test_series = build_series(series)
        result_series = test_series.shift(periods=self._periods, freq=self._freq,
                                          axis=self._axis, fill_value=self._fill_value)

        index_value = series.index_value
        if self._freq is not None:
            # shift index
            index_value = parse_index(index_value.to_pandas(),
                                      self._periods, self._freq)

        return self.new_series([series], shape=series.shape,
                               index_value=index_value,
                               dtype=result_series.dtype,
                               name=series.name)

    def __call__(self, df_or_series):
        if df_or_series.op.object_type == ObjectType.dataframe:
            self._object_type = ObjectType.dataframe
            return self._call_dataframe(df_or_series)
        else:
            assert df_or_series.op.object_type == ObjectType.series
            self._object_type = ObjectType.series
            return self._call_series(df_or_series)

    @classmethod
    def _tile_dataframe(cls, op):
        from ..indexing.iloc import DataFrameIlocGetItem
        from ..merge.concat import DataFrameConcat

        inp = op.input
        out = op.outputs[0]
        axis = op.axis

        out_chunks = []
        if op.freq is not None:
            cum_nsplit = [0] + np.cumsum(inp.nsplits[axis]).tolist()
            # shift index
            for c in inp.chunks:
                chunk_op = op.copy().reset_key()
                i = c.index[axis]
                start, end = cum_nsplit[i], cum_nsplit[i + 1]
                if axis == 0:
                    index_value = parse_index(c.index_value.to_pandas(),
                                              op.periods, op.freq)
                    columns_value = c.columns_value
                    dtypes = out.dtypes.iloc[start: end]
                else:
                    dtypes = out.dtypes.iloc[start: end]
                    columns_value = parse_index(dtypes.index, store_data=True)
                    index_value = c.index_value
                out_chunk = chunk_op.new_chunk([c], index=c.index, shape=c.shape,
                                               index_value=index_value,
                                               columns_value=columns_value,
                                               dtypes=dtypes)
                out_chunks.append(out_chunk)
        else:
            if np.isnan(np.sum(inp.nsplits[axis])):  # pragma: no cover
                raise TilesError('input has unknown chunk shape '
                                 'on axis {}'.format(axis))

            # shift data
            inc = op.periods > 0
            cum_nsplit = [0] + np.cumsum(inp.nsplits[axis]).tolist()
            for j in range(inp.chunk_shape[1 - axis]):
                for i in range(inp.chunk_shape[axis]):
                    index = [None, None]
                    index[axis] = i
                    index[1 - axis] = j
                    index = tuple(index)

                    start, end = cum_nsplit[i], cum_nsplit[i + 1]

                    c = inp.cix[index]
                    to_concats = [c]
                    left = abs(op.periods)
                    prev_i = i - 1 if inc else i + 1
                    while left > 0 and 0 <= prev_i < inp.chunk_shape[axis]:
                        prev_index = [None, None]
                        prev_index[axis] = prev_i
                        prev_index[1 - axis] = j
                        prev_index = tuple(prev_index)

                        prev_chunk = inp.cix[prev_index]
                        size = min(prev_chunk.shape[axis], left)
                        left -= size
                        prev_i = prev_i - 1 if inc else prev_i + 1

                        if size == prev_chunk.shape[axis]:
                            to_concat = prev_chunk
                        else:
                            slcs = [slice(None)] * 2
                            slc = slice(-size, None) if inc else slice(size)
                            slcs[axis] = slc
                            slc_op = DataFrameIlocGetItem(indexes=slcs)
                            to_concat = slc_op.new_chunk([prev_chunk])

                        if inc:
                            to_concats.insert(0, to_concat)
                        else:
                            to_concats.append(to_concat)

                    if len(to_concats) > 1:
                        concat_op = DataFrameConcat(axis=axis,
                                                    object_type=ObjectType.dataframe)
                        to_shift_chunk = concat_op.new_chunk(to_concats)
                    else:
                        to_shift_chunk = to_concats[0]

                    chunk_op = op.copy().reset_key()
                    out_chunk = chunk_op.new_chunk([to_shift_chunk],
                                                   index=index, shape=c.shape,
                                                   index_value=c.index_value,
                                                   columns_value=c.columns_value,
                                                   dtypes=c.dtypes.iloc[start: end])
                    out_chunks.append(out_chunk)

        params = out.params
        params['chunks'] = out_chunks
        params['nsplits'] = inp.nsplits
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def _tile_series(cls, op):
        from ..indexing.iloc import SeriesIlocGetItem
        from ..merge import DataFrameConcat

        check_chunks_unknown_shape(op.inputs, TilesError)

        inp = op.input
        out = op.outputs[0]

        out_chunks = []

        for i, c in enumerate(inp.chunks):
            chunk_op = op.copy().reset_key()

            if op.freq is not None:
                # shift index
                index_value = parse_index(c.index_value.to_pandas(),
                                          op.periods, op.freq)
                out_chunk = chunk_op.new_chunk([c], shape=c.shape,
                                               index_value=index_value,
                                               name=c.name,
                                               dtype=out.dtype,
                                               index=c.index)
            else:
                inc = op.periods > 0
                prev_i = i - 1 if inc else i + 1

                to_concats = [c]
                left = abs(op.periods)
                while left > 0 and 0 <= prev_i < inp.chunk_shape[0]:
                    prev_chunk = inp.cix[prev_i, ]
                    size = min(left, prev_chunk.shape[0])
                    left -= size
                    prev_i = prev_i - 1 if inc else prev_i + 1

                    if size == prev_chunk.shape[0]:
                        to_concat = prev_chunk
                    else:
                        slc = slice(-size, None) if inc else slice(size)
                        slc_op = SeriesIlocGetItem(indexes=[slc])
                        to_concat = slc_op.new_chunk([prev_chunk])

                    if inc:
                        to_concats.insert(0, to_concat)
                    else:
                        to_concats.append(to_concat)

                if len(to_concats) > 1:
                    concat_op = DataFrameConcat(object_type=ObjectType.series)
                    to_concat = concat_op.new_chunk(to_concats)
                else:
                    to_concat = to_concats[0]

                out_chunk = chunk_op.new_chunk([to_concat],
                                               index=(i,), shape=c.shape,
                                               index_value=c.index_value,
                                               dtype=out.dtype, name=out.name)
            out_chunks.append(out_chunk)

        params = out.params
        params['chunks'] = out_chunks
        params['nsplits'] = inp.nsplits
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def tile(cls, op):
        if op.object_type == ObjectType.dataframe:
            return cls._tile_dataframe(op)
        else:
            return cls._tile_series(op)

    @classmethod
    def execute(cls, ctx, op):
        axis = op.axis
        periods = op.periods

        obj = ctx[op.input.key]
        out = op.outputs[0]

        result = obj.shift(periods=periods, freq=op.freq,
                           axis=axis, fill_value=op.fill_value)
        if result.shape != out.shape:
            slc = [slice(None)] * obj.ndim
            if periods > 0:
                slc[axis] = slice(-out.shape[axis], None)
            else:
                slc[axis] = slice(out.shape[axis])

            result = result.iloc[tuple(slc)]
            assert result.shape == out.shape

        ctx[out.key] = result


def shift(df_or_series, periods=1, freq=None, axis=0, fill_value=None):
    """
    Shift index by desired number of periods with an optional time `freq`.

    When `freq` is not passed, shift the index without realigning the data.
    If `freq` is passed (in this case, the index must be date or datetime,
    or it will raise a `NotImplementedError`), the index will be
    increased using the periods and the `freq`.

    Parameters
    ----------
    periods : int
        Number of periods to shift. Can be positive or negative.
    freq : DateOffset, tseries.offsets, timedelta, or str, optional
        Offset to use from the tseries module or time rule (e.g. 'EOM').
        If `freq` is specified then the index values are shifted but the
        data is not realigned. That is, use `freq` if you would like to
        extend the index when shifting and preserve the original data.
    axis : {0 or 'index', 1 or 'columns', None}, default None
        Shift direction.
    fill_value : object, optional
        The scalar value to use for newly introduced missing values.
        the default depends on the dtype of `self`.
        For numeric data, ``np.nan`` is used.
        For datetime, timedelta, or period data, etc. :attr:`NaT` is used.
        For extension dtypes, ``self.dtype.na_value`` is used.

    Returns
    -------
    DataFrame or Series
        Copy of input object, shifted.

    See Also
    --------
    Index.shift : Shift values of Index.
    DatetimeIndex.shift : Shift values of DatetimeIndex.
    PeriodIndex.shift : Shift values of PeriodIndex.
    tshift : Shift the time index, using the index's frequency if
        available.

    Examples
    --------
    >>> import mars.dataframe as md

    >>> df = md.DataFrame({'Col1': [10, 20, 15, 30, 45],
    ...                    'Col2': [13, 23, 18, 33, 48],
    ...                    'Col3': [17, 27, 22, 37, 52]})

    >>> df.shift(periods=3).execute()
       Col1  Col2  Col3
    0   NaN   NaN   NaN
    1   NaN   NaN   NaN
    2   NaN   NaN   NaN
    3  10.0  13.0  17.0
    4  20.0  23.0  27.0

    >>> df.shift(periods=1, axis='columns').execute()
       Col1  Col2  Col3
    0   NaN  10.0  13.0
    1   NaN  20.0  23.0
    2   NaN  15.0  18.0
    3   NaN  30.0  33.0
    4   NaN  45.0  48.0

    >>> df.shift(periods=3, fill_value=0).execute()
       Col1  Col2  Col3
    0     0     0     0
    1     0     0     0
    2     0     0     0
    3    10    13    17
    4    20    23    27
    """
    axis = validate_axis(axis, df_or_series)
    if periods == 0:
        return df_or_series.copy()

    op = DataFrameShift(periods=periods, freq=freq,
                        axis=axis, fill_value=fill_value)
    return op(df_or_series)


def tshift(df_or_series, periods: int = 1, freq=None, axis=0):
    """
    Shift the time index, using the index's frequency if available.

    Parameters
    ----------
    periods : int
        Number of periods to move, can be positive or negative.
    freq : DateOffset, timedelta, or str, default None
        Increment to use from the tseries module
        or time rule expressed as a string (e.g. 'EOM').
    axis : {0 or ‘index’, 1 or ‘columns’, None}, default 0
        Corresponds to the axis that contains the Index.

    Returns
    -------
    shifted : Series/DataFrame

    Notes
    -----
    If freq is not specified then tries to use the freq or inferred_freq
    attributes of the index. If neither of those attributes exist, a
    ValueError is thrown
    """
    axis = validate_axis(axis, df_or_series)
    index = df_or_series.index_value.to_pandas() if axis == 0 else \
        df_or_series.columns_value.to_pandas()

    if freq is None:
        freq = getattr(index, "freq", None)

    if freq is None:  # pragma: no cover
        freq = getattr(index, "inferred_freq", None)

    if freq is None:
        raise ValueError('Freq was not given and was not set in the index')

    return shift(df_or_series, periods=periods, freq=freq, axis=axis)
