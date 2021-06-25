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

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import ENTITY_TYPE, recursive_tile
from ...serialization.serializables import BoolField, AnyField, KeyField, Int32Field
from ...tensor.utils import filter_inputs
from ..core import SERIES_TYPE, DATAFRAME_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_empty_df, validate_axis, parse_index


class DataFrameCorr(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.CORR

    _other = KeyField('other')
    _method = AnyField('method')
    _min_periods = Int32Field('min_periods')
    _axis = Int32Field('axis')
    _drop = BoolField('drop')

    def __init__(self, other=None, method=None, min_periods=None, axis=None,
                 drop=None, **kw):
        super().__init__(_other=other, _method=method, _min_periods=min_periods,
                         _axis=axis, _drop=drop, **kw)

    @property
    def other(self):
        return self._other

    @property
    def method(self):
        return self._method

    @property
    def min_periods(self):
        return self._min_periods

    @property
    def axis(self):
        return self._axis

    @property
    def drop(self):
        return self._drop

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        next(inputs_iter)
        if isinstance(self._other, ENTITY_TYPE):
            self._other = next(inputs_iter)

    def __call__(self, df_or_series):
        if isinstance(df_or_series, SERIES_TYPE):
            inputs = filter_inputs([df_or_series, self.other])
            return self.new_scalar(inputs, dtype=np.dtype(np.float_))
        else:
            def _filter_numeric(obj):
                if not isinstance(obj, DATAFRAME_TYPE):
                    return obj
                num_dtypes = build_empty_df(obj.dtypes)._get_numeric_data().dtypes
                if len(num_dtypes) != len(obj.dtypes):
                    return obj[list(num_dtypes.index)]
                return obj

            df_or_series = _filter_numeric(df_or_series)
            self._other = _filter_numeric(self._other)

            inputs = filter_inputs([df_or_series, self.other])
            if self.axis is None:
                dtypes = pd.Series([np.dtype(np.float_)] * len(df_or_series.dtypes),
                                   index=df_or_series.dtypes.index)
                return self.new_dataframe(inputs, shape=(df_or_series.shape[1],) * 2, dtypes=dtypes,
                                          index_value=df_or_series.columns_value,
                                          columns_value=df_or_series.columns_value)
            else:
                new_index_value = df_or_series.axes[1 - self.axis].index_value
                if isinstance(self.other, DATAFRAME_TYPE):
                    align_dtypes = pd.concat([self.other.dtypes, df_or_series.dtypes], axis=1)
                    align_shape = (np.nan, align_dtypes.shape[0])
                    new_index_value = parse_index(align_dtypes.index)
                else:
                    align_shape = df_or_series.shape

                shape = (np.nan,) if self.drop else (align_shape[1 - self.axis],)
                return self.new_series(inputs, shape=shape, dtype=np.dtype(np.float_),
                                       index_value=new_index_value)

    @classmethod
    def _tile_single(cls, op: "DataFrameCorr"):
        out = op.outputs[0]

        new_op = op.copy().reset_key()
        chunk = new_op.new_chunk([inp.chunks[0] for inp in op.inputs],
                                 index=(0,) * len(out.shape), **out.params)

        new_op = op.copy().reset_key()
        return new_op.new_tileables(op.inputs, chunks=[chunk], nsplits=((s,) for s in out.shape),
                                    **out.params)

    @staticmethod
    def _tile_pearson_cross(left, right, min_periods):
        left_tensor, right_tensor = left.fillna(0).to_tensor(), right.fillna(0).to_tensor()

        nna_left = left.notna().to_tensor().astype(np.float_)
        nna_right = right.notna().to_tensor().astype(np.float_)

        sum_left = left_tensor.T.dot(nna_right)
        sum_right = right_tensor.T.dot(nna_left)
        sum_left2 = (left_tensor.T ** 2).dot(nna_right)
        sum_right2 = (right_tensor.T ** 2).dot(nna_left)
        sum_mul = left_tensor.T.dot(right_tensor)
        data_count = nna_left.T.dot(nna_right)

        divisor = np.sqrt(data_count * sum_left2 - sum_left * sum_left).T \
            * np.sqrt(data_count * sum_right2 - sum_right * sum_right)

        result = (data_count * sum_mul - sum_left * sum_right.T) / divisor
        if min_periods is not None:
            result = np.where(data_count >= min_periods, result, np.nan)
        return result

    @classmethod
    def _tile_pearson_align(cls, left, right, axis):
        nna_left = left.notna().astype(np.float_)
        nna_right = right.notna().astype(np.float_)

        left, right = left.fillna(0), right.fillna(0)

        sum_left = left.mul(nna_right, axis=axis).sum(axis=axis)
        sum_right = nna_left.mul(right, axis=axis).sum(axis=axis)
        sum_left2 = (left ** 2).mul(nna_right, axis=axis).sum(axis=axis)
        sum_right2 = nna_left.mul(right ** 2, axis=axis).sum(axis=axis)
        sum_mul = left.mul(right, axis=axis).sum(axis=axis)
        data_count = nna_left.mul(nna_right, axis=axis).sum(axis=axis)

        divisor = np.sqrt(data_count * sum_left2 - sum_left * sum_left) \
            * np.sqrt(data_count * sum_right2 - sum_right * sum_right)
        return (data_count * sum_mul - sum_left * sum_right) / divisor

    @classmethod
    def _tile_series(cls, op: "DataFrameCorr"):
        left = op.inputs[0]
        right = op.other

        _check_supported_methods(op.method)
        return [(yield from recursive_tile(
            cls._tile_pearson_cross(left, right, min_periods=op.min_periods)))]

    @classmethod
    def _tile_dataframe_cross(cls, op: "DataFrameCorr"):
        from ..initializer import DataFrame as MarsDataFrame

        left = op.inputs[0]
        right = op.other if op.other is not None else op.inputs[0]

        _check_supported_methods(op.method)

        result = cls._tile_pearson_cross(left, right, min_periods=op.min_periods)
        result = MarsDataFrame(result, index=left.dtypes.index, columns=right.dtypes.index)
        return [(yield from recursive_tile(result))]

    @classmethod
    def _tile_dataframe_align(cls, op: "DataFrameCorr"):
        left = op.inputs[0]
        right = op.other

        _check_supported_methods(op.method)
        result = cls._tile_pearson_align(left, right, axis=op.axis)
        if op.drop:
            result = result.dropna(axis=op.axis)
        return [(yield from recursive_tile(result))]

    @classmethod
    def tile(cls, op: "DataFrameCorr"):
        inp = op.inputs[0]
        if len(inp.chunks) == 1 and (op.other is None or len(op.other.chunks) == 1):
            return cls._tile_single(op)
        elif isinstance(inp, SERIES_TYPE):
            return (yield from cls._tile_series(op))
        elif op.axis is None:
            return (yield from cls._tile_dataframe_cross(op))
        else:
            return (yield from cls._tile_dataframe_align(op))

    @classmethod
    def execute(cls, ctx, op: "DataFrameCorr"):
        inp = op.inputs[0]
        out = op.outputs[0]
        inp_data = ctx[inp.key]

        if inp.ndim == 1:
            ctx[out.key] = inp_data.corr(ctx[op.other.key], method=op.method,
                                         min_periods=op.min_periods)
        elif op.axis is None:
            ctx[out.key] = inp_data.corr(method=op.method, min_periods=op.min_periods)
        else:
            ctx[out.key] = inp_data.corrwith(ctx[op.other.key], method=op.method,
                                             axis=op.axis, drop=op.drop)


def _check_supported_methods(method):
    if method != 'pearson':
        raise NotImplementedError(f'Correlation method {method!r} not supported')


def df_corr(df, method='pearson', min_periods=1):
    """
    Compute pairwise correlation of columns, excluding NA/null values.

    Parameters
    ----------
    method : {'pearson', 'kendall', 'spearman'} or callable
        Method of correlation:

        * pearson : standard correlation coefficient
        * kendall : Kendall Tau correlation coefficient
        * spearman : Spearman rank correlation
        * callable: callable with input two 1d ndarrays
            and returning a float. Note that the returned matrix from corr
            will have 1 along the diagonals and will be symmetric
            regardless of the callable's behavior.

        .. note::
            kendall, spearman and callables not supported on multiple chunks yet.

    min_periods : int, optional
        Minimum number of observations required per pair of columns
        to have a valid result. Currently only available for Pearson
        and Spearman correlation.

    Returns
    -------
    DataFrame
        Correlation matrix.

    See Also
    --------
    DataFrame.corrwith : Compute pairwise correlation with another
        DataFrame or Series.
    Series.corr : Compute the correlation between two Series.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
    ...                   columns=['dogs', 'cats'])
    >>> df.corr(method='pearson').execute()
              dogs      cats
    dogs  1.000000 -0.851064
    cats -0.851064  1.000000
    """
    op = DataFrameCorr(method=method, min_periods=min_periods)
    return op(df)


def df_corrwith(df, other, axis=0, drop=False, method='pearson'):
    """
    Compute pairwise correlation.

    Pairwise correlation is computed between rows or columns of
    DataFrame with rows or columns of Series or DataFrame. DataFrames
    are first aligned along both axes before computing the
    correlations.

    Parameters
    ----------
    other : DataFrame, Series
        Object with which to compute correlations.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        The axis to use. 0 or 'index' to compute column-wise, 1 or 'columns' for
        row-wise.
    drop : bool, default False
        Drop missing indices from result.
    method : {'pearson', 'kendall', 'spearman'} or callable
        Method of correlation:

        * pearson : standard correlation coefficient
        * kendall : Kendall Tau correlation coefficient
        * spearman : Spearman rank correlation
        * callable: callable with input two 1d ndarrays
            and returning a float.

        .. note::
            kendall, spearman and callables not supported on multiple chunks yet.

    Returns
    -------
    Series
        Pairwise correlations.

    See Also
    --------
    DataFrame.corr : Compute pairwise correlation of columns.
    """
    axis = validate_axis(axis, df)
    if drop:
        # TODO implement with df.align(method='inner')
        raise NotImplementedError('drop=True not implemented')
    op = DataFrameCorr(other=other, method=method, axis=axis, drop=drop)
    return op(df)


def series_corr(series, other, method='pearson', min_periods=None):
    """
    Compute correlation with `other` Series, excluding missing values.

    Parameters
    ----------
    other : Series
        Series with which to compute the correlation.
    method : {'pearson', 'kendall', 'spearman'} or callable
        Method used to compute correlation:

        - pearson : Standard correlation coefficient
        - kendall : Kendall Tau correlation coefficient
        - spearman : Spearman rank correlation
        - callable: Callable with input two 1d ndarrays and returning a float.

        .. note::
            kendall, spearman and callables not supported on multiple chunks yet.

    min_periods : int, optional
        Minimum number of observations needed to have a valid result.

    Returns
    -------
    float
        Correlation with other.

    See Also
    --------
    DataFrame.corr : Compute pairwise correlation between columns.
    DataFrame.corrwith : Compute pairwise correlation with another
        DataFrame or Series.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> s1 = md.Series([.2, .0, .6, .2])
    >>> s2 = md.Series([.3, .6, .0, .1])
    >>> s1.corr(s2, method='pearson').execute()
    -0.8510644963469898
    """
    op = DataFrameCorr(other=other, method=method, min_periods=min_periods)
    return op(series)


def series_autocorr(series, lag=1):
    """
    Compute the lag-N autocorrelation.

    This method computes the Pearson correlation between
    the Series and its shifted self.

    Parameters
    ----------
    lag : int, default 1
        Number of lags to apply before performing autocorrelation.

    Returns
    -------
    float
        The Pearson correlation between self and self.shift(lag).

    See Also
    --------
    Series.corr : Compute the correlation between two Series.
    Series.shift : Shift index by desired number of periods.
    DataFrame.corr : Compute pairwise correlation of columns.
    DataFrame.corrwith : Compute pairwise correlation between rows or
        columns of two DataFrame objects.

    Notes
    -----
    If the Pearson correlation is not well defined return 'NaN'.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> s = md.Series([0.25, 0.5, 0.2, -0.05])
    >>> s.autocorr().execute()  # doctest: +ELLIPSIS.execute()
    0.10355...
    >>> s.autocorr(lag=2).execute()  # doctest: +ELLIPSIS.execute()
    -0.99999...

    If the Pearson correlation is not well defined, then 'NaN' is returned.

    >>> s = md.Series([1, 0, 0, 0])
    >>> s.autocorr().execute()
    nan
    """
    op = DataFrameCorr(other=series.shift(lag), method='pearson')
    return op(series)
