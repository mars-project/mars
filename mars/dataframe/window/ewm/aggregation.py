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

from collections import namedtuple

import numpy as np
import pandas as pd

from .... import opcodes
from ....serialization.serializables import BoolField, Float64Field
from ..aggregation import BaseDataFrameExpandingAgg

_stage_info = namedtuple('_stage_info', ('map_groups', 'map_sources', 'combine_sources',
                                         'combine_columns', 'combine_funcs', 'key_to_funcs',
                                         'valid_columns', 'min_periods_func_name'))

_cum_alpha_coeff_func = '_cum_alpha_coeff'
_cum_square_alpha_coeff_func = '_cum_square_alpha_coeff'


def _add_pred_results(pred_results, local_results, axis=0, alpha=None, order=1,
                      alpha_ignore_na=False, pred_exponent=None, alpha_data=None):
    if pred_results[0].ndim == 1:
        df_filler = 0
    else:
        df_filler = pred_results[0].iloc[-1, :].dropna()
        df_filler[:] = 0

    new_locals = []
    combine_axis = pred_results[0].ndim - axis - 1
    weight = (1 - alpha) ** order
    pred_coeff = weight ** pred_exponent
    for idx, (pred_result, local_result) in enumerate(zip(pred_results, local_results)):
        local_result.fillna(df_filler, inplace=True)
        pred_result = pred_result.mul(pred_coeff).sum(axis=axis)

        if alpha_ignore_na:
            pred_df = pred_result * weight ** alpha_data.notna().cumsum()
        else:
            weights = np.arange(1, len(local_result) + 1)
            if local_result.ndim == 2:
                weights_df = pd.DataFrame(
                    np.repeat(weights.reshape((len(local_result), 1)), len(local_result.columns), axis=1),
                    columns=local_result.columns, index=local_result.index)
            else:
                weights_df = pd.Series(weights, index=local_result.index)
            weights_df[alpha_data.isna()] = np.nan
            weights_df.ffill(inplace=True)
            weights_df.fillna(0, inplace=True)

            weights_df = weight ** weights_df
            pred_df = weights_df.mul(pred_result, axis=combine_axis)

        new_locals.append(local_result.add(pred_df, axis=combine_axis))
    return new_locals


def _combine_mean(pred_results, local_results, axis=0, alpha=None, alpha_ignore_na=False,
                  pred_exponent=None):
    if pred_results is None:
        return (local_results[0] / local_results[1]).ffill()

    alpha_data = local_results[1]
    local_results[0] = local_results[0].ffill()
    local_results[1] = alpha_data.ffill()

    local_sum_data, local_count_data = local_results

    if pred_results is not None:
        local_sum_data, local_count_data = _add_pred_results(
            pred_results, local_results, axis=axis, alpha=alpha, alpha_ignore_na=alpha_ignore_na,
            pred_exponent=pred_exponent, alpha_data=alpha_data
        )
    return local_sum_data / local_count_data


def _combine_var(pred_results, local_results, axis=0, alpha=None, alpha_ignore_na=False,
                 pred_exponent=None):
    local_results[0] = local_results[0].ffill()
    alpha_data = local_results[1]
    local_results[1] = alpha_data.ffill()

    local_results[2] = local_results[2].ffill()
    alpha2_data = local_results[3]
    local_results[3] = alpha2_data.ffill()

    local_sum_data, local_count_data, local_sum_square, local_count2_data = local_results
    if pred_results is None:
        return (local_sum_square - local_sum_data ** 2 / local_count_data) \
               / (local_count_data - local_count2_data / local_count_data)

    pred_sum_data, pred_count_data, pred_sum_square, pred_count2_data = pred_results

    local_count2_data, = _add_pred_results(
        [pred_count2_data], [local_count2_data], axis=axis, alpha=alpha, order=2,
        alpha_ignore_na=alpha_ignore_na, pred_exponent=pred_exponent, alpha_data=alpha_data)

    local_sum_square, local_sum_data, local_count_data = \
        _add_pred_results(
            [pred_sum_square, pred_sum_data, pred_count_data],
            [local_sum_square, local_sum_data, local_count_data],
            axis=axis, alpha=alpha, alpha_ignore_na=alpha_ignore_na,
            pred_exponent=pred_exponent, alpha_data=alpha_data
        )

    return (local_sum_square - local_sum_data ** 2 / local_count_data) \
        / (local_count_data - local_count2_data / local_count_data)


def _combine_std(pred_results, local_results, axis=0, alpha=None, alpha_ignore_na=False,
                 pred_exponent=None):
    return np.sqrt(_combine_var(
        pred_results, local_results, axis=axis, alpha=alpha, alpha_ignore_na=alpha_ignore_na,
        pred_exponent=pred_exponent))


def _combine_data_count(pred_results, local_results, axis=0, **__):
    if pred_results is None:
        return local_results[0]
    return local_results[0].add(pred_results[0].sum(), axis=pred_results[0].ndim - axis - 1)


class DataFrameEwmAgg(BaseDataFrameExpandingAgg):
    _op_type_ = opcodes.EWM_AGG

    _alpha = Float64Field('alpha')
    _adjust = BoolField('adjust')
    _alpha_ignore_na = BoolField('alpha_ignore_na')

    _validate_columns = BoolField('_validate_columns')

    _exec_cache = dict()

    def __init__(self, alpha=None, adjust=None, alpha_ignore_na=None, validate_columns=None, **kw):
        super().__init__(_alpha=alpha, _adjust=adjust, _alpha_ignore_na=alpha_ignore_na,
                         _validate_columns=validate_columns, **kw)

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def adjust(self) -> bool:
        return self._adjust

    @property
    def alpha_ignore_na(self) -> bool:
        return self._alpha_ignore_na

    @property
    def validate_columns(self) -> bool:
        return self._validate_columns

    @classmethod
    def _get_stage_functions(cls, op: "DataFrameEwmAgg", func):
        if func == '_data_count':
            return ['_data_count'], _combine_data_count
        elif func == 'mean':
            return ['cumsum', _cum_alpha_coeff_func], _combine_mean
        elif func in {'var', 'std'}:
            return ['cumsum', _cum_alpha_coeff_func, 'cumsum2', _cum_square_alpha_coeff_func], \
                   _combine_var if func == 'var' else _combine_std
        else:  # pragma: no cover
            raise NotImplementedError

    @classmethod
    def _calc_data_alphas(cls, op: "DataFrameEwmAgg", in_data, order):
        exec_cache = cls._exec_cache[op.key]
        cache_key = ('_calc_data_alphas', order, id(in_data))
        try:
            return exec_cache[cache_key]
        except KeyError:
            pass

        cum_df = in_data.copy()
        cum_df[cum_df.notna()] = 1
        if not op.alpha_ignore_na:
            cum_df.ffill(inplace=True)
        cum_df = cum_df.cumsum(axis=op.axis) - 1
        if not op.alpha_ignore_na:
            cum_df[in_data.isna()] = np.nan

        result = exec_cache[cache_key] = (1 - op.alpha) ** (order * cum_df)
        return result

    @classmethod
    def _execute_cum_alpha_coeff(cls, op: "DataFrameEwmAgg", in_data, order, final=True):
        exec_cache = cls._exec_cache[op.key]
        cache_key = ('cum_alpha_coeff', order, id(in_data))
        summary = None

        try:
            result = exec_cache[cache_key]
        except KeyError:
            alphas = cls._calc_data_alphas(op, in_data, order)
            result = alphas.cumsum()
            exec_cache[cache_key] = result

        if final:
            if op.output_agg:
                summary = result.ffill()[-1:]
        return result, summary

    @classmethod
    def _execute_cumsum(cls, op: "DataFrameEwmAgg", in_data):
        exec_cache = cls._exec_cache[op.key]
        cache_key = ('cumsum', id(in_data))
        summary = None

        try:
            result = exec_cache[cache_key]
        except KeyError:
            min_periods = 1 if op.min_periods > 0 else 0

            try:
                data = in_data.ewm(alpha=op.alpha, ignore_na=op.alpha_ignore_na, adjust=op.adjust,
                                   min_periods=min_periods).mean()
            except ValueError:
                in_data = in_data.copy()
                data = in_data.ewm(alpha=op.alpha, ignore_na=op.alpha_ignore_na, adjust=op.adjust,
                                   min_periods=min_periods).mean()

            alpha_sum, _ = op._execute_cum_alpha_coeff(op, in_data, 1, final=False)
            result = exec_cache[cache_key] = data * alpha_sum

        if op.output_agg:
            summary = result.ffill()[-1:]
        return result, summary

    @classmethod
    def _execute_cumsum2(cls, op: "DataFrameEwmAgg", in_data):
        summary = None
        min_periods = 1 if op.min_periods > 0 else 0

        try:
            data = in_data.ewm(alpha=op.alpha, ignore_na=op.alpha_ignore_na, adjust=op.adjust,
                               min_periods=min_periods).var(bias=True)
        except ValueError:
            in_data = in_data.copy()
            data = in_data.ewm(alpha=op.alpha, ignore_na=op.alpha_ignore_na, adjust=op.adjust,
                               min_periods=min_periods).var(bias=True)

        alpha_sum, _ = op._execute_cum_alpha_coeff(op, in_data, 1)
        cumsum, _ = op._execute_cumsum(op, in_data)
        result = alpha_sum * data + cumsum ** 2 / alpha_sum

        if op.output_agg:
            summary = result.ffill()[-1:]

        return result, summary

    @classmethod
    def _execute_map_function(cls, op: "DataFrameEwmAgg", func, in_data):
        in_data = in_data._get_numeric_data()

        summary = None
        min_periods = 1 if op.min_periods > 0 else 0
        if func == '_data_count':
            result = in_data.expanding(min_periods=min_periods).count()
        elif func in (_cum_alpha_coeff_func, _cum_square_alpha_coeff_func):
            order = 1 if func == _cum_alpha_coeff_func else 2
            result, summary = cls._execute_cum_alpha_coeff(op, in_data, order)
        elif func == 'cumsum':
            result, summary = cls._execute_cumsum(op, in_data)
        elif func == 'cumsum2':
            result, summary = cls._execute_cumsum2(op, in_data)
        else:  # pragma: no cover
            raise ValueError('Map function %s not supported')

        if op.output_agg:
            summary = summary if summary is not None else result.iloc[-1:]
        else:
            summary = None
        return result, summary

    @classmethod
    def _execute_map(cls, ctx, op: "DataFrameEwmAgg"):
        try:
            cls._exec_cache[op.key] = dict()

            super()._execute_map(ctx, op)
            if op.output_agg:
                in_data = ctx[op.inputs[0].key]
                summaries = list(ctx[op.outputs[1].key])

                if op.alpha_ignore_na:
                    in_count = in_data.count()
                    if not isinstance(in_count, pd.Series):
                        in_count = pd.Series([in_count])
                    summary = in_count
                    if in_data.ndim == 2:
                        summary = in_count.to_frame().T
                    summary.index = summaries[-1].index
                else:
                    remain_counts = in_data.notna()[::-1].to_numpy().argmax(axis=0)
                    if in_data.ndim > 1:
                        remain_counts = remain_counts.reshape((1, len(in_data.columns)))
                        summary = pd.DataFrame(remain_counts, columns=in_data.columns, index=summaries[-1].index)
                    else:
                        summary = pd.Series(remain_counts, index=summaries[-1].index)
                summaries.insert(-1, summary)

                ctx[op.outputs[1].key] = tuple(summaries)
        finally:
            cls._exec_cache.pop(op.key, None)

    @classmethod
    def _execute_combine_function(cls, op: "DataFrameEwmAgg", func, prev_inputs, local_inputs,
                                  func_cols):
        exec_cache = cls._exec_cache[op.key]
        pred_exponent = exec_cache.get('pred_exponent')
        if func_cols and pred_exponent is not None:
            pred_exponent = pred_exponent[func_cols] if pred_exponent is not None else None
        return func(prev_inputs, local_inputs, axis=op.axis, alpha=op.alpha,
                    alpha_ignore_na=op.alpha_ignore_na, pred_exponent=pred_exponent)

    @classmethod
    def _execute_combine(cls, ctx, op: "DataFrameEwmAgg"):
        try:
            cls._exec_cache[op.key] = dict()

            if len(op.inputs) != 1:
                pred_data = ctx[op.inputs[1].key]

                if op.alpha_ignore_na:
                    pred_exponent = pred_data[-2].shift(-1)[::-1].cumsum()[::-1].fillna(0)
                else:
                    succ_counts = pred_data[-1].shift(-1)
                    succ_counts.iloc[-1] = 0
                    pred_exponent = pred_data[-2].add(succ_counts[::-1].cumsum()[::-1], axis=op.axis)

                cls._exec_cache[op.key]['pred_exponent'] = pred_exponent

            super()._execute_combine(ctx, op)
        finally:
            cls._exec_cache.pop(op.key, None)

    @classmethod
    def _execute_raw_function(cls, op: "DataFrameEwmAgg", in_data):
        for _ in range(2):
            ewm = in_data.ewm(alpha=op.alpha, min_periods=op.min_periods, adjust=op.adjust,
                              ignore_na=op.alpha_ignore_na)
            try:
                val = ewm.agg(op.func)
                if in_data.ndim == 2 and op.validate_columns \
                        and len(val.columns) != len(op.outputs[0].columns_value.to_pandas()):
                    raise ValueError('Columns not consistent')
                return val
            except ValueError:
                in_data = in_data.copy()
        else:  # pragma: no cover
            raise ValueError
