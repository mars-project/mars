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

from collections import namedtuple

import numpy as np
import pandas as pd

from .... import opcodes
from ....serialize import BoolField, Float64Field
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
        return local_results[0]

    alpha_data = local_results[1]
    local_results[1] = alpha_data.expanding(1).sum()
    local_results[0] = local_results[0] * local_results[1]
    if pred_results is not None:
        pred_results[0] = pred_results[0] * pred_results[1]

    local_sum_data, local_count_data = local_results

    if pred_results is not None:
        local_sum_data, local_count_data = _add_pred_results(
            pred_results, local_results, axis=axis, alpha=alpha, alpha_ignore_na=alpha_ignore_na,
            pred_exponent=pred_exponent, alpha_data=alpha_data
        )
    return local_sum_data / local_count_data


def _combine_var(pred_results, local_results, axis=0, alpha=None, alpha_ignore_na=False,
                 pred_exponent=None):
    alpha_data = local_results[1]
    local_results[1] = alpha_data.expanding(1).sum()

    alpha2_data = local_results[3]
    local_results[3] = alpha2_data.expanding(1).sum()

    local_results[0] = local_results[0] * local_results[1]
    if pred_results is not None:
        pred_results[0] = pred_results[0] * pred_results[1]

    local_sum_data, local_count_data, local_var_data, local_count2_data = local_results
    if pred_results is None:
        return local_var_data / (1 - local_count2_data / local_count_data ** 2)

    pred_sum_data, pred_count_data, pred_var_data, pred_count2_data = pred_results

    local_sum_square = local_count_data * local_var_data + local_sum_data ** 2 / local_count_data
    pred_sum_square = pred_count_data * pred_var_data + pred_sum_data ** 2 / pred_count_data

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
    return np.sqrt(_combine_var(pred_results, local_results, axis=axis, alpha=alpha,
                                alpha_ignore_na=alpha_ignore_na, pred_exponent=pred_exponent))


class DataFrameEwmAgg(BaseDataFrameExpandingAgg):
    _op_type_ = opcodes.EWM_AGG

    _alpha = Float64Field('alpha')
    _adjust = BoolField('adjust')
    _alpha_ignore_na = BoolField('alpha_ignore_na')

    _exec_cache = dict()

    def __init__(self, alpha=None, adjust=None, alpha_ignore_na=None, **kw):
        super().__init__(_alpha=alpha, _adjust=adjust, _alpha_ignore_na=alpha_ignore_na, **kw)

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def adjust(self) -> bool:
        return self._adjust

    @property
    def alpha_ignore_na(self) -> bool:
        return self._alpha_ignore_na

    @classmethod
    def _get_stage_functions(cls, op: "DataFrameEwmAgg", func):
        if func == 'mean':
            return ['mean', _cum_alpha_coeff_func], _combine_mean
        elif func in {'var', 'std'}:
            return ['mean', _cum_alpha_coeff_func, 'var', _cum_square_alpha_coeff_func], \
                   _combine_var if func == 'var' else _combine_std
        else:  # pragma: no cover
            raise NotImplementedError

    @classmethod
    def _execute_map_function(cls, op: "DataFrameEwmAgg", func, in_data):
        min_periods = 1 if op.min_periods > 0 else 0
        in_data = in_data._get_numeric_data()

        summary = None
        if func in (_cum_alpha_coeff_func, _cum_square_alpha_coeff_func):
            weight = (1 - op.alpha) if func == _cum_alpha_coeff_func else (1 - op.alpha) ** 2
            cum_df = in_data.copy()
            cum_df[cum_df.notna()] = 1
            if not op.alpha_ignore_na:
                cum_df.ffill(inplace=True)
            cum_df = cum_df.cumsum(axis=op.axis) - 1
            if not op.alpha_ignore_na:
                cum_df[in_data.isna()] = np.nan

            result = weight ** cum_df

            if op.output_agg:
                if result.ndim == 2:
                    summary = result.sum().to_frame().T
                    summary.index = result.index[-1:]
                else:
                    summary = pd.Series(result.sum(), index=result.index[-1:])
        else:
            for _ in range(2):
                expanding = in_data.ewm(alpha=op.alpha, min_periods=min_periods, adjust=op.adjust,
                                        ignore_na=op.alpha_ignore_na)
                try:
                    if func == 'var':
                        result = expanding.var(bias=True)
                    else:
                        result = expanding.agg(func)
                    break
                except ValueError:
                    in_data = in_data.copy()
            else:  # pragma: no cover
                raise ValueError

        if op.output_agg:
            summary = summary if summary is not None else result.iloc[len(result) - 1:len(result)]
        else:
            summary = None
        return result, summary

    @classmethod
    def _execute_combine_function(cls, op: "DataFrameEwmAgg", func, prev_inputs, local_inputs,
                                  func_cols):
        exec_cache = cls._exec_cache[op.key]
        pred_exponent = exec_cache.get('pred_exponent')
        if func_cols:
            pred_exponent = pred_exponent[func_cols]
        return func(prev_inputs, local_inputs, axis=op.axis, alpha=op.alpha,
                    alpha_ignore_na=op.alpha_ignore_na, pred_exponent=pred_exponent)

    @classmethod
    def _execute_map(cls, ctx, op: "DataFrameEwmAgg"):
        super()._execute_map(ctx, op)
        if op.output_agg:
            in_data = ctx[op.inputs[0].key]
            summaries = list(ctx[op.outputs[1].key])

            if op.alpha_ignore_na:
                in_count = in_data.count()
                if not isinstance(in_count, pd.Series):
                    in_count = pd.Series([in_count])
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
                return ewm.agg(op.func)
            except ValueError:
                in_data = in_data.copy()
        else:  # pragma: no cover
            raise ValueError
