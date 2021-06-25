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
from functools import partial

import numpy as np
import pandas as pd

from .... import opcodes
from ....serialization.serializables import BoolField
from ..aggregation import BaseDataFrameExpandingAgg

_stage_info = namedtuple('_stage_info', ('map_groups', 'map_sources', 'combine_sources',
                                         'combine_columns', 'combine_funcs', 'key_to_funcs',
                                         'valid_columns', 'min_periods_func_name'))

_cum_alpha_coeff_func = '_cum_alpha_coeff'
_cum_square_alpha_coeff_func = '_cum_square_alpha_coeff'


def _add_pred_results(pred_results, local_results, axis=0):
    if pred_results[0].ndim == 1:
        df_filler = 0
    else:
        df_filler = pred_results[0].iloc[-1, :].dropna()
        df_filler[:] = 0

    new_locals = []
    combine_axis = pred_results[0].ndim - axis - 1
    for pred_result, local_result in zip(pred_results, local_results):
        local_result = local_result.fillna(df_filler, axis=axis)
        new_locals.append(local_result.add(pred_result.sum(axis=axis), axis=combine_axis))
    return new_locals


def _combine_arithmetic(pred_results, local_results, axis=0):
    if pred_results is None:
        return local_results[0]
    return _add_pred_results(pred_results, local_results, axis=axis)[0]


def _combine_minmax(pred_results, local_results, axis=0, fun_name=None):
    if pred_results is None:
        return local_results[0]

    pred_size = len(pred_results[0])
    con = pd.concat([pred_results[0], local_results[0]], axis=axis)
    result = con.expanding(axis=axis).agg(fun_name)
    if result.ndim == 2:
        return result.iloc[pred_size:, :] if axis == 0 else result.iloc[:, pred_size:]
    else:
        return result.iloc[pred_size:]


def _combine_mean(pred_results, local_results, axis=0):
    local_sum_data, local_count_data = local_results

    if pred_results is not None:
        local_sum_data, local_count_data = _add_pred_results(
            pred_results, local_results, axis=axis)
    return local_sum_data / local_count_data


def _combine_var(pred_results, local_results, axis=0):
    local_sum_data, local_count_data, local_var_data = local_results
    if pred_results is None:
        return local_var_data * local_count_data / (local_count_data - 1)

    pred_sum_data, pred_count_data, pred_var_data = pred_results

    local_sum_square = local_count_data * local_var_data + local_sum_data ** 2 / local_count_data
    pred_sum_square = pred_count_data * pred_var_data + pred_sum_data ** 2 / pred_count_data

    local_sum_square, local_sum_data, local_count_data = \
        _add_pred_results([pred_sum_square, pred_sum_data, pred_count_data],
                          [local_sum_square, local_sum_data, local_count_data], axis=axis)

    return (local_sum_square - local_sum_data ** 2 / local_count_data) / (local_count_data - 1)


def _combine_std(pred_results, local_results, axis=0):
    return np.sqrt(_combine_var(pred_results, local_results, axis=axis))


class DataFrameExpandingAgg(BaseDataFrameExpandingAgg):
    _op_type_ = opcodes.EXPANDING_AGG

    _center = BoolField('center')

    def __init__(self, center=None, **kw):
        super().__init__(_center=center, **kw)

    @property
    def center(self):
        return self._center

    @classmethod
    def _get_stage_functions(cls, op: "DataFrameExpandingAgg", func):
        if func == '_data_count':
            return ['count'], _combine_arithmetic
        elif func in ('sum', 'prod', 'count'):
            return [func], _combine_arithmetic
        elif func in ('min', 'max'):
            return [func], partial(_combine_minmax, fun_name=func)
        elif func == 'mean':
            return ['sum', 'count'], _combine_mean
        elif func in {'var', 'std'}:
            return ['sum', 'count', 'var'], _combine_var if func == 'var' else _combine_std
        else:  # pragma: no cover
            raise NotImplementedError

    @classmethod
    def _execute_map_function(cls, op: "DataFrameExpandingAgg", func, in_data):
        min_periods = 1 if op.min_periods > 0 else 0

        expanding = in_data.expanding(min_periods=min_periods, center=op.center, axis=op.axis)
        if func == 'var':
            result = expanding.var(ddof=0)
        else:
            result = expanding.agg(func)

        if op.output_agg:
            summary = result.iloc[len(result) - 1:len(result)]
        else:
            summary = None
        return result, summary

    @classmethod
    def _execute_combine_function(cls, op: "DataFrameExpandingAgg", func, pred_inputs,
                                  local_inputs, func_cols):
        return func(pred_inputs, local_inputs, axis=op.axis)

    @classmethod
    def _execute_raw_function(cls, op: "DataFrameExpandingAgg", in_data):
        expanding = in_data.expanding(min_periods=op.min_periods, center=op.center, axis=op.axis)
        return expanding.agg(op.func)
