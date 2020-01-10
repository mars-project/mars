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

from .sum import DataFrameSum
from .prod import DataFrameProd
from .max import DataFrameMax
from .min import DataFrameMin
from .count import DataFrameCount
from .mean import DataFrameMean


def _install():
    from ..core import DATAFRAME_TYPE, SERIES_TYPE
    from .sum import sum_series, sum_dataframe
    from .prod import prod_series, prod_dataframe
    from .max import max_series, max_dataframe
    from .min import min_series, min_dataframe
    from .count import count_series, count_dataframe
    from .mean import mean_series, mean_dataframe

    func_names = ['sum', 'prod', 'max', 'min', 'count', 'mean']
    series_funcs = [sum_series, prod_series, max_series, min_series, count_series, mean_series]
    df_funcs = [sum_dataframe, prod_dataframe, max_dataframe, min_dataframe, count_dataframe, mean_dataframe]
    for func_name, series_func, df_func in zip(func_names, series_funcs, df_funcs):
        for t in DATAFRAME_TYPE:
            setattr(t, func_name, df_func)
        for t in SERIES_TYPE:
            setattr(t, func_name, series_func)


_install()
del _install
