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
from .var import DataFrameVar
from .all import DataFrameAll
from .any import DataFrameAny

from .cummax import DataFrameCummax
from .cummin import DataFrameCummin
from .cumprod import DataFrameCumprod
from .cumsum import DataFrameCumsum

from .nunique import DataFrameNunique
from .unique import DataFrameUnique, unique


def _install():
    from ..core import DATAFRAME_TYPE, SERIES_TYPE
    from .aggregation import aggregate
    from .sum import sum_series, sum_dataframe
    from .prod import prod_series, prod_dataframe
    from .max import max_series, max_dataframe
    from .min import min_series, min_dataframe
    from .count import count_series, count_dataframe
    from .mean import mean_series, mean_dataframe
    from .var import var_series, var_dataframe
    from .std import std_series, std_dataframe
    from .all import all_series, all_dataframe
    from .any import any_series, any_dataframe
    from .cummax import cummax
    from .cummin import cummin
    from .cumprod import cumprod
    from .cumsum import cumsum
    from .nunique import nunique_dataframe, nunique_series

    funcs = [
        ('sum', sum_series, sum_dataframe),
        ('prod', prod_series, prod_dataframe),
        ('product', prod_series, prod_dataframe),
        ('max', max_series, max_dataframe),
        ('min', min_series, min_dataframe),
        ('count', count_series, count_dataframe),
        ('mean', mean_series, mean_dataframe),
        ('var', var_series, var_dataframe),
        ('std', std_series, std_dataframe),
        ('all', all_series, all_dataframe),
        ('any', any_series, any_dataframe),
        ('cummax', cummax, cummax),
        ('cummin', cummin, cummin),
        ('cumprod', cumprod, cumprod),
        ('cumsum', cumsum, cumsum),
        ('agg', aggregate, aggregate),
        ('aggregate', aggregate, aggregate),
        ('nunique', nunique_series, nunique_dataframe),
        ('unique', unique, None),
    ]
    for func_name, series_func, df_func in funcs:
        if df_func is not None:  # pragma: no branch
            for t in DATAFRAME_TYPE:
                setattr(t, func_name, df_func)
        if series_func is not None:  # pragma: no branch
            for t in SERIES_TYPE:
                setattr(t, func_name, series_func)


_install()
del _install
