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

from .map import series_map, index_map
from .to_gpu import to_gpu
from .to_cpu import to_cpu
from .rechunk import rechunk
from .describe import describe
from .apply import df_apply, series_apply
from .transform import df_transform, series_transform
from .isin import series_isin, df_isin
from .cut import cut
from .qcut import qcut
from .shift import shift, tshift
from .diff import df_diff, series_diff
from .value_counts import value_counts
from .astype import astype, index_astype
from .drop import df_drop, df_pop, series_drop, index_drop
from .drop_duplicates import df_drop_duplicates, \
    series_drop_duplicates, index_drop_duplicates
from .duplicated import df_duplicated, series_duplicated, index_duplicated
from .melt import melt
from .memory_usage import df_memory_usage, series_memory_usage, \
    index_memory_usage
from .select_dtypes import select_dtypes
from .map_chunk import map_chunk
from .cartesian_chunk import cartesian_chunk
from .rebalance import rebalance
from .stack import stack
from .explode import df_explode, series_explode
from .eval import df_eval, df_query
from .check_monotonic import check_monotonic, is_monotonic, \
    is_monotonic_increasing, is_monotonic_decreasing
from .pct_change import pct_change


def _install():
    from ..core import DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE
    from .standardize_range_index import ChunkStandardizeRangeIndex
    from .string_ import _string_method_to_handlers
    from .datetimes import _datetime_method_to_handlers
    from .accessor import StringAccessor, DatetimeAccessor, CachedAccessor

    for t in DATAFRAME_TYPE:
        setattr(t, 'to_gpu', to_gpu)
        setattr(t, 'to_cpu', to_cpu)
        setattr(t, 'rechunk', rechunk)
        setattr(t, 'describe', describe)
        setattr(t, 'apply', df_apply)
        setattr(t, 'transform', df_transform)
        setattr(t, 'isin', df_isin)
        setattr(t, 'shift', shift)
        setattr(t, 'tshift', tshift)
        setattr(t, 'diff', df_diff)
        setattr(t, 'astype', astype)
        setattr(t, 'drop', df_drop)
        setattr(t, 'pop', df_pop)
        setattr(t, '__delitem__', lambda df, items: df_drop(df, items, axis=1, inplace=True))
        setattr(t, 'drop_duplicates', df_drop_duplicates)
        setattr(t, 'duplicated', df_duplicated)
        setattr(t, 'melt', melt)
        setattr(t, 'memory_usage', df_memory_usage)
        setattr(t, 'select_dtypes', select_dtypes)
        setattr(t, 'map_chunk', map_chunk)
        setattr(t, 'cartesian_chunk', cartesian_chunk)
        setattr(t, 'rebalance', rebalance)
        setattr(t, 'stack', stack)
        setattr(t, 'explode', df_explode)
        setattr(t, 'eval', df_eval)
        setattr(t, 'query', df_query)
        setattr(t, 'pct_change', pct_change)

    for t in SERIES_TYPE:
        setattr(t, 'to_gpu', to_gpu)
        setattr(t, 'to_cpu', to_cpu)
        setattr(t, 'rechunk', rechunk)
        setattr(t, 'map', series_map)
        setattr(t, 'describe', describe)
        setattr(t, 'apply', series_apply)
        setattr(t, 'transform', series_transform)
        setattr(t, 'isin', series_isin)
        setattr(t, 'shift', shift)
        setattr(t, 'tshift', tshift)
        setattr(t, 'diff', series_diff)
        setattr(t, 'value_counts', value_counts)
        setattr(t, 'astype', astype)
        setattr(t, 'drop', series_drop)
        setattr(t, 'drop_duplicates', series_drop_duplicates)
        setattr(t, 'duplicated', series_duplicated)
        setattr(t, 'memory_usage', series_memory_usage)
        setattr(t, 'map_chunk', map_chunk)
        setattr(t, 'cartesian_chunk', cartesian_chunk)
        setattr(t, 'rebalance', rebalance)
        setattr(t, 'explode', series_explode)
        setattr(t, 'check_monotonic', check_monotonic)
        setattr(t, 'is_monotonic', property(fget=is_monotonic))
        setattr(t, 'is_monotonic_increasing', property(fget=is_monotonic_increasing))
        setattr(t, 'is_monotonic_decreasing', property(fget=is_monotonic_decreasing))
        setattr(t, 'pct_change', pct_change)

    for t in INDEX_TYPE:
        setattr(t, 'map', index_map)
        setattr(t, 'rechunk', rechunk)
        setattr(t, 'rebalance', rebalance)
        setattr(t, 'drop', index_drop)
        setattr(t, 'drop_duplicates', index_drop_duplicates)
        setattr(t, 'duplicated', index_duplicated)
        setattr(t, 'memory_usage', index_memory_usage)
        setattr(t, 'astype', index_astype)
        setattr(t, 'value_counts', value_counts)
        setattr(t, 'check_monotonic', check_monotonic)
        setattr(t, 'is_monotonic', property(fget=is_monotonic))
        setattr(t, 'is_monotonic_increasing', property(fget=is_monotonic_increasing))
        setattr(t, 'is_monotonic_decreasing', property(fget=is_monotonic_decreasing))

    for method in _string_method_to_handlers:
        if not hasattr(StringAccessor, method):
            StringAccessor._register(method)

    for method in _datetime_method_to_handlers:
        if not hasattr(DatetimeAccessor, method):
            DatetimeAccessor._register(method)

    for series in SERIES_TYPE:
        series.str = CachedAccessor('str', StringAccessor)
        series.dt = CachedAccessor('dt', DatetimeAccessor)


_install()
del _install
