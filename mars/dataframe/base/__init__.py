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

from .map import map_
from .to_gpu import to_gpu
from .to_cpu import to_cpu
from .rechunk import rechunk
from .describe import describe
from .apply import df_apply, series_apply
from .fillna import fillna, ffill, bfill
from .transform import df_transform, series_transform
from .isin import isin
from .checkna import isna, notna, isnull, notnull
from .dropna import df_dropna, series_dropna
from .cut import cut
from .shift import shift, tshift
from .diff import df_diff, series_diff
from .value_counts import value_counts
from .astype import astype
from .drop import df_drop, df_pop, series_drop, index_drop
from .drop_duplicates import df_drop_duplicates, \
    series_drop_duplicates, index_drop_duplicates
from .melt import melt
from .memory_usage import df_memory_usage, series_memory_usage, index_memory_usage


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
        setattr(t, 'fillna', fillna)
        setattr(t, 'ffill', ffill)
        setattr(t, 'bfill', bfill)
        setattr(t, 'isna', isna)
        setattr(t, 'isnull', isnull)
        setattr(t, 'notna', notna)
        setattr(t, 'notnull', notnull)
        setattr(t, 'dropna', df_dropna)
        setattr(t, 'shift', shift)
        setattr(t, 'tshift', tshift)
        setattr(t, 'diff', df_diff)
        setattr(t, 'astype', astype)
        setattr(t, 'drop', df_drop)
        setattr(t, 'pop', df_pop)
        setattr(t, '__delitem__', lambda df, items: df_drop(df, items, axis=1, inplace=True))
        setattr(t, 'drop_duplicates', df_drop_duplicates)
        setattr(t, 'melt', melt)
        setattr(t, 'memory_usage', df_memory_usage)

    for t in SERIES_TYPE:
        setattr(t, 'to_gpu', to_gpu)
        setattr(t, 'to_cpu', to_cpu)
        setattr(t, 'rechunk', rechunk)
        setattr(t, 'map', map_)
        setattr(t, 'describe', describe)
        setattr(t, 'apply', series_apply)
        setattr(t, 'transform', series_transform)
        setattr(t, 'fillna', fillna)
        setattr(t, 'ffill', ffill)
        setattr(t, 'bfill', bfill)
        setattr(t, 'isin', isin)
        setattr(t, 'isna', isna)
        setattr(t, 'isnull', isnull)
        setattr(t, 'notna', notna)
        setattr(t, 'notnull', notnull)
        setattr(t, 'dropna', series_dropna)
        setattr(t, 'shift', shift)
        setattr(t, 'tshift', tshift)
        setattr(t, 'diff', series_diff)
        setattr(t, 'value_counts', value_counts)
        setattr(t, 'astype', astype)
        setattr(t, 'drop', series_drop)
        setattr(t, 'drop_duplicates', series_drop_duplicates)
        setattr(t, 'memory_usage', series_memory_usage)

    for t in INDEX_TYPE:
        setattr(t, 'rechunk', rechunk)
        setattr(t, 'drop', index_drop)
        setattr(t, 'drop_duplicates', index_drop_duplicates)
        setattr(t, 'memory_usage', index_memory_usage)

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
