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

# noinspection PyUnresolvedReferences
from ..core import GroupBy, DataFrameGroupBy, SeriesGroupBy


def _install():
    from ..core import DATAFRAME_TYPE, SERIES_TYPE, GROUPBY_TYPE, DATAFRAME_GROUPBY_TYPE
    from .core import groupby
    from .aggregation import agg
    from .apply import groupby_apply
    from .cum import cumcount, cummin, cummax, cumprod, cumsum
    from .getitem import df_groupby_getitem
    from .head import head
    from .transform import groupby_transform
    from .sample import groupby_sample

    for cls in DATAFRAME_TYPE:
        setattr(cls, 'groupby', groupby)

    for cls in SERIES_TYPE:
        setattr(cls, 'groupby', groupby)

    for cls in GROUPBY_TYPE:
        setattr(cls, 'agg', agg)
        setattr(cls, 'aggregate', agg)

        setattr(cls, 'sum', lambda groupby, **kw: agg(groupby, 'sum', **kw))
        setattr(cls, 'prod', lambda groupby, **kw: agg(groupby, 'prod', **kw))
        setattr(cls, 'max', lambda groupby, **kw: agg(groupby, 'max', **kw))
        setattr(cls, 'min', lambda groupby, **kw: agg(groupby, 'min', **kw))
        setattr(cls, 'count', lambda groupby, **kw: agg(groupby, 'count', **kw))
        setattr(cls, 'size', lambda groupby, **kw: agg(groupby, 'size', **kw))
        setattr(cls, 'mean', lambda groupby, **kw: agg(groupby, 'mean', **kw))
        setattr(cls, 'var', lambda groupby, **kw: agg(groupby, 'var', **kw))
        setattr(cls, 'std', lambda groupby, **kw: agg(groupby, 'std', **kw))
        setattr(cls, 'all', lambda groupby, **kw: agg(groupby, 'all', **kw))
        setattr(cls, 'any', lambda groupby, **kw: agg(groupby, 'any', **kw))
        setattr(cls, 'skew', lambda groupby, **kw: agg(groupby, 'skew', **kw))
        setattr(cls, 'kurt', lambda groupby, **kw: agg(groupby, 'kurt', **kw))
        setattr(cls, 'kurtosis', lambda groupby, **kw: agg(groupby, 'kurtosis', **kw))
        setattr(cls, 'sem', lambda groupby, **kw: agg(groupby, 'sem', **kw))

        setattr(cls, 'apply', groupby_apply)
        setattr(cls, 'transform', groupby_transform)

        setattr(cls, 'cumcount', cumcount)
        setattr(cls, 'cummin', cummin)
        setattr(cls, 'cummax', cummax)
        setattr(cls, 'cumprod', cumprod)
        setattr(cls, 'cumsum', cumsum)

        setattr(cls, 'head', head)

        setattr(cls, 'sample', groupby_sample)

    for cls in DATAFRAME_GROUPBY_TYPE:
        setattr(cls, '__getitem__', df_groupby_getitem)


_install()
del _install
