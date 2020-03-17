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


def _install():
    from ..operands import DATAFRAME_TYPE, GROUPBY_TYPE
    from .core import dataframe_groupby
    from .aggregation import agg
    from .cum import cumcount, cummin, cummax, cumprod, cumsum

    for cls in DATAFRAME_TYPE:
        setattr(cls, 'groupby', dataframe_groupby)
    for cls in GROUPBY_TYPE:
        setattr(cls, 'agg', agg)
        setattr(cls, 'aggregate', agg)

        setattr(cls, 'sum', lambda groupby, **kw: agg(groupby, 'sum', **kw))
        setattr(cls, 'prod', lambda groupby, **kw: agg(groupby, 'prod', **kw))
        setattr(cls, 'max', lambda groupby, **kw: agg(groupby, 'max', **kw))
        setattr(cls, 'min', lambda groupby, **kw: agg(groupby, 'min', **kw))

        setattr(cls, 'cumcount', cumcount)
        setattr(cls, 'cummin', cummin)
        setattr(cls, 'cummax', cummax)
        setattr(cls, 'cumprod', cumprod)
        setattr(cls, 'cumsum', cumsum)


_install()
del _install
