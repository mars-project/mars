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

from .fillna import fillna, ffill, bfill, index_fillna
from .checkna import isna, notna, isnull, notnull
from .dropna import df_dropna, series_dropna, index_dropna
from .replace import df_replace, series_replace


def _install():
    from ..core import DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE

    for cls in DATAFRAME_TYPE + SERIES_TYPE:
        setattr(cls, 'fillna', fillna)
        setattr(cls, 'ffill', ffill)
        setattr(cls, 'pad', ffill)
        setattr(cls, 'backfill', bfill)
        setattr(cls, 'bfill', bfill)
        setattr(cls, 'isna', isna)
        setattr(cls, 'isnull', isnull)
        setattr(cls, 'notna', notna)
        setattr(cls, 'notnull', notnull)

    for cls in DATAFRAME_TYPE:
        setattr(cls, 'dropna', df_dropna)
        setattr(cls, 'replace', df_replace)

    for cls in SERIES_TYPE:
        setattr(cls, 'dropna', series_dropna)
        setattr(cls, 'replace', series_replace)

    for cls in INDEX_TYPE:
        setattr(cls, 'fillna', index_fillna)
        setattr(cls, 'dropna', index_dropna)
        setattr(cls, 'isna', isna)
        setattr(cls, 'notna', notna)


_install()
del _install
