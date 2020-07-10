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
    from pandas.util import cache_readonly
    from .at import at
    from .getitem import dataframe_getitem, series_getitem
    from .iat import iat
    from .iloc import iloc, head, tail
    from .insert import df_insert
    from .loc import loc
    from .rename import df_rename, series_rename, index_rename
    from .reset_index import df_reset_index, series_reset_index
    from .set_index import set_index
    from .setitem import dataframe_setitem
    from ..operands import DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE

    for cls in DATAFRAME_TYPE + SERIES_TYPE:
        setattr(cls, 'iloc', cache_readonly(iloc))
        setattr(cls, 'loc', cache_readonly(loc))
        setattr(cls, 'iat', cache_readonly(iat))
        setattr(cls, 'at', cache_readonly(at))
        setattr(cls, 'head', head)
        setattr(cls, 'tail', tail)

    for cls in DATAFRAME_TYPE:
        setattr(cls, 'set_index', set_index)
        setattr(cls, '__getitem__', dataframe_getitem)
        setattr(cls, '__setitem__', dataframe_setitem)
        setattr(cls, 'insert', df_insert)
        setattr(cls, 'reset_index', df_reset_index)
        setattr(cls, 'rename', df_rename)

    for cls in SERIES_TYPE:
        setattr(cls, '__getitem__', series_getitem)
        setattr(cls, 'reset_index', series_reset_index)
        setattr(cls, 'rename', series_rename)

    for cls in INDEX_TYPE:
        setattr(cls, 'rename', index_rename)


_install()
del _install
