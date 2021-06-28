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


def _install():
    from pandas.util import cache_readonly
    from ..operands import DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE
    from .at import at
    from .getitem import dataframe_getitem, series_getitem
    from .iat import iat
    from .iloc import iloc, head, tail, index_getitem, index_setitem
    from .insert import df_insert
    from .loc import loc
    from .reindex import reindex, reindex_like
    from .rename import df_rename, series_rename, index_rename, index_set_names
    from .rename_axis import rename_axis
    from .reset_index import df_reset_index, series_reset_index
    from .set_index import set_index
    from .setitem import dataframe_setitem
    from .where import mask, where
    from .set_axis import df_set_axis, series_set_axis
    from .sample import sample
    from .add_prefix import df_add_prefix, series_add_prefix

    for cls in DATAFRAME_TYPE + SERIES_TYPE:
        setattr(cls, 'iloc', cache_readonly(iloc))
        setattr(cls, 'loc', cache_readonly(loc))
        setattr(cls, 'iat', cache_readonly(iat))
        setattr(cls, 'at', cache_readonly(at))
        setattr(cls, 'head', head)
        setattr(cls, 'reindex', reindex)
        setattr(cls, 'reindex_like', reindex_like)
        setattr(cls, 'rename_axis', rename_axis)
        setattr(cls, 'tail', tail)
        setattr(cls, 'mask', mask)
        setattr(cls, 'where', where)
        setattr(cls, 'sample', sample)

    for cls in DATAFRAME_TYPE:
        setattr(cls, 'set_index', set_index)
        setattr(cls, '__getitem__', dataframe_getitem)
        setattr(cls, '__setitem__', dataframe_setitem)
        setattr(cls, 'insert', df_insert)
        setattr(cls, 'reset_index', df_reset_index)
        setattr(cls, 'rename', df_rename)
        setattr(cls, 'set_axis', df_set_axis)
        setattr(cls, 'add_prefix', df_add_prefix)

    for cls in SERIES_TYPE:
        setattr(cls, '__getitem__', series_getitem)
        setattr(cls, 'reset_index', series_reset_index)
        setattr(cls, 'rename', series_rename)
        setattr(cls, 'set_axis', series_set_axis)
        setattr(cls, 'add_prefix', series_add_prefix)

    for cls in INDEX_TYPE:
        setattr(cls, '__getitem__', index_getitem)
        setattr(cls, '__setitem__', index_setitem)
        setattr(cls, 'rename', index_rename)
        setattr(cls, 'set_names', index_set_names)


_install()
del _install
