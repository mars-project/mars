# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from .iloc import iloc, DataFrameIloc, DataFrameIlocGetItem, DataFrameIlocSetItem
from .set_index import set_index, DataFrameSetIndex


def _install():
    from .getitem import dataframe_getitem, series_getitem
    from ..operands import DATAFRAME_TYPE, SERIES_TYPE

    for cls in DATAFRAME_TYPE:
        setattr(cls, 'iloc', property(iloc))
        setattr(cls, 'set_index', set_index)
        setattr(cls, '__getitem__', dataframe_getitem)
    for cls in SERIES_TYPE:
        setattr(cls, '__getitem__', series_getitem)


_install()
del _install
