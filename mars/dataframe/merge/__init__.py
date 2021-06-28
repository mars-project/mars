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

from .concat import DataFrameConcat, concat
from .merge import join, merge, DataFrameShuffleMerge, DataFrameMergeAlign
from .append import DataFrameAppend, append


def _install():
    from ..core import DATAFRAME_TYPE, SERIES_TYPE

    for cls in DATAFRAME_TYPE:
        setattr(cls, 'join', join)
        setattr(cls, 'merge', merge)

    for cls in DATAFRAME_TYPE + SERIES_TYPE:
        setattr(cls, 'append', append)


_install()
del _install
