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

from .sort_values import DataFrameSortValues
from .sort_index import DataFrameSortIndex


def _install():
    from ..core import DATAFRAME_TYPE, SERIES_TYPE
    from .sort_values import dataframe_sort_values, series_sort_values
    from .sort_index import sort_index

    for cls in DATAFRAME_TYPE:
        setattr(cls, 'sort_values', dataframe_sort_values)
        setattr(cls, 'sort_index', sort_index)

    for cls in SERIES_TYPE:
        setattr(cls, 'sort_values', series_sort_values)
        setattr(cls, 'sort_index', sort_index)


_install()
del _install
