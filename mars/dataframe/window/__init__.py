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
    from .rolling.aggregation import DataFrameRollingAgg
    from ..core import DATAFRAME_TYPE, SERIES_TYPE
    from .rolling.core import rolling

    for t in DATAFRAME_TYPE + SERIES_TYPE:
        t.rolling = rolling

    del DataFrameRollingAgg


_install()
del _install
