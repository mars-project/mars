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

from mars.dataframe.base.to_gpu import to_gpu
from mars.dataframe.base.to_cpu import to_cpu


def _install():
    from ..core import DataFrameData, DataFrame, SeriesData, Series

    DataFrameData.to_gpu = to_gpu
    DataFrame.to_gpu = to_gpu
    DataFrameData.to_cpu = to_cpu
    DataFrame.to_cpu = to_cpu
    SeriesData.to_gpu = to_gpu
    Series.to_gpu = to_gpu
    SeriesData.to_cpu = to_cpu
    Series.to_cpu = to_cpu


_install()
del _install
