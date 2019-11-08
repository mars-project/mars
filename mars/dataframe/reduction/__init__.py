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

from .sum import DataFrameSum
from .prod import DataFrameProd
from .max import DataFrameMax
from .min import DataFrameMin


def _install():
    from ..core import DataFrame, Series
    from .sum import sum_series, sum_dataframe
    from .prod import prod_series, prod_dataframe
    from .max import max_series, max_dataframe
    from .min import min_series, min_dataframe

    setattr(DataFrame, 'sum', sum_dataframe)
    setattr(Series, 'sum', sum_series)
    setattr(DataFrame, 'prod', prod_dataframe)
    setattr(Series, 'prod', prod_series)
    setattr(DataFrame, 'max', max_dataframe)
    setattr(Series, 'max', max_series)
    setattr(DataFrame, 'min', min_dataframe)
    setattr(Series, 'min', min_series)


_install()
del _install
