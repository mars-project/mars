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
    import pandas as pd

    from ..base.accessor import CachedAccessor
    from ..core import DATAFRAME_TYPE, SERIES_TYPE
    from .core import PlotAccessor

    for t in DATAFRAME_TYPE + SERIES_TYPE:
        t.plot = CachedAccessor('plot', PlotAccessor)

    for method in dir(pd.DataFrame.plot):
        if not method.startswith('_'):
            PlotAccessor._register(method)

    PlotAccessor.__doc__ = pd.DataFrame.plot.__doc__.replace('pd.', 'md.')


_install()
del _install
