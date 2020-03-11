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

from .sort_values import DataFrameSortValues


def _install():
    from ..core import DATAFRAME_TYPE
    from .sort_values import sort_values

    def handle_inplace(sort_func):
        def inner(*args, **kwargs):
            if not kwargs.get('inplace'):
                return sort_func(*args, **kwargs)
            else:
                args[0].data = sort_func(*args, **kwargs).data
        return inner

    for cls in DATAFRAME_TYPE:
        setattr(cls, 'sort_values', handle_inplace(sort_values))


_install()
del _install
