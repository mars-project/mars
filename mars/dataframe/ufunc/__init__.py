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
    from ..core import DataFrame, Series
    from .ufunc import _array_ufunc
    from .tensor import _tensor_ufunc

    for Entity in (DataFrame, Series):
        Entity.__array_ufunc__ = _array_ufunc
        Entity.__tensor_ufunc__ = _tensor_ufunc


_install()
del _install
