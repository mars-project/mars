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

from numbers import Integral

from .iloc import DataFrameIloc


class DataFrameIat:
    def __init__(self, obj):
        self._obj = obj
        self._iloc = DataFrameIloc(self._obj)

    def __getitem__(self, indexes):
        if not isinstance(indexes, tuple):
            indexes = (indexes,)

        for index in indexes:
            if not isinstance(index, Integral):
                raise ValueError('Invalid call for scalar access (getting)!')

        return self._iloc[indexes]


def iat(a):
    return DataFrameIat(a)
