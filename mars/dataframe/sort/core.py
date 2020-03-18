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


from ...serialize import Int32Field, StringField, ListField, BoolField, ValueType
from ..operands import DataFrameOperand


class DataFrameSortOperand(DataFrameOperand):
    _axis = Int32Field('axis')
    _ascending = BoolField('ascending')
    _inplace = BoolField('inplace')
    _kind = StringField('kind')
    _na_position = StringField('na_position')
    _ignore_index = BoolField('ignore_index')
    _parallel_kind = StringField('parallel_kind')
    _psrs_kinds = ListField('psrs_kinds', ValueType.string)

    def __init__(self, axis=None, ascending=None, inplace=None, kind=None,
                 na_position=None, ignore_index=None, parallel_kind=None, psrs_kinds=None, **kw):
        super(DataFrameSortOperand, self).__init__(_axis=axis, _ascending=ascending,
                                                   _inplace=inplace, _kind=kind,
                                                   _na_position=na_position,
                                                   _ignore_index=ignore_index,
                                                   _parallel_kind=parallel_kind,
                                                   _psrs_kinds=psrs_kinds, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def ascending(self):
        return self._ascending

    @property
    def inplace(self):
        return self._inplace

    @property
    def kind(self):
        return self._kind

    @property
    def na_position(self):
        return self._na_position

    @property
    def ignore_index(self):
        return self._ignore_index

    @property
    def parallel_kind(self):
        return self._parallel_kind

    @property
    def psrs_kinds(self):
        return self._psrs_kinds
