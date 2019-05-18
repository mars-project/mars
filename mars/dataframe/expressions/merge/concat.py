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

from ....serialize import ValueType, ListField, StringField, BoolField, AnyField
from .... import opcodes as OperandDef
from ..core import DataFrameOperand, DataFrameOperandMixin


class DataFrameConcat(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.CONCATENATE

    _axis = AnyField('axis')
    _join = StringField('join')
    _join_axes = ListField('join_axes', ValueType.key)
    _ignore_index = BoolField('ignore_index')
    _keys = ListField('keys')
    _levels = ListField('levels')
    _names = ListField('names')
    _verify_integrity = BoolField('verify_integrity')
    _sort = BoolField('sort')
    _copy = BoolField('copy')

    def __init__(self, axis=None, join=None, join_axes=None, ignore_index=None,
                 keys=None, levels=None, names=None, verify_integrity=None,
                 sort=None, copy=None, sparse=None, **kw):
        super(DataFrameConcat, self).__init__(
            _axis=axis, _join=join, _join_axes=join_axes, _ignore_index=ignore_index,
            _keys=keys, _levels=levels, _names=names,
            _verify_integrity=verify_integrity, _sort=sort, _copy=copy,
            _sparse=sparse, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def join(self):
        return self._join

    @property
    def join_axes(self):
        return self._join_axes

    @property
    def ignore_index(self):
        return self._ignore_index

    @property
    def keys(self):
        return self._keys

    @property
    def level(self):
        return self._levels

    @property
    def name(self):
        return self._name

    @property
    def verify_integrity(self):
        return self._verify_integrity

    @property
    def sort(self):
        return self._sort

    @property
    def copy_(self):
        return self._copy

