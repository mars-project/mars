#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from .core import Operand
from .. import opcodes as OperandDef
from ..serialize import StringField, ListField, ValueType
from ..utils import to_str


class Fetch(Operand):
    _op_type_ = OperandDef.FETCH

    _to_fetch_key = StringField('to_fetch_key', on_serialize=to_str)

    @property
    def to_fetch_key(self):
        return self._to_fetch_key


class FetchShuffle(Operand):
    _op_type_ = OperandDef.FETCH_SHUFFLE

    _to_fetch_keys = ListField('to_fetch_keys', ValueType.string,
                               on_serialize=lambda v: [to_str(i) for i in v])

    @property
    def to_fetch_keys(self):
        return self._to_fetch_keys
