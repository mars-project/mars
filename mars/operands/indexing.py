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
from .base import HasInput
from .. import opcodes as OperandDef
from ..serialize import ValueType, KeyField, ListField, StringField


class Choose(Operand):
    _op_type_ = OperandDef.CHOOSE

    _a = KeyField('a')
    _choices = ListField('choices', ValueType.key)
    _mode = StringField('mode')

    def __setattr__(self, key, value):
        if key == '_mode' and value not in ('raise', 'wrap', 'clip'):
            raise ValueError('mode should be raise, wrap or clip')

        super(Choose, self).__setattr__(key, value)

    @property
    def a(self):
        return self._a

    @property
    def choices(self):
        return self._choices

    @property
    def mode(self):
        return self._mode


class Nonzero(HasInput):
    _op_type_ = OperandDef.NONZERO

    _input = KeyField('input')

    @property
    def output_limit(self):
        return float('inf')
