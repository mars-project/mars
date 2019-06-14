#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from ....operands import Slice
from ..core import TensorOperandMixin


class TensorSlice(Slice, TensorOperandMixin):
    def __init__(self, slices=None, dtype=None, sparse=False, **kw):
        super(TensorSlice, self).__init__(_slices=slices, _dtype=dtype,
                                          _sparse=sparse, **kw)

    def _set_inputs(self, inputs):
        super(TensorSlice, self)._set_inputs(inputs)
        self._input = self._inputs[0]
