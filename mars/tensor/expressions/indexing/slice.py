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

import numpy as np

from ....operands import Slice
from ..core import TensorOperandMixin
from ..utils import calc_sliced_size


class TensorSlice(Slice, TensorOperandMixin):
    def __init__(self, slices=None, dtype=None, sparse=False, **kw):
        super(TensorSlice, self).__init__(_slices=slices, _dtype=dtype,
                                          _sparse=sparse, **kw)

    def calc_shape(self, *inputs_shape):
        input_shape = inputs_shape[0]
        shape = []
        idx = 0
        for s in self._slices:
            if s is None:
                shape.append(1)
            elif isinstance(s, slice):
                if np.isnan(input_shape[idx]):
                    shape.append(np.nan)
                else:
                    shape.append(calc_sliced_size(input_shape[idx], s))
                idx += 1
        shape.extend(list(input_shape[idx:]))
        return tuple(shape)

    def _set_inputs(self, inputs):
        super(TensorSlice, self)._set_inputs(inputs)
        self._input = self._inputs[0]
