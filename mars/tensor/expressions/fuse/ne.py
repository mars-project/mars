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

from ....serialize import DataTypeField
from .core import TensorFuse, TensorFuseChunkMixin


class TensorNeFuseChunk(TensorFuse, TensorFuseChunkMixin):
    _op_type_ = None  # no opcode, cannot be serialized
    _dtype = DataTypeField('dtype')

    # use for numexpr-fused operand
    def __init__(self, dtype=None, **kw):
        super(TensorNeFuseChunk, self).__init__(_dtype=dtype, **kw)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None)
