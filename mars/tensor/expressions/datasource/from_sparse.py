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

from .... import opcodes as OperandDef
from ....serialize import KeyField
from .core import TensorHasInput
from .array import tensor


class SparseToDense(TensorHasInput):
    _op_type_ = OperandDef.SPARSE_TO_DENSE

    _input = KeyField('_input')

    def __init__(self, dtype=None, gpu=None, **kw):
        super(SparseToDense, self).__init__(_dtype=dtype, _gpu=gpu,
                                            _sparse=False, **kw)


def fromsparse(a):
    a = tensor(a)
    if not a.issparse():
        return a

    op = SparseToDense(dtype=a.dtype, gpu=a.op.gpu)
    return op(a)
