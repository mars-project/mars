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

from .... import operands
from .core import TensorBinOp, TensorConstant


class TensorSetReal(operands.SetReal, TensorBinOp):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorSetReal, self).__init__(_casting=casting, _err=err,
                                            _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, x1, x2):
        return x1.issparse() and x2.issparse()

    @classmethod
    def constant_cls(cls):
        return TensorSetRealConstant


class TensorSetRealConstant(operands.SetRealConstant, TensorConstant):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorSetRealConstant, self).__init__(_casting=casting, _err=err,
                                                    _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse() and np.isscalar(x2) and x2 == 0:
            return True
        if hasattr(x2, 'issparse') and x2.issparse() and np.isscalar(x1) and x1 == 0:
            return True
        return False


def set_real(val, real):
    op = TensorSetReal(dtype=val.dtype)
    return op(val, real)
