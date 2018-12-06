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
from ..utils import infer_dtype
from .core import TensorUnaryOp


class TensorPositive(operands.Positive, TensorUnaryOp):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorPositive, self).__init__(_casting=casting, _err=err,
                                             _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, x):
        return x.issparse()


@infer_dtype(np.positive)
def positive(x, out=None, where=None, **kwargs):
    """
    Numerical positive, element-wise.

    Parameters
    ----------
    x : array_like or scalar
        Input tensor.

    Returns
    -------
    y : Tensor or scalar
        Returned array or scalar: `y = +x`.
    """
    op = TensorPositive(**kwargs)
    return op(x, out=out, where=where)
