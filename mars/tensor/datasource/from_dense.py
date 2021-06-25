#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

from collections.abc import Iterable
from functools import reduce
from operator import and_

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...serialization.serializables import KeyField, AnyField
from ...lib.sparse import SparseNDArray
from ...lib.sparse.core import naked, cps, sps
from .core import TensorHasInput
from .array import tensor


class DenseToSparse(TensorHasInput):
    _op_type_ = OperandDef.DENSE_TO_SPARSE

    _input = KeyField('input')
    _missing = AnyField('missing')

    def __init__(self, missing=None, **kw):
        super().__init__(sparse=True, _missing=missing, **kw)

    @property
    def missing(self):
        return self._missing

    @staticmethod
    def _get_mask(data, missing):
        if isinstance(missing, Iterable):
            return reduce(and_, (DenseToSparse._get_mask(data, m) for m in missing))
        elif pd.isna(missing):
            return ~pd.isna(data)
        else:
            return data != missing

    @classmethod
    def execute(cls, ctx, op):
        out = op.outputs[0]
        in_data = naked(ctx[op.inputs[0].key])
        missing = op.missing
        shape = in_data.shape \
            if any(np.isnan(s) for s in out.shape) else out.shape

        xps = cps if op.gpu else sps
        if missing is None:
            ctx[out.key] = \
                SparseNDArray(xps.csr_matrix(in_data), shape=shape)
        else:
            mask = cls._get_mask(in_data, missing)
            spmatrix = xps.csr_matrix((in_data[mask], mask.nonzero()),
                                      shape=shape)
            ctx[out.key] = SparseNDArray(spmatrix)


def fromdense(a, missing=None):
    a = tensor(a)
    if a.issparse():
        return a

    op = DenseToSparse(dtype=a.dtype, gpu=a.op.gpu, missing=missing)
    return op(a)
