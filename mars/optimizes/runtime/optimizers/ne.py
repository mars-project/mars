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

from ....tensor import arithmetic
from ....tensor import reduction
from ....tensor.fuse import TensorNeFuseChunk

REDUCTION_OP = {reduction.TensorSum, reduction.TensorProd,
                reduction.TensorMax, reduction.TensorMin}
SUPPORT_OP = {
    arithmetic.TensorAdd,
    arithmetic.TensorSubtract,
    arithmetic.TensorMultiply,
    arithmetic.TensorDivide,
    arithmetic.TensorPower,
    arithmetic.TensorMod,
    arithmetic.TensorNegative,
    arithmetic.TensorAbs,
    arithmetic.TensorConj,
    arithmetic.TensorExp,
    arithmetic.TensorLog,
    arithmetic.TensorLog10,
    arithmetic.TensorExpm1,
    arithmetic.TensorLog1p,
    arithmetic.TensorSqrt,

    arithmetic.TensorEqual,
    arithmetic.TensorNotEqual,
    arithmetic.TensorLessThan,
    arithmetic.TensorLessEqual,
    arithmetic.TensorGreaterThan,
    arithmetic.TensorGreaterEqual,

    arithmetic.TensorSin,
    arithmetic.TensorCos,
    arithmetic.TensorTan,
    arithmetic.TensorArcsin,
    arithmetic.TensorArccos,
    arithmetic.TensorArctan,
    arithmetic.TensorSinh,
    arithmetic.TensorCosh,
    arithmetic.TensorTanh,
    arithmetic.TensorArcsinh,
    arithmetic.TensorArccosh,
    arithmetic.TensorArctanh,

    arithmetic.TensorLshift,
    arithmetic.TensorRshift,

    arithmetic.TensorTreeAdd,
    arithmetic.TensorTreeMultiply,

    reduction.TensorSum,
    reduction.TensorProd,
    reduction.TensorMax,
    reduction.TensorMin
}


class NeOptimizer(object):
    def __init__(self, graph):
        self._graph = graph

    @property
    def graph(self):
        return self._graph

    def optimize(self, keys=None):
        from .utils import Composer
        return Composer(self, keys).compose()
