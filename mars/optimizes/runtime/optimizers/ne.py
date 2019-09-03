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
from .base import Optimizer

NE_REDUCTION_OP = (reduction.TensorSum, reduction.TensorProd,
                   reduction.TensorMax, reduction.TensorMin)
NE_SUPPORT_OP = (
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
)


def check_reduction_axis(node):
    return len(node.op.axis) == 1 or len(node.op.axis) == node.ndim


class NeOptimizer(Optimizer):
    def __init__(self, graph):
        super(NeOptimizer, self).__init__(graph)

    def optimize(self, keys=None):
        return self.compose(keys)

    def _can_break(self, node):
        return self.graph.count_successors(node) != 1 or isinstance(node.op, NE_REDUCTION_OP)

    def _support(self, node):
        op_type = type(node.op)
        if op_type in NE_REDUCTION_OP:
            return check_reduction_axis(node)
        return op_type in NE_SUPPORT_OP

    def _can_skip(self, node):
        if super(NeOptimizer, self)._can_skip(node):
            return True
        return not isinstance(node.op, NE_SUPPORT_OP)

    @staticmethod
    def _get_fused_chunk(tail_node):
        from ....tensor.fuse import TensorNeFuseChunk
        return TensorNeFuseChunk(dtype=tail_node.dtype)