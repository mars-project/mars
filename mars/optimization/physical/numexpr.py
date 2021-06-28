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

from ...core import ChunkType
from ...tensor import arithmetic
from ...tensor import reduction
from ...tensor.fuse import TensorNeFuseChunk
from ...tensor.fuse.numexpr import NUMEXPR_INSTALLED
from .core import RuntimeOptimizer, register_optimizer


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


def _check_reduction_axis(node: ChunkType):
    return len(node.op.axis) == 1 or len(node.op.axis) == node.ndim


def _support(node: ChunkType):
    op_type = type(node.op)
    if op_type in REDUCTION_OP:
        return _check_reduction_axis(node)
    return op_type in SUPPORT_OP


def _transfer_op(node: ChunkType):
    op = node.op
    if type(op) in REDUCTION_OP and not _check_reduction_axis(node):
        return op
    return op


@register_optimizer
class NumexprRuntimeOptimizer(RuntimeOptimizer):
    engine = 'numexpr'

    @classmethod
    def is_available(cls) -> bool:
        return NUMEXPR_INSTALLED

    def optimize(self):
        fuses = []
        explored = set()

        graph = self._graph
        for node in graph.topological_iter():
            if node.op.gpu or node.op.sparse:
                # break
                return [], []
            if type(node.op) not in SUPPORT_OP or \
                    node in graph.results:
                continue
            if node in explored or type(node.op) in REDUCTION_OP:
                # TODO: check logic here
                continue
            if graph.count_successors(node) != 1:
                continue

            selected = [node]
            # add successors
            cur_node = graph.successors(node)[0]
            while graph.count_predecessors(cur_node) == 1 and _support(cur_node):
                selected.append(cur_node)
                if graph.count_successors(cur_node) != 1 or \
                        type(cur_node.op) in REDUCTION_OP or \
                        cur_node in graph.results:
                    break
                else:
                    cur_node = graph.successors(cur_node)[0]
            if len(selected) > 1:
                explored.update(selected)
                fuses.append(list(selected))

        return self._fuse_nodes(fuses, TensorNeFuseChunk)
