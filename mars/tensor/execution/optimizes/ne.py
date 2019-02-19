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

from ...expressions import arithmetic, reduction
from ...expressions.fuse import TensorNeFuseChunk

REDUCTION_OP = {reduction.TensorSum, reduction.TensorProd,
                reduction.TensorMax, reduction.TensorMin}
SUPPORT_OP = {
    arithmetic.TensorAdd, arithmetic.TensorAddConstant,
    arithmetic.TensorSubtract, arithmetic.TensorSubConstant,
    arithmetic.TensorMultiply, arithmetic.TensorMulConstant,
    arithmetic.TensorDivide, arithmetic.TensorDivConstant,
    arithmetic.TensorPower, arithmetic.TensorPowConstant,
    arithmetic.TensorMod, arithmetic.TensorModConstant,
    arithmetic.TensorNegative,
    arithmetic.TensorAbs,
    arithmetic.TensorConj,
    arithmetic.TensorExp,
    arithmetic.TensorLog,
    arithmetic.TensorLog10,
    arithmetic.TensorExpm1,
    arithmetic.TensorLog1p,
    arithmetic.TensorSqrt,

    arithmetic.TensorEqual, arithmetic.TensorEqConstant,
    arithmetic.TensorNotEqual, arithmetic.TensorNeConstant,
    arithmetic.TensorLessThan, arithmetic.TensorLtConstant,
    arithmetic.TensorLessEqual, arithmetic.TensorLeConstant,
    arithmetic.TensorGreaterThan, arithmetic.TensorGtConstant,
    arithmetic.TensorGreaterEqual, arithmetic.TensorGeConstant,

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

    arithmetic.TensorLshift, arithmetic.TensorLshiftConstant,
    arithmetic.TensorRshift, arithmetic.TensorRshiftConstant,

    arithmetic.TensorTreeAdd,
    arithmetic.TensorTreeMultiply,

    reduction.TensorSum,
    reduction.TensorProd,
    reduction.TensorMax,
    reduction.TensorMin
}


def _check_reduction_axis(node):
    return len(node.op.axis) == 1 or len(node.op.axis) == node.ndim


def _support(node):
    op_type = type(node.op)
    if op_type in REDUCTION_OP:
        return _check_reduction_axis(node)
    return op_type in SUPPORT_OP


def _transfer_op(node):
    op = node.op
    if type(op) in REDUCTION_OP and not _check_reduction_axis(node):
        return op
    return op


class NeOptimizer(object):
    def __init__(self, graph):
        self._graph = graph

    def optimize(self, keys=None):
        self.compose(keys=keys)

    def _compose_graph(self, composes):
        graph = self._graph
        composed_nodes = []

        for c in composes:
            head_node = c[0]
            tail_node = c[-1]

            op = TensorNeFuseChunk(dtype=tail_node.dtype)
            composed_chunk = op(c).data
            graph.add_node(composed_chunk)
            for node in graph.iter_successors(tail_node):
                graph.add_edge(composed_chunk, node)
            for node in graph.iter_predecessors(head_node):
                graph.add_edge(node, composed_chunk)
            for node in c:
                graph.remove_node(node)
            composed_nodes.append(composed_chunk)

        return composed_nodes

    def compose(self, keys=None):
        composes = []
        explored = set()
        keys = set(keys or [])

        graph = self._graph
        for v in graph.bfs():
            if v.op.gpu or v.op.sparse:
                # break out
                return []
            if type(v.op) not in SUPPORT_OP or v.key in keys:
                continue
            if v in explored or type(v.op) in REDUCTION_OP:  # TODO: check logic here
                continue
            if graph.count_successors(v) != 1:
                continue
            selected = [v]
            # add successors
            cur_node = graph.successors(v)[0]
            while graph.count_predecessors(cur_node) == 1\
                    and _support(cur_node) and cur_node.key not in keys:
                selected.append(cur_node)
                if graph.count_successors(cur_node) != 1 \
                        or type(cur_node.op) in REDUCTION_OP:
                    break
                else:
                    cur_node = graph.successors(cur_node)[0]
            if len(selected) > 1:
                explored.update(selected)
                composes.append(list(selected))
        return self._compose_graph(composes)

