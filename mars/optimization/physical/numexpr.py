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

import dataclasses
import functools
import logging
from typing import List, Set

import numpy as np

from ...core import ChunkType, ChunkGraph
from ...tensor import arithmetic
from ...tensor import reduction
from ...tensor.fuse import TensorNeFuseChunk
from ...tensor.fuse.numexpr import NUMEXPR_INSTALLED
from .core import RuntimeOptimizer, register_optimizer


logger = logging.getLogger(__name__)


REDUCTION = object()
REDUCTION_OP = {
    reduction.TensorSum,
    reduction.TensorProd,
    reduction.TensorMax,
    reduction.TensorMin,
}
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
    arithmetic.TensorFloor,
    arithmetic.TensorCeil,
    arithmetic.TensorAnd,
    arithmetic.TensorOr,
    arithmetic.TensorNot,
    reduction.TensorSum,
    reduction.TensorProd,
    reduction.TensorMax,
    reduction.TensorMin,
}


@dataclasses.dataclass
class _Fuse:
    graph: ChunkGraph
    heads: List[ChunkType]
    tails: List[ChunkType]


def _can_fuse(node: ChunkType):
    op = node.op
    op_type = type(op)
    if op_type in REDUCTION_OP:
        if len(op.axis) == 1 or len(op.axis) == node.ndim:
            return REDUCTION
        else:
            return False
    # return op_type in SUPPORT_OP
    if op_type not in SUPPORT_OP:
        return False
    if op_type in (arithmetic.TensorOr, arithmetic.TensorAnd):
        # numexpr only support logical and or:
        # https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/user_guide.html#supported-operators
        if np.isscalar(op.lhs) or np.isscalar(op.rhs):
            return False
    return True


def _collect_fuse(
    graph: ChunkGraph,
    node: ChunkType,
    graph_results: Set[ChunkType],
    cached_can_fuse,
):
    fuse_graph = ChunkGraph()
    fuse_graph.add_node(node)
    fuse_heads = []
    fuse_tails = []
    tail_reduction_node = None

    stack = [node]
    # Do a full search of sub graph even the fuse tails > 1
    while len(stack) != 0:
        node = stack.pop()
        is_head = graph.count_predecessors(node) == 0
        for n in graph.iter_predecessors(node):
            can_fuse = cached_can_fuse(n)
            if can_fuse is False or can_fuse is REDUCTION:
                is_head = True
            elif not fuse_graph.contains(n):
                stack.append(n)
                fuse_graph.add_node(n)
            else:
                fuse_graph.add_edge(n, node)
        if is_head:
            fuse_heads.append(node)
        # Skip the successors of tail reduction node.
        if node is tail_reduction_node:
            continue
        is_tail = graph.count_successors(node) == 0 or node in graph_results
        for n in graph.iter_successors(node):
            can_fuse = cached_can_fuse(n)
            if can_fuse is False:
                is_tail = True
            elif can_fuse is REDUCTION:
                if tail_reduction_node is None:
                    tail_reduction_node = n
                    fuse_tails.append(n)
                    stack.append(n)
                    fuse_graph.add_node(n)
                elif n is tail_reduction_node:
                    fuse_graph.add_edge(node, n)
                else:
                    is_tail = True
            elif not fuse_graph.contains(n):
                stack.append(n)
                fuse_graph.add_node(n)
            else:
                fuse_graph.add_edge(node, n)
        if is_tail:
            fuse_tails.append(node)

    return _Fuse(fuse_graph, fuse_heads, fuse_tails)


@register_optimizer
class NumexprRuntimeOptimizer(RuntimeOptimizer):
    engine = "numexpr"

    @classmethod
    def is_available(cls) -> bool:
        return NUMEXPR_INSTALLED

    def optimize(self):
        fuses = []
        explored = set()
        cached_can_fuse = functools.lru_cache(maxsize=None)(_can_fuse)

        graph = self._graph
        graph_results = set(graph.results)
        for node in graph.topological_iter():
            if node.op.gpu or node.op.sparse:
                # break
                return [], []
            if node in explored or node in graph_results:
                continue
            can_fuse = cached_can_fuse(node)
            if can_fuse is True:
                fuse = _collect_fuse(graph, node, graph_results, cached_can_fuse)
                if len(fuse.graph) > 1:
                    explored.update(fuse.graph)
                    if len(fuse.tails) == 1:
                        fuses.append(fuse)
                    else:
                        logger.info(
                            "Refused fusing for numexpr because the tail node count > 1."
                        )

        return self._fuse_nodes(fuses, TensorNeFuseChunk)

    def _fuse_nodes(self, fuses: List[_Fuse], fuse_cls):
        graph = self._graph
        fused_nodes = []

        for fuse in fuses:
            fuse_graph = fuse.graph
            tail_nodes = fuse.tails
            head_nodes = fuse.heads
            inputs = [
                inp for n in head_nodes for inp in n.inputs if inp not in fuse_graph
            ]

            tail_chunk = tail_nodes[0]
            tail_chunk_op = tail_chunk.op
            fuse_op = fuse_cls(
                sparse=tail_chunk_op.sparse,
                gpu=tail_chunk_op.gpu,
                _key=tail_chunk_op.key,
                fuse_graph=fuse_graph,
                dtype=tail_chunk.dtype,
            )
            fused_chunk = fuse_op.new_chunk(
                inputs,
                kws=[tail_chunk.params],
                _key=tail_chunk.key,
                _chunk=tail_chunk,
            ).data

            graph.add_node(fused_chunk)
            for node in graph.iter_successors(tail_chunk):
                graph.add_edge(fused_chunk, node)
            for head_chunk in head_nodes:
                for node in graph.iter_predecessors(head_chunk):
                    if not fuse_graph.contains(node):
                        graph.add_edge(node, fused_chunk)
            for node in fuse_graph:
                graph.remove_node(node)
            fused_nodes.append(fused_chunk)

            try:
                # check tail node if it's in results
                i = graph.results.index(tail_chunk)
                graph.results[i] = fused_chunk
            except ValueError:
                pass

        return fuses, fused_nodes
