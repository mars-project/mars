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

from ...expressions import arithmetic
from ...expressions.fuse import TensorCpFuseChunk


CP_ELEMENTWISE_OP = {
    arithmetic.TensorSubtract,
    arithmetic.TensorSubConstant,
    arithmetic.TensorMultiply,
    arithmetic.TensorMulConstant,
    arithmetic.TensorTrueDiv,
    arithmetic.TensorSqrt
}
CP_OP = CP_ELEMENTWISE_OP


class CpOptimizer(object):
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

            op = TensorCpFuseChunk(dtype=tail_node.dtype)
            composed_chunk = op(c)
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
            if type(v.op) not in CP_OP:
                continue
            if v in explored:
                continue
            if graph.count_successors(v) != 1:
                continue
            if v.key in keys:
                continue
            selected = [v]
            # add successors
            cur_node = graph.successors(v)[0]
            while graph.count_predecessors(cur_node) == 1\
                    and type(cur_node.op) in CP_OP \
                    and cur_node.key not in keys:
                selected.append(cur_node)
                if graph.count_successors(cur_node) != 1:
                    break
                else:
                    cur_node = graph.successors(cur_node)[0]
            if len(selected) > 1:
                explored.update(selected)
                composes.append(list(selected))
        return self._compose_graph(composes)
