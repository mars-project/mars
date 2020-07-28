#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

from collections import deque

from ...operands import Fuse, VirtualOperand, Fetch


class Fusion:
    def __init__(self, graph):
        self._graph = graph

    @property
    def graph(self):
        return self._graph

    def _compose_graph(self, composes):
        from ...utils import build_fuse_chunk
        composed_nodes = []

        for c in composes:
            head_node = c[0]
            tail_node = c[-1]
            fuse_chunk = build_fuse_chunk(c)
            self._graph.add_node(fuse_chunk)
            for node in self._graph.iter_successors(tail_node):
                self._graph.add_edge(fuse_chunk, node)
                # replace inputs
                node_inputs = node.inputs
                new_node_inputs = []
                for inp in node_inputs:
                    if inp is tail_node:
                        new_node_inputs.append(fuse_chunk)
                    else:
                        new_node_inputs.append(inp)
                node.inputs = new_node_inputs
            for node in self._graph.iter_predecessors(head_node):
                self._graph.add_edge(node, fuse_chunk)
            # TODO:judge compose is independent?
            for node in c:
                self._graph.remove_node(node)
            composed_nodes.append(fuse_chunk)

        return composed_nodes

    def compose(self, keys=None):
        composes = []
        explored = set()
        # for those chunk in result sets, we should not do any fuse
        keys_set = set(keys or [])

        for v in self._graph.topological_iter():
            if v in explored or v.key in keys_set:
                continue
            if self._graph.count_successors(v) != 1:
                continue
            if len(v.op.outputs) != 1:
                continue
            if isinstance(v.op, (VirtualOperand, Fetch)):
                # cannot fuse virtual operand or fetch
                continue
            if v.op.expect_worker is not None:
                # don't fuse operand that has explicit worker assignment
                continue
            selected = [v]
            # add successors
            cur_node = self._graph.successors(v)[0]
            while self._graph.count_predecessors(cur_node) == 1 and \
                    not isinstance(cur_node.op, (VirtualOperand, Fetch)):
                selected.append(cur_node)
                if self._graph.count_successors(cur_node) != 1 or \
                        cur_node.key in keys_set:
                    break
                else:
                    cur_node = self._graph.successors(cur_node)[0]
            if len(selected) > 1:
                explored.update(selected)
                composes.append(list(selected))
        return self._compose_graph(composes)

    def _decompose_node(self, node):
        def get_node(n):
            if n.composed:
                return n.composed[-1]
            return n

        composed_nodes = node.composed
        nodes_set = set(composed_nodes)
        tail_node = composed_nodes[-1]
        self._graph.add_node(tail_node)
        q = deque()
        q.append(tail_node)
        while len(q) > 0:
            cur_node = q.pop()
            if not cur_node.inputs:
                continue
            inputs = cur_node.inputs
            for pre in inputs:
                pre = get_node(pre)
                if pre in nodes_set:
                    self._graph.add_node(pre)
                if pre in self._graph:
                    self._graph.add_edge(pre, cur_node)
                if pre in nodes_set:
                    q.appendleft(pre)
        for n in self._graph.iter_successors(node):
            self._graph.add_edge(composed_nodes[-1], n)
            # replace inputs for successors
            n_inputs = n.inputs
            new_n_inputs = []
            node_data = getattr(node, 'data') if hasattr(node, 'data') else node
            for inp in n_inputs:
                if inp is node_data:
                    new_n_inputs.append(composed_nodes[-1])
                else:
                    new_n_inputs.append(inp)
            # replace inputs first
            n.inputs = new_n_inputs
            # if the successor is a fuse,
            # replace the first compose node as well
            if isinstance(n.op, Fuse):
                n.composed[0].inputs = new_n_inputs
        self._graph.remove_node(node)

    def decompose(self, nodes=None):
        if nodes is None:
            nodes = list(self._graph.traverse())
        for v in nodes:
            if isinstance(v.op, Fuse):
                self._decompose_node(v)
