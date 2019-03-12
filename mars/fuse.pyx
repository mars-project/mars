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

from collections import deque

from .operands import Fuse


class InvalidComposedNodeError(Exception):
    pass


class Fusion(object):
    def __init__(self, graph):
        self._graph = graph

    @property
    def graph(self):
        return self._graph

    def _compose_graph(self, composes):
        cdef:
            list composed_nodes = []

        from .tensor.expressions.fuse.core import TensorFuseChunk
        composed_nodes = []

        for c in composes:
            head_node = c[0]
            tail_node = c[-1]
            fuse_op = TensorFuseChunk(dtype=tail_node.dtype, sparse=tail_node.op.sparse, _key=tail_node.op.key)
            fuse_chunk = fuse_op.new_chunk(head_node.inputs, tail_node.shape,
                                           index=tail_node.index, _key=tail_node.key,
                                           _composed=c)
            self._graph.add_node(fuse_chunk)
            for node in self._graph.iter_successors(tail_node):
                self._graph.add_edge(fuse_chunk, node)
                # change inputs
                node.inputs = [i if i is not tail_node else fuse_chunk for i in node.inputs]
            for node in self._graph.iter_predecessors(head_node):
                self._graph.add_edge(node, fuse_chunk)
            # TODO:judge compose is independent?
            for node in c:
                self._graph.remove_node(node)
            composed_nodes.append(fuse_chunk)

        return composed_nodes

    def compose(self, list keys=None):
        def _visit_predicate(n, visited):
            cond = any if getattr(n.op, '_loose_require', False) else all
            preds = self._graph.predecessors(n)
            return not preds or cond(pred in visited for pred in preds)

        composes = []
        explored = set()
        # for those chunk in result sets, we should not do any fuse
        keys_set = set(keys or [])

        for v in self._graph.bfs(visit_predicate=_visit_predicate):
            if v in explored or v.key in keys_set:
                continue
            if self._graph.count_successors(v) != 1:
                continue
            if len(v.op.outputs) != 1:
                continue
            selected = [v]
            # add successors
            cur_node = self._graph.successors(v)[0]
            while self._graph.count_predecessors(cur_node) == 1:
                selected.append(cur_node)
                if self._graph.count_successors(cur_node) != 1:
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

        observed = set()

        composed_nodes = node.composed
        nodes_set = set(composed_nodes)
        tail_node = composed_nodes[-1]
        self._graph.add_node(tail_node)
        observed.add(tail_node)
        q = deque()
        q.append(tail_node)
        while len(q) > 0:
            cur_node = q.pop()
            if not cur_node.inputs:
                continue
            for pre in cur_node.inputs:
                pre = get_node(pre)
                if pre in nodes_set:
                    self._graph.add_node(pre)
                if pre in self._graph:
                    self._graph.add_edge(pre, cur_node)
                if pre in nodes_set:
                    q.appendleft(pre)
                    observed.add(pre)
        if len(observed) != len(nodes_set):
            raise InvalidComposedNodeError("Invalid composed node data")
        for n in self._graph.iter_successors(node):
            self._graph.add_edge(composed_nodes[-1], n)
        self._graph.remove_node(node)

    def decompose(self, nodes=None):
        if nodes is None:
            nodes = list(self._graph.traverse())
        for v in nodes:
            if isinstance(v.op, Fuse):
                self._decompose_node(v)
