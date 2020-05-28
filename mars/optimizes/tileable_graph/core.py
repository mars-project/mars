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

import weakref
from collections import defaultdict

from ...graph import DAG
from ...graph_builder import TileableGraphBuilder
from ...utils import copy_tileables, kernel_mode, enter_build_mode

_rules = defaultdict(list)

tileable_optimized = weakref.WeakKeyDictionary()


class TileableOptimizeRule(object):
    def __init__(self, optimized_context):
        self._optimizer_context = optimized_context

    def match(self, node):
        raise NotImplementedError

    def apply(self, node):
        raise NotImplementedError


class OptimizeContext(weakref.WeakKeyDictionary):
    def __init__(self, dict=None):
        weakref.WeakKeyDictionary.__init__(self, dict=dict)
        self._result_tileables = []

    @property
    def result_tileables(self):
        return self._result_tileables

    def append_result_tileables(self, tileables):
        self._result_tileables.extend(tileables)


class OptimizeIntegratedTileableGraphBuilder(TileableGraphBuilder):
    def __init__(self, **kw):
        self._optimizer_context = OptimizeContext()
        super().__init__(**kw)
        self._node_processor = self._apply_rules(self._node_processor, self._optimizer_context)

    @staticmethod
    def _apply_rules(node_processor, optimizer_context):
        def inner(node):
            node = node_processor(node) if node_processor is not None else node
            if node in tileable_optimized:
                node = tileable_optimized[node]
            elif len(node.inputs or []) > 0 and \
                    any(inp in tileable_optimized for inp in node.inputs):
                new_inputs = []
                for inp in node.inputs:
                    if inp in tileable_optimized:
                        new_inputs.append(tileable_optimized[inp])
                    else:
                        new_inputs.append(inp)
                node.inputs = new_inputs
            elif type(node.op) in _rules:
                for rule in _rules[type(node.op)]:
                    ruler = rule(optimizer_context)
                    if ruler.match(node):
                        node = rule(optimizer_context).apply(node)
            return node

        return inner

    def _mapping_tileables(self, tileables):
        for t in tileables:
            if t in self._optimizer_context:
                tileable_optimized[t] = self._optimizer_context[t]

    def _replace_copied_tilebale(self, graph):
        if len(self._optimizer_context) == 0:
            return graph

        new_graph = DAG()
        replaced_tileables = weakref.WeakKeyDictionary()
        for n in graph.topological_iter():
            if graph.count_predecessors(n) == 0:
                if n in self._optimizer_context and \
                        all(suc in self._optimizer_context for suc in graph.successors(n)):
                    replaced_tileables[n] = new_node = self._optimizer_context[n]
                else:
                    new_node = n
            elif any(inp in replaced_tileables for inp in n.inputs):
                new_inputs = [replaced_tileables.get(i, i) for i in n.inputs]
                new_tileables = copy_tileables(n.op.outputs, inputs=new_inputs)
                for t, new_t in zip(n.op.outputs, new_tileables):
                    replaced_tileables[t] = new_t.data
                    if t is n:
                        new_node = new_t.data
            else:
                new_node = n
            new_graph.add_node(new_node)
            for inp in new_node.inputs:
                new_graph.add_node(inp)
                new_graph.add_edge(inp, new_node)
        self._optimizer_context.update(replaced_tileables)
        return new_graph

    @kernel_mode
    @enter_build_mode
    def build(self, tileables, tileable_graph=None):
        self._optimizer_context.append_result_tileables(tileables)
        graph = super().build(tileables, tileable_graph=tileable_graph)
        graph = self._replace_copied_tilebale(graph)
        self._mapping_tileables(tileables)
        return graph


def register(op_type, rule):
    _rules[op_type].append(rule)
