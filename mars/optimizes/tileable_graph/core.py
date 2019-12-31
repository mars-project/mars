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
            if type(node.op) in _rules:
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
        reversed_mapping = weakref.WeakKeyDictionary((v, k) for k, v in self._optimizer_context.items())
        for n in graph.topological_iter():
            if n in reversed_mapping:
                new_node = n
            elif any(inp in self._optimizer_context for inp in n.inputs):
                new_inputs = [self._optimizer_context.get(i, i) for i in n.inputs]
                new_tileables = copy_tileables(n.op.outputs, inputs=new_inputs)
                for t, new_t in zip(n.op.outputs, new_tileables):
                    self._optimizer_context[t] = new_t.data
                    if t is n:
                        new_node = new_t.data
            else:
                new_node = n
            new_graph.add_node(new_node)
            for inp in new_node.inputs:
                new_graph.add_node(inp)
                new_graph.add_edge(inp, new_node)
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
