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

from ...config import options
from ...graph import DAG
from ...graph_builder import TileableGraphBuilder

_rules = defaultdict(list)
_reuslt_tileable_mapping = weakref.WeakKeyDictionary()


class TileableOptimizeRule(object):
    def __init__(self, optimized_context):
        self._optimizer_context = optimized_context

    def _match(self, node):
        raise NotImplementedError

    def _process(self, node):
        raise NotImplementedError

    def apply(self, node):
        if self._match(node):
            return self._process(node)
        else:
            return node


def apply_rules(optimize_context):
    def inner(node):
        if type(node.op) in _rules:
            for rule in _rules[type(node.op)]:
                node = rule(optimize_context).apply(node)
        return node
    return inner


class OptimizedTileableGraphBuilder(TileableGraphBuilder):
    def __init__(self, **kw):
        self._optimizer_context = weakref.WeakKeyDictionary()
        kw.pop('node_processor', None)
        super(OptimizedTileableGraphBuilder, self).__init__(
            node_processor=apply_rules(self._optimizer_context), **kw)

    def get_before_optimized_tileable(self, tileable):
        if tileable in self._optimizer_context:
            return self._optimizer_context[tileable]
        else:
            return tileable

    def _mapping_tileables(self, tileables):
        for t in tileables:
            if t in self._optimizer_context:
                _reuslt_tileable_mapping[t] = self._optimizer_context[t]

    def _replace_copied_tilebale(self, graph):
        if len(self._optimizer_context) == 0:
            return graph
        else:
            new_graph = DAG()
            for n in graph.topological_iter():
                if any(inp in self._optimizer_context for inp in n.inputs) and \
                        n not in self._optimizer_context.values():
                    new_inputs = [
                        self._optimizer_context[i] if i in self._optimizer_context else i for i in n.inputs]
                    params = []
                    op = n.op.copy().reset_key()
                    for o in n.op.outputs:
                        p = o.params.copy()
                        p['_key'] = o.key
                        p.update(o.extra_params)
                        params.append(p)
                    tds = op.new_tileables(new_inputs, kws=params, output_limit=len(params))
                    for t, new_t in zip(n.op.outputs, tds):
                        self._optimizer_context[t] = new_t.data
                    new_node = tds[n.op.outputs.index(n)].data
                else:
                    new_node = n
                new_graph.add_node(new_node)
                for inp in new_node.inputs:
                    new_graph.add_edge(inp, new_node)
            return new_graph

    def build(self, tileables, tileable_graph=None):
        graph = super(OptimizedTileableGraphBuilder, self).build(tileables, tileable_graph=tileable_graph)
        graph = self._replace_copied_tilebale(graph)
        self._mapping_tileables(tileables)
        return graph


def get_tileable_mapping():
    if options.tileable.optimize:
        return _reuslt_tileable_mapping
    else:
        return None


def register(op_type, rule):
    _rules[op_type].append(rule)
