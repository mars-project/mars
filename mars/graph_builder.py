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

import itertools

from .graph import DAG
from .utils import kernel_mode, enter_build_mode


class GraphBuilder(object):
    def __init__(self, graph=None, graph_cls=DAG, node_processor=None,
                 inputs_selector=None):
        self._graph_cls = graph_cls
        if graph is not None:
            self._graph = graph
        else:
            self._graph = graph_cls()
        self._node_processor = node_processor
        if inputs_selector is None:
            inputs_selector = lambda x: x
        self._inputs_selector = inputs_selector

    def _add_nodes(self, nodes, visited):
        graph = self._graph
        visited.update(nodes)

        while len(nodes) > 0:
            node = nodes.pop()
            if self._node_processor:
                # if node processor registered, process the node first
                node = self._node_processor(node)

            visited.add(node)
            if not graph.contains(node):
                graph.add_node(node)
            children = self._inputs_selector(node.inputs or [])
            for c in children:
                if self._node_processor:
                    c = self._node_processor(c)
                if not graph.contains(c):
                    graph.add_node(c)
                if not graph.has_successor(c, node):
                    graph.add_edge(c, node)
                for n in c.op.outputs:
                    if n not in visited:
                        nodes.append(n)

    def build(self, tileables, tileable_graph=None):
        raise NotImplementedError


class TileableGraphBuilder(GraphBuilder):
    def __init__(self, graph=None, graph_cls=DAG, node_processor=None,
                 inputs_selector=None):
        super().__init__(graph=graph, graph_cls=graph_cls,
                         node_processor=node_processor,
                         inputs_selector=inputs_selector)

    @kernel_mode
    @enter_build_mode
    def build(self, tileables, tileable_graph=None):
        if tileable_graph is not None:  # pragma: no cover
            return tileable_graph

        visited = set()
        nodes = list(itertools.chain(
            *(tileable.op.outputs for tileable in tileables)))
        self._add_nodes(nodes, visited)
        return self._graph
