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

from .graph import DirectedGraph
from .utils import kernel_mode


class Tileable(object):
    __slots__ = ()

    def is_coarse(self):
        raise NotImplementedError

    def copy_from(self, other):
        raise NotImplementedError

    def to_graph(self, graph=None, graph_cls=DirectedGraph, substitutes=None):
        visited = set()
        graph = graph_cls() if graph is None else graph
        substitutes = substitutes or dict()

        if self in substitutes:
            graph.add_node(substitutes[self])
            return graph

        q = [self]
        while len(q) > 0:
            obj = q.pop()
            if obj in visited:
                continue

            if obj not in graph:
                graph.add_node(obj)
            for input_obj in obj.inputs or []:
                sub = False
                if input_obj in substitutes:
                    input_obj = substitutes[input_obj]
                    sub = True
                in_graph = True
                if input_obj not in graph:
                    graph.add_node(input_obj)
                    in_graph = False
                graph.add_edge(input_obj, obj)
                if not sub and not in_graph:
                    q.append(input_obj)

            visited.add(obj)

        return graph

    def _repr_svg_(self):
        return self.to_graph()._repr_svg_()


class TilesError(Exception):
    pass


class DataNotReady(TilesError):
    pass


class NotSupportTile(Exception):
    pass


class OperandTilesHandler(object):
    def __init__(self):
        self._handlers = {}

    @classmethod
    def _get_op_cls(cls, op):
        if isinstance(op, type):
            return op
        return type(op)

    @classmethod
    def _assign_to(cls, tile_after_tensor_datas, tile_before_tensor_datas):
        assert len(tile_after_tensor_datas) == len(tile_before_tensor_datas)

        for tile_after_tensor_data, tile_before_tensor_data in \
                zip(tile_after_tensor_datas, tile_before_tensor_datas):
            if tile_before_tensor_data is None:
                # garbage collected
                continue
            tile_after_tensor_data.copy_to(tile_before_tensor_data)
            tile_before_tensor_data.op.outputs = tile_before_tensor_datas

    def register(self, op, handler):
        self._handlers[self._get_op_cls(op)] = handler

    @kernel_mode
    def _dispatch(self, op):
        op_cls = self._get_op_cls(op)
        try:
            handler = self._handlers[op_cls]
            return handler(op)
        except KeyError as e:
            if hasattr(op_cls, 'tile'):
                # has tile implementation
                return op_cls.tile(op)
            for op_clz in self._handlers.keys():
                if issubclass(op_cls, op_clz):
                    self._handlers[op_cls] = self._handlers[op_clz]
                    return self._handlers[op_cls](op)

            raise e

    def dispatch(self, to_tiles):
        return self._dispatch(to_tiles.op)

    def single_tiles(self, to_tiles):
        if to_tiles.is_coarse() and to_tiles.op:
            dispatched = self._dispatch(to_tiles.op)
            self._assign_to([d.data for d in dispatched], to_tiles.op.outputs)

        return to_tiles

    def tiles(self, tiles_obj):
        graph = DirectedGraph()
        visited = {id(tiles_obj)}
        loose_requires = set()
        q = deque([tiles_obj])

        while q:
            to_tiles = q.popleft()
            if to_tiles not in graph:
                graph.add_node(to_tiles)
            if getattr(to_tiles.op, '_loose_require', False):
                loose_requires.add(to_tiles)
            objs = to_tiles.inputs or []
            for o in objs:
                if not isinstance(o, Tileable):
                    continue
                if o not in graph:
                    graph.add_node(o)
                graph.add_edge(o, to_tiles)

                if id(o) in visited:
                    continue
                visited.add(id(o))

                q.append(o)

        visited = set()
        q = deque(graph.iter_indep())

        while q:
            node = q.popleft()
            if node in visited:
                continue
            preds = graph.predecessors(node)
            if node in loose_requires:
                accessible = any(pred in visited for pred in preds)
            else:
                accessible = all(pred in visited for pred in preds)
            if not preds or accessible:
                if node.is_coarse() and node.op:
                    tiled = self._dispatch(node.op)
                    self._assign_to([t.data for t in tiled], node.op.outputs)
                visited.add(node)
                q.extend(n for n in graph[node] if n not in visited)
            else:
                q.append(node)
                q.extend(n for n in preds if n not in visited)

        for to_tiles in loose_requires:
            tiled = self._dispatch(to_tiles.op)
            self._assign_to(tiled, to_tiles.op.outputs)

        return tiles_obj


handler = OperandTilesHandler()


def register(op, func):
    handler.register(op, func)
