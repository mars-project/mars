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

import sys
import weakref

from .graph import DAG
from .graph_builder import GraphBuilder, TileableGraphBuilder
from .config import options
from .utils import kernel_mode, enter_build_mode, copy_tileables


class Tileable(object):
    __slots__ = ()

    @property
    def op(self):
        raise NotImplementedError

    @property
    def inputs(self):
        raise NotImplementedError

    @property
    def chunks(self):
        raise NotImplementedError

    def is_coarse(self):
        raise NotImplementedError

    def copy_from(self, other):
        raise NotImplementedError

    def tiles(self):
        return handler.tiles(self)

    def _inplace_tile(self):
        return handler.inplace_tile(self)

    @kernel_mode
    def build_graph(self, graph=None, cls=DAG, tiled=False, compose=True,
                    **build_chunk_graph_kwargs):
        tileable_graph = graph if not tiled else None
        tileable_graph_builder = TileableGraphBuilder(graph=tileable_graph, graph_cls=cls)
        tileable_graph = tileable_graph_builder.build([self])
        if not tiled:
            return tileable_graph
        chunk_graph_builder = ChunkGraphBuilder(
            graph=graph, graph_cls=cls, compose=compose,
            **build_chunk_graph_kwargs)
        return chunk_graph_builder.build([self], tileable_graph=tileable_graph)

    def visualize(self, graph_attrs=None, node_attrs=None, **kw):
        from graphviz import Source

        g = self.build_graph(**kw)
        dot = g.to_dot(graph_attrs=graph_attrs, node_attrs=node_attrs,
                       result_chunk_keys={c.key for c in self.chunks})

        return Source(dot)


class TilesError(Exception):
    pass


class NotSupportTile(Exception):
    pass


class OperandTilesHandler(object):
    _handlers = {}

    @classmethod
    def _get_op_cls(cls, op):
        if isinstance(op, type):
            return op
        return type(op)

    @classmethod
    def register(cls, op, tile_handler):
        cls._handlers[cls._get_op_cls(op)] = tile_handler

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

    @kernel_mode
    def dispatch(self, op):
        op_cls = self._get_op_cls(op)
        if op_cls in self._handlers:
            return self._handlers[op_cls](op)
        try:
            return op_cls.tile(op)
        except NotImplementedError as ex:
            cause = ex
            for registered_op_cls in self._handlers:
                if issubclass(op_cls, registered_op_cls):
                    self._handlers[op_cls] = self._handlers[registered_op_cls]
                    return self._handlers[op_cls][op]

        raise NotImplementedError('{} does not support tile'.format(type(op))) from cause

    def inplace_tile(self, to_tile):
        if not to_tile.is_coarse():
            return to_tile
        dispatched = self.dispatch(to_tile.op)
        self._assign_to([d.data for d in dispatched], to_tile.op.outputs)
        return to_tile

    @classmethod
    def tiles(cls, to_tile):
        to_tile.build_graph(tiled=True, compose=False)
        return get_tiled(to_tile)


handler = OperandTilesHandler()
register = OperandTilesHandler.register

_tileable_data_to_tiled = weakref.WeakKeyDictionary()
_op_to_copied = weakref.WeakKeyDictionary()


@enter_build_mode
def get_tiled(tileable, mapping=None, raise_err_if_not_tiled=True):
    tileable_data = tileable.data if hasattr(tileable, 'data') else tileable
    if mapping:
        tileable_data = mapping.get(tileable_data, tileable_data)
    if raise_err_if_not_tiled:
        return _tileable_data_to_tiled[tileable_data]
    else:
        return _tileable_data_to_tiled.get(tileable_data)


class ChunkGraphBuilder(GraphBuilder):
    def __init__(self, graph=None, graph_cls=DAG, node_processor=None,
                 inputs_selector=None, compose=True,
                 on_tile=None, on_tile_success=None, on_tile_failure=None):
        super().__init__(graph=graph, graph_cls=graph_cls, node_processor=node_processor,
                         inputs_selector=inputs_selector)
        self._compose = compose
        self._on_tile = on_tile
        self._on_tile_success = on_tile_success
        self._on_tile_failure = on_tile_failure

    @property
    def is_compose(self):
        return self._compose

    def _tile(self, tileable_data, tileable_graph):
        cache = _tileable_data_to_tiled
        on_tile = self._on_tile

        if tileable_data in cache:
            return [cache[o] for o in tileable_data.op.outputs]

        # copy tileable
        if tileable_data.op in _op_to_copied:
            tds = _op_to_copied[tileable_data.op]
        else:
            tds = copy_tileables(tileable_data.op.outputs,
                                 inputs=[cache[inp] for inp in tileable_data.inputs],
                                 copy_key=True, copy_id=False)
            _op_to_copied[tileable_data.op] = tds
        if not tileable_data.is_coarse():
            # the tileable is already tiled
            # could only happen when executor.execute_tileable(tileable.tiles())
            for o, t in zip(tileable_data.op.outputs, tds):
                t._chunks = o.chunks
                t._nsplits = o.nsplits
        elif on_tile is None:
            tds[0]._inplace_tile()
        else:
            tds = on_tile(tileable_data.op.outputs, tds)
            if not isinstance(tds, (list, tuple)):
                tds = [tds]
            assert len(tileable_data.op.outputs) == len(tds)
        for t, td in zip(tileable_data.op.outputs, tds):
            cache[t] = td.data if hasattr(td, 'data') else td
        return tds

    def _get_tileable_data_graph(self, tileables, tileable_graph):
        from .optimizes.tileable_graph import OptimizeIntegratedTileableGraphBuilder

        if tileable_graph is None:
            # if tileable_data graph not provided
            # create a new one via GraphBuilder
            if options.optimize_tileable_graph:
                builder_cls = OptimizeIntegratedTileableGraphBuilder
            else:
                builder_cls = TileableGraphBuilder
            tileable_graph_builder = builder_cls(
                graph_cls=type(self._graph),
                node_processor=self._node_processor)
            tileable_graph = tileable_graph_builder.build(tileables)
        return tileable_graph

    @kernel_mode
    @enter_build_mode
    def build(self, tileables, tileable_graph=None):
        tileable_graph = self._get_tileable_data_graph(tileables, tileable_graph)

        # do tiles and add nodes or edges to chunk graph
        tileables_set = set(tileables)
        keys = []
        visited = set()
        tiled_op = set()
        for tileable_data in tileable_graph.topological_iter():
            nodes = []
            # do tiling
            if tileable_data.op in tiled_op:
                continue
            try:
                tiled = self._tile(tileable_data, tileable_graph)
                tiled_op.add(tileable_data.op)
                for t, td in zip(tileable_data.op.outputs, tiled):
                    if self._on_tile_success is not None:
                        td = self._on_tile_success(t, td)
                        if td is None:
                            # if return None after calling `on_tile_success`,
                            # the chunks will not be added into chunk graph any more
                            continue
                    nodes.extend(c.data for c in td.chunks)
                    if t in tileables_set:
                        keys.extend(c.key for c in td.chunks)
                    self._add_nodes(nodes, visited)
            except:  # noqa: E722
                exc_info = sys.exc_info()
                if self._on_tile_failure:
                    # partial tiled chunks can be returned
                    # here they will be added to the chunk graph
                    # for further execution
                    partial_tiled_chunks = \
                        self._on_tile_failure(tileable_data.op, exc_info)
                    if partial_tiled_chunks is not None and \
                            len(partial_tiled_chunks) > 0:
                        self._add_nodes(partial_tiled_chunks, visited)
                    tiled_op.add(tileable_data.op)
                else:
                    raise
        if self._compose:
            self._graph.compose(keys=keys)
        return self._graph


class IterativeChunkGraphBuilder(ChunkGraphBuilder):
    def __init__(self, graph=None, graph_cls=DAG, node_processor=None, inputs_selector=None,
                 compose=True, on_tile=None, on_tile_success=None, on_tile_failure=None):
        self._interrupted_ops = set()
        self._prev_tileable_graph = None
        self._cur_tileable_graph = None
        self._iterative_chunk_graphs = []
        self._done = False
        super().__init__(
            graph=graph, graph_cls=graph_cls, node_processor=node_processor,
            inputs_selector=inputs_selector, compose=compose, on_tile=on_tile,
            on_tile_success=self._wrap_on_tile_success(on_tile_success),
            on_tile_failure=self._wrap_on_tile_failure(on_tile_failure))
        if self._graph_cls is None:
            self._graph_cls = type(self._graph)

    def _wrap_on_tile_failure(self, on_tile_failure):
        def inner(op, exc_info):
            if isinstance(exc_info[1], TilesError):
                self._interrupted_ops.add(op)
                partial_tiled_chunks = getattr(exc_info[1], 'partial_tiled_chunks', None)
                if partial_tiled_chunks is not None:
                    return partial_tiled_chunks
            else:
                if on_tile_failure is not None:
                    on_tile_failure(op, exc_info)
                else:
                    raise exc_info[1].with_traceback(exc_info[2]) from None
        return inner

    def _wrap_on_tile_success(self, on_tile_success):
        def inner(tile_before, tile_after):
            # if tile succeed, add the node before tiling
            # to current iterative tileable graph
            if on_tile_success is not None:
                tile_after = on_tile_success(tile_before, tile_after)
            iterative_tileable_graph = self._cur_tileable_graph
            iterative_tileable_graph.add_node(tile_before)
            for inp in self._prev_tileable_graph.iter_predecessors(tile_before):
                if inp in iterative_tileable_graph:
                    iterative_tileable_graph.add_edge(inp, tile_before)
            return tile_after
        return inner

    @property
    def interrupted_ops(self):
        return self._interrupted_ops

    @property
    def prev_tileable_graph(self):
        return self._prev_tileable_graph

    @property
    def iterative_chunk_graphs(self):
        return self._iterative_chunk_graphs

    @property
    def done(self):
        return self._done

    def _tile(self, tileable_data, tileable_graph):
        if any(inp.op in self._interrupted_ops for inp in tileable_data.inputs):
            raise TilesError('Tile fail due to failure of inputs')
        return super()._tile(tileable_data, tileable_graph)

    @kernel_mode
    @enter_build_mode
    def build(self, tileables, tileable_graph=None):
        tileable_graph = self._get_tileable_data_graph(tileables, tileable_graph)
        self._graph = self._graph_cls()
        self._interrupted_ops.clear()
        self._prev_tileable_graph = tileable_graph
        self._cur_tileable_graph = type(tileable_graph)()

        chunk_graph = super().build(
            tileables, tileable_graph=tileable_graph)
        self._iterative_chunk_graphs.append(chunk_graph)
        if len(self._interrupted_ops) == 0:
            self._done = True
        self._prev_tileable_graph = self._cur_tileable_graph
        self._cur_tileable_graph = None
        return chunk_graph
