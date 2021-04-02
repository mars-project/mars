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
import sys
import weakref
from abc import ABC, abstractmethod
from typing import Callable, List, Set, Union, Tuple, Iterable

from ...utils import enter_mode, copy_tileables
from ..typing import OperandType, TileableType, EntityType
from ..entity import TilesError
from .entity import EntityGraph, TileableGraph, ChunkGraph


def _default_inputs_selector(inputs: List[EntityType]) -> List[EntityType]:
    return inputs


class AbstractGraphBuilder(ABC):
    _graph: EntityGraph
    _node_processor: Callable[[EntityType], EntityType]
    _inputs_selector: Callable[[List], List]

    def __init__(self,
                 graph: EntityGraph,
                 node_processor: Callable[[EntityType], EntityType] = None,
                 inputs_selector: Callable[[List[EntityType]], List[EntityType]] = None,
                 ):
        self._graph = graph
        self._node_processor = node_processor
        self._inputs_selector = inputs_selector or _default_inputs_selector

    def _add_nodes(self,
                   graph: Union[ChunkGraph, TileableGraph],
                   nodes: List[EntityType],
                   visited: Set):
        # update visited
        visited.update(nodes)

        while len(nodes) > 0:
            node = nodes.pop()
            if self._node_processor:
                # if node processor registered, process the node first
                node = self._node_processor(node)

            # mark node as visited
            visited.add(node)

            # add node to graph if possible
            if not graph.contains(node):
                graph.add_node(node)

            children = self._inputs_selector(node.inputs or [])
            for c in children:
                if self._node_processor:
                    # process if node processor registered
                    c = self._node_processor(c)
                if not graph.contains(c):
                    graph.add_node(c)
                if not graph.has_successor(c, node):
                    graph.add_edge(c, node)
                for out in c.op.outputs:
                    if out not in visited:
                        nodes.append(out)

    @abstractmethod
    def build(self) -> Iterable[Union[EntityGraph, ChunkGraph]]:
        """
        Build a entity graph.

        Returns
        -------
        graph : EntityGraph
            Entity graph.
        """


class TileableGraphBuilder(AbstractGraphBuilder):
    _graph: TileableGraph

    def __init__(self,
                 graph: TileableGraph,
                 node_processor: Callable[[EntityType], EntityType] = None,
                 inputs_selector: Callable[[List[EntityType]], List[EntityType]] = None,
                 ):
        super().__init__(graph=graph,
                         node_processor=node_processor,
                         inputs_selector=inputs_selector)

    @enter_mode(build=True, kernel=True)
    def _build(self) -> Union[TileableGraph, ChunkGraph]:
        self._add_nodes(self._graph, list(self._graph.result_tileables), set())
        return self._graph

    def build(self) -> Iterable[Union[TileableGraph, ChunkGraph]]:
        yield self._build()


_tileable_data_to_tiled = weakref.WeakKeyDictionary()
_op_to_copied = weakref.WeakKeyDictionary()


@enter_mode(build=True)
def get_tiled(tileable, mapping=None):
    tileable_data = tileable.data if hasattr(tileable, 'data') else tileable
    if mapping:
        tileable_data = mapping.get(tileable_data, tileable_data)
    return _tileable_data_to_tiled[tileable_data]


class ChunkGraphBuilder(AbstractGraphBuilder):
    _graph: TileableGraph

    def __init__(self,
                 graph: TileableGraph,
                 node_processor: Callable[[EntityType], EntityType] = None,
                 inputs_selector: Callable[[List[EntityType]], List[EntityType]] = None,
                 on_tile: Callable[[List[TileableType], List[TileableType]], List[TileableType]] = None,
                 on_tile_success: Callable[[TileableType, TileableType], TileableType] = None,
                 on_tile_failure: Callable[[OperandType, Tuple], List[TileableType]] = None,
                 fuse_enabled: bool = True):
        super().__init__(graph=graph, node_processor=node_processor,
                         inputs_selector=inputs_selector)
        self._fuse_enabled = fuse_enabled
        self._on_tile = on_tile
        self._on_tile_success = on_tile_success
        self._on_tile_failure = on_tile_failure

    @property
    def fused_enabled(self):
        return self._fuse_enabled

    def _tile(self,
              tileable_data: TileableType) -> List[TileableType]:
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

    @enter_mode(build=True, kernel=True)
    def _build(self) -> Union[TileableGraph, ChunkGraph]:
        tileable_graph = self._graph
        tileables = tileable_graph.result_tileables

        result_chunks = []
        graph = ChunkGraph(result_chunks)

        # do tiles and add nodes or edges to chunk graph
        tileables_set = set(tileables)
        keys = list()
        visited = set()
        tiled_op = set()
        for tileable_data in tileable_graph.topological_iter():
            nodes = list()
            # do tiling
            if tileable_data.op in tiled_op:
                continue
            try:
                tiled = self._tile(tileable_data)
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
                        result_chunks.extend(c.data for c in td.chunks)
                        keys.extend(c.key for c in td.chunks)
                    self._add_nodes(graph, nodes, visited)
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except  # pragma: no cover
                exc_info = sys.exc_info()
                if self._on_tile_failure:
                    # partial tiled chunks can be returned
                    # here they will be added to the chunk graph
                    # for further execution
                    partial_tiled_chunks = \
                        self._on_tile_failure(tileable_data.op, exc_info)
                    if partial_tiled_chunks is not None and \
                            len(partial_tiled_chunks) > 0:
                        self._add_nodes(graph, partial_tiled_chunks, visited)
                    tiled_op.add(tileable_data.op)
                else:
                    raise

        if self._fuse_enabled:
            graph.compose(keys=keys)

        return graph

    def build(self) -> Iterable[Union[EntityGraph, ChunkGraph]]:
        yield self._build()


@enter_mode(kernel=True)
def _build_graph(tileables: List[TileableType],
                 tiled: bool = False,
                 fuse_enabled: bool = True,
                 **chunk_graph_build_kwargs) -> Union[TileableGraph, ChunkGraph]:
    """
    Helper for test purpose.
    """
    tileables = list(itertools.chain(
        *(tileable.op.outputs for tileable in tileables)))
    tileable_graph = TileableGraph(tileables)
    tileable_graph_builder = TileableGraphBuilder(tileable_graph)
    tileable_graph = next(tileable_graph_builder.build())
    if not tiled:
        return tileable_graph
    chunk_graph_builder = ChunkGraphBuilder(
        tileable_graph, fuse_enabled=fuse_enabled,
        **chunk_graph_build_kwargs)
    return next(chunk_graph_builder.build())


class IterativeChunkGraphBuilder(ChunkGraphBuilder):
    def __init__(self,
                 graph: TileableGraph,
                 node_processor: Callable[[EntityType], EntityType] = None,
                 inputs_selector: Callable[[List[EntityType]], List[EntityType]] = None,
                 on_tile: Callable[[List[TileableType], List[TileableType]], List[TileableType]] = None,
                 on_tile_success: Callable[[TileableType, TileableType], TileableType] = None,
                 on_tile_failure: Callable[[OperandType, Tuple], List[TileableType]] = None,
                 fuse_enabled: bool = True):

        super().__init__(
            graph=graph, node_processor=node_processor,
            inputs_selector=inputs_selector, on_tile=on_tile,
            on_tile_success=self._wrap_on_tile_success(on_tile_success),
            on_tile_failure=self._wrap_on_tile_failure(on_tile_failure),
            fuse_enabled=fuse_enabled)

        self._interrupted_ops = set()
        self._prev_tileable_graph = None
        self._cur_tileable_graph = None
        self._iterative_chunk_graphs = []
        self._done = False

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
            if tile_before not in self._prev_tileable_graph:
                return
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

    def _tile(self,
              tileable_data: TileableType):
        if any(inp.op in self._interrupted_ops for inp in tileable_data.inputs):
            raise TilesError('Tile fail due to failure of inputs')
        return super()._tile(tileable_data)

    @enter_mode(build=True, kernel=True)
    def _build(self) -> Union[TileableGraph, ChunkGraph]:
        self._interrupted_ops.clear()
        self._prev_tileable_graph = self._graph
        self._cur_tileable_graph = type(self._graph)(self._graph.result_tileables)

        chunk_graph = super()._build()
        self._iterative_chunk_graphs.append(chunk_graph)
        if len(self._interrupted_ops) == 0:
            self._done = True
        self._prev_tileable_graph = self._cur_tileable_graph
        self._cur_tileable_graph = None
        return chunk_graph

    def build(self) -> Iterable[Union[EntityGraph, ChunkGraph]]:
        while not self._done:
            yield self._build()
