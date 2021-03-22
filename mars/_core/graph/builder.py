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
from abc import ABC, abstractmethod
from typing import Callable, List, Set, Union, Tuple, Iterable

from ...core import EntityType, Tileable, TileableData
from ...operands import Operand
from ...utils import enter_mode, copy_tileables
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
    def build(self) -> Iterable[Union[TileableGraph, ChunkGraph]]:
        self._add_nodes(self._graph, self._graph.result_tileables, set())
        yield self._graph


_tileable_data_to_tiled = weakref.WeakKeyDictionary()
_op_to_copied = weakref.WeakKeyDictionary()


class ChunkGraphBuilder(AbstractGraphBuilder):
    _graph: TileableGraph

    def __init__(self,
                 graph: TileableGraph,
                 node_process: Callable[[EntityType], EntityType] = None,
                 inputs_selector: Callable[[List[EntityType]], List[EntityType]] = None,
                 on_tile: Callable[[List[Tileable], List[Tileable]], List[Tileable]] = None,
                 on_tile_success: Callable[[Tileable, Tileable], Tileable] = None,
                 on_tile_failure: Callable[[Operand, Tuple], List[Tileable]] = None,
                 fuse_enabled: bool = True,
                 ):
        super().__init__(graph=graph, node_processor=node_process,
                         inputs_selector=inputs_selector)
        self._fuse_enabled = fuse_enabled
        self._on_tile = on_tile
        self._on_tile_success = on_tile_success
        self._on_tile_failure = on_tile_failure

    @property
    def fused_enabled(self):
        return self._fuse_enabled

    def _tile(self,
              tileable_data: TileableData) -> List[Tileable]:
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
    def build(self) -> Iterable[Union[TileableGraph, ChunkGraph]]:
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
                        result_chunks.extend(td.chunks)
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

        yield graph
