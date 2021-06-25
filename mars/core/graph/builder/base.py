# Copyright 1999-2021 Alibaba Group Holding Ltd.
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


from abc import ABC, abstractmethod
from typing import List, Set, Union, Generator

from ....typing import EntityType
from ..entity import EntityGraph, TileableGraph, ChunkGraph


def _default_inputs_selector(inputs: List[EntityType]) -> List[EntityType]:
    return inputs


class AbstractGraphBuilder(ABC):
    _graph: EntityGraph

    def __init__(self, graph: EntityGraph):
        self._graph = graph

    def _process_node(self, entity: EntityType):
        return entity

    def _select_inputs(self, inputs: List[EntityType]):
        return inputs

    def _if_add_node(self, node: EntityType, visited: Set):  # pylint: disable=no-self-use
        return node not in visited

    def _add_nodes(self,
                   graph: Union[ChunkGraph, TileableGraph],
                   nodes: List[EntityType],
                   visited: Set):
        # update visited
        visited.update(nodes)

        while len(nodes) > 0:
            node = nodes.pop()
            node = self._process_node(node)

            # mark node as visited
            visited.add(node)

            # add node to graph if possible
            if not graph.contains(node):
                graph.add_node(node)

            children = self._select_inputs(node.inputs or [])
            for c in children:
                c = self._process_node(c)
                if not graph.contains(c):
                    graph.add_node(c)
                if not graph.has_successor(c, node):
                    graph.add_edge(c, node)
                for out in c.op.outputs:
                    if self._if_add_node(out, visited):
                        nodes.append(out)

    @abstractmethod
    def build(self) -> Generator[Union[EntityGraph, ChunkGraph], None, None]:
        """
        Build a entity graph.

        Returns
        -------
        graph : EntityGraph
            Entity graph.
        """
