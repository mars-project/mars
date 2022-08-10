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

from abc import ABCMeta, abstractmethod
from typing import List, Dict, Tuple, Union, Iterable

from ...core import Tileable, Chunk
from ...serialization.core import buffered
from ...serialization.serializables import Serializable, DictField, ListField, BoolField
from ...serialization.serializables.core import SerializableSerializer
from ...utils import tokenize
from .core import DAG


class EntityGraph(DAG, metaclass=ABCMeta):
    @property
    @abstractmethod
    def results(self):
        """
        Return result tileables or chunks.

        Returns
        -------
        results
        """

    @results.setter
    @abstractmethod
    def results(self, new_results):
        """
        Set result tileables or chunks.

        Parameters
        ----------
        new_results

        Returns
        -------

        """

    def copy(self) -> "EntityGraph":
        graph = super().copy()
        graph.results = self.results.copy()
        return graph


class TileableGraph(EntityGraph, Iterable[Tileable]):
    _result_tileables: List[Tileable]
    # logic key is a unique and deterministic key for `TileableGraph`. For
    # multiple runs the logic key will remain same if the computational logic
    # doesn't change. And it can be used to some optimization when running a
    # same `execute`, like HBO.
    _logic_key: str

    def __init__(self, result_tileables: List[Tileable] = None):
        super().__init__()
        self._result_tileables = result_tileables

    @property
    def result_tileables(self):
        return self._result_tileables

    @property
    def results(self):
        return self._result_tileables

    @results.setter
    def results(self, new_results):
        self._result_tileables = new_results

    @property
    def logic_key(self):
        if not hasattr(self, "_logic_key") or self._logic_key is None:
            token_keys = []
            for node in self.bfs():
                logic_key = node.op.get_logic_key()
                if hasattr(node.op, "logic_key") and node.op.logic_key is None:
                    node.op.logic_key = logic_key
                token_keys.append(
                    tokenize(logic_key, **node.extra_params)
                    if node.extra_params
                    else logic_key
                )
            self._logic_key = tokenize(*token_keys)
        return self._logic_key


class ChunkGraph(EntityGraph, Iterable[Chunk]):
    _result_chunks: List[Chunk]

    def __init__(self, result_chunks: List[Chunk] = None):
        super().__init__()
        self._result_chunks = result_chunks

    @property
    def result_chunks(self):
        return self._result_chunks

    @property
    def results(self):
        return self._result_chunks

    @results.setter
    def results(self, new_results):
        self._result_chunks = new_results


class SerializableGraph(Serializable):
    _is_chunk = BoolField("is_chunk")
    # TODO(qinxuye): remove this logic when we handle fetch elegantly,
    # now, the node in the graph and inputs for operand may be inconsistent,
    # for example, an operand's inputs may be chunks,
    # but in the graph, the predecessors are all fetch chunks,
    # we serialize the fetch chunks first to make sure when operand's inputs
    # are serialized, they will just be marked as serialized and skip serialization.
    _fetch_nodes = ListField("fetch_nodes")
    _nodes = DictField("nodes")
    _predecessors = DictField("predecessors")
    _successors = DictField("successors")
    _results = ListField("results")

    @classmethod
    def from_graph(cls, graph: Union[TileableGraph, ChunkGraph]) -> "SerializableGraph":
        from ..operand import Fetch

        is_chunk = isinstance(graph, ChunkGraph)
        return SerializableGraph(
            _is_chunk=is_chunk,
            _fetch_nodes=[chunk for chunk in graph if isinstance(chunk.op, Fetch)],
            _nodes=graph._nodes,
            _predecessors=graph._predecessors,
            _successors=graph._successors,
            _results=graph.results,
        )

    def to_graph(self) -> Union[TileableGraph, ChunkGraph]:
        graph_cls = ChunkGraph if self._is_chunk else TileableGraph
        graph = graph_cls(self._results)
        graph._nodes.update(self._nodes)
        graph._predecessors.update(self._predecessors)
        graph._successors.update(self._successors)
        return graph


class GraphSerializer(SerializableSerializer):
    @buffered
    def serial(self, obj: Union[TileableGraph, ChunkGraph], context: Dict):
        serializable_graph = SerializableGraph.from_graph(obj)
        return (), [serializable_graph], False

    def deserial(
        self, serialized: Tuple, context: Dict, subs: List
    ) -> Union[TileableGraph, ChunkGraph]:
        serializable_graph: SerializableGraph = subs[0]
        return serializable_graph.to_graph()


GraphSerializer.register(EntityGraph)
