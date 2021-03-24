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

from typing import List, Dict, Union

from ...core import Tileable, Chunk
from ...serialization.serializables import Serializable, DictField, ListField, BoolField
from ...serialization.serializables.core import SerializableSerializer
from .core import DAG


class EntityGraph(DAG):
    pass


class TileableGraph(EntityGraph):
    _result_tileables: List[Tileable]

    def __init__(self, result_tileables: List[Tileable]):
        super().__init__()
        self._result_tileables = result_tileables

    @property
    def result_tileables(self):
        return self._result_tileables

    @property
    def results(self):
        return self._result_tileables


class ChunkGraph(EntityGraph):
    _result_chunks: List[Chunk]

    def __init__(self, result_chunks: List[Chunk]):
        super().__init__()
        self._result_chunks = result_chunks

    @property
    def result_chunks(self):
        return self._result_chunks

    @property
    def results(self):
        return self._result_chunks


class SerializableGraph(Serializable):
    _is_chunk = BoolField('is_chunk')
    _nodes = DictField('nodes')
    _predecessors = DictField('predecessors')
    _successors = DictField('successors')
    _results = ListField('results')

    @classmethod
    def from_graph(cls, graph: Union[TileableGraph, ChunkGraph]):
        is_chunk = isinstance(graph, ChunkGraph)
        return SerializableGraph(
            _is_chunk=is_chunk,
            _nodes=graph._nodes,
            _predecessors=graph._predecessors,
            _successors=graph._successors,
            _results=graph.results
        )

    def to_graph(self) -> Union[TileableGraph, ChunkGraph]:
        graph_cls = ChunkGraph if self._is_chunk else TileableGraph
        graph = graph_cls(self._results)
        graph._nodes.update(self._nodes)
        graph._predecessors.update(self._predecessors)
        graph._successors.update(self._successors)
        return graph


class GraphSerializer(SerializableSerializer):
    serializer_name = 'graph'

    def serialize(self, obj: Union[TileableGraph, ChunkGraph], context: Dict):
        serializable_graph = SerializableGraph.from_graph(obj)
        return super().serialize(serializable_graph, context)

    def deserialize(self, header: Dict, buffers: List, context: Dict) \
            -> Union[TileableGraph, ChunkGraph]:
        serializable_graph: SerializableGraph = \
            super().deserialize(header, buffers, context)
        return serializable_graph.to_graph()


GraphSerializer.register(EntityGraph)
