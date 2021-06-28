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

from typing import Union, Generator

from ...mode import enter_mode
from ..entity import TileableGraph, ChunkGraph
from .base import AbstractGraphBuilder


class TileableGraphBuilder(AbstractGraphBuilder):
    _graph: TileableGraph

    def __init__(self, graph: TileableGraph):
        super().__init__(graph=graph)

    @enter_mode(build=True, kernel=True)
    def _build(self) -> Union[TileableGraph, ChunkGraph]:
        self._add_nodes(self._graph, list(self._graph.result_tileables), set())
        return self._graph

    def build(self) -> Generator[Union[TileableGraph, ChunkGraph], None, None]:
        yield self._build()
