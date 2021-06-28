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
from typing import List, Tuple, Dict, Type

from ...core import ChunkGraph, ChunkType, OperandType
from ...utils import build_fuse_chunk


class RuntimeOptimizer(ABC):
    engine = None

    def __init__(self,
                 graph: ChunkGraph):
        self._graph = graph

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """
        Check this optimizer is available.

        Returns
        -------
        is_available : bool
            Available.
        """

    @abstractmethod
    def optimize(self):
        """
        Optimize chunk graph.
        """

    def _fuse_nodes(self,
                    fuses: List[List[ChunkType]],
                    fuse_cls: OperandType) -> \
            Tuple[List[List[ChunkType]], List[ChunkType]]:
        graph = self._graph
        fused_nodes = []

        for fuse in fuses:
            head_node = fuse[0]
            tail_node = fuse[-1]

            fused_chunk = build_fuse_chunk(
                fuse, fuse_cls,
                op_kw={'dtype': tail_node.dtype}).data
            graph.add_node(fused_chunk)
            for node in graph.iter_successors(tail_node):
                graph.add_edge(fused_chunk, node)
            for node in graph.iter_predecessors(head_node):
                graph.add_edge(node, fused_chunk)
            for node in fuse:
                graph.remove_node(node)
            fused_nodes.append(fused_chunk)

            try:
                # check tail node if it's in results
                i = graph.results.index(tail_node)
                graph.results[i] = fused_chunk
            except ValueError:
                pass

        return fuses, fused_nodes


_engine_to_optimizers: Dict[str, Type[RuntimeOptimizer]] = dict()


def register_optimizer(optimizer_cls: Type[RuntimeOptimizer]):
    _engine_to_optimizers[optimizer_cls.engine] = optimizer_cls
    return optimizer_cls


def optimize(graph: ChunkGraph,
             engines: List[str] = None) -> ChunkGraph:
    if engines is None:
        engines = ['numexpr', 'cupy']

    for engine in engines:
        optimizer_cls = _engine_to_optimizers[engine]
        optimizer = optimizer_cls(graph)
        if not optimizer.is_available():
            continue
        optimizer.optimize()

    return graph
