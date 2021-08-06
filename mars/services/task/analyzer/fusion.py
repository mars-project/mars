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

from typing import List, Tuple

from ....core import ChunkGraph, ChunkType
from ....core.operand import VirtualOperand, Fetch
from ....utils import build_fuse_chunk


class Fusion:
    """
    Fuse chunks in a chunk graph into a single node.
    """

    def __init__(self, graph: ChunkGraph):
        self._graph = graph

    def _fuse_nodes(self, nodes_list: List[List[ChunkType]]) -> \
            Tuple[List[List[ChunkType]], List[ChunkType]]:
        fused_nodes = []

        for nodes in nodes_list:
            head_node = nodes[0]
            tail_node = nodes[-1]
            fuse_chunk = build_fuse_chunk(
                nodes, tail_node.op.get_fuse_op_cls(tail_node), None, None).data
            self._graph.add_node(fuse_chunk)
            try:
                result_index = self._graph.results.index(tail_node)
                self._graph.results[result_index] = fuse_chunk
            except ValueError:
                pass
            for node in self._graph.iter_successors(tail_node):
                self._graph.add_edge(fuse_chunk, node)
                # replace inputs
                node_inputs = node.inputs
                new_node_inputs = []
                for inp in node_inputs:
                    if inp is tail_node:
                        new_node_inputs.append(fuse_chunk)
                    else:
                        new_node_inputs.append(inp)
                node.inputs = new_node_inputs
            for node in self._graph.iter_predecessors(head_node):
                self._graph.add_edge(node, fuse_chunk)
            # TODO:judge compose is independent?
            for node in nodes:
                self._graph.remove_node(node)
            fused_nodes.append(fuse_chunk)

        return nodes_list, fused_nodes

    def fuse(self) -> Tuple[List[List[ChunkType]], List[ChunkType]]:
        fuses = []
        explored = set()
        # for those chunk in result sets, we should not do any fuse
        result_chunk_set = set(self._graph.result_chunks)

        for v in self._graph.topological_iter():
            if v in explored or v in result_chunk_set:
                continue
            if self._graph.count_successors(v) != 1:
                continue
            if len(v.op.outputs) != 1:  # pragma: no cover
                continue
            if v.op.gpu != self._graph.successors(v)[0].op.gpu:
                continue
            if isinstance(v.op, (VirtualOperand, Fetch)):  # pragma: no cover
                # cannot fuse virtual operand or fetch
                continue
            if v.op.scheduling_hint is not None and \
                    not v.op.scheduling_hint.can_be_fused():  # pragma: no cover
                # don't fuse operand that cannot be fused
                continue
            selected = [v]
            # add successors
            cur_node = self._graph.successors(v)[0]
            while self._graph.count_predecessors(cur_node) == 1 and \
                    not isinstance(cur_node.op, (VirtualOperand, Fetch)):
                selected.append(cur_node)
                if self._graph.count_successors(cur_node) != 1 or \
                        cur_node in result_chunk_set:
                    break
                else:
                    cur_node = self._graph.successors(cur_node)[0]
            if len(selected) > 1:
                explored.update(selected)
                fuses.append(list(selected))
        return self._fuse_nodes(fuses)
