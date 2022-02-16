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

import itertools
from collections import defaultdict
from typing import Dict, List, Tuple

from ....config import options
from ....core import ChunkGraph
from ....core.operand import VirtualOperand, Fetch
from ....typing import BandType, ChunkType, OperandType
from ....utils import build_fuse_chunk


class Fusion:
    """
    Fuse chunks in a chunk graph into a single node.
    """

    def __init__(self, graph: ChunkGraph):
        self._graph = graph

    def _fuse_nodes(
        self, nodes_list: List[List[ChunkType]]
    ) -> Tuple[List[List[ChunkType]], List[ChunkType]]:
        fused_nodes = []
        replace_dict = dict()

        for nodes in nodes_list:
            head_node = nodes[0]
            tail_node = nodes[-1]
            fuse_chunk = build_fuse_chunk(
                nodes, tail_node.op.get_fuse_op_cls(tail_node), None, None
            ).data
            self._graph.add_node(fuse_chunk)
            replace_dict[tail_node] = fuse_chunk
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

        # replace outputs
        self._graph.results = [
            replace_dict.get(out, out) for out in self._graph.results
        ]

        return nodes_list, fused_nodes

    def fuse(self) -> Tuple[List[List[ChunkType]], List[ChunkType]]:
        fuses = []
        explored = set()
        # for those chunk in result sets, we should not do any fuse
        result_chunk_set = set(self._graph.result_chunks)

        for v in self._graph.topological_iter():
            if v in explored or v in result_chunk_set:
                continue
            successors = self._graph.successors(v)
            if len(successors) != 1:
                continue

            cur_node = successors[0]
            if len(v.op.outputs) != 1:  # pragma: no cover
                continue
            if v.op.gpu != cur_node.op.gpu:
                continue
            if isinstance(v.op, (VirtualOperand, Fetch)):  # pragma: no cover
                # cannot fuse virtual operand or fetch
                continue
            if (
                v.op.scheduling_hint is not None
                and not v.op.scheduling_hint.can_be_fused()
            ):  # pragma: no cover
                # don't fuse operand that cannot be fused
                continue
            selected = [v]
            # add successors
            while self._graph.count_predecessors(cur_node) == 1 and not isinstance(
                cur_node.op, (VirtualOperand, Fetch)
            ):
                selected.append(cur_node)
                if (
                    self._graph.count_successors(cur_node) != 1
                    or cur_node in result_chunk_set
                ):
                    break
                else:
                    cur_node = self._graph.successors(cur_node)[0]
            if len(selected) > 1:
                explored.update(selected)
                fuses.append(list(selected))
        return self._fuse_nodes(fuses)


class Coloring:
    """
    Coloring a chunk graph according to an algorithm
    described in https://github.com/mars-project/mars/issues/2435
    """

    def __init__(
        self,
        chunk_graph: ChunkGraph,
        all_bands: List[BandType],
        chunk_to_bands: Dict[ChunkType, BandType],
        initial_same_color_num: int = None,
        successor_same_color_num: int = None,
    ):
        self.chunk_graph = chunk_graph
        self.all_bands = all_bands
        self.chunk_to_bands = chunk_to_bands
        if initial_same_color_num is None:
            initial_same_color_num = max(options.combine_size // 2, 1)
        self.initial_same_color_num = initial_same_color_num
        if successor_same_color_num is None:
            successor_same_color_num = options.combine_size * 2
        self.successor_same_color_num = successor_same_color_num

        self._coloring_iter = itertools.count()

    def _next_color(self) -> int:
        return next(self._coloring_iter)

    @classmethod
    def _can_color_same(cls, chunk: ChunkType, predecessors: List[ChunkType]) -> bool:
        if (
            # VirtualOperand cannot be fused
            any(isinstance(n.op, VirtualOperand) for n in [chunk] + predecessors)
            # allocated on different bands
            or len({n.op.gpu for n in [chunk] + predecessors}) > 1
            # expect worker changed
            or len({n.op.expect_worker for n in [chunk] + predecessors}) > 1
            # scheduling hint tells that cannot be fused
            or (
                chunk.op.scheduling_hint is not None
                and not chunk.op.scheduling_hint.can_be_fused()
            )
        ):
            return False
        return True

    def _color_init_nodes(self) -> Dict[OperandType, int]:
        # for initial op with same band but different priority
        # we color them w/ different colors,
        # to prevent from wrong fusion.
        band_priority_to_colors = dict()
        for chunk, band in self.chunk_to_bands.items():
            band_priority = (band, chunk.op.priority)
            if band_priority not in band_priority_to_colors:
                band_priority_to_colors[band_priority] = self._next_color()

        band_priority_to_color_list = defaultdict(list)
        for (band, priority), color in band_priority_to_colors.items():
            band_priority_to_color_list[band, priority].append(color)
        color_to_size = defaultdict(lambda: 0)
        op_to_colors = dict()
        for chunk, band in self.chunk_to_bands.items():
            priority = chunk.op.priority
            color = band_priority_to_color_list[band, priority][-1]
            size = color_to_size[color]
            if size >= self.initial_same_color_num:
                color = self._next_color()
                band_priority_to_color_list[band, priority].append(color)
            color_to_size[color] += 1
            op_to_colors[chunk.op] = color
        return op_to_colors

    def color(self) -> Dict[ChunkType, int]:
        chunk_to_colors = dict()

        # step 1: Coloring the initial nodes according to the bands that assigned by assigner
        op_to_colors = self._color_init_nodes()

        # step2: Propagate color in the topological order,
        # if the input nodes have same color, color it with the same color;
        # otherwise, color with a new color.
        chunk_to_is_broadcaster = dict()
        for chunk in self.chunk_graph.topological_iter():
            if self.chunk_graph.count_successors(chunk) > self.successor_same_color_num:
                # is broadcaster
                chunk_to_is_broadcaster[chunk] = True

            if chunk.op in op_to_colors:
                # colored
                chunk_to_colors[chunk] = op_to_colors[chunk.op]
                continue

            predecessors = self.chunk_graph.predecessors(chunk)
            pred_colors = {op_to_colors[pred.op] for pred in predecessors}
            if len(predecessors) == 1 and chunk_to_is_broadcaster.get(predecessors[0]):
                # predecessor is broadcaster, just allocate a new color
                color = self._next_color()
            elif len(pred_colors) == 1:
                if self._can_color_same(chunk, predecessors):
                    # predecessors have only 1 color, will color with same one
                    color = next(iter(pred_colors))
                else:
                    color = self._next_color()
            else:
                # has more than 1 color, color a new one
                assert len(pred_colors) > 1
                color = self._next_color()

            op_to_colors[chunk.op] = chunk_to_colors[chunk] = color

        # step 3: Propagate with reversed topological order,
        # check a node with its inputs, if all inputs have different color with itself, skip;
        # otherwise, if some of inputs have the same color, but some others have different color,
        # color the input nodes with same one with a new color, and propagate to its inputs and so on.
        for chunk in self.chunk_graph.topological_iter(reverse=True):
            pred_colors = {
                op_to_colors[pred.op]
                for pred in self.chunk_graph.iter_successors(chunk)
            }
            prev_color = curr_color = chunk_to_colors[chunk]
            if curr_color in pred_colors and len(pred_colors) > 1:
                # conflict
                curr_color = self._next_color()
                # color the descendants with same color to the new one
                stack = list(self.chunk_graph.iter_successors(chunk))
                while len(stack) > 0:
                    node = stack.pop()
                    node_color = chunk_to_colors[node]
                    if node_color == prev_color:
                        # same color, recolor to the new one
                        chunk_to_colors[node] = op_to_colors[node.op] = curr_color
                        stack.extend(self.chunk_graph.successors(node))

        return chunk_to_colors
