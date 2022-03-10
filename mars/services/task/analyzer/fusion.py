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
from typing import Dict, List

from ....config import options
from ....core import ChunkGraph
from ....core.operand import VirtualOperand
from ....typing import BandType, ChunkType, OperandType


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
        as_broadcaster_successor_num: int = None,
    ):
        self.chunk_graph = chunk_graph
        self.all_bands = all_bands
        self.chunk_to_bands = chunk_to_bands
        if initial_same_color_num is None:
            initial_same_color_num = max(options.combine_size // 2, 1)
        self.initial_same_color_num = initial_same_color_num
        if as_broadcaster_successor_num is None:
            as_broadcaster_successor_num = options.combine_size * 2
        self.successor_same_color_num = as_broadcaster_successor_num

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
        # e.g. md.read_csv ensure incremental index by generating
        # chunks with ascending priorities (smaller one has higher priority),
        # chunk 0 has higher priority than chunk 1,
        # so that when chunk 1 executing, it would know chunk 0's shape
        # TODO: make it general instead handle priority as a special case
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
        broadcaster_chunk_set = set()
        for chunk in self.chunk_graph.topological_iter():
            if self.chunk_graph.count_successors(chunk) > self.successor_same_color_num:
                # is broadcaster
                broadcaster_chunk_set.add(chunk)

            if chunk.op in op_to_colors:
                # colored
                chunk_to_colors[chunk] = op_to_colors[chunk.op]
                continue

            predecessors = self.chunk_graph.predecessors(chunk)
            pred_colors = {op_to_colors[pred.op] for pred in predecessors}
            if len(predecessors) == 1 and predecessors[0] in broadcaster_chunk_set:
                # TODO: handle situation that chunks which specify reassign_workers
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
            chunk_color = chunk_to_colors[chunk]
            if chunk_color in pred_colors and len(pred_colors) > 1:
                # conflict
                # color the successors with new colors
                stack = []
                for succ in self.chunk_graph.iter_successors(chunk):
                    if chunk_to_colors[succ] == chunk_color:
                        chunk_to_colors[succ] = op_to_colors[
                            succ.op
                        ] = self._next_color()
                        stack.extend(self.chunk_graph.successors(succ))
                # color the descendants with same color to the new one
                # the descendants will not be visited more than 2 times
                while len(stack) > 0:
                    node = stack.pop()
                    node_color = chunk_to_colors[node]
                    if node_color == chunk_color:
                        # same color, recolor to the new one
                        node_pred_colors = list(
                            {
                                op_to_colors[inp.op]
                                for inp in self.chunk_graph.iter_predecessors(node)
                            }
                        )
                        node_input_same_color = len(node_pred_colors) == 1
                        if node_input_same_color:
                            node_new_color = node_pred_colors[0]
                        else:
                            node_new_color = self._next_color()
                        chunk_to_colors[node] = op_to_colors[node.op] = node_new_color
                        stack.extend(self.chunk_graph.successors(node))

        return chunk_to_colors
