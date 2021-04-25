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

from abc import ABC, abstractmethod
from collections import deque
from typing import Type, Dict, List

from ....core import ChunkGraph, ChunkType
from ....core.operand import VirtualOperand, Fuse
from ....utils import implements, build_fetch
from ...core import BandType
from ..core import SubtaskGraph, Subtask, new_task_id
from .assigner import AbstractGraphAssigner, GraphAssigner
from .fusion import Fusion


class AbstractGraphAnalyzer(ABC):
    def __init__(self,
                 chunk_graph: ChunkGraph,
                 band_slots: Dict[BandType, int],
                 fuse_enabled: bool,
                 extra_config: dict,
                 task_stage_info,
                 graph_assigner_cls: Type[AbstractGraphAssigner] = None):
        self._chunk_graph = chunk_graph
        self._band_slots = band_slots
        self._fuse_enabled = fuse_enabled
        self._extra_config = extra_config
        self._task_stage_info = task_stage_info
        if graph_assigner_cls is None:
            graph_assigner_cls = GraphAssigner
        self._graph_assigner_cls = graph_assigner_cls

    @abstractmethod
    def gen_subtask_graph(self) -> SubtaskGraph:
        """
        Analyze chunk graph and generate subtask graph.

        Returns
        -------
        subtask_graph: SubtaskGraph
            Subtask graph.
        """


class GraphAnalyzer(AbstractGraphAnalyzer):
    def _iter_start_ops(self):
        visited = set()
        op_keys = set()
        start_chunks = deque(self._chunk_graph.iter_indep())
        stack = deque([start_chunks.popleft()])

        while stack:
            chunk = stack.popleft()
            if chunk not in visited:
                inp_chunks = self._chunk_graph.predecessors(chunk)
                if not inp_chunks or \
                        all(inp_chunk in visited for inp_chunk in inp_chunks):
                    if len(inp_chunks) == 0:
                        op_key = chunk.op.key
                        if op_key not in op_keys:
                            op_keys.add(op_key)
                            yield chunk.op
                    visited.add(chunk)
                    stack.extend(c for c in self._chunk_graph[chunk]
                                 if c not in visited)
                else:
                    stack.appendleft(chunk)
                    stack.extendleft(
                        reversed([c for c in self._chunk_graph.predecessors(chunk)
                                  if c not in visited]))
            if not stack and start_chunks:
                stack.appendleft(start_chunks.popleft())

    def _fuse(self, chunk_to_bands: Dict[ChunkType, BandType]) \
            -> Dict[ChunkType, BandType]:
        fusion = Fusion(self._chunk_graph)
        to_fuse_nodes_list, fused_nodes = fusion.fuse()
        assert len(to_fuse_nodes_list) == len(fused_nodes)
        fuse_to_fused = dict()
        for to_fuse_nodes, fused_node in zip(to_fuse_nodes_list, fused_nodes):
            for to_fuse_node in to_fuse_nodes:
                fuse_to_fused[to_fuse_node] = fused_node
        # modify chunk_to_bands if some chunk fused
        new_chunk_to_bands = dict()
        for chunk, band in chunk_to_bands.items():
            if chunk in fuse_to_fused:
                new_chunk_to_bands[fuse_to_fused[chunk]] = band
            else:
                new_chunk_to_bands[chunk] = band
        return new_chunk_to_bands

    @classmethod
    def _gen_input_chunks(cls,
                          inp_chunks: List[ChunkType],
                          chunk_to_fetch_chunk: Dict[ChunkType, ChunkType]) \
            -> List[ChunkType]:
        # gen fetch chunks for input chunks
        inp_fetch_chunks = []
        for inp_chunk in inp_chunks:
            if inp_chunk in chunk_to_fetch_chunk:
                inp_fetch_chunks.append(chunk_to_fetch_chunk[inp_chunk])
            else:
                fetch_chunk = build_fetch(inp_chunk).data
                chunk_to_fetch_chunk[inp_chunk] = fetch_chunk
                inp_fetch_chunks.append(fetch_chunk)

        return inp_fetch_chunks

    def _build_subtask_chunk_graph(self,
                                   chunk: ChunkType,
                                   chunk_to_fetch_chunk: Dict[ChunkType, ChunkType]) \
            -> ChunkGraph:
        virtual = isinstance(chunk.op, VirtualOperand)
        inp_chunks = self._chunk_graph.predecessors(chunk)
        inp_fetch_chunks = self._gen_input_chunks(inp_chunks, chunk_to_fetch_chunk)

        # gen chunk graph for each subtask
        result_chunks = []
        subtask_chunk_graph = ChunkGraph(result_chunks)
        outs = chunk.op.outputs
        out_params = [out.params for out in outs]
        copied_op = chunk.op.copy()
        copied_op._key = chunk.op.key
        copied_chunks = [c.data for c in
                         copied_op.new_chunks(inp_fetch_chunks, kws=out_params)]
        for out, copied_chunk in zip(outs, copied_chunks):
            copied_chunk._key = out.key
            result_chunks.append(copied_chunk)
            subtask_chunk_graph.add_node(copied_chunk)
            if not virtual:
                # skip adding fetch chunk to chunk graph when op is virtual operand
                for inp_fetch_chunk in inp_fetch_chunks:
                    subtask_chunk_graph.add_node(inp_fetch_chunk)
                    subtask_chunk_graph.add_edge(inp_fetch_chunk, copied_chunk)

        return subtask_chunk_graph

    def _build_fuse_subtask_chunk_graph(self,
                                        chunk: ChunkType,
                                        chunk_to_fetch_chunk: Dict[ChunkType, ChunkType]) \
            -> ChunkGraph:
        # gen chunk graph for each subtask
        result_chunks = []
        subtask_chunk_graph = ChunkGraph(result_chunks)
        fused_chunks = chunk.composed
        fuse_to_copied = dict()
        for i, fuse_chunk in enumerate(fused_chunks):
            copied_op = fuse_chunk.op.copy()
            copied_op._key = fuse_chunk.op.key
            if i == 0:
                # the first chunk
                inp_chunks = self._chunk_graph.predecessors(chunk)
                inp_fetch_chunks = self._gen_input_chunks(
                    inp_chunks, chunk_to_fetch_chunk)
                copied_fuse_chunk = copied_op.new_chunk(
                    inp_fetch_chunks, kws=[fuse_chunk.params.copy()]).data
                copied_fuse_chunk._key = fuse_chunk.key
                subtask_chunk_graph.add_node(copied_fuse_chunk)
                for inp_fetch_chunk in inp_fetch_chunks:
                    subtask_chunk_graph.add_node(inp_fetch_chunk)
                    subtask_chunk_graph.add_edge(
                        inp_fetch_chunk, copied_fuse_chunk)
            else:
                inp_chunk = fuse_to_copied[fused_chunks[i - 1]]
                copied_fuse_chunk = copied_op.new_chunk(
                    [inp_chunk], kws=[fuse_chunk.params.copy()]).data
                copied_fuse_chunk._key = fuse_chunk.key
                subtask_chunk_graph.add_node(copied_fuse_chunk)
                subtask_chunk_graph.add_edge(inp_chunk, copied_fuse_chunk)
                if i == len(fused_chunks) - 1:
                    # the last chunk
                    result_chunks.append(copied_fuse_chunk)
            fuse_to_copied[fuse_chunk] = copied_fuse_chunk
        return subtask_chunk_graph

    def _gen_subtask(self,
                     chunk: ChunkType,
                     chunk_to_priorities: Dict[ChunkType, int],
                     chunk_to_bands: Dict[ChunkType, BandType],
                     chunk_to_fetch_chunk: Dict[ChunkType, ChunkType]) -> Subtask:
        virtual = isinstance(chunk.op, VirtualOperand)
        inp_chunks = self._chunk_graph.predecessors(chunk)
        # calculate priority
        if inp_chunks:
            priority = max(chunk_to_priorities[inp_chunk]
                           for inp_chunk in inp_chunks) + 1
        else:
            priority = 0
        for out in chunk.op.outputs:
            chunk_to_priorities[out] = priority

        band = chunk_to_bands.get(chunk)
        if not isinstance(chunk.op, Fuse):
            subtask_chunk_graph = self._build_subtask_chunk_graph(
                chunk, chunk_to_fetch_chunk)
        else:
            subtask_chunk_graph = self._build_fuse_subtask_chunk_graph(
                chunk, chunk_to_fetch_chunk)

        subtask = Subtask(
            subtask_id=new_task_id(),
            session_id=self._task_stage_info.task_info.session_id,
            task_id=self._task_stage_info.task_id,
            chunk_graph=subtask_chunk_graph,
            expect_bands=[band],
            virtual=virtual,
            priority=priority,
            extra_config=self._extra_config)
        return subtask

    @implements(AbstractGraphAnalyzer.gen_subtask_graph)
    def gen_subtask_graph(self) -> SubtaskGraph:
        start_ops = list(self._iter_start_ops())

        # assign start chunks
        assigner = self._graph_assigner_cls(
            self._chunk_graph, start_ops, self._band_slots)
        chunk_to_bands = assigner.assign()

        # fuse node
        if self._fuse_enabled:
            chunk_to_bands = self._fuse(chunk_to_bands)

        subtask_graph = SubtaskGraph()
        chunk_to_priorities = dict()
        chunk_to_fetch_chunk = dict()
        chunk_to_subtask = dict()
        visited = set()
        for chunk in self._chunk_graph.topological_iter():
            if chunk in visited:
                continue

            # gen subtask
            subtask = self._gen_subtask(
                chunk, chunk_to_priorities,
                chunk_to_bands, chunk_to_fetch_chunk)

            for out in chunk.op.outputs:
                chunk_to_subtask[out] = subtask
                visited.add(out)
            subtask_graph.add_node(subtask)
            inp_subtasks = \
                [chunk_to_subtask[inp_chunk] for inp_chunk
                 in self._chunk_graph.predecessors(chunk)]
            for inp_subtask in inp_subtasks:
                subtask_graph.add_edge(inp_subtask, subtask)

        return subtask_graph
