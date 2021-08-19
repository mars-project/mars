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

import logging

from collections import deque
from typing import Dict, List, Tuple, Type, Union

from ....core import ChunkGraph, ChunkType, enter_mode
from ....core.operand import Fetch, Fuse, VirtualOperand
from ....typing import BandType
from ....utils import build_fetch
from ...subtask import SubtaskGraph, Subtask
from ..core import Task, new_task_id
from .assigner import AbstractGraphAssigner, GraphAssigner
from .fusion import Fusion

logger = logging.getLogger(__name__)


class GraphAnalyzer:
    def __init__(self,
                 chunk_graph: ChunkGraph,
                 band_slots: Dict[BandType, int],
                 task: Task,
                 graph_assigner_cls: Type[AbstractGraphAssigner] = None):
        self._chunk_graph = chunk_graph
        self._band_slots = band_slots
        self._task = task
        self._fuse_enabled = task.fuse_enabled
        self._extra_config = task.extra_config
        if graph_assigner_cls is None:
            graph_assigner_cls = GraphAssigner
        self._graph_assigner_cls = graph_assigner_cls
        self._chunk_to_copied = dict()

    @classmethod
    def _iter_start_ops(cls, chunk_graph: ChunkGraph):
        visited = set()
        op_keys = set()
        start_chunks = deque(chunk_graph.iter_indep())
        stack = deque([start_chunks.popleft()])

        while stack:
            chunk = stack.popleft()
            if chunk not in visited:
                inp_chunks = chunk_graph.predecessors(chunk)
                if not inp_chunks or \
                        all(inp_chunk in visited for inp_chunk in inp_chunks):
                    if len(inp_chunks) == 0:
                        op_key = chunk.op.key
                        if op_key not in op_keys:
                            op_keys.add(op_key)
                            yield chunk.op
                    visited.add(chunk)
                    stack.extend(c for c in chunk_graph[chunk]
                                 if c not in visited)
                else:
                    stack.appendleft(chunk)
                    stack.extendleft(
                        reversed([c for c in chunk_graph.predecessors(chunk)
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
            priority = None
            for to_fuse_node in to_fuse_nodes:
                if to_fuse_node.op.priority:
                    assert priority is None or priority == to_fuse_node.op.priority
                    priority = to_fuse_node.op.priority
                fuse_to_fused[to_fuse_node] = fused_node
            if priority:
                fused_node.op.priority = priority
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
            elif isinstance(inp_chunk.op, Fetch):
                chunk_to_fetch_chunk[inp_chunk] = inp_chunk
                inp_fetch_chunks.append(inp_chunk)
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
        inp_chunks = chunk.inputs
        assert all(inp_chunk in self._chunk_graph
                   for inp_chunk in inp_chunks)
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
            self._chunk_to_copied[out] = copied_chunk
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
                inp_chunks = chunk.inputs
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
                copied_fuse_chunk = copied_op.new_chunks(
                    [inp_chunk] * len(fuse_chunk.inputs),
                    kws=[fuse_chunk.params.copy()])[0].data
                copied_fuse_chunk._key = fuse_chunk.key
                subtask_chunk_graph.add_node(copied_fuse_chunk)
                subtask_chunk_graph.add_edge(inp_chunk, copied_fuse_chunk)
                if i == len(fused_chunks) - 1:
                    # the last chunk
                    result_chunks.append(copied_fuse_chunk)
            fuse_to_copied[fuse_chunk] = copied_fuse_chunk
        self._chunk_to_copied[chunk.chunk] = self._chunk_to_copied[chunk] = \
            fuse_to_copied[chunk.chunk]
        return subtask_chunk_graph

    def _gen_subtask(self,
                     chunk: ChunkType,
                     chunk_to_priorities: Dict[ChunkType, Tuple[int, ...]],
                     chunk_to_bands: Dict[ChunkType, BandType],
                     chunk_to_fetch_chunk: Dict[ChunkType, ChunkType]) -> Subtask:
        virtual = isinstance(chunk.op, VirtualOperand)
        inp_chunks = [inp_chunk for inp_chunk in self._chunk_graph.predecessors(chunk)
                      if not isinstance(inp_chunk.op, Fetch)]
        # calculate priority
        if inp_chunks:
            depth = max(chunk_to_priorities[inp_chunk][0]
                        for inp_chunk in inp_chunks) + 1
        else:
            depth = 0
        priority = (depth, chunk.op.priority or 0)
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
            session_id=self._task.session_id,
            task_id=self._task.task_id,
            chunk_graph=subtask_chunk_graph,
            expect_bands=[band] if band is not None else None,
            virtual=virtual,
            priority=priority,
            extra_config=self._extra_config)
        return subtask

    @staticmethod
    def _to_band(band_or_worker: Union[BandType, str]) -> BandType:
        if isinstance(band_or_worker, tuple) and len(band_or_worker) == 2:
            # band already
            return band_or_worker
        else:
            return band_or_worker, 'numa-0'

    @enter_mode(build=True)
    def gen_subtask_graph(self) -> SubtaskGraph:
        """
        Analyze chunk graph and generate subtask graph.

        Returns
        -------
        subtask_graph: SubtaskGraph
            Subtask graph.
        """
        fetch_removed_chunk_graph = self._chunk_graph.copy()
        reassign_worker_ops = []
        for chunk in self._chunk_graph:
            if isinstance(chunk.op, Fetch):
                fetch_removed_chunk_graph.remove_node(chunk)
            elif chunk.op.reassign_worker:
                reassign_worker_ops.append(chunk.op)

        start_ops = list(self._iter_start_ops(fetch_removed_chunk_graph)) \
            if len(fetch_removed_chunk_graph) > 0 else []

        # assign start chunks
        to_assign_ops = start_ops + reassign_worker_ops
        assigner = self._graph_assigner_cls(
            fetch_removed_chunk_graph, to_assign_ops, self._band_slots)
        # assign expect workers
        cur_assigns = {op.key: self._to_band(op.expect_worker)
                       for op in start_ops if op.expect_worker is not None}
        logger.info('Start to assign %s start chunks.', len(start_ops))
        chunk_to_bands = assigner.assign(cur_assigns=cur_assigns)
        logger.info('Assigned %s start chunks.', len(start_ops))

        # fuse node
        if self._fuse_enabled:
            logger.info('Start to fuse chunks.')
            chunk_to_bands = self._fuse(chunk_to_bands)
            logger.info('Fused chunks.')

        subtask_graph = SubtaskGraph()
        chunk_to_priorities = dict()
        chunk_to_fetch_chunk = dict()
        chunk_to_subtask = dict()
        visited = set()
        for chunk in self._chunk_graph.topological_iter():
            if chunk in visited:
                continue
            if isinstance(chunk.op, Fetch):
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
                 in self._chunk_graph.predecessors(chunk)
                 if not isinstance(inp_chunk.op, Fetch)]
            for inp_subtask in inp_subtasks:
                subtask_graph.add_edge(inp_subtask, subtask)

        return subtask_graph
