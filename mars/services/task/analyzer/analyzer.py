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
from typing import Type, Dict, Tuple

from ....core import ChunkGraph
from ....utils import implements, build_fetch
from ..core import SubtaskGraph, Subtask, new_task_id
from .assigner import AbstractGraphAssigner, GraphAssigner


class AbstractGraphAnalyzer(ABC):
    def __init__(self,
                 chunk_graph: ChunkGraph,
                 band_slots: Dict[Tuple[str, str], int],
                 task_stage_info,
                 graph_assigner_cls: Type[AbstractGraphAssigner] = None):
        self._chunk_graph = chunk_graph
        self._band_slots = band_slots
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

    @implements(AbstractGraphAnalyzer.gen_subtask_graph)
    def gen_subtask_graph(self) -> SubtaskGraph:
        start_ops = list(self._iter_start_ops())

        # assign start chunks
        assigner = self._graph_assigner_cls(
            self._chunk_graph, start_ops, self._band_slots)
        chunk_to_bands = assigner.assign()

        subtask_graph = SubtaskGraph()
        chunk_to_priorities = dict()
        chunk_to_fetch_chunk = dict()
        chunk_to_subtask = dict()
        visited = set()
        for chunk in self._chunk_graph.topological_iter():
            if chunk in visited:
                continue

            inp_chunks = self._chunk_graph.predecessors(chunk)

            # calculate priority
            if inp_chunks:
                priority = max(chunk_to_priorities[inp_chunk]
                               for inp_chunk in inp_chunks) + 1
            else:
                priority = 0
            chunk_to_priorities[chunk] = priority

            band = chunk_to_bands.get(chunk)

            # gen fetch chunks for input chunks
            inp_fetch_chunks = []
            for inp_chunk in inp_chunks:
                if inp_chunk in chunk_to_fetch_chunk:
                    inp_fetch_chunks.append(chunk_to_fetch_chunk[inp_chunk])
                else:
                    fetch_chunk = build_fetch(inp_chunk).data
                    chunk_to_fetch_chunk[inp_chunk] = fetch_chunk
                    inp_fetch_chunks.append(fetch_chunk)

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
                visited.add(out)
                result_chunks.append(copied_chunk)
                subtask_chunk_graph.add_node(copied_chunk)
                for inp_fetch_chunk in inp_fetch_chunks:
                    subtask_chunk_graph.add_node(inp_fetch_chunk)
                    subtask_chunk_graph.add_edge(inp_fetch_chunk, copied_chunk)

            # gen subtask
            subtask = Subtask(
                subtask_id=new_task_id(),
                session_id=self._task_stage_info.task_info.session_id,
                task_id=self._task_stage_info.task_id,
                chunk_graph=subtask_chunk_graph,
                expect_band=band,
                priority=priority)
            for out in outs:
                chunk_to_subtask[out] = subtask
            subtask_graph.add_node(subtask)
            inp_subtasks = [chunk_to_subtask[inp_chunk]
                            for inp_chunk in inp_chunks]
            for inp_subtask in inp_subtasks:
                subtask_graph.add_edge(inp_subtask, subtask)

        return subtask_graph
