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
from collections import defaultdict
from operator import itemgetter
from typing import List, Dict, Union

import numpy as np

from ....core import ChunkGraph, ChunkData
from ....core.operand import Operand, Fetch
from ....lib.ordered_set import OrderedSet
from ....resource import Resource
from ....typing import BandType
from ....utils import implements


class AbstractGraphAssigner(ABC):
    """
    Assign start nodes.
    """

    def __init__(
        self,
        chunk_graph: ChunkGraph,
        start_ops: List[Operand],
        band_resource: Dict[BandType, Resource],
    ):
        self._chunk_graph = chunk_graph
        self._start_ops = start_ops
        self._band_resource = band_resource

    @abstractmethod
    def assign(self, cur_assigns: Dict[str, str] = None) -> Dict[ChunkData, BandType]:
        """
        Assign start nodes to bands.

        cur_assigns : dict
            op already assigned.

        Returns
        -------
        node_to_bands : dict
            From node to band.
        """

    def _is_gpu_band(self) -> bool:
        gpu_ops = (
            [op for op in self._start_ops if not isinstance(op, Fetch)]
            if self._start_ops
            else []
        )
        if gpu_ops and all(op.gpu for op in gpu_ops):
            return True
        return False

    def get_device_band_slots(self) -> Dict[BandType, int]:
        if self._is_gpu_band():  # pragma: no cover
            band_prefix = "gpu"
        else:
            band_prefix = "numa"
        return {
            band: resource.num_cpus or resource.num_gpus
            for band, resource in self._band_resource.items()
            if band[1].startswith(band_prefix)
        }


class GraphAssigner(AbstractGraphAssigner):
    def __init__(
        self,
        chunk_graph: ChunkGraph,
        start_ops: List[Operand],
        band_resource: Dict[BandType, Resource],
    ):
        super().__init__(chunk_graph, start_ops, band_resource)
        self._op_keys: OrderedSet = OrderedSet([start_op.key for start_op in start_ops])

    def _calc_band_assign_limits(
        self, initial_count: int, occupied: Dict[BandType, int]
    ) -> Dict[BandType, int]:
        """
        Calculate limitation of number of initial operands for bands.

        Parameters
        ----------
        initial_count : int
            Number of nodes that is ready for running.
        occupied : dict
           Band to those initials that already assigned.

        Returns
        -------
        slot_assign_limits: dict
            Slot to limitation of number of initial operands.
        """
        actual_count: int = initial_count - sum(occupied.values())
        band_slots = sorted(
            self.get_device_band_slots().items(), key=itemgetter(1), reverse=True
        )
        bands: List[BandType] = [it[0] for it in band_slots]
        slots = np.asarray([it[1] for it in band_slots], dtype=np.float32)

        # remove assigned nodes from limitations
        counts = initial_count * slots / slots.sum()
        for i, band in enumerate(bands):
            counts[i] = max(0, counts[i] - occupied.get(band, 0))

        # all assigned, nothing to do
        if counts.sum() == 0:
            return {band: 0 for band in bands}

        # assign remaining nodes
        counts = (actual_count * counts / counts.sum()).astype(np.int32)
        pos = 0
        rest = actual_count - counts.sum()
        while rest > 0:
            counts[pos] += 1
            rest -= 1
            pos = (pos + 1) % len(counts)
        return dict(zip(bands, counts))

    @classmethod
    def _assign_by_bfs(
        cls,
        undirected_chunk_graph: ChunkGraph,
        start: ChunkData,
        band: BandType,
        initial_sizes: Dict[BandType, int],
        spread_limits: Dict[BandType, float],
        key_to_assign: OrderedSet,
        assigned_record: Dict[str, Union[str, BandType]],
    ):
        """
        Assign initial nodes using breath-first search given initial sizes and
        limitations of spread range.
        """
        if initial_sizes[band] <= 0:
            return

        assigned = 0
        spread_range = 0
        for chunk in undirected_chunk_graph.bfs(start=start, visit_predicate="all"):
            op_key = chunk.op.key
            if op_key in assigned_record:
                continue
            spread_range += 1
            # `op_key` may not be in `key_to_assign`,
            # but we need to record it to avoid iterate the node repeatedly.
            assigned_record[op_key] = band
            if op_key not in key_to_assign:
                continue
            assigned += 1
            if spread_range >= spread_limits[band] or assigned >= initial_sizes[band]:
                break
        initial_sizes[band] -= assigned

    def _build_undirected_chunk_graph(
        self, chunk_to_assign: List[ChunkData]
    ) -> ChunkGraph:
        chunk_graph = self._chunk_graph.copy()
        # remove edges for all chunk_to_assign which may contain chunks
        # that need be reassigned
        for chunk in chunk_to_assign:
            if chunk_graph.count_predecessors(chunk) > 0:
                for pred in list(chunk_graph.predecessors(chunk)):
                    chunk_graph.remove_edge(pred, chunk)
        return chunk_graph.build_undirected()

    @implements(AbstractGraphAssigner.assign)
    def assign(
        self, cur_assigns: Dict[str, BandType] = None
    ) -> Dict[ChunkData, BandType]:
        graph = self._chunk_graph
        assign_result = dict()
        cur_assigns = cur_assigns or dict()
        # assigned by expect worker or band
        initial_assigned_op_keys = set(cur_assigns)

        op_key_to_chunks = defaultdict(list)
        for chunk in graph:
            op_key_to_chunks[chunk.op.key].append(chunk)

        op_keys = OrderedSet(self._op_keys)
        chunk_to_assign = [
            op_key_to_chunks[op_key][0]
            for op_key in op_keys
            if op_key not in cur_assigns
        ]
        assigned_counts = defaultdict(lambda: 0)
        for band in cur_assigns.values():
            assigned_counts[band] += 1

        # build undirected graph
        undirected_chunk_graph = self._build_undirected_chunk_graph(chunk_to_assign)

        # calculate the number of chunks to be assigned to each band
        # given number of bands and existing assignments
        band_quotas = self._calc_band_assign_limits(
            len(chunk_to_assign) + sum(assigned_counts.values()), assigned_counts
        )

        # calculate expected descendant count (spread range) of
        # every band and subtract assigned number from it
        average_spread_range = len(graph) * 1.0 / len(self.get_device_band_slots())
        spread_ranges = defaultdict(lambda: average_spread_range)
        # assign from other chunks to be assigned
        # TODO: sort by what?
        sorted_candidates = chunk_to_assign.copy()
        while max(band_quotas.values()):
            band = max(band_quotas, key=lambda k: band_quotas[k])
            cur = sorted_candidates.pop()
            while cur.op.key in cur_assigns:
                cur = sorted_candidates.pop()
            self._assign_by_bfs(
                undirected_chunk_graph,
                cur,
                band,
                band_quotas,
                spread_ranges,
                op_keys,
                cur_assigns,
            )

        key_to_assign = {n.op.key for n in chunk_to_assign} | initial_assigned_op_keys
        for op_key, band in cur_assigns.items():
            if op_key in key_to_assign:
                for chunk in op_key_to_chunks[op_key]:
                    assign_result[chunk] = band

        return assign_result
