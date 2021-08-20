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
from typing import List, Dict, Set

import numpy as np

from ....core import ChunkGraph, ChunkData
from ....core.operand import Operand
from ....typing import BandType
from ....utils import implements


class AbstractGraphAssigner(ABC):
    """
    Assign start nodes.
    """

    def __init__(self,
                 chunk_graph: ChunkGraph,
                 start_ops: List[Operand],
                 band_slots: Dict[BandType, int]):
        self._chunk_graph = chunk_graph
        self._start_ops = start_ops
        self._band_slots = band_slots

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

    def get_device_band_slots(self) -> Dict[BandType, int]:
        if self._start_ops and all(op.gpu for op in self._start_ops):  # pragma: no cover
            band_prefix = 'gpu'
        else:
            band_prefix = 'numa'
        return {band: slots for band, slots in self._band_slots.items()
                if band[1].startswith(band_prefix)}


class GraphAssigner(AbstractGraphAssigner):
    def __init__(self,
                 chunk_graph: ChunkGraph,
                 start_ops: List[Operand],
                 band_slots: Dict[BandType, int]):
        super().__init__(chunk_graph, start_ops, band_slots)
        self._undirected_chunk_graph = None
        self._op_keys: Set[str] = {start_op.key for start_op in start_ops}

    def _calc_band_assign_limits(self,
                                 initial_count: int,
                                 occupied: Dict[BandType, int]) \
            -> Dict[BandType, int]:
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
            Slot to limination of number of initial operands.
        """
        actual_count: int = initial_count - sum(occupied.values())
        band_slots = sorted(self.get_device_band_slots().items(),
                            key=itemgetter(1), reverse=True)
        bands: List[BandType] = [it[0] for it in band_slots]
        slots = np.asarray([it[1] for it in band_slots], dtype=np.float32)

        # remove assigned nodes from limitatins
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

    def _assign_by_bfs(self,
                       start: ChunkData,
                       band: BandType,
                       initial_sizes: Dict[BandType, int],
                       spread_limits: Dict[BandType, float],
                       key_to_assign: Set[str],
                       assigned_record: Dict[str, int]):
        """
        Assign initial nodes using breath-first search given initial sizes and
        limitations of spread range.
        """
        if initial_sizes[band] <= 0:
            return

        graph = self._chunk_graph
        if self._undirected_chunk_graph is None:
            self._undirected_chunk_graph = graph.build_undirected()
        undirected_chunk_graph = self._undirected_chunk_graph

        assigned = 0
        spread_range = 0
        for chunk in undirected_chunk_graph.bfs(start=start,
                                                visit_predicate='all'):
            op_key = chunk.op.key
            if op_key in assigned_record:
                continue
            spread_range += 1
            if op_key not in key_to_assign:
                continue
            assigned_record[op_key] = band
            assigned += 1
            if spread_range >= spread_limits[band] or \
                    assigned >= initial_sizes[band]:
                break
        initial_sizes[band] -= assigned

    @implements(AbstractGraphAssigner.assign)
    def assign(self, cur_assigns: Dict[str, str] = None) -> Dict[ChunkData, BandType]:
        graph = self._chunk_graph
        assign_result = dict()
        cur_assigns = cur_assigns or dict()
        # assigned by expect worker
        initial_assigned_op_keys = set(cur_assigns)

        op_key_to_chunks = defaultdict(list)
        for chunk in graph:
            op_key_to_chunks[chunk.op.key].append(chunk)

        op_keys = set(self._op_keys)
        chunk_to_assign = [op_key_to_chunks[op_key][0]
                           for op_key in op_keys
                           if op_key not in cur_assigns]
        assigned_counts = defaultdict(lambda: 0)
        for band in cur_assigns.values():
            assigned_counts[band] += 1

        # calculate the number of chunks to be assigned to each band
        # given number of bands and existing assignments
        band_quotas = self._calc_band_assign_limits(
            len(chunk_to_assign), assigned_counts)

        # calculate expected descendant count (spread range) of
        # every band and subtract assigned number from it
        average_spread_range = len(graph) * 1.0 / len(self.get_device_band_slots())
        spread_ranges = defaultdict(lambda: average_spread_range)
        # assign from other chunks to be assigned
        sorted_candidates = [v for v in chunk_to_assign]
        while max(band_quotas.values()):
            band = max(band_quotas, key=lambda k: band_quotas[k])
            cur = sorted_candidates.pop()
            while cur.op.key in cur_assigns:
                cur = sorted_candidates.pop()
            self._assign_by_bfs(cur, band, band_quotas, spread_ranges,
                                op_keys, cur_assigns)

        key_to_assign = \
            {n.op.key for n in chunk_to_assign} | initial_assigned_op_keys
        for op_key, band in cur_assigns.items():
            if op_key in key_to_assign:
                for chunk in op_key_to_chunks[op_key]:
                    assign_result[chunk] = band

        return assign_result
