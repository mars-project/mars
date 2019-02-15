# Copyright 1999-2018 Alibaba Group Holding Ltd.
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
import operator
import random
from collections import defaultdict, OrderedDict

import numpy as np

from ..compat import reduce
from .utils import remove_shuffle_chunks, OperandState

logger = logging.getLogger(__name__)


class GraphAnalyzer(object):
    """
    Analyzer for chunk graph, supporting optimization analysis
    as well as fail-over analysis.
    """
    def __init__(self, graph, worker_slots, fixed_assigns=None, op_states=None,
                 lost_chunks=None):
        """
        :param graph: chunk graph
        :param worker_slots: dict mapping worker endpoint to slots available
        :param fixed_assigns: dict mapping operands to workers fixed
        :param op_states:  dict recording operand states
        :param lost_chunks: keys of lost chunks, for fail-over analysis
        """
        self._graph = graph
        self._undigraph = None
        self._worker_slots = OrderedDict(worker_slots)
        self._op_states = op_states or dict()
        self._lost_chunks = lost_chunks or []
        self._descendant_sizes = defaultdict(lambda: 0)

        self._fixed_assigns = fixed_assigns or dict()
        self._fixed_assigns = dict((k, v) for k, v in self._fixed_assigns.items() if v in self._worker_slots)

    def update_operand_states(self, new_states):
        self._op_states.update(new_states)

    def calc_depths(self):
        """
        Calculate depths of every operand
        :return: dict mapping operand keys into depth
        """
        graph = self._graph
        depth_cache = dict()

        for n in graph.topological_iter():
            preds = graph.predecessors(n)
            if not preds:
                depth_cache[n.op.key] = 0
            else:
                depth_cache[n.op.key] = 1 + max(depth_cache[ni.op.key] for ni in preds)
        return depth_cache

    def calc_descendant_sizes(self):
        """
        Estimate descendant sizes of every operand
        Note that due to performance requirements, the calculated descendant
        sizes are not accurate.
        :return: dict mapping operand keys into estimated descendant size
        """
        graph = self._graph
        sizes = self._descendant_sizes
        for n in graph.topological_iter(reverse=True):
            sizes[n.op.key] += 1
            if not graph.count_predecessors(n):
                continue
            for ni in set(graph.predecessors(n)):
                sizes[ni.op.key] += sizes[n.op.key]
        return sizes

    def _iter_successor_assigns(self, existing_assigns):
        """Iterate over all successors to get allocations of successor nodes"""
        graph = self._graph

        worker_assigns = existing_assigns.copy()
        for n in graph.topological_iter():
            if not graph.count_predecessors(n):
                continue
            pred_sizes = defaultdict(lambda: 0)
            total_size = 0
            worker_involved = set()
            # calculate ratios of every worker in predecessors
            for pred in graph.iter_predecessors(n):
                op_key = pred.op.key
                pred_bytes = pred.rough_nbytes
                total_size += pred_bytes
                if op_key in worker_assigns:
                    pred_sizes[worker_assigns[op_key]] += pred_bytes
                    worker_involved.add(worker_assigns[op_key])
                else:
                    worker_involved.add(None)
            # get the worker occupying most of the data
            max_worker = None
            max_size = 0
            for w, size in pred_sizes.items():
                if size > max_size:
                    max_size = size
                    max_worker = w
            # if there is a dominant worker, return it
            if max_size > total_size / max(2, len(worker_involved)) and \
                    max_worker is not None:
                worker_assigns[n.op.key] = max_worker
                yield n.op.key, max_worker

    def _calc_worker_initial_limits(self, initial_count, occupied=None):
        occupied = occupied or dict()
        actual_count = initial_count - sum(occupied.values())

        endpoint_res = sorted(self._worker_slots.items(), key=operator.itemgetter(1),
                              reverse=True)

        endpoints = [t[0] for t in endpoint_res]
        endpoint_cores = np.array([t[1] for t in endpoint_res]).astype(np.float32)

        # remove assigned nodes from limitations
        counts = initial_count * endpoint_cores / endpoint_cores.sum()
        for idx, ep in enumerate(endpoints):
            counts[idx] = max(0, counts[idx] - occupied.get(ep, 0))
        counts = (actual_count * counts / counts.sum()).astype(np.int32)

        # assign remaining nodes
        pos = 0
        rest = actual_count - counts.sum()
        while rest > 0:
            counts[pos] += 1
            rest -= 1
            pos = (pos + 1) % len(counts)
        return dict(zip(endpoints, counts))

    def _collect_zero_degrees(self):
        """Collect unassigned initial nodes"""
        if not self._descendant_sizes:
            self.calc_descendant_sizes()

        graph = self._graph
        descendant_sizes = self._descendant_sizes
        fixed_assigns = self._fixed_assigns

        zero_degrees = list()
        descendant_readies = dict()
        visited_keys = set()
        for n in graph:
            if n.op.key in visited_keys:
                continue
            visited_keys.add(n.op.key)
            if not remove_shuffle_chunks(graph.predecessors(n)) and n.op.key not in fixed_assigns:
                zero_degrees.append(n)
        random.shuffle(zero_degrees)

        assigned_initials = defaultdict(lambda: 0)
        for worker in descendant_readies.values():
            assigned_initials[worker] += 1

        # note that different orders can contribute to different efficiency
        return sorted(zero_degrees, key=lambda n: descendant_sizes[n.op.key])

    def _assign_by_bfs(self, start, worker, initial_sizes, spread_limits, assigned_record):
        """
        Assign initial nodes using Breadth-first Search given initial sizes and
        limitations of spread range.
        """
        graph = self._graph
        if self._undigraph is None:
            undigraph = self._undigraph = graph.build_undirected()
        else:
            undigraph = self._undigraph

        def _successor_fun(n):
            chunks = remove_shuffle_chunks(undigraph.iter_successors(n))
            random.shuffle(chunks)
            return chunks

        assigned = 0
        spread_range = 0
        for v in undigraph.bfs(start=start, visit_predicate='all',
                               successors=_successor_fun):
            if v.op.key in assigned_record:
                continue
            spread_range += 1
            if graph.predecessors(v):
                continue
            assigned_record[v.op.key] = worker
            assigned += 1
            if spread_range >= spread_limits[worker] \
                    or assigned >= initial_sizes[worker]:
                break
        initial_sizes[worker] -= assigned

    def calc_initial_assignments(self):
        """
        Decide target worker for initial chunks. This function works when
        initializing a new graph, or recovering from worker losses.
        :return: dict mapping operand keys into worker endpoints
        """
        graph = self._graph
        op_states = self._op_states
        cur_assigns = self._fixed_assigns.copy()

        # collect chunks with no inputs or all inputs from shuffle, and nodes
        zero_degrees = self._collect_zero_degrees()

        descendant_readies = set()

        assigned_initial_counts = defaultdict(lambda: 0)
        worker_op_keys = defaultdict(list)
        if cur_assigns:
            # calculate ranges of nodes already assigned
            for op_key, worker in self._iter_successor_assigns(cur_assigns):
                if op_states.get(op_key) == OperandState.READY:
                    descendant_readies.add(op_key)
                    assigned_initial_counts[worker] += 1
                cur_assigns[op_key] = worker
                worker_op_keys[worker].append(op_key)

        # calculate the number of initial nodes to be assigned to every worker
        # given number of workers and existing assignments
        worker_quotas = self._calc_worker_initial_limits(
            len(zero_degrees) + len(descendant_readies), assigned_initial_counts)

        logger.debug('Worker initial quotas: %r', worker_quotas)

        # calculate expected descendant count (spread range) of
        # every worker and subtract assigned number from it
        average_spread_range = len(graph) * 1.0 / len(self._worker_slots)
        spread_ranges = defaultdict(lambda: average_spread_range)
        for worker in cur_assigns.values():
            spread_ranges[worker] -= 1

        logger.debug('Scan spread ranges: %r', dict(spread_ranges))

        key_to_chunks = defaultdict(list)
        for n in graph:
            key_to_chunks[n.op.key].append(n)

        # assign pass 1: assign from fixed groups
        sorted_workers = sorted(worker_op_keys, reverse=True, key=lambda k: len(worker_op_keys[k]))
        for worker in sorted_workers:
            start_chunks = reduce(operator.add, (key_to_chunks[op_key] for op_key in worker_op_keys[worker]))
            self._assign_by_bfs(start_chunks, worker, worker_quotas, spread_ranges, cur_assigns)

        # assign pass 2: assign from initial nodes
        sorted_initials = [v for v in zero_degrees]
        while max(worker_quotas.values()):
            worker = max(worker_quotas, key=lambda k: worker_quotas[k])
            cur = sorted_initials.pop()
            while cur.op.key in cur_assigns:
                cur = sorted_initials.pop()
            self._assign_by_bfs(cur, worker, worker_quotas, spread_ranges, cur_assigns)

        return dict((n.op.key, cur_assigns[n.op.key]) for n in zero_degrees)

    def analyze_state_changes(self):
        """
        Update operand states when some chunks are lost.
        :return: dict mapping operand keys into changed states
        """
        graph = self._graph
        lost_chunks = set(self._lost_chunks)
        op_states = self._op_states

        op_key_to_chunks = defaultdict(list)
        lost_ops = set()
        for n in graph:
            op_key_to_chunks[n.op.key].append(n)
            if n.key in lost_chunks:
                lost_ops.add(n.op.key)

        # check data on finished operands. when data lost, mark the operand
        # and its successors as affected.
        affected_op_keys = set()
        for op_key, state in op_states.items():
            if state != OperandState.FINISHED or op_key not in lost_ops:
                continue
            affected_op_keys.add(op_key)
            for n in op_key_to_chunks[op_key]:
                affected_op_keys.update(succ.op.key for succ in graph.iter_successors(n))

        # scan the graph from bottom and reassign new states
        new_states = dict()
        for chunk in graph.topological_iter(reverse=True):
            op_key = chunk.op.key
            if chunk.op.key not in affected_op_keys:
                continue

            can_be_ready = True
            stop_spread_states = (OperandState.RUNNING, OperandState.FINISHED)
            for pred in graph.iter_predecessors(chunk):
                pred_op_key = pred.op.key
                # mark affected, if
                # 1. data of the operand is lost
                # 2. state does not hold data, or is calculating,
                #    for instance, operand is freed.
                if pred.key in lost_chunks or op_states.get(pred_op_key) not in stop_spread_states:
                    affected_op_keys.add(pred_op_key)
                    can_be_ready = False

            # update state given data preservation of prior nodes
            chunk_op_state = op_states.get(op_key)
            if can_be_ready and chunk_op_state != OperandState.READY:
                new_states[op_key] = OperandState.READY
            elif not can_be_ready and chunk_op_state != OperandState.UNSCHEDULED:
                new_states[op_key] = OperandState.UNSCHEDULED
        op_states.update(new_states)
        return new_states
