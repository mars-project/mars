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

import logging
import operator
import random
from collections import defaultdict, OrderedDict
from functools import reduce

import numpy as np

from ..operands import VirtualOperand
from .operands import OperandState

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

    def collect_external_input_chunks(self, initial=True):
        """
        Collect keys of input chunks not in current graph, for instance,
        chunks as inputs of initial operands in eager mode.
        :param initial: collect initial chunks only
        :return: dict mapping operand key to its input keys
        """
        graph = self._graph
        chunk_keys = set(n.key for n in graph)
        visited = set()
        results = dict()
        for n in graph:
            op_key = n.op.key
            if op_key in visited or not n.inputs:
                continue
            if initial and graph.count_predecessors(n):
                continue
            ext_keys = [c.key for c in n.inputs if c.key not in chunk_keys]
            if not ext_keys:
                continue
            visited.add(op_key)
            results[op_key] = ext_keys
        return results

    @staticmethod
    def _get_workers_with_max_size(worker_to_size):
        """Get workers with maximal size"""
        max_workers = set()
        max_size = 0
        for w, size in worker_to_size.items():
            if size > max_size:
                max_size = size
                max_workers = {w}
            elif size == max_size:
                max_workers.add(w)
        max_workers.difference_update([None])
        return max_size, list(max_workers)

    def _iter_successor_assigns(self, existing_assigns):
        """Iterate over all successors to get allocations of successor nodes"""
        graph = self._graph

        worker_assigns = existing_assigns.copy()
        for n in graph.topological_iter():
            if not graph.count_predecessors(n):
                continue
            pred_sizes = defaultdict(lambda: 0)
            total_count = 0
            worker_involved = set()
            # calculate ratios of every worker in predecessors
            for pred in graph.iter_predecessors(n):
                op_key = pred.op.key
                total_count += 1
                if op_key in worker_assigns:
                    pred_sizes[worker_assigns[op_key]] += 1
                    worker_involved.add(worker_assigns[op_key])
                else:
                    worker_involved.add(None)
            # get the worker occupying most of the data
            max_size, max_workers = self._get_workers_with_max_size(pred_sizes)
            # if there is a dominant worker, return it
            if max_size > total_count / max(2, len(worker_involved)) and max_workers:
                max_worker = random.choice(max_workers)
                worker_assigns[n.op.key] = max_worker
                yield n.op.key, max_worker

    def _calc_worker_assign_limits(self, initial_count, occupied=None):
        """
        Calculate limitation of number of initial operands for workers
        :param initial_count: num of nodes in READY state
        :param occupied: worker -> num of initials already assigned
        """
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

        # all assigned, nothing to do
        if counts.sum() == 0:
            return dict((ep, 0) for ep in endpoints)

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
        visited_keys = set()
        for n in graph:
            if n.op.key in visited_keys:
                continue
            visited_keys.add(n.op.key)
            if not graph.predecessors(n) and n.op.key not in fixed_assigns:
                zero_degrees.append(n)
        random.shuffle(zero_degrees)

        # note that different orders can contribute to different efficiency
        return sorted(zero_degrees, key=lambda n: descendant_sizes[n.op.key])

    def get_initial_operand_keys(self):
        return list(set(n.op.key for n in self._collect_zero_degrees()))

    def _iter_assignments_by_transfer_sizes(self, worker_quotas, input_chunk_metas):
        """
        Assign chunks by input sizes
        :type input_chunk_metas: dict[str, dict[str, mars.scheduler.chunkmeta.WorkerMeta]]
        """
        total_transfers = dict((k, sum(v.chunk_size for v in chunk_to_meta.values()))
                               for k, chunk_to_meta in input_chunk_metas.items())
        # operands with largest amount of data will be allocated first
        sorted_chunks = sorted(total_transfers.keys(), reverse=True,
                               key=lambda k: total_transfers[k])
        for op_key in sorted_chunks:
            # compute data amounts held in workers
            worker_stores = defaultdict(lambda: 0)
            for meta in input_chunk_metas[op_key].values():
                for w in meta.workers:
                    worker_stores[w] += meta.chunk_size

            max_size, max_workers = self._get_workers_with_max_size(worker_stores)
            if max_workers and max_size > 0.5 * total_transfers[op_key]:
                max_worker = random.choice(max_workers)
                if worker_quotas.get(max_worker, 0) <= 0:
                    continue
                worker_quotas[max_worker] -= 1
                yield op_key, max_worker

    def _assign_by_bfs(self, start, worker, initial_sizes, spread_limits,
                       keys_to_assign, assigned_record, graph=None):
        """
        Assign initial nodes using Breadth-first Search given initial sizes and
        limitations of spread range.
        """
        if initial_sizes[worker] <= 0:
            return

        graph = graph or self._graph
        if self._undigraph is None:
            undigraph = self._undigraph = graph.build_undirected()
        else:
            undigraph = self._undigraph

        assigned = 0
        spread_range = 0
        for v in undigraph.bfs(start=start, visit_predicate='all'):
            op_key = v.op.key
            if op_key in assigned_record:
                continue
            spread_range += 1
            if op_key not in keys_to_assign:
                continue
            assigned_record[op_key] = worker
            assigned += 1
            if spread_range >= spread_limits[worker] \
                    or assigned >= initial_sizes[worker]:
                break
        initial_sizes[worker] -= assigned

    def calc_operand_assignments(self, op_keys, input_chunk_metas=None):
        """
        Decide target worker for given chunks.

        :param op_keys: keys of operands to assign
        :param input_chunk_metas: chunk metas for graph-level inputs, grouped by initial chunks
        :type input_chunk_metas: dict[str, dict[str, mars.scheduler.chunkmeta.WorkerMeta]]
        :return: dict mapping operand keys into worker endpoints
        """
        graph = self._graph
        op_states = self._op_states
        cur_assigns = OrderedDict(self._fixed_assigns)

        key_to_chunks = defaultdict(list)
        for n in graph:
            key_to_chunks[n.op.key].append(n)

        descendant_readies = set()
        op_keys = set(op_keys)
        chunks_to_assign = [key_to_chunks[k][0] for k in op_keys]

        if any(graph.count_predecessors(c) for c in chunks_to_assign):
            graph = graph.copy()
            for c in graph:
                if c.op.key not in op_keys:
                    continue
                for pred in graph.predecessors(c):
                    graph.remove_edge(pred, c)

        assigned_counts = defaultdict(lambda: 0)
        worker_op_keys = defaultdict(set)
        if cur_assigns:
            for op_key, state in op_states.items():
                if op_key not in op_keys and state == OperandState.READY \
                        and op_key in cur_assigns:
                    descendant_readies.add(op_key)
                    assigned_counts[cur_assigns[op_key]] += 1

        # calculate the number of nodes to be assigned to every worker
        # given number of workers and existing assignments
        pre_worker_quotas = self._calc_worker_assign_limits(
            len(chunks_to_assign) + len(descendant_readies), assigned_counts)

        # pre-assign nodes given pre-determined transfer sizes
        if not input_chunk_metas:
            worker_quotas = pre_worker_quotas
        else:
            for op_key, worker in self._iter_assignments_by_transfer_sizes(
                    pre_worker_quotas, input_chunk_metas):
                if op_key in cur_assigns or op_key not in op_keys:
                    continue
                assigned_counts[worker] += 1
                cur_assigns[op_key] = worker
                worker_op_keys[worker].add(op_key)

            worker_quotas = self._calc_worker_assign_limits(
                len(chunks_to_assign) + len(descendant_readies), assigned_counts)

        if cur_assigns:
            # calculate ranges of nodes already assigned
            for op_key, worker in self._iter_successor_assigns(cur_assigns):
                cur_assigns[op_key] = worker
                worker_op_keys[worker].add(op_key)

        logger.debug('Worker assign quotas: %r', worker_quotas)

        # calculate expected descendant count (spread range) of
        # every worker and subtract assigned number from it
        average_spread_range = len(graph) * 1.0 / len(self._worker_slots)
        spread_ranges = defaultdict(lambda: average_spread_range)
        for worker in cur_assigns.values():
            spread_ranges[worker] -= 1

        logger.debug('Scan spread ranges: %r', dict(spread_ranges))

        # assign pass 1: assign from fixed groups
        sorted_workers = sorted(worker_op_keys, reverse=True, key=lambda k: len(worker_op_keys[k]))
        for worker in sorted_workers:
            start_chunks = reduce(operator.add, (key_to_chunks[op_key] for op_key in worker_op_keys[worker]))
            self._assign_by_bfs(start_chunks, worker, worker_quotas, spread_ranges,
                                op_keys, cur_assigns, graph=graph)

        # assign pass 2: assign from other nodes to be assigned
        sorted_candidates = [v for v in chunks_to_assign]
        while max(worker_quotas.values()):
            worker = max(worker_quotas, key=lambda k: worker_quotas[k])
            cur = sorted_candidates.pop()
            while cur.op.key in cur_assigns:
                cur = sorted_candidates.pop()
            self._assign_by_bfs(cur, worker, worker_quotas, spread_ranges, op_keys,
                                cur_assigns, graph=graph)

        # FIXME: force to assign vineyard source/sink ops to their `expect_worker` even when
        # the worker has an input.
        # The special case should be fixed by respecting the `expect_worker` attribute of an
        # op, regardless it has inputs or not.
        #
        # See also: "TODO refine this to support mixed scenarios here" in graph.py.
        keys_to_assign = {n.op.key: n.op for n in chunks_to_assign}
        assignments = OrderedDict()
        for k, v in cur_assigns.items():
            if k in keys_to_assign:
                if keys_to_assign[k].expect_worker is not None and \
                        'Vineyard' in type(keys_to_assign[k]).__name__:
                    assignments[k] = keys_to_assign[k].expect_worker
                else:
                    assignments[k] = v
        return assignments

    def analyze_state_changes(self):
        """
        Update operand states when some chunks are lost.
        :return: dict mapping operand keys into changed states
        """
        graph = self._graph
        lost_chunks = set(self._lost_chunks)
        op_states = self._op_states

        # mark lost virtual nodes as lost when some preds are lost
        for n in graph:
            if not isinstance(n.op, VirtualOperand) \
                    or op_states.get(n.op.key) == OperandState.UNSCHEDULED:
                continue
            if any(pred.key in lost_chunks for pred in graph.iter_predecessors(n)):
                lost_chunks.add(n.key)

        # collect operands with lost data
        op_key_to_chunks = defaultdict(list)
        lost_ops = set()
        for n in graph:
            op_key_to_chunks[n.op.key].append(n)
            if n.key in lost_chunks:
                lost_ops.add(n.op.key)

        # check data on finished operands. when data lost, mark the operand
        # and its successors as affected.
        affected_op_keys = set()
        for op_key in lost_ops:
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
                # 2. state does not hold data, or data is lost,
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
