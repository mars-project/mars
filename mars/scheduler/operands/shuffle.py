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
from collections import defaultdict

from ...operands import ShuffleProxy
from ...errors import WorkerDead
from .base import BaseOperandActor
from .core import register_operand_class, rewrite_worker_errors, OperandState

logger = logging.getLogger(__name__)


class ShuffleProxyActor(BaseOperandActor):
    def __init__(self, session_id, graph_id, op_key, op_info, **kwargs):
        super().__init__(session_id, graph_id, op_key, op_info, **kwargs)
        self._session_id = session_id
        self._graph_id = graph_id
        self._op_key = op_key

        io_meta = self._io_meta
        self._shuffle_keys_to_op = dict(zip(io_meta['shuffle_keys'], io_meta['successors']))
        self._op_to_shuffle_keys = dict(zip(io_meta['successors'], io_meta['shuffle_keys']))

        self._worker_to_mappers = defaultdict(set)
        self._reducer_workers = dict()

        self._all_deps_built = False
        self._mapper_op_to_chunk = dict()
        self._reducer_to_mapper = defaultdict(dict)

    def add_finished_predecessor(self, op_key, worker, output_sizes=None, output_shapes=None):
        super().add_finished_predecessor(op_key, worker, output_sizes=output_sizes,
                                         output_shapes=output_shapes)

        from ..chunkmeta import WorkerMeta
        chunk_key = next(iter(output_sizes.keys()))[0]
        self._mapper_op_to_chunk[op_key] = chunk_key
        if op_key not in self._worker_to_mappers[worker]:
            self._worker_to_mappers[worker].add(op_key)
            self.chunk_meta.add_worker(self._session_id, chunk_key, worker, _tell=True)

        shuffle_keys_to_op = self._shuffle_keys_to_op

        if not self._reducer_workers:
            self._reducer_workers = self._graph_refs[0].assign_operand_workers(
                self._succ_keys, input_chunk_metas=self._reducer_to_mapper)
        reducer_workers = self._reducer_workers
        data_to_addresses = dict()

        for (chunk_key, shuffle_key), data_size in output_sizes.items() or ():
            succ_op_key = shuffle_keys_to_op[shuffle_key]
            meta = self._reducer_to_mapper[succ_op_key][op_key] = \
                WorkerMeta(chunk_size=data_size, workers=(worker,),
                           chunk_shape=output_shapes.get((chunk_key, shuffle_key)))
            reducer_worker = reducer_workers.get(succ_op_key)
            if reducer_worker and reducer_worker != worker:
                data_to_addresses[(chunk_key, shuffle_key)] = [reducer_worker]
                meta.workers += (reducer_worker,)

        if data_to_addresses:
            try:
                with rewrite_worker_errors():
                    self._get_raw_execution_ref(address=worker) \
                        .send_data_to_workers(self._session_id, data_to_addresses, _tell=True)
            except WorkerDead:
                self._resource_ref.detach_dead_workers([worker], _tell=True)

        if all(k in self._finish_preds for k in self._pred_keys):
            self._start_successors()

    def _start_successors(self):
        self._all_deps_built = True
        futures = []

        logger.debug('Predecessors of shuffle proxy %s done, notifying successors', self._op_key)
        for succ_key in self._succ_keys:
            if succ_key in self._finish_succs:
                continue

            shuffle_key = self._op_to_shuffle_keys[succ_key]
            input_data_metas = dict(((self._mapper_op_to_chunk[k], shuffle_key), meta)
                                    for k, meta in self._reducer_to_mapper[succ_key].items())

            futures.append(self._get_operand_actor(succ_key).start_operand(
                OperandState.READY, io_meta=dict(input_data_metas=input_data_metas),
                target_worker=self._reducer_workers.get(succ_key),
                _tell=True, _wait=False))

        [f.result() for f in futures]
        self.ref().start_operand(OperandState.FINISHED, _tell=True)

    def add_finished_successor(self, op_key, worker):
        super().add_finished_successor(op_key, worker)
        shuffle_key = self._op_to_shuffle_keys[op_key]

        # input data in reduce nodes can be freed safely
        data_keys = []
        workers_list = []
        for pred_key, meta in self._reducer_to_mapper[op_key].items():
            data_keys.append((self._mapper_op_to_chunk[pred_key], shuffle_key))
            workers_list.append((self._reducer_workers[op_key],))
        self._free_data_in_worker(data_keys, workers_list)

        if all(k in self._finish_succs for k in self._succ_keys):
            self.free_predecessors()

    def free_predecessors(self):
        can_be_freed, deterministic = self.check_can_be_freed()
        if not deterministic:
            # if we cannot determine whether to do failover, just delay and retry
            self.ref().free_predecessors(_delay=1, _tell=True)
            return
        elif not can_be_freed:
            return

        futures = []
        for k in self._pred_keys:
            futures.append(self._get_operand_actor(k).start_operand(
                OperandState.FREED, _tell=True, _wait=False))

        data_keys = []
        workers_list = []
        for op_key in self._succ_keys:
            shuffle_key = self._op_to_shuffle_keys[op_key]
            for pred_key, meta in self._reducer_to_mapper[op_key].items():
                data_keys.append((self._mapper_op_to_chunk[pred_key], shuffle_key))
                workers_list.append(tuple(set(meta.workers + (self._reducer_workers[op_key],))))
        self._free_data_in_worker(data_keys, workers_list)

        inp_chunk_keys = [self._mapper_op_to_chunk[k] for k in self._pred_keys
                          if k in self._mapper_op_to_chunk]
        self.chunk_meta.batch_delete_meta(
            self._session_id, inp_chunk_keys, _tell=True, _wait=False)
        self._finish_preds = set()
        [f.result() for f in futures]

        self.ref().start_operand(OperandState.FREED, _tell=True)

    def update_demand_depths(self, depth):
        pass

    def propose_descendant_workers(self, input_key, worker_scores, depth=1):
        pass

    def move_failover_state(self, from_states, state, new_target, dead_workers):
        if self.state not in from_states:
            return

        dead_workers = set(dead_workers)
        for w in dead_workers:
            self._finish_preds.difference_update(self._worker_to_mappers[w])
            del self._worker_to_mappers[w]

        for op_key in self._succ_keys:
            if op_key not in self._reducer_to_mapper:
                continue
            new_mapper_metas = dict()
            for pred_key, meta in self._reducer_to_mapper[op_key].items():
                meta.workers = tuple(w for w in meta.workers if w not in dead_workers)
                if meta.workers:
                    new_mapper_metas[pred_key] = meta
            self._reducer_to_mapper[op_key] = new_mapper_metas

        missing_succs = []
        for op, w in self._reducer_workers.items():
            if w in dead_workers:
                missing_succs.append(op)
        self._finish_succs.difference_update(missing_succs)

        if missing_succs:
            self._reducer_workers.update(self._graph_refs[0].assign_operand_workers(
                missing_succs, input_chunk_metas=self._reducer_to_mapper))

        super().move_failover_state(
            from_states, state, new_target, dead_workers)

    def free_data(self, state=OperandState.FREED, check=True):
        pass


register_operand_class(ShuffleProxy, ShuffleProxyActor)
