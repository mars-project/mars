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

from collections import defaultdict

from ...operands import ShuffleProxy
from .base import BaseOperandActor
from .core import register_operand_class, OperandState


class ShuffleProxyActor(BaseOperandActor):
    def __init__(self, session_id, graph_id, op_key, op_info, **kwargs):
        super(ShuffleProxyActor, self).__init__(session_id, graph_id, op_key, op_info, **kwargs)
        self._session_id = session_id
        self._graph_id = graph_id
        self._op_key = op_key

        io_meta = self._io_meta
        self._shuffle_keys_to_op = dict(zip(io_meta['shuffle_keys'], io_meta['successors']))
        self._op_to_shuffle_keys = dict(zip(io_meta['successors'], io_meta['shuffle_keys']))

        self._reducer_workers = dict()

        self._all_deps_built = False
        self._mapper_to_reducer = defaultdict(set)
        self._reducer_to_mapper = defaultdict(dict)

    def _submit_successor_operand(self, reducer_op_key, wait=True):
        if not self._reducer_workers:
            self._reducer_workers = self._graph_ref.assign_operand_workers(
                self._succ_keys, input_chunk_metas=self._reducer_to_mapper)

        shuffle_key = self._op_to_shuffle_keys[reducer_op_key]
        input_data_keys = [(k, shuffle_key) for k in self._reducer_to_mapper[reducer_op_key]]
        return self._get_operand_actor(reducer_op_key).start_operand(
            OperandState.READY, io_meta=dict(input_data_keys=input_data_keys),
            target_worker=self._reducer_workers.get(reducer_op_key),
            _tell=True, _wait=wait)

    def add_running_predecessor(self, op_key, worker):
        pass

    def add_finished_predecessor(self, op_key, worker, output_sizes=None):
        super(ShuffleProxyActor, self).add_finished_predecessor(op_key, worker, output_sizes=output_sizes)

        from ..chunkmeta import WorkerMeta
        shuffle_keys_to_op = self._shuffle_keys_to_op

        for (op_key, shuffle_key), data_size in output_sizes.items() or ():
            succ_op_key = shuffle_keys_to_op[shuffle_key]
            self._mapper_to_reducer[op_key].add(succ_op_key)
            self._reducer_to_mapper[succ_op_key][op_key] = WorkerMeta(chunk_size=data_size, workers=(worker,))

        if len(self._mapper_to_reducer[op_key]) == len(self._succ_keys):
            del self._mapper_to_reducer[op_key]

        if all(k in self._finish_preds for k in self._pred_keys):
            self._all_deps_built = True
            futures = []

            for op_key in self._succ_keys:
                futures.append(self._submit_successor_operand(op_key, wait=False))

            [f.result() for f in futures]
            self.ref().start_operand(OperandState.FINISHED, _tell=True)

    def add_finished_successor(self, op_key):
        super(ShuffleProxyActor, self).add_finished_successor(op_key)
        shuffle_key = self._op_to_shuffle_keys[op_key]

        data_keys = [(pred_key, shuffle_key) for pred_key in self._reducer_to_mapper[op_key]]
        self._free_data_in_worker(data_keys)

        futures = []
        if all(k in self._finish_succs for k in self._succ_keys):
            for k in self._pred_keys:
                futures.append(self._get_operand_actor(k).start_operand(
                    OperandState.FREED, _tell=True, _wait=False))
        [f.result() for f in futures]

        self.ref().start_operand(OperandState.FREED, _tell=True)

    def update_demand_depths(self, depth):
        pass

    def propose_descendant_workers(self, input_key, worker_scores, depth=1):
        pass

    def move_failover_state(self, from_states, state, new_target, dead_workers):
        pass

    def free_data(self, state=OperandState.FREED):
        pass


register_operand_class(ShuffleProxy, ShuffleProxyActor)
