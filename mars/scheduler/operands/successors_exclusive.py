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

from ...operands import SuccessorsExclusive
from .base import BaseOperandActor
from .core import OperandState, register_operand_class


class SuccessorsExclusiveOperandActor(BaseOperandActor):
    """
    Actor makes sure only one of successors can be executed at the same time.
    """

    def __init__(self, session_id, graph_id, op_key, op_info, **kwargs):
        super().__init__(session_id, graph_id, op_key, op_info, **kwargs)

        io_meta = self._io_meta
        self._predecessors_to_sucessors = io_meta['predecessors_to_successors']
        self._ready_successors_queue = []
        self._finished_sucessors = set()
        self._is_successor_running = False

    def add_finished_predecessor(self, op_key, worker, output_sizes=None, output_shapes=None):
        super().add_finished_predecessor(op_key, worker, output_sizes=output_sizes,
                                         output_shapes=output_shapes)

        from ..chunkmeta import WorkerMeta
        data_meta = {k: WorkerMeta(chunk_size=v, workers=(worker,), chunk_shape=output_shapes.get(k))
                     for k, v in output_sizes.items()}
        sucessor_op_key = self._predecessors_to_sucessors[op_key]
        self._ready_successors_queue.append((sucessor_op_key, data_meta))
        self._pick_successor_to_run()

    def add_finished_successor(self, op_key, worker):
        super().add_finished_successor(op_key, worker)
        self._finished_sucessors.add(op_key)
        if self._finished_sucessors == set(self._predecessors_to_sucessors.values()):
            self.ref().start_operand(OperandState.FINISHED, _tell=True)
        else:
            self._is_successor_running = False
            self._pick_successor_to_run()

    def _pick_successor_to_run(self):
        if not self._is_successor_running and len(self._ready_successors_queue) > 0:
            to_run_successor_op_key, data_meta = self._ready_successors_queue.pop(0)
            self._get_operand_actor(to_run_successor_op_key).start_operand(
                OperandState.READY, io_meta=dict(input_data_metas=data_meta), _tell=True)
            self._is_successor_running = True

    def update_demand_depths(self, depth):
        pass

    def propose_descendant_workers(self, input_key, worker_scores, depth=1):
        pass

    def free_data(self, state=OperandState.FREED, check=True):
        pass


register_operand_class(SuccessorsExclusive, SuccessorsExclusiveOperandActor)
