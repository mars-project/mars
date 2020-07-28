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

from ...errors import WorkerDead
from ..utils import SchedulerActor
from .core import OperandState, rewrite_worker_errors

logger = logging.getLogger(__name__)


class BaseOperandActor(SchedulerActor):
    @staticmethod
    def gen_uid(session_id, op_key):
        return 's:h1:operand$%s$%s' % (session_id, op_key)

    def __init__(self, session_id, graph_id, op_key, op_info, worker=None,
                 with_kvstore=True, schedulers=None):
        super().__init__()
        self._session_id = session_id
        self._graph_ids = [graph_id]
        self._info = op_info
        self._op_key = op_key
        self._op_path = '/sessions/%s/operands/%s' % (self._session_id, self._op_key)

        self._is_initial = op_info.get('is_initial') or False
        self._is_terminal = op_info.get('is_terminal') or False
        # worker actually assigned
        self._worker = worker

        self._op_name = op_info['op_name']
        self._state = self._last_state = op_info['state']
        io_meta = self._io_meta = op_info['io_meta']
        self._pred_keys = set(io_meta['predecessors'])
        self._succ_keys = set(io_meta['successors'])

        self._executable_dag = op_info.pop('executable_dag', None)

        # set of running predecessors, used to broadcast priority changes
        self._running_preds = set()
        # set of finished predecessors, used to decide whether we should move the operand to ready
        self._finish_preds = set()
        # set of finished successors, used to detect whether we can do clean up
        self._finish_succs = set()

        # handlers of states. will be called when the state of the operand switches
        # from one to another
        self._state_handlers = {
            OperandState.UNSCHEDULED: self._on_unscheduled,
            OperandState.READY: self._on_ready,
            OperandState.RUNNING: self._on_running,
            OperandState.FINISHED: self._on_finished,
            OperandState.FREED: self._on_freed,
            OperandState.FATAL: self._on_fatal,
            OperandState.CANCELLING: self._on_cancelling,
            OperandState.CANCELLED: self._on_cancelled,
        }

        self._graph_refs = []
        self._cluster_info_ref = None
        self._assigner_ref = None
        self._resource_ref = None

        self._with_kvstore = with_kvstore
        self._kv_store_ref = None

        if schedulers:  # pragma: no branch
            self.set_schedulers(schedulers)

    def post_create(self):
        from ..graph import GraphActor
        from ..assigner import AssignerActor
        from ..kvstore import KVStoreActor
        from ..resource import ResourceActor

        self.set_cluster_info_ref()
        self._assigner_ref = self.get_promise_ref(AssignerActor.gen_uid(self._session_id))
        self._graph_refs.append(self.get_actor_ref(GraphActor.gen_uid(self._session_id, self._graph_ids[0])))
        self._resource_ref = self.get_actor_ref(ResourceActor.default_uid())

        if self._with_kvstore:
            self._kv_store_ref = self.ctx.actor_ref(KVStoreActor.default_uid())

        self.ref().start_operand(_tell=True)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._last_state = self._state
        if value != self._last_state:
            logger.debug('Operand %s(%s) state from %s to %s.', self._op_key, self._op_name,
                         self._last_state, value)
        self._state = value
        self._info['state'] = value.name
        for graph_ref in self._graph_refs:
            graph_ref.set_operand_state(self._op_key, value, _tell=True, _wait=False)
        if self._kv_store_ref is not None:
            self._kv_store_ref.write(
                '%s/state' % self._op_path, value.name, _tell=True, _wait=False)

    @property
    def worker(self):
        return self._worker

    @worker.setter
    def worker(self, value):
        futures = []
        for graph_ref in self._graph_refs:
            futures.append(graph_ref.set_operand_worker(self._op_key, value, _tell=True, _wait=False))
        if self._kv_store_ref is not None:
            if value:
                futures.append(self._kv_store_ref.write(
                    '%s/worker' % self._op_path, value, _tell=True, _wait=False))
            elif self._worker is not None:
                futures.append(self._kv_store_ref.delete(
                    '%s/worker' % self._op_path, silent=True, _tell=True, _wait=False))
        [f.result() for f in futures]
        self._worker = value

    def append_graph(self, graph_key, op_info):
        from ..graph import GraphActor

        graph_ref = self.get_actor_ref(GraphActor.gen_uid(self._session_id, graph_key))
        self._graph_refs.append(graph_ref)

        self._pred_keys.update(op_info['io_meta']['predecessors'])
        self._succ_keys.update(op_info['io_meta']['successors'])

        if self.state not in OperandState.STORED_STATES and self._state != OperandState.RUNNING:
            if op_info['state'] == OperandState.UNSCHEDULED and set(self._pred_keys) == self._finish_preds:
                self.start_operand(OperandState.READY)
            else:
                self.state = op_info['state']
                logger.debug('State of %s(%s) reset to %s', self._op_key, self._op_name, self.state)
        else:
            # make sure states synchronized among graphs
            logger.debug('State of %s(%s) kept as %s', self._op_key, self._op_name, self.state)
            self.state = self.state

    def get_state(self):
        return self._state

    def _get_raw_execution_ref(self, uid=None, address=None):
        """
        Get raw ref of ExecutionActor on assigned worker. This method can be patched on debug
        """
        from ...worker import ExecutionActor
        uid = uid or ExecutionActor.default_uid()

        return self.ctx.actor_ref(uid, address=address)

    def _get_operand_actor(self, key):
        """
        Get ref of OperandActor by operand key
        """
        op_uid = self.gen_uid(self._session_id, key)
        return self.ctx.actor_ref(op_uid, address=self.get_scheduler(op_uid))

    def _wait_worker_futures(self, worker_futures):
        dead_workers = []
        for ep, future in worker_futures:
            try:
                with rewrite_worker_errors():
                    future.result()
            except WorkerDead:
                dead_workers.append(ep)
        if dead_workers:
            self._resource_ref.detach_dead_workers(dead_workers, _tell=True)
        return dead_workers

    def _free_data_in_worker(self, data_keys, workers_list=None):
        """
        Free data on single worker
        :param data_keys: keys of data in chunk meta
        """
        if not workers_list:
            workers_list = self.chunk_meta.batch_get_workers(self._session_id, data_keys)
        worker_data = defaultdict(list)
        for data_key, endpoints in zip(data_keys, workers_list):
            if endpoints is None:
                continue
            for ep in endpoints:
                worker_data[ep].append(data_key)

        self.chunk_meta.batch_delete_meta(self._session_id, data_keys, _tell=True, _wait=False)

        worker_futures = []
        for ep, data_keys in worker_data.items():
            ref = self._get_raw_execution_ref(address=ep)
            worker_futures.append((ep, ref.delete_data_by_keys(
                self._session_id, data_keys, _tell=True, _wait=False)))

        return self._wait_worker_futures(worker_futures)

    def start_operand(self, state=None, **kwargs):
        """
        Start handling operand given self.state
        """
        if state:
            self.state = state

        kwargs = dict((k, v) for k, v in kwargs.items() if v is not None)
        io_meta = kwargs.pop('io_meta', None)
        if io_meta:
            self._io_meta.update(io_meta)
            self._info['io_meta'] = self._io_meta
        self._info.update(kwargs)

        self._state_handlers[self.state]()

    def stop_operand(self, state=OperandState.CANCELLING):
        """
        Stop operand by starting CANCELLING procedure
        """
        if self.state == OperandState.CANCELLING or self.state == OperandState.CANCELLED:
            return
        if self.state != state:
            self.start_operand(state)

    def add_running_predecessor(self, op_key, worker):
        self._running_preds.add(op_key)

    def add_finished_predecessor(self, op_key, worker, output_sizes=None, output_shapes=None):
        self._finish_preds.add(op_key)

    def add_finished_successor(self, op_key, worker):
        self._finish_succs.add(op_key)

    def remove_finished_predecessor(self, op_key):
        try:
            self._finish_preds.remove(op_key)
        except KeyError:
            pass

    def remove_finished_successor(self, op_key):
        try:
            self._finish_succs.remove(op_key)
        except KeyError:
            pass

    def propose_descendant_workers(self, input_key, worker_scores, depth=1):
        pass

    def move_failover_state(self, from_states, state, new_target, dead_workers):
        if dead_workers:
            futures = []
            # remove executed traces in neighbor operands
            for out_key in self._succ_keys:
                futures.append(self._get_operand_actor(out_key).remove_finished_predecessor(
                    self._op_key, _tell=True, _wait=False))
            for in_key in self._pred_keys:
                futures.append(self._get_operand_actor(in_key).remove_finished_successor(
                    self._op_key, _tell=True, _wait=False))
            if self._is_terminal:
                for graph_ref in self._graph_refs:
                    futures.append(graph_ref.remove_finished_terminal(
                        self._op_key, _tell=True, _wait=False))
            [f.result() for f in futures]

        # actual start the new state
        self.start_operand(state)

    def check_can_be_freed(self, target_state=OperandState.FREED):
        """
        Check if the data of the operand can be freed.
        :param target_state: The state to move into, FREED by default
        :return: a tuple. The first value indicates whether data cleaning can be performed,
            and the last value indicates whether the result is deterministic.
        """
        if self.state == OperandState.FREED:
            return False, True
        if target_state == OperandState.CANCELLED:
            can_be_freed = True
        else:
            can_be_freed_states = [graph_ref.check_operand_can_be_freed(self._succ_keys) for
                                   graph_ref in self._graph_refs]
            if None in can_be_freed_states:
                can_be_freed = None
            else:
                can_be_freed = all(can_be_freed_states)
        if can_be_freed is None:
            return False, False
        elif not can_be_freed:
            return False, True
        return True, True

    def _on_unscheduled(self):
        pass

    def _on_ready(self):
        pass

    def _on_running(self):
        pass

    def _on_finished(self):
        pass

    def _on_freed(self):
        pass

    def _on_fatal(self):
        pass

    def _on_cancelling(self):
        pass

    def _on_cancelled(self):
        pass
