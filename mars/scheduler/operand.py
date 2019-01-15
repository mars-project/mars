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

import base64
import copy
import functools
import logging
import time

from .assigner import AssignerActor
from .chunkmeta import ChunkMetaActor
from .graph import GraphActor
from .kvstore import KVStoreActor
from .resource import ResourceActor
from .utils import SchedulerActor, OperandState, GraphState, array_to_bytes
from ..config import options
from ..errors import *
from ..utils import log_unhandled

logger = logging.getLogger(__name__)


class OperandActor(SchedulerActor):
    """
    Actor handling the whole lifecycle of a particular operand instance
    """
    @staticmethod
    def gen_uid(session_id, op_key):
        return 's:operator$%s$%s' % (session_id, op_key)

    def __init__(self, session_id, graph_id, op_key, op_info, worker=None,
                 is_terminal=False):
        super(OperandActor, self).__init__()
        op_info = copy.deepcopy(op_info)

        self._session_id = session_id
        self._graph_id = graph_id
        self._op_key = op_key
        self._op_path = '/sessions/%s/operands/%s' % (self._session_id, self._op_key)

        self._cluster_info_ref = None
        self._assigner_ref = None
        self._graph_ref = None
        self._resource_ref = None
        self._kv_store_ref = None
        self._chunk_meta_ref = None

        self._op_name = op_info['op_name']
        self._state = OperandState(op_info['state'].lower())
        self._last_state = self._state
        self._retries = op_info['retries']
        self._is_terminal = is_terminal
        self._io_meta = op_info['io_meta']
        self._pred_keys = self._io_meta['predecessors']
        self._succ_keys = self._io_meta['successors']
        self._input_chunks = self._io_meta['input_chunks']
        self._chunks = self._io_meta['chunks']
        self._info = op_info

        # worker the operand expected to be executed on
        self._target_worker = op_info.get('target_worker')
        self._assigned_workers = set()
        # worker actually assigned
        self._worker = worker

        # ref of ExecutionActor on worker
        self._execution_ref = None
        # set of finished predecessors, used to decide whether we should move the operand to ready
        self._finish_preds = set()
        # set of finished successors, used to detect whether we can do clean up
        self._finish_succs = set()

        self._input_worker_scores = dict()
        self._worker_scores = dict()

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

    def post_create(self):
        self.set_cluster_info_ref()
        self._assigner_ref = self.ctx.actor_ref(AssignerActor.default_name())
        self._chunk_meta_ref = self.ctx.actor_ref(ChunkMetaActor.default_name())
        self._graph_ref = self.get_actor_ref(GraphActor.gen_name(self._session_id, self._graph_id))
        self._resource_ref = self.get_actor_ref(ResourceActor.default_name())

        self._kv_store_ref = self.ctx.actor_ref(KVStoreActor.default_name())
        if not self.ctx.has_actor(self._kv_store_ref):
            self._kv_store_ref = None

    def get_state(self):
        return self._state

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
        futures = [
            self._graph_ref.set_operand_state(self._op_key, value.value, _tell=True, _wait=False),
        ]
        if self._kv_store_ref is not None:
            futures.append(self._kv_store_ref.write('%s/state' % self._op_path, value.name, _tell=True, _wait=False))
        [f.result() for f in futures]

    @property
    def worker(self):
        return self._worker

    @worker.setter
    def worker(self, value):
        futures = [
            self._graph_ref.set_operand_worker(self._op_key, value, _tell=True, _wait=False)
        ]
        if self._kv_store_ref is not None:
            if value:
                futures.append(self._kv_store_ref.write('%s/worker' % self._op_path, value, _tell=True, _wait=False))
            elif self._worker is not None:
                futures.append(self._kv_store_ref.delete(
                    '%s/worker' % self._op_path, silent=True, _tell=True, _wait=False))
        [f.result() for f in futures]
        self._worker = value

    @property
    def retries(self):
        return self._retries

    @retries.setter
    def retries(self, value):
        futures = []
        self._retries = value
        self._info['retries'] = value

        if self._kv_store_ref is not None:
            futures.append(self._kv_store_ref.write('%s/retries' % self._op_path, str(value), _tell=True, _wait=False))

        retry_timestamp = time.time()
        self._info['retry_timestamp'] = retry_timestamp
        if self._kv_store_ref is not None:
            futures.append(self._kv_store_ref.write('%s/retry_timestamp' % self._op_path, str(value),
                                                    _tell=True, _wait=False))
        [f.result() for f in futures]

    def add_finished_predecessor(self, op_key):
        self._finish_preds.add(op_key)
        if all(k in self._finish_preds for k in self._pred_keys):
            if self.state in (OperandState.CANCELLED, OperandState.CANCELLING):
                return True
            # all predecessors done, the operand can be executed now
            self.start_operand(OperandState.READY)
            return True
        self.update_demand_depths(self._info.get('optimize', {}).get('depth', 0))
        return False

    def remove_finished_predecessor(self, op_key):
        try:
            self._finish_preds.remove(op_key)
        except KeyError:
            pass

    def add_finished_successor(self, op_key):
        self._finish_succs.add(op_key)
        if not self._is_terminal and all(k in self._finish_succs for k in self._succ_keys):
            # non-terminal operand with all successors done, the data can be freed
            self.ref().free_data(_tell=True)

    def update_demand_depths(self, depth):
        """
        Update the depth of operand demanding data, or demanding the descendant of data
        produced by the current operand
        :param depth: depth to update
        """
        demand_depths = list(self._info.get('optimize', {}).get('demand_depths', ()))
        if not demand_depths:
            demand_depths = [depth]
        else:
            idx = 0
            for idx, v in enumerate(demand_depths):
                if v <= depth:
                    break
            if demand_depths[idx] == depth:
                return
            elif demand_depths[idx] > depth:
                demand_depths.append(depth)
            else:
                demand_depths.insert(idx, depth)
        try:
            optimize_data = self._info['optimize']
        except KeyError:
            optimize_data = self._info['optimize'] = dict()
        optimize_data['demand_depths'] = tuple(demand_depths)
        if self._kv_store_ref is not None:
            self._kv_store_ref.write(
                '%s/optimize/demand_depths' % self._op_path,
                base64.b64encode(array_to_bytes('I', demand_depths)), _tell=True, _wait=False)
        futures = []
        if self.state == OperandState.READY:
            # if the operand is already submitted to AssignerActor, we need to update the priority
            for w in self._assigned_workers:
                futures.append(self._get_execution_ref(address=w).update_priority(
                    self._session_id, self._op_key, optimize_data, _tell=True, _wait=False))
        else:
            # send update command to predecessors
            for in_key in self._pred_keys:
                futures.append(self._get_operand_actor(in_key).update_demand_depths(
                    depth, _tell=True, _wait=False))
        [f.result() for f in futures]

    def propose_descendant_workers(self, input_key, worker_scores, depth=1):
        """
        Calculate likelihood of the operand being sent to workers
        :param input_key: input key that carries the scores
        :param worker_scores: score of every worker on input key
        :param depth: maximal propagate depth
        """
        if self.worker:
            # worker already assigned, there should be no other possibilities
            self._worker_scores = {self.worker: 1.0}
        elif self._target_worker:
            # worker already proposed, there should be no other possibilities
            self._worker_scores = {self._target_worker: 1.0}
        else:
            # aggregate the score from input to the score of current operand
            old_scores = self._input_worker_scores.get(input_key, {})
            self._input_worker_scores[input_key] = worker_scores
            all_keys = set(old_scores.keys()) | set(worker_scores)
            for k in all_keys:
                delta = (worker_scores.get(k, 0) - old_scores.get(k, 0)) * 1.0 / len(self._pred_keys)
                self._worker_scores[k] = self._worker_scores.get(k, 0) + delta
                if self._worker_scores[k] < 1e-6:
                    del self._worker_scores[k]

        if depth:
            # push down to successors
            futures = []
            for succ_key in self._succ_keys:
                futures.append(self._get_operand_actor(succ_key).propose_descendant_workers(
                    self._op_key, self._worker_scores, depth=depth - 1, _tell=True, _wait=False))
            [f.result() for f in futures]
        # pick the worker with largest likelihood
        max_score = 0
        max_worker = None
        for k, v in self._worker_scores.items():
            if v > max_score:
                max_score = v
                max_worker = k
        if max_score > 0.5:
            logger.debug('Operand %s(%s) now owning a dominant worker %s. scores=%r',
                         self._op_key, self._op_name, max_worker, self._worker_scores)
            return self._input_chunks, max_worker

    def start_operand(self, state=None):
        """
        Start handling operand given self.state
        """
        if state:
            self.state = state
        self._state_handlers[self.state]()

    def stop_operand(self):
        """
        Stop operand by starting CANCELLING procedure
        """
        if self.state == OperandState.CANCELLING or self.state == OperandState.CANCELLED:
            return
        self.start_operand(OperandState.CANCELLING)

    def _free_worker_data(self, ep, chunk_key):
        """
        Free data on single worker
        :param ep: worker endpoint
        :param chunk_key: chunk key
        """
        from ..worker.chunkholder import ChunkHolderActor

        worker_cache_ref = self.ctx.actor_ref(ep, ChunkHolderActor.default_name())
        return worker_cache_ref.unregister_chunk(self._session_id, chunk_key,
                                                 _tell=True, _wait=False)

    def free_data(self, state=OperandState.FREED):
        """
        Free output data of current operand
        :param state: target state
        """
        if self.state == OperandState.FREED:
            return
        endpoint_lists = self._chunk_meta_ref.batch_get_workers(self._session_id, self._chunks)
        futures = []
        for chunk_key, endpoints in zip(self._chunks, endpoint_lists):
            if endpoints is None:
                continue
            for ep in endpoints:
                futures.append(self._free_worker_data(ep, chunk_key))
        futures.append(self._chunk_meta_ref.batch_delete_meta(
            self._session_id, self._chunks, _tell=True, _wait=False))
        [f.result() for f in futures]
        self.start_operand(state)

    def propagate_state(self, state):
        """
        Change state of current operand by other operands
        :param state: target state
        """
        if self.state == OperandState.CANCELLING or self.state == OperandState.CANCELLED:
            return
        if self.state != state:
            self.start_operand(state)

    def _get_raw_execution_ref(self, uid, address):
        """
        Get raw ref of ExecutionActor on assigned worker. This method can be patched on debug
        """
        return self.ctx.actor_ref(uid, address=address)

    def _get_execution_ref(self, uid=None, address=None):
        """
        Get ref of ExecutionActor on assigned worker
        """
        from ..worker import ExecutionActor
        uid = uid or ExecutionActor.default_name()

        if address is None and self._execution_ref is not None:
            return self._execution_ref
        ref = self.promise_ref(self._get_raw_execution_ref(uid, address=address or self.worker))
        if address is None:
            self._execution_ref = ref
        return ref

    def _get_operand_actor(self, key):
        """
        Get ref of OperandActor by operand key
        """
        op_uid = self.gen_uid(self._session_id, key)
        return self.ctx.actor_ref(op_uid, address=self.get_scheduler(op_uid))

    def _get_target_predicts(self, worker):
        target_predicts = dict()
        if options.scheduler.enable_active_push:
            # if active push enabled, we calculate the most possible target
            futures = []
            for succ_key in self._succ_keys:
                futures.append(self._get_operand_actor(succ_key).propose_descendant_workers(
                    self._op_key, {worker: 1.0}, _wait=False))
            for succ_key, future in zip(self._succ_keys, futures):
                succ_worker_predict = future.result()
                if not succ_worker_predict:
                    continue
                keys, target = succ_worker_predict
                if target == worker:
                    continue
                for k in keys:
                    if k not in self._chunks:
                        continue
                    if k not in target_predicts:
                        target_predicts[k] = set()
                    target_predicts[k].add(target)
        if not target_predicts:
            target_predicts = None
        else:
            logger.debug('Receive active pushing list for operand %s: %r',
                         self._op_key, target_predicts)
        return target_predicts

    @log_unhandled
    def _handle_worker_accept(self, worker):
        if (self.worker and self.worker != worker) or \
                (self._target_worker and worker != self._target_worker):
            logger.debug('Cancelling running operand %s on %s, op_worker %s, op_target %s',
                         self._op_key, worker, self.worker, self._target_worker)
            self._get_execution_ref(address=worker).dequeue_graph(
                self._session_id, self._op_key)
            self._assigned_workers.difference_update((worker,))
            return
        elif self.worker is not None:
            logger.debug('Worker for operand %s already assigned', self._op_key)
            return

        # worker assigned, submit job
        if self.state in (OperandState.CANCELLED, OperandState.CANCELLING):
            self.ref().start_operand(_tell=True)
            return

        if worker != self.worker:
            self._execution_ref = None
        self.worker = worker
        cancel_futures = []
        for w in self._assigned_workers:
            if w != worker:
                logger.debug('Cancelling running operand %s on %s, when deciding to run on %s',
                             self._op_key, w, worker)
                cancel_futures.append(self._get_execution_ref(address=w).dequeue_graph(
                    self._session_id, self._op_key, _wait=False))

        [f.result() for f in cancel_futures]
        self._assigned_workers = set()

        target_predicts = self._get_target_predicts(worker)

        # prepare meta broadcasts
        broadcast_eps = set()
        for succ_key in self._succ_keys:
            broadcast_eps.add(self.get_scheduler(self.gen_uid(self._session_id, succ_key)))
        broadcast_eps.difference_update({self.address})
        broadcast_eps = tuple(broadcast_eps)

        for chunk_key in self._chunks:
            self._chunk_meta_ref.set_chunk_broadcasts(
                self._session_id, chunk_key, broadcast_eps, _tell=True, _wait=False)

        # submit job
        logger.debug('Start running operand %s on %s', self._op_key, worker)
        self._execution_ref = self._get_execution_ref()
        try:
            self._execution_ref.start_execution(
                self._session_id, self._op_key, send_addresses=target_predicts, _promise=True)
        except:
            raise
        self.ref().start_operand(OperandState.RUNNING, _tell=True)

    @log_unhandled
    def _on_ready(self):
        self.worker = None
        self._execution_ref = None

        # if under retry, give application a delay
        delay = options.scheduler.retry_delay if self.retries else 0
        # Send resource application. Submit job when worker assigned
        try:
            new_assignment = self._assigner_ref.get_worker_assignments(
                self._session_id, self._info)
        except DependencyMissing:
            logger.warning('DependencyMissing met, operand %s will be back to UNSCHEDULED.',
                           self._op_key)
            self.ref().start_operand(OperandState.UNSCHEDULED, _tell=True)
            return

        chunk_sizes = self._chunk_meta_ref.batch_get_chunk_size(self._session_id, self._input_chunks)
        if any(v is None for v in chunk_sizes):
            logger.warning('DependencyMissing met, operand %s will be back to UNSCHEDULED.',
                           self._op_key)
            self.ref().start_operand(OperandState.UNSCHEDULED, _tell=True)
            return

        new_assignment = [a for a in new_assignment if a not in self._assigned_workers]
        self._assigned_workers.update(new_assignment)
        logger.debug('Operand %s assigned to run on workers %r, now it has %r',
                     self._op_key, new_assignment, self._assigned_workers)

        data_sizes = dict(zip(self._input_chunks, chunk_sizes))

        serialized_exec_graph = self._graph_ref.get_executable_operand_dag(self._op_key)
        for worker_ep in new_assignment:
            try:
                self._get_execution_ref(address=worker_ep).enqueue_graph(
                    self._session_id, self._op_key, serialized_exec_graph, self._io_meta,
                    data_sizes, self._info['optimize'], _delay=delay, _promise=True) \
                    .then(functools.partial(self._handle_worker_accept, worker_ep))
            except:
                self._assigned_workers.difference_update([worker_ep])

    @log_unhandled
    def _on_running(self):
        self._execution_ref = self._get_execution_ref()

        @log_unhandled
        def _acceptor(*_):
            # handling success of operand execution
            self.start_operand(OperandState.FINISHED)

        @log_unhandled
        def _rejecter(*exc):
            # handling exception occurrence of operand execution
            if exc and not isinstance(exc[0], type):
                raise TypeError('Unidentified rejection args: %r', exc)

            if self.state == OperandState.CANCELLING:
                logger.warning('Execution of operand %s interrupted.', self._op_key)
                self.free_data(OperandState.CANCELLED)
                return

            if exc and issubclass(exc[0], ResourceInsufficient):
                # resource insufficient: just set to READY and continue
                self.state = OperandState.READY
                self.ref().start_operand(_tell=True)
            elif exc and issubclass(exc[0], ExecutionInterrupted):
                # job cancelled: switch to cancelled
                logger.warning('Execution of operand %s interrupted.', self._op_key)
                self.free_data(OperandState.CANCELLED)
            else:
                if exc:
                    logger.exception('Attempt %d: Unexpected error occurred in executing operand %s in %s',
                                     self.retries + 1, self._op_key, self.worker, exc_info=exc)
                else:
                    logger.error('Attempt %d: Unexpected error occurred in executing operand %s in %s '
                                 'without details.', self.retries + 1, self._op_key, self.worker)
                # increase retry times
                self.retries += 1
                if self.retries >= options.scheduler.retry_num:
                    # no further trial
                    self.state = OperandState.FATAL
                else:
                    self.state = OperandState.READY
                self.ref().start_operand(_tell=True)

        self._execution_ref.add_finish_callback(self._session_id, self._op_key, _promise=True) \
            .then(_acceptor, _rejecter)

    @log_unhandled
    def _on_finished(self):
        if self._last_state == OperandState.CANCELLING:
            self.start_operand(OperandState.CANCELLING)
            return

        futures = []
        # update pred & succ finish records to trigger further actions
        # record if successors can be executed
        for out_key in self._succ_keys:
            futures.append(self._get_operand_actor(out_key).add_finished_predecessor(
                self._op_key, _wait=False))
        for in_key in self._pred_keys:
            futures.append(self._get_operand_actor(in_key).add_finished_successor(
                self._op_key, _tell=True, _wait=False))
        # require more chunks to execute if the completion caused no successors to run
        if self._is_terminal:
            # update records in GraphActor to help decide if the whole graph finished execution
            futures.append(self._graph_ref.mark_terminal_finished(
                self._op_key, _tell=True, _wait=False))
        [f.result() for f in futures]

    @log_unhandled
    def _on_fatal(self):
        if self._last_state == OperandState.FATAL:
            return

        futures = []
        if self._is_terminal:
            # update records in GraphActor to help decide if the whole graph finished execution
            futures.append(self._graph_ref.mark_terminal_finished(
                self._op_key, final_state=GraphState.FAILED, _tell=True, _wait=False))
        # set successors to FATAL
        for k in self._succ_keys:
            futures.append(self._get_operand_actor(k).propagate_state(
                OperandState.FATAL, _tell=True, _wait=False))
        [f.result() for f in futures]

    @log_unhandled
    def _on_cancelling(self):
        if self._last_state == OperandState.CANCELLING:
            return
        elif self._last_state == OperandState.CANCELLED:
            self.state = OperandState.CANCELLED
        elif self._last_state == OperandState.RUNNING:
            # send stop to worker
            self._execution_ref = self._get_execution_ref()
            logger.debug('Sending stop on operand %s to %s', self._op_key, self.worker)
            self._execution_ref.stop_execution(self._session_id, self._op_key, _tell=True)
        elif self._last_state == OperandState.FINISHED:
            # delete data on cancelled
            self.ref().free_data(state=OperandState.CANCELLED, _tell=True)
        elif self._last_state == OperandState.READY:
            # stop application on workers
            cancel_futures = []
            for w in self._assigned_workers:
                logger.debug('Cancelling running operand %s on %s', self._op_key, w)
                cancel_futures.append(self._get_execution_ref(address=w).dequeue_graph(
                    self._session_id, self._op_key, _wait=False))
            [f.result() for f in cancel_futures]

            self._assigned_workers = set()
            self.state = OperandState.CANCELLED
            self.ref().start_operand(OperandState.CANCELLED, _tell=True)
        else:
            self.ref().start_operand(OperandState.CANCELLED, _tell=True)

    @log_unhandled
    def _on_cancelled(self):
        futures = []
        if self._is_terminal:
            futures.append(self._graph_ref.mark_terminal_finished(
                self._op_key, final_state=GraphState.CANCELLED, _tell=True, _wait=False))
        for k in self._succ_keys:
            futures.append(self._get_operand_actor(k).propagate_state(
                OperandState.CANCELLING, _tell=True, _wait=False))
        [f.result() for f in futures]

    def _on_unscheduled(self):
        self.worker = None

    _on_freed = lambda self: None
