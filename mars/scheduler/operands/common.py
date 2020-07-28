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

import base64
import logging
import time

from ...config import options
from ...errors import ExecutionInterrupted, DependencyMissing, WorkerDead
from ...operands import Operand
from ...utils import log_unhandled, insert_reversed_tuple
from ..utils import GraphState, array_to_bytes
from .base import BaseOperandActor
from .core import OperandState, register_operand_class, rewrite_worker_errors

logger = logging.getLogger(__name__)


class OperandActor(BaseOperandActor):
    """
    Actor handling the whole lifecycle of a particular operand instance
    """
    def __init__(self, session_id, graph_id, op_key, op_info, worker=None, allocated=False, **kwargs):
        super().__init__(
            session_id, graph_id, op_key, op_info, worker=worker, **kwargs)

        io_meta = self._io_meta
        self._input_chunks = io_meta['input_chunks']
        self._chunks = io_meta['chunks']

        # worker the operand expected to be executed on
        self._target_worker = op_info.get('target_worker')
        self._calc_device = op_info.get('calc_device', 'cpu')
        self._retries = op_info['retries']
        self._assigned_workers = set()

        # ref of ExecutionActor on worker
        self._execution_ref = None

        # set of running predecessors and workers of predecessors,
        # used to decide whether to pre-push to a worker
        self._running_preds = set()
        self._pred_workers = set()

        self._data_sizes = None
        self._data_shapes = None

        self._input_worker_scores = dict()
        self._worker_scores = dict()

        self._allocated = allocated
        self._submit_promise = None

        # record the exception info when failed to execute the graph
        self._exc = None

    @property
    def retries(self):
        return self._retries

    @retries.setter
    def retries(self, value):
        futures = []
        self._retries = value
        self._info['retries'] = value

        if self._kv_store_ref is not None:
            futures.append(self._kv_store_ref.write(
                '%s/retries' % self._op_path, str(value), _tell=True, _wait=False))

        retry_timestamp = time.time()
        self._info['retry_timestamp'] = retry_timestamp
        if self._kv_store_ref is not None:
            futures.append(self._kv_store_ref.write('%s/retry_timestamp' % self._op_path, str(value),
                                                    _tell=True, _wait=False))
        [f.result() for f in futures]

    @log_unhandled
    def append_graph(self, graph_key, op_info):
        super().append_graph(graph_key, op_info)

        if not self._is_terminal:
            self._is_terminal = op_info.get('is_terminal')

        if self.state not in OperandState.TERMINATED_STATES:
            for in_key in self._pred_keys:
                self._get_operand_actor(in_key).remove_finished_successor(
                    self._op_key, _tell=True, _wait=False)
            self.start_operand()
        elif self.state in OperandState.STORED_STATES:
            for out_key in self._succ_keys:
                self._get_operand_actor(out_key).add_finished_predecessor(
                    self._op_key, self.worker, _tell=True, _wait=False)
            # require more chunks to execute if the completion caused no successors to run
            if self._is_terminal:
                # update records in GraphActor to help decide if the whole graph finished execution
                self._add_finished_terminal()

    def start_operand(self, state=None, **kwargs):
        target_worker = kwargs.get('target_worker')
        if target_worker:
            self._target_worker = target_worker
        return super().start_operand(state=state, **kwargs)

    @log_unhandled
    def add_running_predecessor(self, op_key, worker):
        super().add_running_predecessor(op_key, worker)
        self.update_demand_depths(self._info.get('optimize', {}).get('depth', 0))

    @log_unhandled
    def add_finished_predecessor(self, op_key, worker, output_sizes=None, output_shapes=None):
        """
        This function shall return whether current node is ready. The return values will
        be collected by the predecessor to judge if a node with lower-priority can be
        scheduled.
        """
        super().add_finished_predecessor(op_key, worker, output_sizes=output_sizes,
                                         output_shapes=output_shapes)
        if all(k in self._finish_preds for k in self._pred_keys):
            # all predecessors done, the operand can be executed now
            if self.state == OperandState.UNSCHEDULED:
                self.start_operand(OperandState.READY)
            return True
        return False

    @log_unhandled
    def add_finished_successor(self, op_key, worker):
        super().add_finished_successor(op_key, worker)
        if not self._is_terminal and \
                all(k in self._finish_succs for k in self._succ_keys):
            # make sure that all prior states are terminated (in case of failover)
            states = []
            for graph_ref in self._graph_refs:
                states.extend(graph_ref.get_operand_states(self._succ_keys))
            # non-terminal operand with all successors done, the data can be freed
            if all(k in OperandState.TERMINATED_STATES for k in states) and self._is_worker_alive():
                self.ref().free_data(_tell=True)

    def update_demand_depths(self, depth):
        """
        Update the depth of operand demanding data, or demanding the descendant of data
        produced by the current operand
        :param depth: depth to update
        """
        try:
            optimize_data = self._info['optimize']
        except KeyError:
            optimize_data = self._info['optimize'] = dict()

        demand_depths = optimize_data.get('demand_depths', ())
        new_demand_depths = insert_reversed_tuple(demand_depths, depth)
        if demand_depths == new_demand_depths:
            return
        optimize_data['demand_depths'] = new_demand_depths
        if self._kv_store_ref is not None:
            self._kv_store_ref.write(
                '%s/optimize/demand_depths' % self._op_path,
                base64.b64encode(array_to_bytes('I', new_demand_depths)), _tell=True, _wait=False)

        if self.state == OperandState.READY:
            # if the operand is already submitted to AssignerActor, we need to update the priority
            self._assigner_ref.update_priority(self._op_key, optimize_data, _tell=True, _wait=False)
        else:
            # send update command to predecessors
            for in_key in self._pred_keys:
                if in_key not in self._finish_preds and in_key not in self._running_preds:
                    self._get_operand_actor(in_key).update_demand_depths(depth, _tell=True, _wait=False)

    @log_unhandled
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

        live_workers = set(self._assigner_ref.filter_alive_workers(list(self._worker_scores.keys())))
        self._worker_scores = dict((k, v) for k, v in self._worker_scores.items()
                                   if k in live_workers)

        if self._worker_scores and depth:
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

    def _is_worker_alive(self):
        return bool(self._assigner_ref.filter_alive_workers([self.worker], refresh=True))

    @log_unhandled
    def move_failover_state(self, from_states, state, new_target, dead_workers):
        """
        Move the operand into new state when executing fail-over step
        :param from_states: the source states the operand should be in, when not match, we stopped.
        :param state: the target state to move
        :param new_target: new target worker proposed for worker
        :param dead_workers: list of dead workers
        :return:
        """
        dead_workers = set(dead_workers)
        if self.state not in from_states:
            logger.debug('From state not matching (%s not in %r), operand %s skips failover step',
                         self.state.name, [s.name for s in from_states], self._op_key)
            return
        if self.state in (OperandState.RUNNING, OperandState.FINISHED):
            if state != OperandState.UNSCHEDULED and self.worker not in dead_workers:
                logger.debug('Worker %s of operand %s still alive, skip failover step',
                             self.worker, self._op_key)
                return
            elif state == OperandState.RUNNING:
                # move running operand in dead worker to ready
                state = OperandState.READY

        if new_target and self._target_worker != new_target:
            logger.debug('Target worker of %s reassigned to %s', self._op_key, new_target)
            self._target_worker = new_target
            self._info['target_worker'] = new_target

            for succ_key in self._succ_keys:
                self._get_operand_actor(succ_key).propose_descendant_workers(
                    self._op_key, {new_target: 1.0}, _tell=True, _wait=False)

            target_updated = True
        else:
            target_updated = False

        if self.state == state == OperandState.READY:
            if not self._target_worker:
                if self._assigned_workers - dead_workers:
                    logger.debug('Operand %s still have alive workers assigned %r, skip failover step',
                                 self._op_key, list(self._assigned_workers - dead_workers))
                    return
            else:
                if not target_updated and self._target_worker not in dead_workers:
                    logger.debug('Target of operand %s (%s) not dead, skip failover step',
                                 self._op_key, self._target_worker)
                    return

        self._allocated = False
        super().move_failover_state(from_states, state, new_target, dead_workers)

    @log_unhandled
    def free_data(self, state=OperandState.FREED, check=True):
        """
        Free output data of current operand
        :param state: target state
        :param check: ask GraphActor if we can free the data, useful when failover
        """
        if check:
            can_be_freed, deterministic = self.check_can_be_freed(state)
            if not deterministic:
                self.ref().free_data(state, _delay=1, _tell=True)
                return
            elif not can_be_freed:
                return

        self.start_operand(state)

        for out_key in self._succ_keys:
            self._get_operand_actor(out_key).remove_finished_predecessor(
                self._op_key, _tell=True, _wait=False)

        stored_keys = self._io_meta.get('data_targets')
        if stored_keys:
            self._free_data_in_worker(stored_keys)

    def _get_execution_ref(self, uid=None, address=None):
        """
        Get ref of ExecutionActor on assigned worker
        """
        if address is None and self._execution_ref is not None:
            return self._execution_ref
        ref = self.promise_ref(self._get_raw_execution_ref(uid, address=address or self.worker))
        if address is None:
            self._execution_ref = ref
        return ref

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
    def submit_to_worker(self, worker, data_metas):
        # worker assigned, submit job
        if self.state in (OperandState.CANCELLED, OperandState.CANCELLING):
            self.start_operand()
            return
        if self.state == OperandState.RUNNING:
            # already running
            return

        self.worker = worker

        target_predicts = self._get_target_predicts(worker)
        try:
            input_metas = self._io_meta['input_data_metas']
            input_chunks = [k[0] if isinstance(k, tuple) else k for k in input_metas]
        except KeyError:
            input_chunks = self._input_chunks

        # submit job
        if set(input_chunks) != set(self._input_chunks) or self._executable_dag is None:
            exec_graph = self._graph_refs[-1].get_executable_operand_dag(self._op_key, input_chunks)
        else:
            exec_graph = self._executable_dag
        self._execution_ref = self._get_execution_ref()
        try:
            with rewrite_worker_errors():
                self._submit_promise = self._execution_ref.execute_graph(
                    self._session_id, self._op_key, exec_graph, self._io_meta, data_metas,
                    calc_device=self._calc_device, send_addresses=target_predicts, _promise=True, _spawn=False)
        except WorkerDead:
            logger.debug('Worker %s dead when submitting operand %s into queue',
                         worker, self._op_key)
            self._resource_ref.detach_dead_workers([worker], _tell=True)
        else:
            self.start_operand(OperandState.RUNNING)

    @log_unhandled
    def _on_ready(self):
        self.worker = None
        self._execution_ref = None

        def _apply_fail(*exc_info):
            if issubclass(exc_info[0], DependencyMissing):
                logger.warning('DependencyMissing met, operand %s will be back to UNSCHEDULED.',
                               self._op_key)
                self.worker = None
                self.ref().start_operand(OperandState.UNSCHEDULED, _tell=True)
            else:
                raise exc_info[1].with_traceback(exc_info[2]) from None

        # if under retry, give application a delay
        delay = options.scheduler.retry_delay if self.retries else 0
        # Send resource application. Submit job when worker assigned
        if not self._allocated:
            self._assigner_ref.apply_for_resource(
                self._session_id, self._op_key, self._info, _delay=delay, _promise=True) \
                .catch(_apply_fail)

    @log_unhandled
    def _on_running(self):
        self._execution_ref = self._get_execution_ref()

        # notify successors to propagate priority changes
        for out_key in self._succ_keys:
            self._get_operand_actor(out_key).add_running_predecessor(
                self._op_key, self.worker, _tell=True, _wait=False)

        @log_unhandled
        def _acceptor(data_sizes, data_shapes):
            self._allocated = False
            if not self._is_worker_alive():
                return
            self._resource_ref.deallocate_resource(
                self._session_id, self._op_key, self.worker, _tell=True, _wait=False)

            self._data_sizes = data_sizes
            self._data_shapes = data_shapes
            self._io_meta['data_targets'] = list(data_sizes)
            self.start_operand(OperandState.FINISHED)

        @log_unhandled
        def _rejecter(*exc):
            self._allocated = False
            # handling exception occurrence of operand execution
            exc_type = exc[0]
            self._resource_ref.deallocate_resource(
                self._session_id, self._op_key, self.worker, _tell=True, _wait=False)

            if self.state == OperandState.CANCELLING:
                logger.warning('Execution of operand %s cancelled.', self._op_key)
                self.free_data(OperandState.CANCELLED)
                return

            if issubclass(exc_type, ExecutionInterrupted):
                # job cancelled: switch to cancelled
                logger.warning('Execution of operand %s interrupted.', self._op_key)
                self.free_data(OperandState.CANCELLED)
            elif issubclass(exc_type, DependencyMissing):
                logger.warning('Operand %s moved to UNSCHEDULED because of DependencyMissing.',
                               self._op_key)
                self.ref().start_operand(OperandState.UNSCHEDULED, _tell=True)
            else:
                logger.exception('Attempt %d: Unexpected error %s occurred in executing operand %s in %s',
                                 self.retries + 1, exc_type.__name__, self._op_key, self.worker, exc_info=exc)
                # increase retry times
                self.retries += 1
                if not self._info['retryable'] or self.retries >= options.scheduler.retry_num:
                    # no further trial
                    self.state = OperandState.FATAL
                    self._exc = exc
                else:
                    self.state = OperandState.READY
                self.ref().start_operand(_tell=True)

        try:
            with rewrite_worker_errors():
                if self._submit_promise is None:
                    self._submit_promise = self._execution_ref.add_finish_callback(
                        self._session_id, self._op_key, _promise=True, _spawn=False)
                self._submit_promise.then(_acceptor, _rejecter)
        except WorkerDead:
            logger.debug('Worker %s dead when adding callback for operand %s',
                         self.worker, self._op_key)
            self._resource_ref.detach_dead_workers([self.worker], _tell=True)
        finally:
            self._submit_promise = None

    @log_unhandled
    def _on_finished(self):
        if self._last_state == OperandState.CANCELLING:
            self.start_operand(OperandState.CANCELLING)
            return

        use_aggressive_assign = options.scheduler.aggressive_assign

        succ_futures = []
        # update pred & succ finish records to trigger further actions
        # record if successors can be executed
        for out_key in self._succ_keys:
            succ_futures.append(self._get_operand_actor(out_key).add_finished_predecessor(
                self._op_key, self.worker, output_sizes=self._data_sizes,
                output_shapes=self._data_shapes, _wait=False))

        pred_futures = []
        for in_key in self._pred_keys:
            pred_futures.append(self._get_operand_actor(in_key).add_finished_successor(
                self._op_key, self.worker, _tell=True, _wait=False))
        # require more chunks to execute if the completion caused no successors to run
        if self._is_terminal:
            # update records in GraphActor to help decide if the whole graph finished execution
            pred_futures.extend(self._add_finished_terminal())
        [f.result() for f in pred_futures]

        if use_aggressive_assign and not any(f.result() for f in succ_futures):
            self._assigner_ref.allocate_top_resources(1, _tell=True)

    @log_unhandled
    def _on_fatal(self):
        if self._last_state == OperandState.FATAL:
            return

        futures = []
        if self._is_terminal:
            # update records in GraphActor to help decide if the whole graph finished execution
            futures.extend(self._add_finished_terminal(final_state=GraphState.FAILED, exc=self._exc))
        # set successors to FATAL
        for k in self._succ_keys:
            futures.append(self._get_operand_actor(k).stop_operand(
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
            with rewrite_worker_errors(ignore_error=True):
                self._execution_ref.stop_execution(
                    self._session_id, self._op_key, _tell=True)
        elif self._last_state == OperandState.FINISHED:
            # delete data on cancelled
            self.ref().free_data(state=OperandState.CANCELLED, _tell=True)
        elif self._last_state == OperandState.READY:
            self._assigned_workers = set()
            self.state = OperandState.CANCELLED
            self.ref().start_operand(OperandState.CANCELLED, _tell=True)
        else:
            self.ref().start_operand(OperandState.CANCELLED, _tell=True)

    @log_unhandled
    def _on_cancelled(self):
        futures = []
        if self._is_terminal:
            futures.extend(self._add_finished_terminal(final_state=GraphState.CANCELLED))
        for k in self._succ_keys:
            futures.append(self._get_operand_actor(k).stop_operand(
                OperandState.CANCELLING, _tell=True, _wait=False))
        [f.result() for f in futures]

    def _on_unscheduled(self):
        self.worker = None

    def _add_finished_terminal(self, final_state=None, exc=None):
        futures = []
        for graph_ref in self._graph_refs:
            futures.append(graph_ref.add_finished_terminal(
                self._op_key, final_state=final_state, exc=exc, _tell=True, _wait=False
            ))

        return futures


register_operand_class(Operand, OperandActor)
