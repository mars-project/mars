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
import contextlib
import copy
import functools
import logging
import time

from .assigner import AssignerActor
from .chunkmeta import ChunkMetaActor
from .graph import GraphActor
from .kvstore import KVStoreActor
from .resource import ResourceActor
from .utils import SchedulerActor, OperandState, OperandPosition, GraphState, array_to_bytes
from ..actors import ActorNotExist
from ..compat import BrokenPipeError, ConnectionRefusedError, TimeoutError  # pylint: disable=W0622
from ..config import options
from ..errors import ExecutionInterrupted, DependencyMissing, WorkerDead
from ..utils import log_unhandled

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _rewrite_worker_errors(ignore_error=False):
    rewrite = False
    try:
        yield
    except (BrokenPipeError, ConnectionRefusedError, ActorNotExist, TimeoutError):
        # we don't raise here, as we do not want
        # the actual stack be dumped
        rewrite = not ignore_error
    if rewrite:
        raise WorkerDead


class OperandActor(SchedulerActor):
    """
    Actor handling the whole lifecycle of a particular operand instance
    """
    @staticmethod
    def gen_uid(session_id, op_key):
        return 's:operator$%s$%s' % (session_id, op_key)

    def __init__(self, session_id, graph_id, op_key, op_info, worker=None,
                 position=None):
        super(OperandActor, self).__init__()
        self._info = op_info = copy.deepcopy(op_info)

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
        self._state = self._last_state = OperandState(op_info['state'].lower())
        self._retries = op_info['retries']
        self._position = position
        self._io_meta = io_meta = op_info['io_meta']
        self._pred_keys = io_meta['predecessors']
        self._succ_keys = io_meta['successors']
        self._input_chunks = io_meta['input_chunks']
        self._chunks = io_meta['chunks']

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

        # set of running predecessors and workers of predecessors,
        # used to decide whether to pre-push to a worker
        self._running_preds = set()
        self._pred_workers = set()

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
            futures.append(self._kv_store_ref.write(
                '%s/state' % self._op_path, value.name, _tell=True, _wait=False))
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
                futures.append(self._kv_store_ref.write(
                    '%s/worker' % self._op_path, value, _tell=True, _wait=False))
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
            futures.append(self._kv_store_ref.write(
                '%s/retries' % self._op_path, str(value), _tell=True, _wait=False))

        retry_timestamp = time.time()
        self._info['retry_timestamp'] = retry_timestamp
        if self._kv_store_ref is not None:
            futures.append(self._kv_store_ref.write('%s/retry_timestamp' % self._op_path, str(value),
                                                    _tell=True, _wait=False))
        [f.result() for f in futures]

    def add_running_predecessor(self, op_key, worker):
        self._running_preds.add(op_key)
        self._pred_workers.add(worker)
        if len(self._pred_workers) > 1:
            # we do not push when multiple workers in input
            self._pred_workers = set()
            self._running_preds = set()
            return

        if self.state != OperandState.UNSCHEDULED:
            return

        if all(k in self._running_preds for k in self._pred_keys):
            try:
                if worker in self._assigned_workers:
                    return
                serialized_exec_graph = self._graph_ref.get_executable_operand_dag(self._op_key)

                self._get_execution_ref(address=worker).enqueue_graph(
                    self._session_id, self._op_key, serialized_exec_graph, self._io_meta,
                    dict(), self._info['optimize'], succ_keys=self._succ_keys,
                    pred_keys=self._pred_keys, _promise=True) \
                    .then(functools.partial(self._handle_worker_accept, worker))
                self._assigned_workers.add(worker)
                logger.debug('Pre-push operand %s into worker %s.', self._op_key, worker)
            except:  # noqa: E722
                logger.exception('Failed to pre-push operand %s', self._op_key)
            finally:
                self._pred_workers = set()
                self._running_preds = set()

    def add_finished_predecessor(self, op_key):
        self._finish_preds.add(op_key)
        if all(k in self._finish_preds for k in self._pred_keys):
            if self.state != OperandState.UNSCHEDULED:
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
        if self._position != OperandPosition.TERMINAL and \
                all(k in self._finish_succs for k in self._succ_keys):
            # make sure that all prior states are terminated (in case of failover)
            states = self._graph_ref.get_operand_states(self._succ_keys)
            # non-terminal operand with all successors done, the data can be freed
            if all(k in OperandState.TERMINATED_STATES for k in states) and self._is_worker_alive():
                self.ref().free_data(_tell=True)

    def remove_finished_successor(self, op_key):
        try:
            self._finish_succs.remove(op_key)
        except KeyError:
            pass

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

        # if the operand is already submitted to AssignerActor, we need to update the priority
        for w in self._assigned_workers:
            futures.append(self._get_execution_ref(address=w).update_priority(
                self._session_id, self._op_key, optimize_data, _tell=True, _wait=False))

        dead_workers = set()
        for w, f in zip(self._assigned_workers, futures):
            try:
                with _rewrite_worker_errors():
                    f.result()
            except WorkerDead:
                dead_workers.add(w)
        if dead_workers:
            self._resource_ref.detach_dead_workers(list(dead_workers), _tell=True)
            self._assigned_workers.difference_update(dead_workers)

        futures = []
        if self.state != OperandState.READY:
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

    def _is_worker_alive(self):
        return self._assigner_ref.is_worker_alive(self.worker)

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

        if dead_workers:
            futures = []
            # remove executed traces in neighbor operands
            for out_key in self._succ_keys:
                futures.append(self._get_operand_actor(out_key).remove_finished_predecessor(
                    self._op_key, _tell=True, _wait=False))
            for in_key in self._pred_keys:
                futures.append(self._get_operand_actor(in_key).remove_finished_successor(
                    self._op_key, _tell=True, _wait=False))
            if self._position == OperandPosition.TERMINAL:
                futures.append(self._graph_ref.remove_finished_terminal(
                    self._op_key, _tell=True, _wait=False))
            [f.result() for f in futures]

        # actual start the new state
        self.start_operand(state)

    def _free_worker_data(self, ep, chunk_key):
        """
        Free data on single worker
        :param ep: worker endpoint
        :param chunk_key: chunk key
        """
        from ..worker.chunkholder import ChunkHolderActor

        worker_cache_ref = self.ctx.actor_ref(ChunkHolderActor.default_name(), address=ep)
        return worker_cache_ref.unregister_chunk(self._session_id, chunk_key,
                                                 _tell=True, _wait=False)

    def free_data(self, state=OperandState.FREED):
        """
        Free output data of current operand
        :param state: target state
        """
        if self.state == OperandState.FREED:
            return
        if state == OperandState.CANCELLED:
            can_be_freed = True
        else:
            can_be_freed = self._graph_ref.check_operand_can_be_freed(self._succ_keys)
        if can_be_freed is None:
            self.ref().free_data(state, _delay=1, _tell=True)
            return
        elif not can_be_freed:
            return

        self.start_operand(state)

        endpoint_lists = self._chunk_meta_ref.batch_get_workers(self._session_id, self._chunks)
        futures = []
        for chunk_key, endpoints in zip(self._chunks, endpoint_lists):
            if endpoints is None:
                continue
            for ep in endpoints:
                futures.append((self._free_worker_data(ep, chunk_key), ep))

        dead_workers = []
        for f, ep in futures:
            try:
                with _rewrite_worker_errors():
                    f.result()
            except WorkerDead:
                dead_workers.append(ep)

        if dead_workers:
            self._resource_ref.detach_dead_workers(list(dead_workers), _tell=True)
            self._assigned_workers.difference_update(dead_workers)

        self._chunk_meta_ref.batch_delete_meta(self._session_id, self._chunks, _tell=True)

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
        def _dequeue_worker(endpoint, wait=True):
            try:
                with _rewrite_worker_errors():
                    return self._get_execution_ref(address=endpoint).dequeue_graph(
                        self._session_id, self._op_key, _tell=True, _wait=wait)
            finally:
                self._assigned_workers.difference_update((worker,))

        if self._position == OperandPosition.INITIAL:
            new_worker = self._graph_ref.get_operand_target_worker(self._op_key)
            if new_worker and new_worker != self._target_worker:
                logger.debug('Cancelling running operand %s on %s, new_target %s',
                             self._op_key, worker, new_worker)
                _dequeue_worker(worker)
                return

        if (self.worker and self.worker != worker) or \
                (self._target_worker and worker != self._target_worker):
            logger.debug('Cancelling running operand %s on %s, op_worker %s, op_target %s',
                         self._op_key, worker, self.worker, self._target_worker)
            _dequeue_worker(worker)
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
        for w in list(self._assigned_workers):
            if w != worker:
                logger.debug('Cancelling running operand %s on %s, when deciding to run on %s',
                             self._op_key, w, worker)
                cancel_futures.append(_dequeue_worker(w, wait=False))

        for f in cancel_futures:
            with _rewrite_worker_errors(ignore_error=True):
                f.result()
        self._assigned_workers = set()

        target_predicts = self._get_target_predicts(worker)

        # prepare meta broadcasts
        broadcast_eps = set()
        for succ_key in self._succ_keys:
            broadcast_eps.add(self.get_scheduler(self.gen_uid(self._session_id, succ_key)))
        broadcast_eps.difference_update({self.address})
        broadcast_eps = tuple(broadcast_eps)

        chunk_keys, broadcast_ep_groups = [], []
        for chunk_key in self._chunks:
            chunk_keys.append(chunk_key)
            broadcast_ep_groups.append(broadcast_eps)

        self._chunk_meta_ref.batch_set_chunk_broadcasts(
            self._session_id, chunk_keys, broadcast_ep_groups, _tell=True, _wait=False)

        # submit job
        logger.debug('Start running operand %s on %s', self._op_key, worker)
        self._execution_ref = self._get_execution_ref()
        try:
            with _rewrite_worker_errors():
                self._execution_ref.start_execution(
                    self._session_id, self._op_key, send_addresses=target_predicts, _promise=True)
        except WorkerDead:
            self._resource_ref.detach_dead_workers([self.worker], _tell=True)
            return
        # here we start running immediately to avoid accidental state change
        # and potential submission
        self.start_operand(OperandState.RUNNING)

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
            self._assigned_workers = set()
            self.ref().start_operand(OperandState.UNSCHEDULED, _tell=True)
            return

        chunk_sizes = self._chunk_meta_ref.batch_get_chunk_size(self._session_id, self._input_chunks)
        if any(v is None for v in chunk_sizes):
            logger.warning('DependencyMissing met, operand %s will be back to UNSCHEDULED.',
                           self._op_key)
            self._assigned_workers = set()
            self.ref().start_operand(OperandState.UNSCHEDULED, _tell=True)
            return

        new_assignment = [a for a in new_assignment if a not in self._assigned_workers]
        self._assigned_workers.update(new_assignment)
        logger.debug('Operand %s assigned to run on workers %r, now it has %r',
                     self._op_key, new_assignment, self._assigned_workers)

        data_sizes = dict(zip(self._input_chunks, chunk_sizes))

        dead_workers = set()
        serialized_exec_graph = self._graph_ref.get_executable_operand_dag(self._op_key)
        for worker_ep in new_assignment:
            try:
                with _rewrite_worker_errors():
                    self._get_execution_ref(address=worker_ep).enqueue_graph(
                        self._session_id, self._op_key, serialized_exec_graph, self._io_meta,
                        data_sizes, self._info['optimize'], succ_keys=self._succ_keys,
                        _delay=delay, _promise=True) \
                        .then(functools.partial(self._handle_worker_accept, worker_ep))
            except WorkerDead:
                logger.debug('Worker %s dead when submitting operand %s into queue',
                             worker_ep, self._op_key)
                dead_workers.add(worker_ep)
                self._assigned_workers.difference_update([worker_ep])
        if dead_workers:
            self._resource_ref.detach_dead_workers(list(dead_workers), _tell=True)
            if not self._assigned_workers:
                self.ref().start_operand(_tell=True)

    @log_unhandled
    def _on_running(self):
        self._execution_ref = self._get_execution_ref()

        @log_unhandled
        def _acceptor(*_):
            if not self._is_worker_alive():
                return
            # handling success of operand execution
            self.start_operand(OperandState.FINISHED)

        @log_unhandled
        def _rejecter(*exc):
            # handling exception occurrence of operand execution
            exc_type = exc[0]
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
                if self.retries >= options.scheduler.retry_num:
                    # no further trial
                    self.state = OperandState.FATAL
                else:
                    self.state = OperandState.READY
                self.ref().start_operand(_tell=True)

        try:
            with _rewrite_worker_errors():
                self._execution_ref.add_finish_callback(self._session_id, self._op_key, _promise=True) \
                    .then(_acceptor, _rejecter)
        except WorkerDead:
            logger.debug('Worker %s dead when adding callback for operand %s',
                         self.worker, self._op_key)
            self._resource_ref.detach_dead_workers([self.worker], _tell=True)
            self.start_operand(OperandState.READY)

        futures = []
        for out_key in self._succ_keys:
            futures.append(self._get_operand_actor(out_key).add_running_predecessor(
                self._op_key, self.worker, _tell=True, _wait=False))
        [f.result() for f in futures]

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
                self._op_key, _tell=True, _wait=False))
        for in_key in self._pred_keys:
            futures.append(self._get_operand_actor(in_key).add_finished_successor(
                self._op_key, _tell=True, _wait=False))
        # require more chunks to execute if the completion caused no successors to run
        if self._position == OperandPosition.TERMINAL:
            # update records in GraphActor to help decide if the whole graph finished execution
            futures.append(self._graph_ref.add_finished_terminal(
                self._op_key, _tell=True, _wait=False))
        [f.result() for f in futures]

    @log_unhandled
    def _on_fatal(self):
        if self._last_state == OperandState.FATAL:
            return

        futures = []
        if self._position == OperandPosition.TERMINAL:
            # update records in GraphActor to help decide if the whole graph finished execution
            futures.append(self._graph_ref.add_finished_terminal(
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
            with _rewrite_worker_errors(ignore_error=True):
                self._execution_ref.stop_execution(
                    self._session_id, self._op_key, _tell=True)
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
        if self._position == OperandPosition.TERMINAL:
            futures.append(self._graph_ref.add_finished_terminal(
                self._op_key, final_state=GraphState.CANCELLED, _tell=True, _wait=False))
        for k in self._succ_keys:
            futures.append(self._get_operand_actor(k).propagate_state(
                OperandState.CANCELLING, _tell=True, _wait=False))
        [f.result() for f in futures]

    def _on_unscheduled(self):
        self.worker = None

    _on_freed = lambda self: None
