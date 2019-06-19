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
import logging
import time

from .assigner import AssignerActor
from .graph import GraphActor
from .kvstore import KVStoreActor
from .resource import ResourceActor
from .utils import SchedulerActor, OperandState, GraphState, array_to_bytes
from ..compat import six
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
        return 's:h1:operator$%s$%s' % (session_id, op_key)

    def __init__(self, session_id, graph_id, op_key, op_info, worker_endpoint=None,
                 is_terminal=False):
        super(OperandActor, self).__init__()
        op_info = copy.deepcopy(op_info)

        self._session_id = session_id
        self._graph_ids = [graph_id]
        self._op_key = op_key
        self._op_path = '/sessions/%s/operands/%s' % (self._session_id, self._op_key)

        self._graph_refs = []
        self._cluster_info_ref = None
        self._assigner_ref = None
        self._resource_ref = None
        self._kv_store_ref = None

        self._op_name = op_info['op_name']
        self._state = op_info['state']
        self._last_state = self._state
        self._retries = op_info['retries']
        self._is_terminal = is_terminal
        self._io_meta = op_info['io_meta']
        self._pred_keys = set(self._io_meta['predecessors'])
        self._succ_keys = set(self._io_meta['successors'])
        self._input_chunks = self._io_meta['input_chunks']
        self._chunks = self._io_meta['chunks']
        self._info = op_info

        self._executable_dag = op_info.pop('executable_dag', None)

        # worker the operand expected to be executed on
        self._expect_worker = op_info.get('target_worker')
        # worker actually assigned
        self._worker_endpoint = worker_endpoint

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
            OperandState.READY: self._handle_ready,
            OperandState.RUNNING: self._handle_running,
            OperandState.FINISHED: self._handle_finished,
            OperandState.FREED: self._handle_freed,
            OperandState.FATAL: self._handle_fatal,
            OperandState.CANCELLING: self._handle_cancelling,
            OperandState.CANCELLED: self._handle_cancelled,
            OperandState.UNSCHEDULED: self._handle_unscheduled
        }

    def post_create(self):
        self.set_cluster_info_ref()
        self._assigner_ref = self.get_promise_ref(AssignerActor.gen_uid(self._session_id))
        self._graph_refs.append(self.get_actor_ref(GraphActor.gen_uid(self._session_id, self._graph_ids[0])))
        self._resource_ref = self.get_actor_ref(ResourceActor.default_uid())

        self._kv_store_ref = self.ctx.actor_ref(KVStoreActor.default_uid())
        if not self.ctx.has_actor(self._kv_store_ref):
            self._kv_store_ref = None

    def add_finished_predecessor(self, op_key):
        self._finish_preds.add(op_key)
        if all(k in self._finish_preds for k in self._pred_keys):
            if self.state in (OperandState.CANCELLED, OperandState.CANCELLING):
                return True
            # all predecessors done, the operand can be executed now
            self.state = OperandState.READY
            self.start_operand()
            return True
        self.update_demand_depths(self._info.get('optimize', {}).get('depth', 0))
        return False

    def append_graph(self, graph_key, op_info, is_terminal):
        from .graph import GraphActor

        if is_terminal:
            self._is_terminal = True
        graph_ref = self.get_actor_ref(GraphActor.gen_uid(self._session_id, graph_key))
        self._graph_refs.append(graph_ref)
        if self._state not in OperandState.STORED_STATES and self._state != OperandState.RUNNING:
            self._pred_keys = set(op_info['io_meta']['predecessors'])
            self._succ_keys = set(op_info['io_meta']['successors'])
            self._state = op_info['state']
        else:
            self._pred_keys.update(op_info['io_meta']['predecessors'])
            self._succ_keys.update(op_info['io_meta']['successors'])

        # remove finished successors and predecessors.
        # This is a workaround for execute existing operand actor and
        # will be removed in the next version.
        futures = []
        for pred_key in op_info['io_meta']['predecessors']:
            futures.append(self._get_operand_actor(pred_key).remove_finished_succs(
                self._op_key, _tell=True, _wait=False))
        for succ_key in op_info['io_meta']['successors']:
            futures.append(self._get_operand_actor(succ_key).remove_finished_preds(
                self._op_key, _tell=True, _wait=False))
        [future.result() for future in futures]

    def remove_finished_succs(self, op_key):
        if op_key in self._finish_succs:
            self._finish_succs.remove(op_key)

    def remove_finished_preds(self, op_key):
        if op_key in self._finish_preds:
            self._finish_preds.remove(op_key)

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
        if self.state == OperandState.READY:
            # if the operand is already submitted to AssignerActor, we need to update the priority
            self._assigner_ref.update_priority(self._op_key, self._info['optimize'], _tell=True, _wait=False)
        else:
            # send update command to predecessors
            for in_key in self._pred_keys:
                self._get_operand_actor(in_key).update_demand_depths(depth, _tell=True, _wait=False)

    def propose_descendant_workers(self, input_key, worker_scores, depth=1):
        """
        Calculate likelihood of the operand being sent to workers
        :param input_key: input key that carries the scores
        :param worker_scores: score of every worker on input key
        :param depth: maximal propagate depth
        """
        if self._worker_endpoint:
            # worker already assigned, there should be no other possibilities
            self._worker_scores = {self._worker_endpoint: 1.0}
        elif self._expect_worker:
            # worker already propsed, there should be no other possibilities
            self._worker_scores = {self._expect_worker: 1.0}
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
            for succ_key in self._succ_keys:
                self._get_operand_actor(succ_key).propose_descendant_workers(
                    self._op_key, self._worker_scores, depth=depth - 1, _tell=True)
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
        futures = []
        for graph_ref in self._graph_refs:
            futures.append(graph_ref.set_operand_state(self._op_key, value.value, _tell=True, _wait=False))
        if self._kv_store_ref is not None:
            futures.append(self._kv_store_ref.write('%s/state' % self._op_path, value.name, _tell=True, _wait=False))
        [f.result() for f in futures]

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

    def get_op_info(self):
        info = dict()
        info['name'] = self._op_name
        info['state'] = self.state
        return info

    def start_operand(self):
        """
        Start handling operand given self.state
        """
        self._state_handlers[self.state]()

    def stop_operand(self):
        """
        Stop operand by starting CANCELLING procedure
        """
        if self.state == OperandState.CANCELLING or self.state == OperandState.CANCELLED:
            return
        self.state = OperandState.CANCELLING
        self.start_operand()

    def _free_worker_data(self, ep, chunk_key):
        """
        Free data on single worker
        :param ep: worker endpoint
        :param chunk_key: chunk key
        """
        from ..worker.chunkholder import ChunkHolderActor

        worker_cache_ref = self.ctx.actor_ref(ep, ChunkHolderActor.default_uid())
        return worker_cache_ref.unregister_chunk(self._session_id, chunk_key,
                                                 _tell=True, _wait=False)

    def free_data(self, state=OperandState.FREED):
        """
        Free output data of current operand
        :param state: target state
        """
        if self.state == OperandState.FREED:
            return
        endpoint_lists = self.chunk_meta.batch_get_workers(self._session_id, self._chunks)
        futures = []
        for chunk_key, endpoints in zip(self._chunks, endpoint_lists):
            if endpoints is None:
                continue
            for ep in endpoints:
                futures.append(self._free_worker_data(ep, chunk_key))
        self.chunk_meta.batch_delete_meta(
            self._session_id, self._chunks, _tell=True)
        [f.result() for f in futures]
        self.state = state
        self.start_operand()

    def propagate_state(self, state):
        """
        Change state of current operand by other operands
        :param state: target state
        """
        if self.state == OperandState.CANCELLING or self.state == OperandState.CANCELLED:
            return
        if self.state != state:
            self.state = state
            self.start_operand()

    def _get_raw_execution_ref(self):
        """
        Get raw ref of ExecutionActor on assigned worker. This method can be patched on debug
        """
        from ..worker.dispatcher import DispatchActor

        dispatch_ref = self.promise_ref(DispatchActor.default_uid(), address=self._worker_endpoint)
        exec_uid = dispatch_ref.get_hash_slot('execution', self._op_key)
        return self.ctx.actor_ref(exec_uid, address=self._worker_endpoint)

    def _get_execution_ref(self):
        """
        Get ref of ExecutionActor on assigned worker
        """
        if self._execution_ref is None:
            self._execution_ref = self.promise_ref(self._get_raw_execution_ref())
        return self._execution_ref

    def _get_operand_actor(self, key):
        """
        Get ref of OperandActor by operand key
        """
        op_uid = self.gen_uid(self._session_id, key)
        return self.ctx.actor_ref(op_uid, address=self.get_scheduler(op_uid))

    @log_unhandled
    def _handle_ready(self):
        @log_unhandled
        def _submit_job(worker):
            # worker assigned, submit job
            if self.state in (OperandState.CANCELLED, OperandState.CANCELLING):
                self.start_operand()
                return

            self._worker_endpoint = worker

            target_predicts = dict()
            if options.scheduler.enable_active_push:
                # if active push enabled, we calculate the most possible target
                for succ_key in self._succ_keys:
                    succ_worker_predict = self._get_operand_actor(succ_key) \
                        .propose_descendant_workers(self._op_key, {worker: 1.0})
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

            data_sizes = dict(zip(
                self._input_chunks,
                self.chunk_meta.batch_get_chunk_size(self._session_id, self._input_chunks),
            ))

            # submit job
            self._execution_ref = self._get_execution_ref()
            if self._executable_dag is not None:
                exec_graph = self._executable_dag
            else:
                exec_graph = self._graph_refs[0].get_executable_operand_dag(self._op_key)

            self._execution_ref.execute_graph(self._session_id, self._op_key, exec_graph, self._io_meta,
                                              data_sizes, send_targets=target_predicts, _promise=True)
            self.state = OperandState.RUNNING
            self.start_operand()

        logger.debug('Applying for resources for operand %r', self._info)
        # if under retry, give application a delay
        delay = options.scheduler.retry_delay if self.retries else 0
        # Send resource application. Submit job when worker assigned
        self._assigner_ref.apply_for_resource(
            self._session_id, self._op_key, self._info, _delay=delay, _promise=True) \
            .then(_submit_job)

    @log_unhandled
    def _handle_running(self):
        self._execution_ref = self._get_execution_ref()

        @log_unhandled
        def _acceptor(*_):
            # handling success of operand execution
            self._resource_ref.deallocate_resource(self._session_id, self._op_key, self._worker_endpoint)
            self.state = OperandState.FINISHED
            self.start_operand()

        @log_unhandled
        def _rejecter(*exc):
            # handling execption occurance of operand execution
            self._resource_ref.deallocate_resource(self._session_id, self._op_key, self._worker_endpoint)
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
                self._resource_ref.deallocate_resource(self._session_id, self._op_key, self._worker_endpoint)
                logger.warning('Execution of operand %s interrupted.', self._op_key)
                self.free_data(OperandState.CANCELLED)
            else:
                self._resource_ref.deallocate_resource(self._session_id, self._op_key, self._worker_endpoint)
                try:
                    if exc:
                        six.reraise(*exc)
                    else:
                        raise SystemError('Worker throws rejection without details')
                except:
                    logger.exception('Attempt %d: Unexpected error occurred in executing operand %s in %s',
                                     self.retries + 1, self._op_key, self._worker_endpoint)
                # increase retry times
                self.retries += 1
                if self.retries >= options.scheduler.retry_num:
                    # no further trial
                    self.state = OperandState.FATAL
                else:
                    self.state = OperandState.READY
                self.start_operand()

        self._execution_ref.add_finish_callback(self._session_id, self._op_key, _promise=True) \
            .then(_acceptor, _rejecter)

    @log_unhandled
    def _handle_finished(self):
        self._resource_ref.deallocate_resource(self._session_id, self._op_key, self._worker_endpoint)
        if self._last_state == OperandState.CANCELLING:
            self.state = OperandState.CANCELLING
            self.start_operand()
            return
        new_ready_generated = False
        # update pred & succ finish records to trigger further actions
        # record if successors can be executed
        for out_key in self._succ_keys:
            new_ready_generated |= self._get_operand_actor(out_key).add_finished_predecessor(self._op_key)
        for in_key in self._pred_keys:
            self._get_operand_actor(in_key).add_finished_successor(self._op_key)
        # require more chunks to execute if the completion caused no successors to run
        if not new_ready_generated:
            self._assigner_ref.allocate_top_resources(_tell=True)
        if self._is_terminal:
            # update records in GraphActor to help decide if the whole graph finished execution
            self._mark_finished_terminal()

    @log_unhandled
    def _handle_fatal(self):
        if self._last_state == OperandState.FATAL:
            return
        if self._worker_endpoint:
            self._resource_ref.deallocate_resource(self._session_id, self._op_key, self._worker_endpoint)

        if self._is_terminal:
            # update records in GraphActor to help decide if the whole graph finished execution
            self._mark_finished_terminal(final_state=GraphState.FAILED)
        # set successors to FATAL
        for k in self._succ_keys:
            self._get_operand_actor(k).propagate_state(OperandState.FATAL, _tell=True)

    @log_unhandled
    def _handle_cancelling(self):
        if self._last_state == OperandState.CANCELLING:
            return
        if self._last_state == OperandState.CANCELLED:
            self.state = OperandState.CANCELLED
            return

        if self._last_state == OperandState.RUNNING:
            # send stop to worker
            self._execution_ref = self._get_execution_ref()
            self._execution_ref.stop_execution(self._op_key, _tell=True)
            return
        if self._last_state == OperandState.FINISHED:
            # delete data on cancelled
            self.ref().free_data(state=OperandState.CANCELLED, _tell=True)
            return
        if self._last_state == OperandState.READY:
            # stop worker application
            self._assigner_ref.remove_apply(self._op_key)
        self.state = OperandState.CANCELLED
        self.ref().start_operand(_tell=True)

    @log_unhandled
    def _handle_cancelled(self):
        if self._worker_endpoint:
            self._resource_ref.deallocate_resource(self._session_id, self._op_key, self._worker_endpoint)
        if self._is_terminal:
            self._mark_finished_terminal(final_state=GraphState.CANCELLED)
        for k in self._succ_keys:
            self._get_operand_actor(k).propagate_state(OperandState.CANCELLING, _tell=True)

    @log_unhandled
    def _handle_freed(self):
        pass

    def _handle_unscheduled(self):
        pass

    def _mark_finished_terminal(self, final_state=None):
        for graph_ref in self._graph_refs:
            graph_ref.mark_terminal_finished(
                self._op_key, final_state=final_state, _tell=True)
