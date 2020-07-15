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

import contextlib
import itertools
import logging
import operator
import os
import random
import sys
import time
from collections import defaultdict, OrderedDict
from functools import lru_cache, reduce

import numpy as np

from .analyzer import GraphAnalyzer
from .assigner import AssignerActor
from .kvstore import KVStoreActor
from .operands import get_operand_actor_class, OperandState
from .resource import ResourceActor
from .session import SessionActor
from .utils import SchedulerActor, GraphState
from ..actors.errors import ActorAlreadyExist
from ..config import options
from ..errors import ExecutionInterrupted, GraphNotExists
from ..graph import DAG
from ..operands import Fetch, ShuffleProxy, VirtualOperand, SuccessorsExclusive
from ..serialize import dataserializer
from ..tiles import handler, IterativeChunkGraphBuilder, \
    TileableGraphBuilder, get_tiled
from ..utils import serialize_graph, deserialize_graph, log_unhandled, \
    build_fetch_chunk, build_fetch_tileable, calc_nsplits, \
    get_chunk_shuffle_key, enter_build_mode, has_unknown_shape
from ..context import DistributedContext

logger = logging.getLogger(__name__)


class GraphMetaActor(SchedulerActor):
    """
    Actor storing metadata of a graph
    """
    @staticmethod
    def gen_uid(session_id, graph_key):
        return 's:0:graph_meta$%s$%s' % (session_id, graph_key)

    def __init__(self, session_id, graph_key):
        super().__init__()
        self._session_id = session_id
        self._graph_key = graph_key

        self._kv_store_ref = None
        self._graph_wait_ref = None

        self._start_time = None
        self._end_time = None
        self._state = None
        self._final_state = None
        self._exc = None

        self._graph_finish_event = None

        self._op_infos = defaultdict(dict)
        self._state_to_infos = defaultdict(dict)

    def post_create(self):
        super().post_create()
        self._kv_store_ref = self.ctx.actor_ref(KVStoreActor.default_uid())
        if not self.ctx.has_actor(self._kv_store_ref):
            self._kv_store_ref = None

        self._graph_finish_event = self.ctx.event()
        graph_wait_uid = GraphWaitActor.gen_uid(self._session_id, self._graph_key)
        try:
            graph_wait_uid = self.ctx.distributor.make_same_process(
                graph_wait_uid, self.uid)
        except AttributeError:
            pass
        self._graph_wait_ref = self.ctx.create_actor(
            GraphWaitActor, self._graph_finish_event, uid=graph_wait_uid)

    def pre_destroy(self):
        self._graph_wait_ref.destroy()
        super().pre_destroy()

    def get_graph_info(self):
        return self._start_time, self._end_time, len(self._op_infos)

    def set_graph_start(self):
        self._start_time = time.time()

    def set_graph_end(self):
        self._end_time = time.time()
        self._graph_finish_event.set()

    def set_state(self, state):
        self._state = state
        if self._kv_store_ref is not None:
            self._kv_store_ref.write(
                '/sessions/%s/graph/%s/state' % (self._session_id, self._graph_key), state.name, _tell=True)

    def get_state(self):
        return self._state

    def set_exc_info(self, exc):
        self._exc = exc

    def get_exc_info(self):
        return self._exc

    def get_wait_ref(self):
        return self._graph_wait_ref

    def set_final_state(self, state):
        self._final_state = state
        if self._kv_store_ref is not None:
            self._kv_store_ref.write(
                '/sessions/%s/graph/%s/final_state' % (self._session_id, self._graph_key), state.name, _tell=True)

    def get_final_state(self):
        return self._final_state

    def get_operand_info(self, state=None):
        return self._op_infos if state is None else self._state_to_infos[state]

    def update_op_state(self, op_key, op_name, op_state):
        new_info = dict(op_name=op_name, state=op_state)
        old_state = self._op_infos[op_key].get('state')
        self._op_infos[op_key].update(new_info)
        if old_state is not None:
            self._state_to_infos[old_state].pop(op_key, None)
        self._state_to_infos[op_state][op_key] = self._op_infos[op_key]

    def update_op_worker(self, op_key, op_name, worker):
        new_info = dict(op_name=op_name, worker=worker)
        self._op_infos[op_key].update(new_info)

    def update_op_infos(self, op_infos):
        for key, info in op_infos.items():
            old_state = self._op_infos[key].get('state')
            if old_state is not None:
                self._state_to_infos[old_state].pop(key, None)
            if info.get('state') is not None:
                self._state_to_infos[info['state']][key] = info

            self._op_infos[key].update(info)

    @log_unhandled
    def calc_stats(self):
        states = list(OperandState.__members__.values())
        state_mapping = OrderedDict((v, idx) for idx, v in enumerate(states))
        state_names = [s.name for s in state_mapping]

        op_stats = OrderedDict()
        finished = 0
        total_count = len(self._op_infos)
        for operand_info in self._op_infos.values():
            if operand_info.get('virtual'):
                total_count -= 1
                continue
            op_name = operand_info['op_name']
            state = operand_info['state'] or OperandState.UNSCHEDULED
            if state in (OperandState.FINISHED, OperandState.FREED):
                finished += 1
            try:
                stats_list = op_stats[op_name]
            except KeyError:
                stats_list = op_stats[op_name] = [0] * len(state_mapping)
            stats_list[state_mapping[state]] += 1

        data_src = OrderedDict([('states', state_names), ])
        for op, state_stats in op_stats.items():
            sum_chunks = sum(state_stats)
            data_src[op] = [v * 100.0 / sum_chunks for v in state_stats]

        ops = list(data_src)[1:]
        states = data_src['states']
        transposed = OrderedDict()
        transposed['ops'] = ops
        for sid, state in enumerate(states):
            transposed[state] = list()
            for op in ops:
                transposed[state].append(data_src[op][sid])

        percentage = finished * 100.0 / total_count if total_count != 0 else 1
        return ops, transposed, percentage


class GraphWaitActor(SchedulerActor):
    @staticmethod
    def gen_uid(session_id, graph_key):
        return 's:0:graph_wait$%s$%s' % (session_id, graph_key)

    def __init__(self, graph_event):
        super().__init__()
        self._graph_event = graph_event

    def wait(self, timeout=None):
        self._graph_event.wait(timeout)


class GraphActor(SchedulerActor):
    """
    Actor handling execution and status of a Mars graph
    """
    @staticmethod
    def gen_uid(session_id, graph_key):
        return 's:0:graph$%s$%s' % (session_id, graph_key)

    def __init__(self, session_id, graph_key, serialized_tileable_graph,
                 target_tileables=None, serialized_chunk_graph=None,
                 state=GraphState.UNSCHEDULED, final_state=None):
        super().__init__()
        self._graph_key = graph_key
        self._session_id = session_id
        self._serialized_tileable_graph = serialized_tileable_graph
        self._serialized_chunk_graph = serialized_chunk_graph
        self._state = state
        self._final_state = final_state
        self._context = None

        self._operand_free_paused = False

        self._cluster_info_ref = None
        self._assigner_actor_ref = None
        self._resource_actor_ref = None
        self._kv_store_ref = None
        self._graph_meta_ref = None
        self._session_ref = None

        self._tileable_graph_cache = None
        self._chunk_graph_cache = None

        # chunk graph builder
        self._chunk_graph_builder = None

        self._resource_actor = None
        # generated once
        self._target_tileable_keys = target_tileables
        self._target_tileable_datas = []
        self._tileable_key_to_opid = dict()
        # used over all iteration
        self._tileable_key_opid_to_tiled = defaultdict(list)
        self._chunk_key_id_to_chunk = dict()
        self._all_terminated_tileable_keys = set()
        # used only for current iteration
        self._tileable_to_fetch = dict()
        self._operand_infos = dict()
        self._op_key_to_chunk = defaultdict(list)
        self._terminal_chunk_op_key_to_tileable_key = defaultdict(set)
        self._terminal_tileable_key_to_chunk_op_keys = defaultdict(set)
        self._terminated_tileable_keys = set()
        self._terminated_chunk_keys = set()
        self._terminal_chunk_keys = set()
        self._target_tileable_finished = dict()

        self._assigned_workers = set()
        self._worker_adds = set()
        self._worker_removes = set()
        self._lost_chunks = set()

        self._graph_analyze_pool = None

    def post_create(self):
        super().post_create()
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        random.seed(int(time.time()))
        self.set_cluster_info_ref()
        self._assigner_actor_ref = self.get_actor_ref(AssignerActor.gen_uid(self._session_id))
        self._resource_actor_ref = self.get_actor_ref(ResourceActor.default_uid())
        self._session_ref = self.get_actor_ref(SessionActor.gen_uid(self._session_id))

        uid = GraphMetaActor.gen_uid(self._session_id, self._graph_key)
        self._graph_meta_ref = self.ctx.create_actor(
            GraphMetaActor, self._session_id, self._graph_key, uid=uid)

        self._kv_store_ref = self.ctx.actor_ref(KVStoreActor.default_uid())
        if not self.ctx.has_actor(self._kv_store_ref):
            self._kv_store_ref = None

        self._graph_analyze_pool = self.ctx.threadpool(1)

    def pre_destroy(self):
        super().pre_destroy()
        self._graph_meta_ref.destroy()

    @contextlib.contextmanager
    def _open_dump_file(self, prefix):  # pragma: no cover
        if options.scheduler.dump_graph_data:
            file_name = '%s-%s-%d.log' % (prefix, self._graph_key, int(time.time()))
            yield open(file_name, 'w')
        else:
            try:
                yield None
            except AttributeError:
                return

    @property
    def state(self):
        """
        running state of the graph
        """
        return self._state

    @state.setter
    def state(self, value):
        if value != self._state:
            logger.debug('Graph %s state from %s to %s.', self._graph_key, self._state, value)
        self._state = value
        self._graph_meta_ref.set_state(value, _tell=True, _wait=False)

    @log_unhandled
    def reload_state(self):
        result = self._graph_meta_ref.get_state()
        if result is None:
            return
        state = self._state = result
        return state

    @property
    def final_state(self):
        return self._final_state

    @property
    def context(self):
        if self._context is None:
            self._context = DistributedContext(self.address, self._session_id,
                                               actor_ctx=self.ctx, address=self.address)
        return self._context

    @final_state.setter
    def final_state(self, value):
        self._final_state = value
        self._graph_meta_ref.set_final_state(value, _tell=True, _wait=False)

    def _detect_cancel(self, callback=None):
        if self.reload_state() == GraphState.CANCELLING:
            logger.info('Cancel detected, stopping')
            if callback:
                callback()
            else:
                self._graph_meta_ref.set_graph_end(_tell=True, _wait=False)
                self.state = GraphState.CANCELLED
            raise ExecutionInterrupted

    @log_unhandled
    def execute_graph(self, compose=True):
        """
        Start graph execution
        """

        self._graph_meta_ref.set_graph_start(_tell=True, _wait=False)
        self.state = GraphState.PREPARING
        self._execute_graph(compose=compose)

    def _dump_graph(self):
        with self._open_dump_file('graph') as outf:  # pragma: no cover
            succ_op_keys = dict()
            graph = self._chunk_graph_cache

            outf.write('DOT:\n%s\n\n' % graph.to_dot())
            outf.write('CONNECTIONS:\n')

            chunks = list(graph)
            for n in graph:
                succ_op_keys[n.key] = succ_op_keys.get(n.key, set()) \
                    | set(succ.op.key for succ in graph.iter_successors(n))
                for c in n.inputs:
                    if isinstance(c.op, Fetch):
                        if c.key not in succ_op_keys:
                            chunks.append(c)
                            succ_op_keys[c.key] = set()
                        succ_op_keys[c.key].add(n.op.key)
            for n in chunks:
                outf.write(
                    '%s(%s)[%s] -> %s\n' % (
                        n.op.key, n.key, type(n.op).__name__, ','.join(sorted(succ_op_keys[n.key])))
                )

    def _execute_graph(self, compose=True):
        try:
            self.prepare_graph(compose=compose)
            self._detect_cancel()

            self._dump_graph()

            self.analyze_graph()
            self._detect_cancel()

            if self.state == GraphState.SUCCEEDED:
                self._graph_meta_ref.set_graph_end(_tell=True, _wait=False)
            else:
                self.create_operand_actors()
                self._detect_cancel(self.stop_graph)
        except ExecutionInterrupted:
            pass
        except:  # noqa: E722
            logger.exception('Failed to start graph execution.')
            self._graph_meta_ref.set_exc_info(sys.exc_info(), _tell=True, _wait=False)
            self.stop_graph()
            self.state = GraphState.FAILED
            self._graph_meta_ref.set_graph_end(_tell=True, _wait=False)
            raise

    @log_unhandled
    def stop_graph(self):
        """
        Stop graph execution
        """
        from .operands import OperandActor
        if self.state == GraphState.CANCELLED:
            return
        self.state = GraphState.CANCELLING

        try:
            chunk_graph = self.get_chunk_graph()
        except (KeyError, GraphNotExists, AttributeError):
            self.state = GraphState.CANCELLED
            return

        has_stopping = False
        for chunk in chunk_graph:
            if chunk.op.key not in self._operand_infos:
                continue
            if self._operand_infos[chunk.op.key].get('state') in \
                    (OperandState.READY, OperandState.RUNNING, OperandState.FINISHED):
                # we only need to stop on ready, running and finished operands
                op_uid = OperandActor.gen_uid(self._session_id, chunk.op.key)
                scheduler_addr = self.get_scheduler(op_uid)
                ref = self.ctx.actor_ref(op_uid, address=scheduler_addr)
                has_stopping = True
                ref.stop_operand(_tell=True)
        if not has_stopping:
            self.state = GraphState.CANCELLED
            self._graph_meta_ref.set_graph_end(_tell=True, _wait=False)

    @log_unhandled
    def get_chunk_graph(self):
        try:
            return self._chunk_graph_builder.iterative_chunk_graphs[-1]
        except (IndexError, AttributeError):
            raise GraphNotExists from None

    def _gen_tileable_graph(self):
        if self._tileable_graph_cache is None:
            tileable_graph = deserialize_graph(self._serialized_tileable_graph, graph_cls=DAG)
            self._tileable_graph_cache = tileable_graph

            logger.debug('Begin preparing graph %s with %d tileables to chunk graph.',
                         self._graph_key, len(tileable_graph))

        return self._tileable_graph_cache

    def _gen_chunk_graph(self):
        if self._chunk_graph_cache is None:
            self._chunk_graph_cache = DAG()
        return self._chunk_graph_cache

    def _scan_tileable_graph(self):
        tileable_graph = self._tileable_graph_cache
        target_tileable_provided = self._target_tileable_keys is not None
        if self._target_tileable_keys is None:
            self._target_tileable_keys = []
            self._target_tileable_datas = []
        for tn in tileable_graph:
            self._tileable_key_to_opid[tn.key] = tn.op.id
            if tileable_graph.count_successors(tn) == 0:
                # no successors
                if not target_tileable_provided:
                    self._target_tileable_keys.append(tn.key)
                    self._target_tileable_datas.append(tn)
            if target_tileable_provided and tn.key in self._target_tileable_keys:
                self._target_tileable_datas.append(tn)

    @classmethod
    def _merge_chunk_graph(cls, chunk_graph, update_chunk_graph):
        chunk_key_id_to_chunk = dict()
        for c in chunk_graph:
            chunk_key_id_to_chunk[c.key, c.id] = c
        for update_c in update_chunk_graph:
            if isinstance(update_c.op, Fetch):
                continue
            chunk_graph.add_node(update_c)
            for inp in update_c.inputs or []:
                if inp in chunk_graph:
                    chunk_graph.add_edge(inp, update_c)
                elif isinstance(inp.op, Fetch):
                    if (inp.key, inp.id) in chunk_key_id_to_chunk:
                        inp = chunk_key_id_to_chunk[inp.key, inp.id]
                    else:
                        chunk_graph.add_node(inp)
                    chunk_graph.add_edge(inp, update_c)
        return chunk_graph

    @classmethod
    def _prune_chunk_graph(cls, chunk_graph, result_chunk_keys):
        result_op_keys = set()
        for c in chunk_graph:
            if c.key in result_chunk_keys:
                result_op_keys.add(c.op.key)

        reverse_chunk_graph = chunk_graph.build_reversed()
        marked = set()
        for c in reverse_chunk_graph.topological_iter():
            if reverse_chunk_graph.count_predecessors(c) == 0 and \
                    c.op.key in result_op_keys:
                marked.add(c)
            elif any(inp in marked for inp in reverse_chunk_graph.iter_predecessors(c)):
                marked.add(c)
        for n in list(chunk_graph):
            if n not in marked:
                chunk_graph.remove_node(n)

    def _clear_for_iteration(self):
        self._tileable_to_fetch.clear()
        self._op_key_to_chunk.clear()
        self._terminal_chunk_op_key_to_tileable_key.clear()
        self._terminal_tileable_key_to_chunk_op_keys.clear()
        self._target_tileable_finished.clear()
        self._terminated_tileable_keys.clear()
        self._terminal_chunk_keys.clear()
        self._terminated_chunk_keys.clear()

    def _gen_target_info(self):
        chunk_graph_builder = self._chunk_graph_builder
        is_done = chunk_graph_builder.done
        cur_tileable_graph = chunk_graph_builder.prev_tileable_graph
        cur_chunk_graph = self.get_chunk_graph()
        chunk_keys = {c.key for c in cur_chunk_graph}

        if is_done:
            target_tileable_keys = self._target_tileable_keys
        else:
            target_tileable_keys = set(self._target_tileable_keys)
            for interrupted_op in chunk_graph_builder.interrupted_ops:
                for inp in interrupted_op.inputs:
                    if inp.op not in chunk_graph_builder.interrupted_ops:
                        target_tileable_keys.add(inp.key)

        # scan tileable graph
        for tn in cur_tileable_graph:
            if not isinstance(tn.op, Fetch) and tn.key in target_tileable_keys:
                self._target_tileable_finished[tn.key] = set()
                tiled_tn = self._tileable_key_opid_to_tiled[tn.key, tn.op.id][-1]
                for c in tiled_tn.chunks:
                    assert c.key in chunk_keys
                    if isinstance(c.op, Fetch):
                        # if the terminal chunk's op is fetch
                        # just add it to terminated
                        self._terminated_chunk_keys.add(c.key)
                    self._terminal_chunk_keys.add(c.key)
                    self._terminal_chunk_op_key_to_tileable_key[c.op.key].add(tn.key)
                    self._terminal_tileable_key_to_chunk_op_keys[tn.key].add(c.op.key)

        # if not done, scan chunk graph again, for the reason that,
        # make sure some chunks partially generated by tiling of some op
        # are added to terminal chunks correctly
        if not is_done:
            for cn in cur_chunk_graph:
                if cur_chunk_graph.count_successors(cn) == 0 and \
                        not isinstance(cn.op, Fetch):
                    assert cn.key in chunk_keys
                    self._terminal_chunk_keys.add(cn.key)

        logger.debug('Terminal chunk keys: %r' % self._terminal_chunk_keys)

    @log_unhandled
    @enter_build_mode
    def prepare_graph(self, compose=True):
        """
        Tile and compose tileable graph into chunk graph
        :param compose: if True, do compose after tiling
        """
        tileable_graph = self._gen_tileable_graph()
        chunk_graph = self._gen_chunk_graph()
        self._clear_for_iteration()

        if self._chunk_graph_builder is None:
            # gen target tileable keys if not provided
            self._scan_tileable_graph()

            def on_tile(raw_tileables, tileds):
                first = tileds[0]
                if any(inp in self._tileable_to_fetch for inp in raw_tileables[0].inputs):
                    new_inputs = []
                    for inp in raw_tileables[0].inputs:
                        if inp in self._tileable_to_fetch:
                            tiled_inp = get_tiled(self._tileable_to_fetch[inp])
                            new_inputs.append(tiled_inp)
                        else:
                            new_inputs.append(get_tiled(inp))
                    first.op.inputs = new_inputs
                if isinstance(first.op, Fetch):
                    raw_first = raw_tileables[0]
                    if (raw_first.key, raw_first.op.id) in self._tileable_key_opid_to_tiled:
                        # fetch op generated by iterative tiling graph builder
                        tiled = self._tileable_key_opid_to_tiled[raw_first.key, raw_first.op.id][-1]
                        return [build_fetch_tileable(tiled)]
                    else:
                        return [self.tile_fetch_tileable(first)]
                else:
                    return self.context.wraps(handler.dispatch)(first.op)

            def on_tile_success(tiled_before, tiled_after):
                if not isinstance(tiled_before.op, Fetch):
                    self._tileable_key_opid_to_tiled[
                        tiled_before.key, tiled_before.op.id].append(tiled_after)
                return tiled_after

            # do not compose here
            self._chunk_graph_builder = IterativeChunkGraphBuilder(
                on_tile=on_tile, on_tile_success=on_tile_success, compose=False)

        chunk_graph_builder = self._chunk_graph_builder
        if chunk_graph_builder.prev_tileable_graph is None:
            # first tile
            fetch_tileables = [t for t in tileable_graph if isinstance(t.op, Fetch)]
            cur_chunk_graph = chunk_graph_builder.build(
                # add fetch tileables to make sure that they won't be fused
                self._target_tileable_datas + fetch_tileables, tileable_graph)
        else:
            # some TilesFail happens before
            # build tileable graph from failed ops and their inputs
            failed_tileable_set = set(itertools.chain(
                *(op.outputs for op in chunk_graph_builder.interrupted_ops)))
            tileable_graph_builder = TileableGraphBuilder(
                inputs_selector=lambda inps: [inp for inp in inps if inp in failed_tileable_set])
            to_run_tileable_graph = tileable_graph_builder.build(failed_tileable_set)
            to_fetch_tileables = []
            for failed_op in chunk_graph_builder.interrupted_ops:
                for inp in failed_op.inputs:
                    if inp not in failed_tileable_set:
                        if inp not in self._tileable_to_fetch:
                            fetch_inp = build_fetch_tileable(inp).data
                            self._tileable_to_fetch[inp] = fetch_inp
                        else:
                            fetch_inp = self._tileable_to_fetch[inp]
                        to_fetch_tileables.append(fetch_inp)
                        to_run_tileable_graph.add_node(fetch_inp)
                        for o in failed_op.outputs:
                            to_run_tileable_graph.add_edge(fetch_inp, o)
            cur_chunk_graph = chunk_graph_builder.build(
                # add to_fetch_tileables to make sure that fetch chunk would not be fused
                self._target_tileable_datas + to_fetch_tileables,
                tileable_graph=to_run_tileable_graph)

        self._gen_target_info()
        if chunk_graph_builder.done:
            self._prune_chunk_graph(cur_chunk_graph, self._terminal_chunk_keys)
        if compose:
            cur_chunk_graph.compose(keys=list(self._terminal_chunk_keys))

        for c in list(cur_chunk_graph):
            if not isinstance(c.op, Fetch):
                self._op_key_to_chunk[c.op.key].append(c)

            # merge current chunk into self.chunk_graph_cache
            if isinstance(c.op, Fetch):
                continue
            chunk_graph.add_node(c)
            self._chunk_key_id_to_chunk[c.key, c.id] = c
            for inp in cur_chunk_graph.iter_predecessors(c):
                if inp in chunk_graph:
                    chunk_graph.add_edge(inp, c)
                elif isinstance(inp.op, Fetch):
                    if (inp.key, inp.id) in self._chunk_key_id_to_chunk:
                        inp = self._chunk_key_id_to_chunk[inp.key, inp.id]
                    else:
                        chunk_graph.add_node(inp)
                    chunk_graph.add_edge(inp, c)

    def _get_worker_slots(self):
        metrics = self._resource_actor_ref.get_workers_meta()
        return dict((ep, int(metrics[ep]['hardware']['cpu_total'])) for ep in metrics)

    def _collect_external_input_metas(self, ext_chunks_to_inputs):
        ext_chunk_keys = reduce(operator.add, ext_chunks_to_inputs.values(), [])
        metas = dict(zip(ext_chunk_keys,
                         self.chunk_meta.batch_get_chunk_meta(self._session_id, ext_chunk_keys)))
        input_chunk_metas = defaultdict(dict)
        for chunk_key, input_chunk_keys in ext_chunks_to_inputs.items():
            chunk_metas = input_chunk_metas[chunk_key]
            for k in input_chunk_keys:
                chunk_metas[k] = metas[k]
        return input_chunk_metas

    def assign_operand_workers(self, op_keys, input_chunk_metas=None, analyzer=None):
        operand_infos = self._operand_infos
        chunk_graph = self.get_chunk_graph()

        initial_chunks = [c for c in chunk_graph
                          if chunk_graph.count_successors(c) == 0]
        # TODO refine this to support mixed scenarios here
        if all(c.op.expect_worker is not None for c in initial_chunks):
            assignments = {c.op.key: c.op.expect_worker for c in initial_chunks}
        else:
            if analyzer is None:
                analyzer = GraphAnalyzer(chunk_graph, self._get_worker_slots())
            assignments = analyzer.calc_operand_assignments(op_keys, input_chunk_metas=input_chunk_metas)
        for idx, (k, v) in enumerate(assignments.items()):
            operand_infos[k]['optimize']['placement_order'] = idx
            operand_infos[k]['target_worker'] = v
        return assignments

    def _assign_initial_workers(self, analyzer):
        # collect external inputs for eager mode
        ext_chunks_to_inputs = analyzer.collect_external_input_chunks(initial=True)
        input_chunk_metas = self._collect_external_input_metas(ext_chunks_to_inputs)

        def _do_assign():
            # do placements
            return self.assign_operand_workers(
                analyzer.get_initial_operand_keys(), input_chunk_metas=input_chunk_metas,
                analyzer=analyzer
            )

        return self._graph_analyze_pool.submit(_do_assign).result()

    @log_unhandled
    def analyze_graph(self, **kwargs):
        operand_infos = self._operand_infos
        chunk_graph = self.get_chunk_graph()

        # remove fetch chunk if exists
        if any(isinstance(c.op, Fetch) for c in chunk_graph):
            chunk_graph = chunk_graph.copy()
            for c in list(chunk_graph):
                if isinstance(c.op, Fetch):
                    chunk_graph.remove_node(c)

        if len(chunk_graph) == 0:
            self.state = GraphState.SUCCEEDED
            return

        for n in chunk_graph:
            k = n.op.key
            succ_size = chunk_graph.count_successors(n)
            if k not in operand_infos:
                operand_infos[k] = dict(optimize=dict(
                    depth=0, demand_depths=(), successor_size=succ_size, descendant_size=0
                ))
            else:
                operand_infos[k]['optimize']['successor_size'] = succ_size

        worker_slots = self._get_worker_slots()
        if not worker_slots:
            raise RuntimeError('No worker attached for execution')

        self._assigned_workers = set(worker_slots)
        analyzer = GraphAnalyzer(chunk_graph, worker_slots)

        for k, v in analyzer.calc_depths().items():
            operand_infos[k]['optimize']['depth'] = v

        for k, v in analyzer.calc_descendant_sizes().items():
            operand_infos[k]['optimize']['descendant_size'] = v

        if kwargs.get('do_placement', True):
            logger.debug('Placing initial chunks for graph %s', self._graph_key)
            self._assign_initial_workers(analyzer)

    @log_unhandled
    def get_executable_operand_dag(self, op_key, input_chunk_keys=None, serialize=True):
        """
        Make an operand into a worker-executable dag
        :param op_key: operand key
        :param input_chunk_keys: actual input chunks, None if use all chunks in input
        :param serialize: whether to return serialized dag
        """
        graph = DAG()
        input_mapping = dict()
        output_keys = set()

        input_chunk_keys = set(input_chunk_keys) if input_chunk_keys is not None else None
        for c in self._op_key_to_chunk[op_key]:
            inputs = []
            for inp in set(c.op.inputs or ()):
                try:
                    inp_chunk = input_mapping[(inp.key, inp.id)]
                except KeyError:
                    inp_chunk = input_mapping[(inp.key, inp.id)] \
                        = build_fetch_chunk(inp, input_chunk_keys).data
                    graph.add_node(inp_chunk)
                inputs.append(inp_chunk)

            for out in set(c.op.outputs or ()):
                if (out.key, out.id) not in output_keys:
                    output_keys.add((out.key, out.id))
                    graph.add_node(out)
                    for inp in inputs:
                        graph.add_edge(inp, out)
        if serialize:
            return serialize_graph(graph)
        else:
            return graph

    @staticmethod
    def _collect_operand_io_meta(graph, chunks):
        # collect operand i/o information
        predecessor_keys = set()
        successor_keys = set()
        input_chunk_keys = set()
        shared_input_chunk_keys = set()
        no_prepare_chunk_keys = set()
        chunk_keys = set()
        shuffle_keys = dict()
        predecessors_to_successors = dict()

        for c in chunks:
            # handling predecessor args
            for pn in graph.iter_predecessors(c):
                if not isinstance(pn.op, Fetch):
                    predecessor_keys.add(pn.op.key)
                input_chunk_keys.add(pn.key)
                if graph.count_successors(pn) > 1:
                    shared_input_chunk_keys.add(pn.key)

            for inp, prep in zip(c.op.inputs or (), c.op.prepare_inputs):
                if not prep and inp.key in input_chunk_keys:
                    no_prepare_chunk_keys.add(inp.key)

            # handling successor args
            for sn in graph.iter_successors(c):
                successor_keys.add(sn.op.key)
            if isinstance(c.op, ShuffleProxy):
                for sn in graph.iter_successors(c):
                    shuffle_keys[sn.op.key] = get_chunk_shuffle_key(sn)
            if isinstance(c.op, SuccessorsExclusive):
                for sn in graph.iter_successors(c):
                    predecessors_to_successors[sn.inputs[0].op.key] = sn.op.key

            chunk_keys.update(co.key for co in c.op.outputs)

        io_meta = dict(
            predecessors=list(predecessor_keys),
            successors=list(successor_keys),
            input_chunks=list(input_chunk_keys),
            no_prepare_chunk_keys=list(no_prepare_chunk_keys),
            shared_input_chunks=list(shared_input_chunk_keys),
            chunks=list(chunk_keys),
        )
        if shuffle_keys:
            io_meta['shuffle_keys'] = [shuffle_keys.get(k) for k in io_meta['successors']]
        if predecessors_to_successors:
            io_meta['predecessors_to_successors'] = predecessors_to_successors
        return io_meta

    @log_unhandled
    def create_operand_actors(self, _clean_info=True, _start=True):
        """
        Create operand actors for all operands
        """
        logger.debug('Creating operand actors for graph %s', self._graph_key)

        chunk_graph = self.get_chunk_graph()
        operand_infos = self._operand_infos

        session_id = self._session_id
        op_refs = dict()
        meta_op_infos = dict()
        initial_keys = []
        to_allocate_op_keys = set()
        for op_key in self._op_key_to_chunk:
            chunks = self._op_key_to_chunk[op_key]
            op = chunks[0].op

            if isinstance(op, VirtualOperand):
                operand_infos[op_key]['virtual'] = True

            op_name = type(op).__name__ if op.stage is None \
                else '%s:%s' % (type(op).__name__, op.stage.name)
            op_info = operand_infos[op_key]
            meta_op_info = meta_op_infos[op_key] = dict()

            io_meta = self._collect_operand_io_meta(chunk_graph, chunks)
            op_info['op_name'] = meta_op_info['op_name'] = op_name
            op_info['io_meta'] = io_meta
            op_info['executable_dag'] = self.get_executable_operand_dag(op_key)
            # todo change this when other calc devices supported
            op_info['calc_device'] = 'cuda' if op.gpu else 'cpu'

            if io_meta['predecessors']:
                state = OperandState.UNSCHEDULED
            else:
                initial_keys.append(op_key)
                state = OperandState.READY
                op_info['is_initial'] = True
            op_info['retryable'] = op.retryable
            op_info['retries'] = 0
            meta_op_info['state'] = op_info['state'] = state
            meta_op_info['worker'] = op_info.get('worker')

            if any(c.key in self._terminal_chunk_keys for c in op.outputs):
                op_info['is_terminal'] = True

            op_cls = get_operand_actor_class(type(op))
            op_uid = op_cls.gen_uid(session_id, op_key)
            scheduler_addr = self.get_scheduler(op_uid)
            kw = {}
            # for the **real** initial chunks, we do batch submitting
            if options.scheduler.batch_enqueue_initials \
                    and chunk_graph.count_predecessors(chunks[0]) == 0:
                kw['allocated'] = True
                to_allocate_op_keys.add(op_key)

            op_refs[op_key] = self.ctx.create_actor(
                op_cls, session_id, self._graph_key, op_key, op_info.copy(),
                with_kvstore=self._kv_store_ref is not None,
                schedulers=self.get_schedulers(),
                uid=op_uid, address=scheduler_addr, wait=False, **kw
            )
            if _clean_info:
                op_info.pop('executable_dag', None)
                del op_info['io_meta']

        self.state = GraphState.RUNNING
        self._graph_meta_ref.update_op_infos(meta_op_infos, _tell=True, _wait=False)

        if _start:
            existing_keys = []
            for op_key, future in op_refs.items():
                try:
                    op_refs[op_key] = future.result()
                except ActorAlreadyExist:
                    existing_keys.append(op_key)

            append_futures = []
            for op_key in existing_keys:
                chunks = self._op_key_to_chunk[op_key]
                op = chunks[0].op
                op_info = operand_infos[op_key]
                op_info['io_meta'] = self._collect_operand_io_meta(chunk_graph, chunks)

                op_cls = get_operand_actor_class(type(op))
                op_uid = op_cls.gen_uid(session_id, op_key)
                scheduler_addr = self.get_scheduler(op_uid)
                op_ref = op_refs[op_key] = self.ctx.actor_ref(op_uid, address=scheduler_addr)
                append_futures.append(op_ref.append_graph(self._graph_key, op_info.copy(), _wait=False))
                if _clean_info:
                    op_info.pop('executable_dag', None)
                    del op_info['io_meta']
            [future.result() for future in append_futures]

            res_applications = [(op_key, op_info) for op_key, op_info in operand_infos.items()
                                if op_key in to_allocate_op_keys]
            self._assigner_actor_ref.apply_for_multiple_resources(
                session_id, res_applications, _tell=True)

    @log_unhandled
    def add_finished_terminal(self, op_key, final_state=None, exc=None):
        """
        Add a terminal operand to finished set. Calling this method
        will change graph state if all terminals are in finished states.
        :param op_key: operand key
        :param final_state: state of the operand
        """
        if self._state not in (GraphState.RUNNING, GraphState.CANCELLING):
            return
        if exc is not None:
            self._graph_meta_ref.set_exc_info(exc, _tell=True, _wait=False)

        tileable_keys = self._terminal_chunk_op_key_to_tileable_key[op_key]
        is_failed = final_state in (GraphState.CANCELLED, GraphState.FAILED)
        terminal_tileable_count = len(self._terminal_tileable_key_to_chunk_op_keys)
        for tileable_key in tileable_keys:
            self._target_tileable_finished[tileable_key].add(op_key)
            if final_state == GraphState.FAILED:
                if self.final_state != GraphState.CANCELLED:
                    self.final_state = GraphState.FAILED
            elif final_state == GraphState.CANCELLED:
                self.final_state = final_state

            if self._target_tileable_finished[tileable_key] == \
                    self._terminal_tileable_key_to_chunk_op_keys[tileable_key]:
                self._terminated_tileable_keys.add(tileable_key)
                self._all_terminated_tileable_keys.add(tileable_key)
                if not is_failed and len(self._terminated_tileable_keys) == terminal_tileable_count:
                    # update shape if tileable or its chunks have unknown shape
                    self._update_tileable_and_its_chunk_shapes()

        terminated_chunks = self._op_key_to_chunk[op_key]
        self._terminated_chunk_keys.update([c.key for c in terminated_chunks
                                            if c.key in self._terminal_chunk_keys])
        if self._terminated_chunk_keys == self._terminal_chunk_keys:
            if self._chunk_graph_builder.done or is_failed:
                if self._chunk_graph_builder.prev_tileable_graph is not None:
                    # if failed before, clear intermediate data
                    to_free_tileable_keys = \
                        self._all_terminated_tileable_keys - set(self._target_tileable_keys)
                    skip_chunk_keys = set()
                    for target_tileable_data in self._target_tileable_datas:
                        tiled_target_tileable_data = \
                            self._tileable_key_opid_to_tiled[target_tileable_data.key,
                                                             target_tileable_data.op.id][-1]
                        skip_chunk_keys.update([c.key for c in tiled_target_tileable_data.chunks])
                    [self.free_tileable_data(k, skip_chunk_keys=skip_chunk_keys)
                     for k in to_free_tileable_keys]
                self.state = self.final_state if self.final_state is not None else GraphState.SUCCEEDED
                self._graph_meta_ref.set_graph_end(_tell=True)
            else:
                self._execute_graph(compose=self._chunk_graph_builder.is_compose)

    def _update_tileable_and_its_chunk_shapes(self):
        need_update_tileable_to_tiled = dict()
        for tileable in self._chunk_graph_builder.prev_tileable_graph:
            if tileable.key in self._target_tileable_finished:
                tiled = self._tileable_key_opid_to_tiled[tileable.key, tileable.op.id][-1]
                if not has_unknown_shape(tiled):
                    continue
                need_update_tileable_to_tiled[tileable] = tiled

        if len(need_update_tileable_to_tiled) == 0:
            return

        need_update_chunks = list(c for t in need_update_tileable_to_tiled.values() for c in t.chunks)
        chunk_metas = self.chunk_meta.batch_get_chunk_meta(
            self._session_id, list(c.key for c in need_update_chunks))
        for chunk, chunk_meta in zip(need_update_chunks, chunk_metas):
            chunk.data._shape = chunk_meta.chunk_shape

        for tileable, tiled in need_update_tileable_to_tiled.items():
            chunk_idx_to_shape = OrderedDict((c.index, c.shape) for c in tiled.chunks)
            nsplits = calc_nsplits(chunk_idx_to_shape)
            tiled._nsplits = nsplits
            if any(np.isnan(s) for s in tileable.shape):
                shape = tuple(sum(ns) for ns in nsplits)
                tileable._update_shape(shape)
                tiled._update_shape(shape)

    @log_unhandled
    def remove_finished_terminal(self, op_key):
        """
        Remove a terminal operand from finished set as the data is lost.
        :param op_key: operand key
        """
        tileable_keys = self._terminal_chunk_op_key_to_tileable_key[op_key]
        for tileable_key in tileable_keys:
            self._target_tileable_finished[tileable_key].difference_update([op_key])
            self._terminated_tileable_keys.difference_update([tileable_key])

    def dump_unfinished_terminals(self):  # pragma: no cover
        """
        Dump unfinished terminal chunks into logger, only for debug purposes.
        """
        unfinished_dict = dict()
        for tileable_key, chunk_ops in self._terminal_tileable_key_to_chunk_op_keys.items():
            executed_ops = self._target_tileable_finished.get(tileable_key, ())
            unfinished = sorted(set(chunk_ops) - set(executed_ops))
            if unfinished:
                unfinished_dict[tileable_key] = unfinished
        logger.debug('Unfinished terminal chunks: %r', unfinished_dict)

    def get_state(self):
        return self.state

    def get_operand_states(self, op_keys):
        return [self._operand_infos[k]['state'] for k in op_keys if k in self._operand_infos]

    def set_operand_state(self, op_key, state):
        if op_key not in self._operand_infos and \
                self._chunk_graph_builder.iterative_chunk_graphs and \
                state == OperandState.FREED:
            # if iterative tiling is entered,
            # the `_operand_infos` will be a completely new one,
            # in this case, we don't actually care about if the op is freed
            return
        if op_key not in self._operand_infos and self.state in GraphState.TERMINATED_STATES:
            # if operand has been cleared in iterative tiling and execute again in another
            # graph, just ignore it.
            return
        op_info = self._operand_infos[op_key]
        op_info['state'] = state
        self._graph_meta_ref.update_op_state(op_key, op_info['op_name'], state,
                                             _tell=True, _wait=False)
        op_info.pop('failover_state', None)

    def get_operand_target_worker(self, op_key):
        return self._operand_infos[op_key]['target_worker']

    def get_operand_info(self, state=None):
        if not state:
            return self._operand_infos
        else:
            return dict((k, info) for k, info in self._operand_infos.items()
                        if info.get('state') == state)

    @log_unhandled
    def set_operand_worker(self, op_key, worker):
        if op_key not in self._operand_infos and self.state in GraphState.TERMINATED_STATES:
            # if operand has been cleared in iterative tiling and execute again in another
            # graph, just ignore it.
            return
        op_info = self._operand_infos[op_key]
        if worker:
            op_info['worker'] = worker
        else:
            op_info.pop('worker', None)
        self._graph_meta_ref.update_op_worker(op_key, op_info['op_name'], worker,
                                              _tell=True, _wait=False)

    @lru_cache(1000)
    def _get_operand_ref(self, key):
        from .operands import OperandActor
        op_uid = OperandActor.gen_uid(self._session_id, key)
        scheduler_addr = self.get_scheduler(op_uid)
        return self.ctx.actor_ref(op_uid, address=scheduler_addr)

    def _get_tileable_by_key(self, key):
        tid = self._tileable_key_to_opid[key]
        return self._tileable_key_opid_to_tiled[(key, tid)][-1]

    @log_unhandled
    def free_tileable_data(self, tileable_key, skip_chunk_keys=None, wait=False):
        tileable = self._get_tileable_by_key(tileable_key)
        futures = []
        for chunk in tileable.chunks:
            if skip_chunk_keys is not None and chunk.key in skip_chunk_keys:
                continue
            futures.append(self._get_operand_ref(chunk.op.key).free_data(
                check=False, _tell=not wait, _wait=False))
        logger.debug('Free tileable data: %s, chunk keys: %r' %
                     (tileable_key, [c.key for c in tileable.chunks]))
        [f.result() for f in futures]

    def get_tileable_metas(self, tileable_keys, filter_fields=None):
        """
        Get tileable meta including nsplits, chunk keys and chunk indexes.
        :param tileable_keys: tileable_keys.
        :param filter_fields: filter the fields('nsplits', 'chunk_keys', 'chunk_indexes') in meta.
        :return: metas list.
        """
        meta_names = ['nsplits', 'chunk_keys', 'chunk_indexes']
        filter_fields = filter_fields or meta_names
        for field in filter_fields:
            if field not in ['nsplits', 'chunk_keys', 'chunk_indexes']:
                raise ValueError('Field {} is invalid'.format(field))
        ret_nsplits = 'nsplits' in filter_fields
        metas = []
        for tileable_key in tileable_keys:
            meta = dict()
            tileable = self._get_tileable_by_key(tileable_key)
            chunk_keys, chunk_indexes = tuple(zip(*[(c.key, c.index) for c in tileable.chunks]))
            meta['chunk_keys'] = chunk_keys
            meta['chunk_indexes'] = chunk_indexes
            if ret_nsplits:
                if hasattr(tileable, 'shape') and np.nan in tileable.shape:
                    chunk_shapes = self.chunk_meta.batch_get_chunk_shape(self._session_id,  chunk_keys)
                    nsplits = calc_nsplits(OrderedDict(zip(chunk_indexes, chunk_shapes)))
                else:
                    nsplits = tileable.nsplits
                meta['nsplits'] = nsplits
                if filter_fields is not None:
                    metas.append([meta[k] for k in filter_fields])
        return metas

    def build_fetch_graph(self, tileable_key):
        """
        Convert single tileable node to tiled fetch tileable node and
        put into a graph which only contains one tileable node
        :param tileable_key: the key of tileable node
        """
        tileable = self._get_tileable_by_key(tileable_key)
        graph = DAG()

        new_tileable = build_fetch_tileable(tileable).data
        graph.add_node(new_tileable)
        return serialize_graph(graph)

    def tile_fetch_tileable(self, tileable):
        """
        Find the owner of the input tileable node and ask for tiling
        """
        tileable_key = tileable.key
        graph_ref = self.ctx.actor_ref(self._session_ref.get_graph_ref_by_tileable_key(tileable_key))
        fetch_graph = deserialize_graph(graph_ref.build_fetch_graph(tileable_key))
        return list(fetch_graph)[0]

    @log_unhandled
    @enter_build_mode
    def fetch_tileable_result(self, tileable_key, _check=True):
        # this function is for test usage
        from ..executor import Executor
        from ..worker.transfer import ResultSenderActor

        tileable = self._get_tileable_by_key(tileable_key)
        if _check and tileable_key not in self._terminated_tileable_keys:
            return None

        ctx = dict()
        for chunk in tileable.chunks:
            endpoints = self.chunk_meta.get_workers(self._session_id, chunk.key)
            if endpoints is None:
                raise KeyError('cannot fetch meta of chunk {}'.format(chunk))
            sender_ref = self.ctx.actor_ref(ResultSenderActor.default_uid(), address=endpoints[-1])
            ctx[chunk.key] = dataserializer.loads(sender_ref.fetch_data(self._session_id, chunk.key))

        if len(tileable.chunks) == 1:
            return dataserializer.dumps(ctx[tileable.chunks[0].key])

        fetch_tileable = build_fetch_tileable(tileable)
        concat_chunk = fetch_tileable.op.concat_tileable_chunks(fetch_tileable).chunks[0].data
        chunk_graph = DAG()
        chunk_graph.add_node(concat_chunk)
        for fetch_chunk in fetch_tileable.chunks:
            chunk_graph.add_node(fetch_chunk.data)
            chunk_graph.add_edge(fetch_chunk.data, concat_chunk)
        executor = Executor(storage=ctx)
        concat_result = executor.execute_graph(chunk_graph, keys=[concat_chunk.key])[0]
        return dataserializer.dumps(concat_result)

    @log_unhandled
    def add_fetch_tileable(self, tileable_key, tileable_id, shape, dtype, chunk_size, chunk_keys):
        from ..tensor.utils import create_fetch_tensor
        tensor = create_fetch_tensor(chunk_size, shape, dtype,
                                     tileable_key, tileable_id, chunk_keys)
        self._tileable_key_to_opid[tileable_key] = tensor.op.id
        self._tileable_key_opid_to_tiled[(tileable_key, tensor.op.id)].append(tensor)

    @log_unhandled
    def check_operand_can_be_freed(self, succ_op_keys):
        """
        Check if the data of an operand can be freed.

        :param succ_op_keys: keys of successor operands
        :return: True if can be freed, False if cannot. None when the result
                 is not determinant and we need to test later.
        """
        operand_infos = self._operand_infos
        for k in succ_op_keys:
            if k not in operand_infos:
                continue
            op_info = operand_infos[k]
            op_state = op_info.get('state')
            if op_state not in OperandState.SUCCESSFUL_STATES:
                return False
            failover_state = op_info.get('failover_state')
            if failover_state and failover_state not in OperandState.SUCCESSFUL_STATES:
                return False
        # if can be freed but blocked by an ongoing fail-over step,
        # we try later.
        if self._operand_free_paused:
            return None
        return True

    @log_unhandled
    def handle_worker_change(self, adds, removes, lost_chunks, handle_later=True):
        """
        Calculate and propose changes of operand states given changes
        in workers and lost chunks.

        :param adds: endpoints of workers newly added to the cluster
        :param removes: endpoints of workers removed to the cluster
        :param lost_chunks: keys of lost chunks
        :param handle_later: run the function later, only used in this actor
        """
        if self._state in GraphState.TERMINATED_STATES:
            return

        if handle_later:
            # Run the fail-over process later.
            # This is the default behavior as we need to make sure that
            # all crucial state changes are received by GraphActor.
            # During the delay, no operands are allowed to be freed.
            self._operand_free_paused = True
            self._worker_adds.update(adds)
            self._worker_removes.update(removes)
            self._lost_chunks.update(lost_chunks)
            self.ref().handle_worker_change(adds, removes, lost_chunks,
                                            handle_later=False, _delay=0.5, _tell=True)
            return
        else:
            self._operand_free_paused = False

        adds = self._worker_adds
        self._worker_adds = set()
        removes = self._worker_removes
        self._worker_removes = set()
        lost_chunks = self._lost_chunks
        self._lost_chunks = set()
        if not adds and not removes:
            return

        if all(ep in self._assigned_workers for ep in adds) \
                and not any(ep in self._assigned_workers for ep in removes):
            return

        worker_slots = self._get_worker_slots()
        self._assigned_workers = set(worker_slots)
        removes_set = set(removes)

        # collect operand states
        operand_infos = self._operand_infos
        fixed_assigns = dict()
        graph_states = dict()
        for key, op_info in operand_infos.items():
            op_worker = op_info.get('worker')
            if op_worker is None:
                continue

            op_state = graph_states[key] = op_info['state']

            # RUNNING nodes on dead workers should be moved to READY first
            if op_state == OperandState.RUNNING and op_worker in removes_set:
                graph_states[key] = OperandState.READY

            if op_worker in worker_slots:
                fixed_assigns[key] = op_info['worker']

        graph = self.get_chunk_graph()
        new_states = dict()
        ordered_states = OrderedDict(
            sorted(((k, v) for k, v in graph_states.items()),
                   key=lambda d: operand_infos[d[0]]['optimize'].get('placement_order', 0))
        )
        analyzer = GraphAnalyzer(graph, worker_slots, fixed_assigns, ordered_states, lost_chunks)
        if removes or lost_chunks:
            new_states = analyzer.analyze_state_changes()
            logger.debug('%d chunks lost. %d operands changed state.', len(lost_chunks),
                         len(new_states))

        logger.debug('Start reallocating initial operands')
        new_targets = dict(self._assign_initial_workers(analyzer))

        futures = []
        # make sure that all readies and runnings are included to be checked
        for key, op_info in operand_infos.items():
            if key in new_states:
                continue
            state = op_info['state']
            if state == OperandState.RUNNING and \
                    operand_infos[key]['worker'] not in removes_set:
                continue
            if state in (OperandState.READY, OperandState.RUNNING):
                new_states[key] = state

        for key, state in new_states.items():
            new_target = new_targets.get(key)

            op_info = operand_infos[key]
            from_state = op_info['state']
            # record the target state in special info key
            # in case of concurrency issues
            op_info['failover_state'] = state

            op_ref = self._get_operand_ref(key)
            # states may easily slip into the next state when we are
            # calculating fail-over states. Hence we need to include them
            # into source states.
            if from_state == OperandState.READY:
                from_states = [from_state, OperandState.RUNNING]
            elif from_state == OperandState.RUNNING:
                from_states = [from_state, OperandState.FINISHED]
            elif from_state == OperandState.FINISHED:
                from_states = [from_state, OperandState.FREED]
            else:
                from_states = [from_state]
            futures.append(op_ref.move_failover_state(
                from_states, state, new_target, removes, _tell=True, _wait=False))
        [f.result() for f in futures]

        self._dump_failover_info(adds, removes, lost_chunks, new_states)

    def _dump_failover_info(self, adds, removes, lost_chunks, new_states):  # pragma: no cover
        if not options.scheduler.dump_graph_data:
            return
        with self._open_dump_file('failover-record') as outf:
            outf.write('ADDED WORKERS:\n')
            for c in adds:
                outf.write(c + '\n')
            outf.write('REMOVED WORKERS:\n')
            for c in removes:
                outf.write(c + '\n')
            outf.write('\n\nLOST CHUNKS:\n')
            for c in lost_chunks:
                outf.write(repr(c) + '\n')
            outf.write('\n\nOPERAND SNAPSHOT:\n')
            for key, op_info in self._operand_infos.items():
                outf.write('Chunk: %s Worker: %r State: %s\n' %
                           (key, op_info.get('worker'), op_info['state'].value))
            outf.write('\n\nSTATE TRANSITIONS:\n')
            for key, state in new_states.items():
                outf.write('%s -> %s\n' % (key, state.name))
