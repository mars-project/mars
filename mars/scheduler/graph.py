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

import contextlib
import itertools
import logging
import operator
import os
import random
import time
from collections import deque, defaultdict

from .analyzer import GraphAnalyzer
from .assigner import AssignerActor
from .chunkmeta import ChunkMetaActor
from .resource import ResourceActor
from .kvstore import KVStoreActor
from .session import SessionActor
from .utils import SchedulerActor, OperandPosition, GraphState, OperandState
from ..compat import six, functools32, reduce, OrderedDict
from ..config import options
from ..errors import ExecutionInterrupted, GraphNotExists
from ..graph import DAG
from ..tiles import handler, DataNotReady
from ..serialize.dataserializer import loads, dumps
from ..utils import serialize_graph, deserialize_graph, merge_tensor_chunks, log_unhandled
from ..tensor.expressions.fetch import TensorFetch
from ..tensor.core import ChunkData

logger = logging.getLogger(__name__)


class ResultReceiverActor(SchedulerActor):
    def __init__(self):
        super(ResultReceiverActor, self).__init__()
        self._chunk_meta_ref = None

    @classmethod
    def default_name(cls):
        return 's:%s' % cls.__name__

    def post_create(self):
        super(ResultReceiverActor, self).post_create()

        self.set_cluster_info_ref()
        self._chunk_meta_ref = self.ctx.actor_ref(ChunkMetaActor.default_name())

    def fetch_tensor(self, session_id, graph_key, tensor_key):
        from ..executor import Executor
        from ..worker.transfer import ResultSenderActor

        graph_actor = self.ctx.actor_ref(GraphActor.gen_name(session_id, graph_key))
        fetch_graph = deserialize_graph(graph_actor.build_tensor_merge_graph(tensor_key))

        if len(fetch_graph) == 1 and isinstance(next(fetch_graph.iter_nodes()).op, TensorFetch):
            c = next(fetch_graph.iter_nodes())
            worker_ip = self._chunk_meta_ref.get_workers(session_id, c.key)[-1]
            sender_ref = self.ctx.actor_ref(ResultSenderActor.default_name(), address=worker_ip)
            future = sender_ref.fetch_data(session_id, c.key, _wait=False)
            return future.result()
        else:
            ctx = dict()
            target_keys = set()
            for c in fetch_graph:
                if isinstance(c.op, TensorFetch):
                    if c.key in ctx:
                        continue
                    endpoints = self._chunk_meta_ref.get_workers(session_id, c.key)
                    sender_ref = self.ctx.actor_ref(ResultSenderActor.default_name(), address=endpoints[-1])
                    future = sender_ref.fetch_data(session_id, c.key, _wait=False)
                    ctx[c.key] = future
                else:
                    target_keys.add(c.key)
            ctx = dict((k, loads(future.result())) for k, future in six.iteritems(ctx))
            executor = Executor(storage=ctx)
            concat_result = executor.execute_graph(fetch_graph, keys=target_keys)
            return dumps(concat_result[0])


class GraphMetaActor(SchedulerActor):
    """
    Actor storing metadata of a graph
    """
    @staticmethod
    def gen_name(session_id, graph_key):
        return 's:graph_meta$%s$%s' % (session_id, graph_key)

    def __init__(self, session_id, graph_key):
        super(GraphMetaActor, self).__init__()
        self._session_id = session_id
        self._graph_key = graph_key

        self._kv_store_ref = None

        self._state = None
        self._final_state = None

    def post_create(self):
        super(GraphMetaActor, self).post_create()
        self._kv_store_ref = self.ctx.actor_ref(KVStoreActor.default_name())
        if not self.ctx.has_actor(self._kv_store_ref):
            self._kv_store_ref = None

    def set_state(self, state):
        self._state = state
        if self._kv_store_ref is not None:
            self._kv_store_ref.write(
                '/sessions/%s/graph/%s/state' % (self._session_id, self._graph_key), state.name, _tell=True)

    def get_state(self):
        return self._state

    def set_final_state(self, state):
        self._final_state = state
        if self._kv_store_ref is not None:
            self._kv_store_ref.write(
                '/sessions/%s/graph/%s/final_state' % (self._session_id, self._graph_key), state.name, _tell=True)

    def get_final_state(self):
        return self._final_state


class GraphActor(SchedulerActor):
    """
    Actor handling execution and status of a Mars graph
    """
    @staticmethod
    def gen_name(session_id, graph_key):
        return 'graph$%s$%s' % (session_id, graph_key)

    def __init__(self, session_id, graph_key, serialized_tensor_graph,
                 target_tensors=None, serialized_chunk_graph=None,
                 state=GraphState.UNSCHEDULED, final_state=None):
        super(GraphActor, self).__init__()
        self._graph_key = graph_key
        self._session_id = session_id
        self._serialized_tensor_graph = serialized_tensor_graph
        self._serialized_chunk_graph = serialized_chunk_graph
        self._state = state
        self._final_state = final_state

        self._operand_free_paused = False

        self._start_time = None
        self._end_time = None
        self._nodes_num = None

        self._cluster_info_ref = None
        self._assigner_actor_ref = None
        self._resource_actor_ref = None
        self._kv_store_ref = None
        self._chunk_meta_ref = None
        self._graph_meta_ref = None
        self._session_ref = None

        self._tensor_graph_cache = None
        self._chunk_graph_cache = None

        self._op_key_to_chunk = defaultdict(list)

        self._resource_actor = None
        self._tensor_key_opid_to_tiled = defaultdict(list)
        self._tensor_key_to_opid = dict()
        self._terminal_chunk_op_tensor = defaultdict(set)
        self._terminated_tensors = set()
        self._operand_infos = dict()
        if target_tensors:
            self._target_tensor_chunk_ops = dict((k, set()) for k in target_tensors)
            self._target_tensor_finished = dict((k, set()) for k in self._target_tensor_chunk_ops)
        else:
            self._target_tensor_chunk_ops = dict()
            self._target_tensor_finished = dict()

    def post_create(self):
        super(GraphActor, self).post_create()
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        random.seed(int(time.time()))
        self.set_cluster_info_ref()
        self._assigner_actor_ref = self.ctx.actor_ref(AssignerActor.default_name())
        self._resource_actor_ref = self.get_actor_ref(ResourceActor.default_name())
        self._chunk_meta_ref = self.ctx.actor_ref(ChunkMetaActor.default_name())
        self._session_ref = self.ctx.actor_ref(SessionActor.gen_name(self._session_id))

        uid = GraphMetaActor.gen_name(self._session_id, self._graph_key)
        self._graph_meta_ref = self.ctx.create_actor(
            GraphMetaActor, self._session_id, self._graph_key,
            uid=uid, address=self.get_scheduler(uid))

        self._kv_store_ref = self.ctx.actor_ref(KVStoreActor.default_name())
        if not self.ctx.has_actor(self._kv_store_ref):
            self._kv_store_ref = None

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
        self._graph_meta_ref.set_state(value, _tell=True)

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

    @final_state.setter
    def final_state(self, value):
        self._final_state = value
        self._graph_meta_ref.set_final_state(value, _tell=True)

    @log_unhandled
    def execute_graph(self):
        """
        Start graph execution
        """
        def _detect_cancel(callback=None):
            if self.reload_state() == GraphState.CANCELLING:
                logger.info('Cancel detected, stopping')
                if callback:
                    callback()
                else:
                    self._end_time = time.time()
                    self.state = GraphState.CANCELLED
                raise ExecutionInterrupted

        self._start_time = time.time()
        self.state = GraphState.PREPARING

        try:
            self.prepare_graph()
            _detect_cancel()

            with self._open_dump_file('graph') as outf:  # pragma: no cover
                graph = self.get_chunk_graph()
                for n in graph:
                    outf.write(
                        '%s[%s] -> %s\n' % (
                            n.op.key, n.key,
                            ','.join(succ.op.key for succ in graph.iter_successors(n)))
                    )

            self.analyze_graph()
            _detect_cancel()

            self.create_operand_actors()
            _detect_cancel(self.stop_graph)
        except ExecutionInterrupted:
            pass
        except:  # noqa: E722
            logger.exception('Failed to start graph execution.')
            self.stop_graph()
            self.state = GraphState.FAILED
            raise

        if len(self._chunk_graph_cache) == 0:
            self.state = GraphState.SUCCEEDED

    @log_unhandled
    def stop_graph(self):
        """
        Stop graph execution
        """
        from .operand import OperandActor
        if self.state == GraphState.CANCELLED:
            return
        self.state = GraphState.CANCELLING

        try:
            chunk_graph = self.get_chunk_graph()
        except (KeyError, GraphNotExists):
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

    @log_unhandled
    def reload_chunk_graph(self):
        """
        Reload chunk graph from kv store
        """
        if self._kv_store_ref is not None:
            chunk_graph_ser = self._kv_store_ref.read('/sessions/%s/graphs/%s/chunk_graph'
                                                      % (self._session_id, self._graph_key)).value
        else:
            raise GraphNotExists
        self._chunk_graph_cache = deserialize_graph(chunk_graph_ser, graph_cls=DAG)

        op_key_to_chunk = defaultdict(list)
        for n in self._chunk_graph_cache:
            op_key_to_chunk[n.op.key].append(n)
        self._op_key_to_chunk = op_key_to_chunk

    @log_unhandled
    def get_chunk_graph(self):
        if self._chunk_graph_cache is None:
            self.reload_chunk_graph()
        return self._chunk_graph_cache

    @log_unhandled
    def prepare_graph(self, compose=True):
        """
        Tile and compose tensor graph into chunk graph
        :param compose: if True, do compose after tiling
        """
        tensor_graph = deserialize_graph(self._serialized_tensor_graph)
        self._tensor_graph_cache = tensor_graph

        logger.debug('Begin preparing graph %s with %d tensors to chunk graph.',
                     self._graph_key, len(tensor_graph))

        # mark target tensor steps
        if not self._target_tensor_chunk_ops:
            for tn in tensor_graph:
                if not tensor_graph.count_successors(tn):
                    self._target_tensor_chunk_ops[tn.key] = set()
                    self._target_tensor_finished[tn.key] = set()

        if self._serialized_chunk_graph:
            serialized_chunk_graph = self._serialized_chunk_graph
            chunk_graph = DAG.from_pb(serialized_chunk_graph)
        else:
            chunk_graph = DAG()

        key_to_chunk = {c.key: c for c in chunk_graph}

        tensor_key_opid_to_tiled = self._tensor_key_opid_to_tiled

        for t in tensor_graph:
            self._tensor_key_to_opid[t.key] = t.op.id
            if (t.key, t.op.id) not in tensor_key_opid_to_tiled:
                continue
            t._chunks = [key_to_chunk[k] for k in [tensor_key_opid_to_tiled[(t.key, t.op.id)][-1]]]

        tq = deque()
        for t in tensor_graph:
            if t.inputs and not all((ti.key, ti.op.id) in tensor_key_opid_to_tiled for ti in t.inputs):
                continue
            tq.append(t)

        while tq:
            tensor = tq.popleft()
            if not tensor.is_coarse() or (tensor.key, tensor.op.id) in tensor_key_opid_to_tiled:
                continue
            inputs = [tensor_key_opid_to_tiled[(it.key, it.op.id)][-1] for it in tensor.inputs or ()]

            op = tensor.op.copy()
            _ = op.new_tensors(inputs, [o.shape for o in tensor.op.outputs],  # noqa: F841
                               dtype=[o.dtype for o in tensor.op.outputs], **tensor.params)

            total_tiled = []
            for j, t, to_tile in zip(itertools.count(0), tensor.op.outputs, op.outputs):
                # replace inputs with tiled ones
                if not total_tiled:
                    try:
                        if isinstance(to_tile.op, TensorFetch):
                            td = self.tile_fetch_tensor(tensor)
                        else:
                            td = handler.dispatch(to_tile)
                    except DataNotReady:
                        continue

                    if isinstance(td, (tuple, list)):
                        total_tiled.extend(td)
                    else:
                        total_tiled.append(td)

                tiled = total_tiled[j]
                tensor_key_opid_to_tiled[(t.key, t.op.id)].append(tiled)

                # add chunks to fine grained graph
                q = deque([tiled_c if isinstance(tiled_c, ChunkData) else tiled_c.data for tiled_c in tiled.chunks])
                input_chunk_keys = set(itertools.chain(*([(it.key, it.id) for it in input.chunks]
                                                         for input in to_tile.inputs)))
                while len(q) > 0:
                    c = q.popleft()
                    if (c.key, c.id) in input_chunk_keys:
                        continue
                    if c not in chunk_graph:
                        chunk_graph.add_node(c)
                    for ic in c.inputs or []:
                        if ic not in chunk_graph:
                            chunk_graph.add_node(ic)
                            q.append(ic)
                        chunk_graph.add_edge(ic, c)

                for succ in tensor_graph.successors(t):
                    if any((t.key, t.op.id) not in tensor_key_opid_to_tiled for t in succ.inputs):
                        continue
                    tq.append(succ)

        # record the chunk nodes in graph
        reserve_chunk = set()
        result_chunk_keys = list()
        for tk, topid in tensor_key_opid_to_tiled:
            if tk not in self._target_tensor_chunk_ops:
                continue
            for n in [c.data for t in tensor_key_opid_to_tiled[(tk, topid)] for c in t.chunks]:
                result_chunk_keys.append(n.key)
                dq_predecessors = deque([n])
                while dq_predecessors:
                    current = dq_predecessors.popleft()
                    reserve_chunk.update(n.op.outputs)
                    predecessors = chunk_graph.predecessors(current)
                    dq_predecessors.extend([p for p in predecessors if p not in reserve_chunk])
                    reserve_chunk.update(predecessors)
        # delete redundant chunk
        for n in list(chunk_graph.iter_nodes()):
            if n not in reserve_chunk:
                chunk_graph.remove_node(n)
            elif isinstance(n.op, TensorFetch):
                chunk_graph.remove_node(n)

        if compose:
            chunk_graph.compose(keys=result_chunk_keys)

        for tk, topid in tensor_key_opid_to_tiled:
            if tk not in self._target_tensor_chunk_ops:
                continue
            for n in tensor_key_opid_to_tiled[(tk, topid)][-1].chunks:
                self._terminal_chunk_op_tensor[n.op.key].add(tk)
                self._target_tensor_chunk_ops[tk].add(n.op.key)

        # sync chunk graph to kv store
        if self._kv_store_ref is not None:
            graph_path = '/sessions/%s/graphs/%s' % (self._session_id, self._graph_key)
            self._kv_store_ref.write('%s/chunk_graph' % graph_path,
                                     serialize_graph(chunk_graph, compress=True), _tell=True, _wait=False)

        self._nodes_num = len(chunk_graph)
        self._chunk_graph_cache = chunk_graph
        for n in self._chunk_graph_cache:
            self._op_key_to_chunk[n.op.key].append(n)

    def _get_worker_slots(self):
        metrics = self._resource_actor_ref.get_workers_meta()
        return dict((ep, int(metrics[ep]['hardware']['cpu_total'])) for ep in metrics)

    def _collect_external_input_metas(self, ext_chunks_to_inputs):
        ext_chunk_keys = reduce(operator.add, ext_chunks_to_inputs.values(), [])
        metas = dict(zip(ext_chunk_keys,
                         self._chunk_meta_ref.batch_get_chunk_meta(self._session_id, ext_chunk_keys)))
        input_chunk_metas = defaultdict(dict)
        for chunk_key, input_chunk_keys in ext_chunks_to_inputs.items():
            chunk_metas = input_chunk_metas[chunk_key]
            for k in input_chunk_keys:
                chunk_metas[k] = metas[k]
        return input_chunk_metas

    @log_unhandled
    def analyze_graph(self, **kwargs):
        operand_infos = self._operand_infos
        chunk_graph = self.get_chunk_graph()

        if len(chunk_graph) == 0:
            return

        for n in chunk_graph:
            k = n.op.key
            succ_size = chunk_graph.count_successors(n)
            if k not in operand_infos:
                operand_infos[k] = dict(
                    optimize=dict(depth=0, demand_depths=(), successor_size=succ_size, descendant_size=0)
                )
            else:
                operand_infos[k]['optimize']['successor_size'] = succ_size

        analyzer = GraphAnalyzer(chunk_graph, self._get_worker_slots())

        for k, v in analyzer.calc_depths().items():
            operand_infos[k]['optimize']['depth'] = v

        for k, v in analyzer.calc_descendant_sizes().items():
            operand_infos[k]['optimize']['descendant_size'] = v

        if kwargs.get('do_placement', True):
            logger.debug('Placing initial chunks for graph %s', self._graph_key)

            # collect external inputs for eager mode
            ext_chunks_to_inputs = analyzer.collect_external_input_chunks(initial=True)
            input_chunk_metas = self._collect_external_input_metas(ext_chunks_to_inputs)
            # do placements
            for k, v in analyzer.calc_initial_assignments(input_chunk_metas=input_chunk_metas).items():
                operand_infos[k]['target_worker'] = v

    @log_unhandled
    def get_executable_operand_dag(self, op_key, serialize=True):
        """
        Make an operand into a worker-executable dag
        :param op_key: operand key
        :param serialize: whether to return serialized dag
        """
        graph = DAG()

        inputs_to_copied = dict()
        for c in self._op_key_to_chunk[op_key]:
            for inp in set(c.inputs or ()):
                op = TensorFetch(dtype=inp.dtype)
                inp_chunk = op.new_chunk(None, inp.shape, _key=inp.key).data
                inputs_to_copied[inp] = inp_chunk
                graph.add_node(inp_chunk)
            inputs = [inputs_to_copied[inp] for inp in (c.inputs or ())]

            new_op = c.op.copy()
            kws = []
            for o in c.op.outputs:
                kw = dict(_key=o.key, dtype=o.dtype, index=o.index)
                composed = []
                # copy composed
                for j, com in enumerate(o.composed or []):
                    new_com_op = com.op.copy()
                    if j == 0:
                        inps = inputs
                    else:
                        # if more than 1 inputs, means they are exactly the same object
                        inps = [composed[j - 1]] * len(com.inputs)
                    new_com = new_com_op.new_chunk(inps, com.shape, index=com.index,
                                                   dtype=com.dtype, _key=com.key)
                    composed.append(new_com)
                kw['_composed'] = composed
                kws.append(kw)

            new_outputs = new_op.new_chunks(inputs, [o.shape for o in c.op.outputs],
                                            kws=kws)
            for co in new_outputs:
                exec_chunk = co.data
                graph.add_node(exec_chunk)
                for inp in inputs:
                    graph.add_edge(inp, exec_chunk)
        if serialize:
            return serialize_graph(graph)
        else:
            return graph

    @log_unhandled
    def create_operand_actors(self, _clean_io_meta=True, _start=True):
        """
        Create operand actors for all operands
        """
        logger.debug('Creating operand actors for graph %s', self._graph_key)
        from .operand import OperandActor

        chunk_graph = self.get_chunk_graph()
        operand_infos = self._operand_infos

        op_refs = dict()
        initial_keys = []
        for op_key in self._op_key_to_chunk:
            op_name = type(self._op_key_to_chunk[op_key][0].op).__name__

            op_info = operand_infos[op_key]

            # collect operand i/o information
            predecessor_keys = set()
            successor_keys = set()
            input_chunk_keys = set()
            shared_input_chunk_keys = set()
            chunk_key_sizes = dict()

            for c in self._op_key_to_chunk[op_key]:
                for pn in chunk_graph.iter_predecessors(c):
                    predecessor_keys.add(pn.op.key)
                    input_chunk_keys.add(pn.key)
                    if chunk_graph.count_successors(pn) > 1:
                        shared_input_chunk_keys.add(pn.key)
                successor_keys.update(pn.op.key for pn in chunk_graph.iter_successors(c))
                chunk_key_sizes.update((co.key, co.rough_nbytes) for co in c.op.outputs)

            io_meta = dict(
                predecessors=list(predecessor_keys),
                successors=list(successor_keys),
                input_chunks=list(input_chunk_keys),
                shared_input_chunks=list(shared_input_chunk_keys),
                chunks=list(chunk_key_sizes.keys()),
            )
            op_info['op_name'] = op_name
            op_info['io_meta'] = io_meta
            output_size = sum(chunk_key_sizes.values())
            op_info['output_size'] = int(output_size)

            if predecessor_keys:
                state = 'UNSCHEDULED'
            else:
                initial_keys.append(op_key)
                state = 'READY'
            op_info['retries'] = 0
            op_info['state'] = state

            position = None
            if op_key in self._terminal_chunk_op_tensor:
                position = OperandPosition.TERMINAL
            elif not predecessor_keys:
                position = OperandPosition.INITIAL

            op_uid = OperandActor.gen_uid(self._session_id, op_key)
            scheduler_addr = self.get_scheduler(op_uid)
            op_refs[op_key] = self.ctx.create_actor(
                OperandActor, self._session_id, self._graph_key, op_key, op_info,
                position=position, uid=op_uid, address=scheduler_addr, wait=False
            )
            op_info['state'] = getattr(OperandState, state.upper())
            if _clean_io_meta:
                del op_info['io_meta']

        self.state = GraphState.RUNNING

        if _start:
            op_refs = dict((k, v.result()) for k, v in op_refs.items())
            start_futures = [op_refs[op_key].start_operand(_tell=True, _wait=False)
                             for op_key in initial_keys]
            [future.result() for future in start_futures]

    @log_unhandled
    def add_finished_terminal(self, op_key, final_state=None):
        """
        Add a terminal operand to finished set. Calling this method
        will change graph state if all terminals are in finished states.
        :param op_key: operand key
        :param final_state: state of the operand
        """
        tensor_keys = self._terminal_chunk_op_tensor[op_key]
        for tensor_key in tensor_keys:
            self._target_tensor_finished[tensor_key].add(op_key)
            if final_state == GraphState.FAILED:
                if self.final_state != GraphState.CANCELLED:
                    self.final_state = GraphState.FAILED
            elif final_state == GraphState.CANCELLED:
                self.final_state = final_state
            if self._target_tensor_finished[tensor_key] == self._target_tensor_chunk_ops[tensor_key]:
                self._terminated_tensors.add(tensor_key)
                if len(self._terminated_tensors) == len(self._target_tensor_chunk_ops):
                    self.state = self.final_state if self.final_state is not None else GraphState.SUCCEEDED
                    self._end_time = time.time()

    @log_unhandled
    def remove_finished_terminal(self, op_key):
        """
        Remove a terminal operand from finished set as the data is lost.
        :param op_key: operand key
        """
        tensor_keys = self._terminal_chunk_op_tensor[op_key]
        for tensor_key in tensor_keys:
            self._target_tensor_finished[tensor_key].difference_update([op_key])
            self._terminated_tensors.difference_update([tensor_key])

    def dump_unfinished_terminals(self):  # pragma: no cover
        """
        Dump unfinished terminal chunks into logger, only for debug purposes.
        """
        unfinished_dict = dict()
        for tensor_key, chunk_ops in self._target_tensor_chunk_ops.items():
            executed_ops = self._target_tensor_finished.get(tensor_key, ())
            unfinished = sorted(set(chunk_ops) - set(executed_ops))
            if unfinished:
                unfinished_dict[tensor_key] = unfinished
        logger.debug('Unfinished terminal chunks: %r', unfinished_dict)

    def get_state(self):
        return self.state

    def get_graph_info(self):
        return self._start_time, self._end_time, len(self._operand_infos)

    def get_operand_states(self, op_keys):
        return [self._operand_infos[k]['state'] for k in op_keys]

    def set_operand_state(self, op_key, state):
        op_info = self._operand_infos[op_key]
        op_info['state'] = OperandState(state)
        try:
            del op_info['failover_state']
        except KeyError:
            pass

    def get_operand_target_worker(self, op_key):
        return self._operand_infos[op_key]['target_worker']

    def get_operand_info(self):
        return self._operand_infos

    @log_unhandled
    def set_operand_worker(self, op_key, worker):
        if worker:
            self._operand_infos[op_key]['worker'] = worker
        else:
            try:
                del self._operand_infos[op_key]['worker']
            except KeyError:
                pass

    @functools32.lru_cache(1000)
    def _get_operand_ref(self, key):
        from .operand import OperandActor
        op_uid = OperandActor.gen_uid(self._session_id, key)
        scheduler_addr = self.get_scheduler(op_uid)
        return self.ctx.actor_ref(op_uid, address=scheduler_addr)

    @log_unhandled
    def calc_stats(self):
        states = list(OperandState.__members__.values())
        state_mapping = OrderedDict((v, idx) for idx, v in enumerate(states))
        state_names = [s.name for s in state_mapping]

        op_stats = OrderedDict()
        finished = 0
        total_count = len(self._operand_infos)
        for operand_info in self._operand_infos.values():
            op_name = operand_info['op_name']
            state = operand_info['state']
            if state in (OperandState.FINISHED, OperandState.FREED):
                finished += 1
            if op_name not in op_stats:
                op_stats[op_name] = [0] * len(state_mapping)
            stats_list = op_stats[op_name]
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

    def _get_tensor_by_key(self, key):
        tid = self._tensor_key_to_opid[key]
        return self._tensor_key_opid_to_tiled[(key, tid)][-1]

    @log_unhandled
    def free_tensor_data(self, tensor_key):
        tiled_tensor = self._get_tensor_by_key(tensor_key)
        for chunk in tiled_tensor.chunks:
            self._get_operand_ref(chunk.op.key).free_data(_tell=True)

    def get_tensor_chunk_indexes(self, tensor_key):
        return OrderedDict((c.key, c.index) for c in self._get_tensor_by_key(tensor_key).chunks)

    @log_unhandled
    def build_tensor_merge_graph(self, tensor_key):
        from ..tensor.expressions.merge.concatenate import TensorConcatenate

        tiled_tensor = self._get_tensor_by_key(tensor_key)
        graph = DAG()
        if len(tiled_tensor.chunks) == 1:
            # only one chunk, just trigger fetch
            c = tiled_tensor.chunks[0]
            op = TensorFetch(dtype=c.dtype)
            fetch_chunk = op.new_chunk(None, c.shape, index=c.index, _key=c.key).data
            graph.add_node(fetch_chunk)
        else:
            fetch_chunks = []
            for c in tiled_tensor.chunks:
                op = TensorFetch(dtype=c.dtype)
                fetch_chunk = op.new_chunk(None, c.shape, index=c.index, _key=c.key).data
                graph.add_node(fetch_chunk)
                fetch_chunks.append(fetch_chunk)
            chunk = TensorConcatenate(dtype=tiled_tensor.op.dtype).new_chunk(
                fetch_chunks, tiled_tensor.shape).data
            graph.add_node(chunk)
            [graph.add_edge(fetch_chunk, chunk) for fetch_chunk in fetch_chunks]

        return serialize_graph(graph)

    def build_fetch_graph(self, tensor_key):
        """
        Convert single tensor to tiled fetch tensor and put into a graph which only contains one tensor
        :param tensor_key: the key of tensor
        """
        tiled_tensor = self._get_tensor_by_key(tensor_key)
        graph = DAG()

        chunks = []
        for c in tiled_tensor.chunks:
            fetch_op = TensorFetch(dtype=c.dtype)
            fetch_chunk = fetch_op.new_chunk(None, c.shape, c.index, _key=c.key)
            chunks.append(fetch_chunk)

        new_op = TensorFetch(dtype=tiled_tensor.dtype)
        new_tensor = new_op.new_tensor(None, tiled_tensor.shape, chunks=chunks,
                                       nsplits=tiled_tensor.nsplits, _key=tiled_tensor.key)
        graph.add_node(new_tensor)
        return serialize_graph(graph)

    def tile_fetch_tensor(self, tensor):
        """
        Find the owner of the input tensor and ask for tiling.
        """
        tensor_key = tensor.key
        graph_ref = self.ctx.actor_ref(self._session_ref.get_graph_ref_by_tensor_key(tensor_key))
        fetch_graph = deserialize_graph(graph_ref.build_fetch_graph(tensor_key))
        return list(fetch_graph)[0]

    @log_unhandled
    def fetch_tensor_result(self, tensor_key):
        from ..worker.transfer import ResultSenderActor

        # TODO for test
        tiled_tensor = self._get_tensor_by_key(tensor_key)
        if tensor_key not in self._terminated_tensors:
            return None

        ctx = dict()
        for chunk_key in [c.key for c in tiled_tensor.chunks]:
            if chunk_key in ctx:
                continue
            endpoints = self._chunk_meta_ref.get_workers(self._session_id, chunk_key)
            sender_ref = self.ctx.actor_ref(ResultSenderActor.default_name(), address=endpoints[-1])
            ctx[chunk_key] = loads(sender_ref.fetch_data(self._session_id, chunk_key))
        return dumps(merge_tensor_chunks(tiled_tensor, ctx))

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
            self.ref().handle_worker_change(adds, removes, lost_chunks,
                                            handle_later=False, _delay=0.5, _tell=True)
            return
        else:
            self._operand_free_paused = False

        worker_slots = self._get_worker_slots()
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
        analyzer = GraphAnalyzer(graph, worker_slots, fixed_assigns, graph_states, lost_chunks)
        if removes or lost_chunks:
            new_states = analyzer.analyze_state_changes()
            logger.debug('%d chunks lost. %d operands changed state.', len(lost_chunks),
                         len(new_states))

        logger.debug('Start reallocating initial operands')
        # collect external inputs for eager mode
        ext_chunks_to_inputs = analyzer.collect_external_input_chunks(initial=True)
        input_chunk_metas = self._collect_external_input_metas(ext_chunks_to_inputs)
        new_targets = analyzer.calc_initial_assignments(input_chunk_metas=input_chunk_metas)

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
            if key in new_targets:
                new_target = operand_infos[key]['target_worker'] = new_targets[key]
            else:
                new_target = None

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
                outf.write(c + '\n')
            outf.write('\n\nOPERAND SNAPSHOT:\n')
            for key, op_info in self._operand_infos.items():
                outf.write('Chunk: %s Worker: %r State: %s\n' %
                           (key, op_info.get('worker'), op_info['state'].value))
            outf.write('\n\nSTATE TRANSITIONS:\n')
            for key, state in new_states.items():
                outf.write('%s -> %s\n' % (key, state.name))
