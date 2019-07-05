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

import itertools
import logging
import random
import time
import os
from collections import deque, defaultdict

from .assigner import AssignerActor
from .resource import ResourceActor
from .kvstore import KVStoreActor
from .utils import SchedulerActor, remove_shuffle_chunks, GraphState, OperandState
from ..compat import six, OrderedDict
from ..errors import ExecutionInterrupted, GraphNotExists
from ..graph import DAG
from ..tiles import handler, DataNotReady
from ..serialize import dataserializer
from ..utils import serialize_graph, deserialize_graph, merge_tensor_chunks
from ..actors.errors import ActorAlreadyExist

logger = logging.getLogger(__name__)


class ResultReceiverActor(SchedulerActor):
    def post_create(self):
        super(ResultReceiverActor, self).post_create()
        self.set_cluster_info_ref()

    def fetch_tensor(self, session_id, graph_key, tensor_key, compressions):
        from ..tensor.expressions.datasource import TensorFetchChunk
        from ..tensor.execution.core import Executor
        from ..worker.transfer import ResultSenderActor

        graph_actor = self.ctx.actor_ref(GraphActor.gen_uid(session_id, graph_key))
        fetch_graph = deserialize_graph(graph_actor.build_tensor_merge_graph(tensor_key))

        if len(fetch_graph) == 1 and isinstance(next(fetch_graph.iter_nodes()).op, TensorFetchChunk):
            c = next(fetch_graph.iter_nodes())
            worker_ip = self.chunk_meta.get_workers(session_id, c.key)[-1]
            sender_ref = self.ctx.actor_ref(ResultSenderActor.default_uid(), address=worker_ip)
            future = sender_ref.fetch_data(session_id, c.key, _wait=False)

            data = future.result()
            header = dataserializer.read_file_header(data)
            if header.compress in compressions:
                return data
            else:
                return dataserializer.dumps(dataserializer.loads(data), compress=max(compressions))
        else:
            ctx = dict()
            target_keys = set()
            for c in fetch_graph:
                if isinstance(c.op, TensorFetchChunk):
                    if c.key in ctx:
                        continue
                    endpoints = self.chunk_meta.get_workers(session_id, c.key)
                    sender_ref = self.ctx.actor_ref(ResultSenderActor.default_uid(), address=endpoints[-1])
                    future = sender_ref.fetch_data(session_id, c.key, _wait=False)
                    ctx[c.key] = future
                else:
                    target_keys.add(c.key)
            ctx = dict((k, dataserializer.loads(future.result())) for k, future in six.iteritems(ctx))
            executor = Executor(storage=ctx)
            concat_result = executor.execute_graph(fetch_graph, keys=target_keys)
            return dataserializer.dumps(concat_result[0], compress=max(compressions))


class GraphMetaActor(SchedulerActor):
    """
    Actor storing metadata of a graph
    """
    @staticmethod
    def gen_uid(session_id, graph_key):
        return 's:0:graph_meta$%s$%s' % (session_id, graph_key)

    def __init__(self, session_id, graph_key):
        super(GraphMetaActor, self).__init__()
        self._session_id = session_id
        self._graph_key = graph_key

        self._kv_store_ref = None

        self._state = None
        self._final_state = None

    def post_create(self):
        super(GraphMetaActor, self).post_create()
        self._kv_store_ref = self.ctx.actor_ref(KVStoreActor.default_uid())
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
    def gen_uid(session_id, graph_key):
        return 's:0:graph$%s$%s' % (session_id, graph_key)

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

        self._start_time = None
        self._end_time = None
        self._nodes_num = None

        self._cluster_info_ref = None
        self._assigner_actor_ref = None
        self._resource_actor_ref = None
        self._kv_store_ref = None
        self._graph_meta_ref = None

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
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        random.seed(int(time.time()))
        self.set_cluster_info_ref()
        self._assigner_actor_ref = self.get_actor_ref(AssignerActor.gen_uid(self._session_id))
        self._resource_actor_ref = self.get_actor_ref(ResourceActor.default_uid())

        uid = GraphMetaActor.gen_uid(self._session_id, self._graph_key)
        self._graph_meta_ref = self.ctx.create_actor(
            GraphMetaActor, self._session_id, self._graph_key,
            uid=uid, address=self.get_scheduler(uid))

        self._kv_store_ref = self.ctx.actor_ref(KVStoreActor.default_uid())
        if not self.ctx.has_actor(self._kv_store_ref):
            self._kv_store_ref = None

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

    def execute_graph(self, compose=True):
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
            self.prepare_graph(compose=compose)
            _detect_cancel()

            self.scan_node()
            _detect_cancel()

            self.place_initial_chunks()
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
            if self._operand_infos[chunk.op.key]['state'] in \
                    (OperandState.READY, OperandState.RUNNING, OperandState.FINISHED):
                # we only need to stop on ready, running and finished operands
                op_uid = OperandActor.gen_uid(self._session_id, chunk.op.key)
                scheduler_addr = self.get_scheduler(op_uid)
                ref = self.ctx.actor_ref(op_uid, address=scheduler_addr)
                has_stopping = True
                ref.stop_operand(_tell=True)
        if not has_stopping:
            self.state = GraphState.CANCELLED

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

    def get_chunk_graph(self):
        if self._chunk_graph_cache is None:
            self.reload_chunk_graph()
        return self._chunk_graph_cache

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
                q = deque([tiled_c.data for tiled_c in tiled.chunks])
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

    def scan_node(self):
        operand_infos = self._operand_infos
        chunk_graph = self.get_chunk_graph()
        depth_cache = dict()

        for n in chunk_graph.topological_iter():
            key = n.op.key
            if key not in operand_infos:
                operand_infos[key] = dict(
                    optimize=dict(depth=0, demand_depths=(), successor_size=0, descendant_size=0)
                )
            operand_infos[key]['optimize']['successor_size'] += chunk_graph.count_successors(n)
            if not n.inputs:
                depth_cache[key] = 0
            else:
                depth = depth_cache[key] = 1 + max(depth_cache[ni.op.key] for ni in n.inputs)
                # record operand information
                operand_infos[key]['optimize']['depth'] = depth

        for n in chunk_graph.topological_iter(reverse=True):
            operand_infos[n.op.key]['optimize']['descendant_size'] += 1
            if not n.inputs:
                continue
            for ni in set(n.inputs):
                operand_infos[ni.op.key]['optimize']['descendant_size'] += \
                    operand_infos[n.op.key]['optimize']['descendant_size']

    def place_initial_chunks(self):
        """
        Decide target worker for initial chunks
        """
        logger.debug('Placing initial chunks for graph %s', self._graph_key)

        import numpy as np
        graph = self.get_chunk_graph()
        undigraph = graph.build_undirected()
        operand_infos = self._operand_infos

        metrics = self._resource_actor_ref.get_workers_meta()

        def _successor_fun(n):
            return remove_shuffle_chunks(undigraph.iter_successors(n))

        key_to_chunks = defaultdict(list)
        for n in graph:
            key_to_chunks[n.op.key].append(n)

        # collect chunks with no inputs, or all inputs from shuffle
        zero_degree_key_chunks = dict()
        zero_degrees = []
        for chunk in graph:
            if not remove_shuffle_chunks(graph.predecessors(chunk)) and chunk.op.key not in zero_degree_key_chunks:
                zero_degree_key_chunks[chunk.op.key] = key_to_chunks[chunk.op.key]
        for op_key in zero_degree_key_chunks:
            zero_degrees.append(key_to_chunks[op_key][0])
        random.shuffle(zero_degrees)

        # sort initials by descendant size
        descendant_counts = dict()
        for key, chunks in zero_degree_key_chunks.items():
            descendant_counts[key] = operand_infos[key]['optimize']['descendant_size']
        # note that different sort orders can contribute to different efficiency
        zero_degrees = sorted(zero_degrees, key=lambda n: descendant_counts[n.op.key])

        endpoint_res = [(ep, int(metrics[ep]['hardware']['cpu_total'])) for ep in metrics]
        endpoint_res.sort(key=lambda t: t[1], reverse=True)

        endpoints = [t[0] for t in endpoint_res]
        endpoint_cores = np.array([t[1] for t in endpoint_res])

        splited_initial_counts = (endpoint_cores / endpoint_cores.sum() * len(zero_degrees)).astype(np.int64)
        if splited_initial_counts.sum() < len(zero_degrees):
            pos = 0
            rest = len(zero_degrees) - splited_initial_counts.sum()
            while rest > 0:
                splited_initial_counts[pos] += 1
                rest -= 1
                pos = (pos + 1) % len(splited_initial_counts)

        full_assigns = dict()
        sorted_initials = [v for v in zero_degrees]
        while splited_initial_counts.max():
            slot_id = splited_initial_counts.argmax()
            cur = sorted_initials.pop()
            while cur.op.key in full_assigns:
                cur = sorted_initials.pop()
            assigned = 0
            spread_range = 0
            for v in undigraph.bfs(start=key_to_chunks[cur.op.key], visit_predicate='all',
                                   successors=_successor_fun):
                if v.op.key in full_assigns:
                    continue
                spread_range += 1
                if graph.predecessors(v):
                    continue
                full_assigns[v.op.key] = endpoints[slot_id]
                assigned += 1
                if spread_range >= len(graph) * 1.0 / len(metrics) \
                        or assigned >= splited_initial_counts[slot_id]:
                    break
            splited_initial_counts[slot_id] -= assigned

        # assign initial workers by seeds
        for v in undigraph.bfs(start=zero_degrees, visit_predicate='all', successors=_successor_fun):
            if v.op.key in full_assigns:
                continue
            key_transfers = defaultdict(lambda: 0)
            for pred in undigraph.iter_predecessors(v):
                if pred.op.key in full_assigns:
                    key_transfers[full_assigns[pred.op.key]] += 1
            if not key_transfers:
                continue
            max_transfer = max(key_transfers.values())
            max_assigns = [assign for assign, cnt in key_transfers.items() if cnt == max_transfer]
            full_assigns[v.op.key] = max_assigns[random.randint(0, len(max_assigns) - 1)]

        for v in zero_degrees:
            operand_infos[v.op.key]['target_worker'] = full_assigns[v.op.key]

    def get_executable_operand_dag(self, op_key, serialize=True):
        """
        Make an operand into a worker-executable dag
        :param op_key: operand key
        :param serialize: whether to return serialized dag
        """
        from ..tensor.expressions.datasource import TensorFetchChunk
        graph = DAG()
        input_mapping = dict()
        output_keys = set()

        for c in self._op_key_to_chunk[op_key]:
            inputs = []
            for inp in set(c.op.inputs or ()):
                try:
                    inp_chunk = input_mapping[(inp.key, inp.id)]
                except KeyError:
                    op = TensorFetchChunk(dtype=inp.dtype, to_fetch_key=inp.key, sparse=inp.op.sparse)
                    inp_chunk = input_mapping[(inp.key, inp.id)] = \
                        op.new_chunk(None, inp.shape, _key=inp.key, _id=inp.id).data
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
        chunk_keys = set()

        for c in chunks:
            for pn in graph.iter_predecessors(c):
                predecessor_keys.add(pn.op.key)
                input_chunk_keys.add(pn.key)
                if graph.count_successors(pn) > 1:
                    shared_input_chunk_keys.add(pn.key)
            successor_keys.update(pn.op.key for pn in graph.iter_successors(c))
            chunk_keys.update(co.key for co in c.op.outputs)

        io_meta = dict(
            predecessors=list(predecessor_keys),
            successors=list(successor_keys),
            input_chunks=list(input_chunk_keys),
            shared_input_chunks=list(shared_input_chunk_keys),
            chunks=list(chunk_keys),
        )
        return io_meta

    def create_operand_actors(self, _clean_info=True):
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

            io_meta = self._collect_operand_io_meta(chunk_graph, self._op_key_to_chunk[op_key])
            op_info['op_name'] = op_name
            op_info['io_meta'] = io_meta
            op_info['executable_dag'] = self.get_executable_operand_dag(op_key)

            if io_meta['predecessors']:
                state = OperandState.UNSCHEDULED
            else:
                initial_keys.append(op_key)
                state = OperandState.READY
            op_info['retries'] = 0
            op_info['state'] = state

            op_uid = OperandActor.gen_uid(self._session_id, op_key)
            scheduler_addr = self.get_scheduler(op_uid)
            op_refs[op_key] = self.ctx.create_actor(
                OperandActor, self._session_id, self._graph_key, op_key, op_info.copy(),
                is_terminal=op_key in self._terminal_chunk_op_tensor,
                uid=op_uid, address=scheduler_addr,
                wait=False
            )
            if _clean_info:
                op_info.pop('executable_dag', None)
                del op_info['io_meta']

        self.state = GraphState.RUNNING

        existing_keys = []
        for op_key, future in op_refs.items():
            try:
                op_refs[op_key] = future.result()
            except ActorAlreadyExist:
                existing_keys.append(op_key)

        append_futures = []
        for op_key in existing_keys:
            op_info = operand_infos[op_key]
            op_info['io_meta'] = self._collect_operand_io_meta(chunk_graph, self._op_key_to_chunk[op_key])
            op_uid = OperandActor.gen_uid(self._session_id, op_key)
            scheduler_addr = self.get_scheduler(op_uid)
            op_ref = op_refs[op_key] = self.ctx.actor_ref(op_uid, address=scheduler_addr)
            is_terminal = op_key in self._terminal_chunk_op_tensor
            append_futures.append(op_ref.append_graph(self._graph_key, op_info.copy(),
                                                      is_terminal=is_terminal, _wait=False))
            if _clean_info:
                op_info.pop('executable_dag', None)
                del op_info['io_meta']
        [future.result() for future in append_futures]

        start_futures = [ref.start_operand(_tell=True, _wait=False) for ref in op_refs.values()]
        [future.result() for future in start_futures]

    def mark_terminal_finished(self, op_key, final_state=None):
        """
        Mark terminal operand as finished. Calling this method will change graph state
        if all terminals are in finished states.
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

    def get_state(self):
        return self.state

    def get_graph_info(self):
        return self._start_time, self._end_time, len(self._operand_infos)

    def set_operand_state(self, op_key, state):
        self._operand_infos[op_key]['state'] = OperandState(state)

    def get_operand_info(self):
        return self._operand_infos

    def calc_stats(self):
        states = list(OperandState.__members__.values())
        state_mapping = OrderedDict((v, idx) for idx, v in enumerate(states))
        state_names = [s.name for s in state_mapping]

        op_stats = OrderedDict()
        finished = 0
        total_count = len(self._operand_infos)
        for operand_info in six.itervalues(self._operand_infos):
            op_name = operand_info['op_name']
            state = operand_info['state']
            if state in (OperandState.FINISHED, OperandState.FREED):
                finished += 1
            if op_name not in op_stats:
                op_stats[op_name] = [0] * len(state_mapping)
            stats_list = op_stats[op_name]
            stats_list[state_mapping[state]] += 1

        data_src = OrderedDict([('states', state_names), ])
        for op, state_stats in six.iteritems(op_stats):
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
        return ops, transposed, finished * 100.0 / total_count

    def _get_tensor_by_key(self, key):
        tid = self._tensor_key_to_opid[key]
        return self._tensor_key_opid_to_tiled[(key, tid)][-1]

    def free_tensor_data(self, tensor_key):
        from .operand import OperandActor
        tiled_tensor = self._get_tensor_by_key(tensor_key)
        for chunk in tiled_tensor.chunks:
            op_uid = OperandActor.gen_uid(self._session_id, chunk.op.key)
            scheduler_addr = self.get_scheduler(op_uid)
            op_ref = self.ctx.actor_ref(op_uid, address=scheduler_addr)
            op_ref.free_data(_tell=True)

    def get_tensor_chunk_indexes(self, tensor_key):
        return OrderedDict((c.key, c.index) for c in self._get_tensor_by_key(tensor_key).chunks)

    def build_tensor_merge_graph(self, tensor_key):
        from ..tensor.expressions.merge.concatenate import TensorConcatenate
        from ..tensor.expressions.datasource import TensorFetchChunk

        tiled_tensor = self._get_tensor_by_key(tensor_key)
        graph = DAG()
        if len(tiled_tensor.chunks) == 1:
            # only one chunk, just trigger fetch
            c = tiled_tensor.chunks[0]
            op = TensorFetchChunk(dtype=c.dtype, to_fetch_key=c.key, sparse=c.op.sparse)
            fetch_chunk = op.new_chunk(None, c.shape, index=c.index, _key=c.key).data
            graph.add_node(fetch_chunk)
        else:
            fetch_chunks = []
            for c in tiled_tensor.chunks:
                op = TensorFetchChunk(dtype=c.dtype, to_fetch_key=c.key, sparse=c.op.sparse)
                fetch_chunk = op.new_chunk(None, c.shape, index=c.index, _key=c.key).data
                graph.add_node(fetch_chunk)
                fetch_chunks.append(fetch_chunk)
            chunk = TensorConcatenate(dtype=tiled_tensor.op.dtype).new_chunk(
                fetch_chunks, tiled_tensor.shape).data
            graph.add_node(chunk)
            [graph.add_edge(fetch_chunk, chunk) for fetch_chunk in fetch_chunks]

        return serialize_graph(graph)

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
            endpoints = self.chunk_meta.get_workers(self._session_id, chunk_key)
            sender_ref = self.ctx.actor_ref(ResultSenderActor.default_uid(), address=endpoints[-1])
            ctx[chunk_key] = dataserializer.loads(sender_ref.fetch_data(self._session_id, chunk_key))
        return dataserializer.dumps(merge_tensor_chunks(tiled_tensor, ctx))
