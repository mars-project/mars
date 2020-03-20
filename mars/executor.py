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

import datetime
import itertools
import logging
import sys
import threading
import weakref
import operator
import contextlib
from collections import deque, defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from numbers import Integral

import numpy as np
try:
    from numpy.core._exceptions import UFuncTypeError
except ImportError:  # pragma: no cover
    UFuncTypeError = None

try:
    import gevent
except ImportError:  # pragma: no cover
    gevent = None

from .operands import Fetch, ShuffleProxy
from .graph import DirectedGraph
from .config import options
from .tiles import IterativeChunkGraphBuilder, ChunkGraphBuilder, get_tiled
from .optimizes.runtime.optimizers.core import Optimizer
from .optimizes.tileable_graph import tileable_optimized, OptimizeIntegratedTileableGraphBuilder
from .graph_builder import TileableGraphBuilder
from .context import LocalContext
from .utils import kernel_mode, enter_build_mode, build_fetch, calc_nsplits,\
    has_unknown_shape

if gevent:
    from .actors.pool.gevent_pool import GeventThreadPool

logger = logging.getLogger(__name__)


class ExecutorSyncProvider(object):
    @classmethod
    def thread_pool_executor(cls, n_workers):
        raise NotImplementedError

    @classmethod
    def semaphore(cls, value):
        raise NotImplementedError

    @classmethod
    def lock(cls):
        raise NotImplementedError

    @classmethod
    def rlock(cls):
        raise NotImplementedError

    @classmethod
    def event(cls):
        raise NotImplementedError


class ThreadExecutorSyncProvider(ExecutorSyncProvider):
    @classmethod
    def thread_pool_executor(cls, n_workers):
        return ThreadPoolExecutor(n_workers)

    @classmethod
    def semaphore(cls, value):
        return threading.Semaphore(value)

    @classmethod
    def lock(cls):
        return threading.Lock()

    @classmethod
    def rlock(cls):
        return threading.RLock()

    @classmethod
    def event(cls):
        return threading.Event()


class GeventExecutorSyncProvider(ExecutorSyncProvider):
    @classmethod
    def thread_pool_executor(cls, n_workers):
        return GeventThreadPool(n_workers)

    @classmethod
    def semaphore(cls, value):
        # as gevent threadpool is the **real** thread, so use threading.Semaphore
        return threading.Semaphore(value)

    @classmethod
    def lock(cls):
        # as gevent threadpool is the **real** thread, so use threading.Lock
        return threading.Lock()

    @classmethod
    def rlock(cls):
        # as gevent threadpool is the **real** thread, so use threading.RLock
        return threading.RLock()

    @classmethod
    def event(cls):
        # as gevent threadpool is the **real** thread, so use threading.Event
        return threading.Event()


class MockThreadPoolExecutor(object):
    class _MockResult(object):
        def __init__(self, result=None, exc_info=None):
            self._result = result
            self._exc_info = exc_info

        def result(self, *_):
            if self._exc_info is not None:
                raise self._exc_info[1] from None
            else:
                return self._result

        def exception_info(self, *_):
            return self._exc_info

    def __init__(self, *_):
        pass

    def submit(self, fn, *args, **kwargs):
        try:
            return self._MockResult(fn(*args, **kwargs))
        except:  # noqa: E722
            return self._MockResult(None, sys.exc_info())


class MockExecutorSyncProvider(ThreadExecutorSyncProvider):
    @classmethod
    def thread_pool_executor(cls, n_workers):
        return MockThreadPoolExecutor(n_workers)


class GraphDeviceAssigner(object):
    # Analyze graph and assign initial chunks to different GPU devices
    # only work when execute on GPU
    def __init__(self, graph, starts, devices):
        self._graph = graph
        self._undigraph = None
        self._op_keys = {start.key for start in starts}
        self._devices = devices
        self._device_slots = {dev: 1 for dev in devices}

    def _calc_device_assign_limits(self, initial_count, occupied=None):
        """
        Calculate limitation of number of initial operands for devices
        :param initial_count: num of nodes in READY state
        :param occupied: device -> num of initials already assigned
        """
        occupied = occupied or dict()
        actual_count = initial_count - sum(occupied.values())

        device_res = sorted(self._device_slots.items(), key=operator.itemgetter(1),
                            reverse=True)

        devices = [t[0] for t in device_res]
        device_cores = np.array([t[1] for t in device_res]).astype(np.float32)

        # remove assigned nodes from limitations
        counts = initial_count * device_cores / device_cores.sum()
        for idx, dev in enumerate(devices):
            counts[idx] = max(0, counts[idx] - occupied.get(dev, 0))

        # all assigned, nothing to do
        if counts.sum() == 0:
            return dict((dev, 0) for dev in devices)

        counts = (actual_count * counts / counts.sum()).astype(np.int32)

        # assign remaining nodes
        pos = 0
        rest = actual_count - counts.sum()
        while rest > 0:
            counts[pos] += 1
            rest -= 1
            pos = (pos + 1) % len(counts)
        return dict(zip(devices, counts))

    def _assign_by_bfs(self, start, device, initial_sizes, spread_limits,
                       keys_to_assign, assigned_record, graph=None):
        """
        Assign initial nodes using Breadth-first Search given initial sizes and
        limitations of spread range.
        """
        if initial_sizes[device] <= 0:
            return

        graph = graph or self._graph
        if self._undigraph is None:
            undigraph = self._undigraph = graph.build_undirected()
        else:
            undigraph = self._undigraph

        assigned = 0
        spread_range = 0
        for v in undigraph.bfs(start=start, visit_predicate='all'):
            op_key = v.op.key
            if op_key in assigned_record:
                continue
            spread_range += 1
            if op_key not in keys_to_assign:
                continue
            assigned_record[op_key] = device
            assigned += 1
            if spread_range >= spread_limits[device] \
                    or assigned >= initial_sizes[device]:
                break
        initial_sizes[device] -= assigned

    def assign(self):
        """
        Decide target device for given chunks.

        :return: dict mapping operand keys into device
        """
        graph = self._graph
        cur_assigns = OrderedDict()

        op_key_to_chunks = defaultdict(list)
        for n in graph:
            op_key_to_chunks[n.op.key].append(n)

        descendant_readies = set()
        op_keys = set(self._op_keys)
        chunks_to_assign = [op_key_to_chunks[k][0] for k in op_keys]
        assigned_counts = defaultdict(lambda: 0)

        # calculate the number of nodes to be assigned to each device
        # given number of devices and existing assignments
        device_quotas = self._calc_device_assign_limits(
            len(chunks_to_assign) + len(descendant_readies), assigned_counts)

        # calculate expected descendant count (spread range) of
        # every device and subtract assigned number from it
        average_spread_range = len(graph) * 1.0 / len(self._device_slots)
        spread_ranges = defaultdict(lambda: average_spread_range)
        # assign from other nodes to be assigned
        sorted_candidates = [v for v in chunks_to_assign]
        while max(device_quotas.values()):
            device = max(device_quotas, key=lambda k: device_quotas[k])
            cur = sorted_candidates.pop()
            while cur.op.key in cur_assigns:
                cur = sorted_candidates.pop()
            self._assign_by_bfs(cur, device, device_quotas, spread_ranges, op_keys,
                                cur_assigns, graph=graph)

        keys_to_assign = set(n.op.key for n in chunks_to_assign)
        for k, v in cur_assigns.items():
            if k in keys_to_assign:
                for chunk in op_key_to_chunks[k]:
                    chunk.op._device = v


class GraphExecution(object):
    """
    Represent an execution for a specified graph.
    """

    def __init__(self, chunk_results, graph, keys, executed_keys, sync_provider,
                 n_parallel=None, engine=None, prefetch=False, print_progress=False,
                 mock=False, mock_max_memory=0, fetch_keys=None, no_intermediate=False):
        self._chunk_results = chunk_results
        self._graph = graph
        self._keys = keys
        self._key_set = set(keys).union(executed_keys)
        self._n_parallel = n_parallel or 1
        self._engine = engine
        self._prefetch = prefetch
        self._print_progress = print_progress
        self._mock = mock
        self._mock_max_memory = mock_max_memory
        self._no_intermediate = no_intermediate
        self._fetch_keys = fetch_keys or set()

        # pool executor for the operand execution
        self._operand_executor = sync_provider.thread_pool_executor(self._n_parallel)
        # pool executor for prefetching
        if prefetch:
            self._prefetch_executor = sync_provider.thread_pool_executor(self._n_parallel)
        else:
            self._prefetch_executor = None
        # global lock
        self._lock = sync_provider.lock()
        # control the concurrence
        self._semaphore = sync_provider.semaphore(self._n_parallel)
        # event for setting error happened
        self._has_error = sync_provider.event()
        self._queue = list(self._order_starts()) if len(graph) > 0 else []
        assert len(self._queue) == graph.count_indep()
        self._chunk_key_ref_counts = self._calc_ref_counts()
        self._op_key_to_ops = self._calc_op_key_to_ops()
        self._submitted_op_keys = set()
        self._executed_op_keys = set()
        # initial assignment for GPU
        self._assign_devices()

    def _order_starts(self):
        visited = set()
        starts = deque(self._graph.iter_indep())
        stack = deque([starts.popleft()])

        while stack:
            node = stack.popleft()
            if node not in visited:
                preds = self._graph.predecessors(node)
                if not preds or all(pred in visited for pred in preds):
                    if len(preds) == 0:
                        yield node.op
                    visited.add(node)
                    stack.extend(n for n in self._graph[node] if n not in visited)
                else:
                    stack.appendleft(node)
                    stack.extendleft(reversed(list(n for n in self._graph.predecessors(node)
                                                   if n not in visited)))
            if not stack and starts:
                stack.appendleft(starts.popleft())

    def _assign_devices(self):
        if self._n_parallel <= 1 or self._engine != 'cupy':
            return

        devices = list(range(self._n_parallel))
        assigner = GraphDeviceAssigner(self._graph, self._queue, devices)
        assigner.assign()

    def _calc_ref_counts(self):
        ref_counts = dict()

        for chunk in self._graph:
            for dep_key in chunk.op.get_dependent_data_keys():
                if dep_key in self._key_set:
                    # only record ref count for those not in results
                    continue
                ref_counts[dep_key] = ref_counts.get(dep_key, 0) + 1

        return ref_counts

    def _calc_op_key_to_ops(self):
        op_key_to_ops = defaultdict(set)

        for chunk in self._graph:
            # operand
            op_key_to_ops[chunk.op.key].add(chunk.op)

        return op_key_to_ops

    def _execute_operand(self, op):
        results = self._chunk_results
        ref_counts = self._chunk_key_ref_counts
        op_keys = self._executed_op_keys
        executed_chunk_keys = set()
        deleted_chunk_keys = set()
        try:
            ops = list(self._op_key_to_ops[op.key])
            # note that currently execution is the chunk-level
            # so we pass the first operand's first output to Executor.handle
            first_op = ops[0]
            Executor.handle(first_op, results, self._mock)

            # update maximal memory usage during execution
            if self._mock:
                output_keys = set(o.key for o in first_op.outputs or ())

                cur_memory = sum(results[op_output.key][1] for op_output in first_op.outputs
                                 if results.get(op_output.key) is not None)
                if not self._no_intermediate:
                    cur_memory += sum(tp[0] for key, tp in results.items()
                                      if key not in self._fetch_keys and key not in output_keys
                                      and isinstance(tp, tuple))
                self._mock_max_memory = max(cur_memory, self._mock_max_memory)

            executed_chunk_keys.update([c.key for c in first_op.outputs])
            op_keys.add(first_op.key)
            # handle other operands
            for rest_op in ops[1:]:
                for op_output, rest_op_output in zip(first_op.outputs, rest_op.outputs):
                    # if the op's outputs have been stored,
                    # other same key ops' results will be the same
                    if rest_op_output.key not in executed_chunk_keys:
                        results[rest_op_output.key] = results[op_output.key]

            with self._lock:
                for output in itertools.chain(*[op.outputs for op in ops]):
                    # the output not in the graph will be skipped
                    if output not in self._graph:
                        continue
                    # in case that operand has multiple outputs
                    # and some of the output not in result keys, delete them
                    if ref_counts.get(output.key) == 0:
                        # if the result has been deleted, it should be skipped
                        if output.key not in deleted_chunk_keys:
                            deleted_chunk_keys.add(output.key)
                            del results[output.key]

                    # clean the predecessors' results if ref counts equals 0
                    for dep_key in output.op.get_dependent_data_keys():
                        if dep_key in ref_counts:
                            ref_counts[dep_key] -= 1
                            if ref_counts[dep_key] == 0:
                                del results[dep_key]
                                del ref_counts[dep_key]

                    # add successors' operands to queue
                    for succ_chunk in self._graph.iter_successors(output):
                        preds = self._graph.predecessors(succ_chunk)
                        if succ_chunk.op.key not in self._submitted_op_keys and \
                                (len(preds) == 0 or all(pred.op.key in op_keys for pred in preds)):
                            self._queue.insert(0, succ_chunk.op)
        except Exception:
            self._has_error.set()
            raise
        finally:
            self._semaphore.release()

    def _fetch_chunks(self, chunks):
        """
        Iterate all the successors of given chunks,
        if the successor's predecessors except that in the chunks have all finished,
        we will try to load the successor's all predecessors into memory in advance.
        """

        for chunk in chunks:
            with self._lock:
                to_fetch_chunk = None
                for succ_chunk in self._graph.iter_successors(chunk):
                    accepted = True
                    for pred_chunk in self._graph.iter_predecessors(succ_chunk):
                        if pred_chunk is chunk:
                            continue
                        if pred_chunk.op.key not in self._executed_op_keys:
                            accepted = False
                            break
                    if accepted:
                        to_fetch_chunk = succ_chunk
                        break
                if to_fetch_chunk is None and len(self._queue) > 0:
                    to_fetch_chunk = self._queue[0]
                for pred_chunk in self._graph.iter_predecessors(to_fetch_chunk):
                    # if predecessor is spilled
                    # the get will pull it back into memory
                    self._chunk_results.get(pred_chunk.key)

    def _submit_operand_to_execute(self):
        self._semaphore.acquire()

        with self._lock:
            try:
                to_submit_op = self._queue.pop(0)
                if to_submit_op.key in self._submitted_op_keys:
                    raise ValueError('Get submitted operand')
                self._submitted_op_keys.add(to_submit_op.key)
            except (IndexError, ValueError):
                self._semaphore.release()
                return

        if self._print_progress:
            i, n = len(self._submitted_op_keys), len(self._op_key_to_ops)
            if i % 30 or i >= n:
                logger.info('[{0}] {1:.2f}% percent of graph has been submitted'.format(
                    str(datetime.datetime.now()), float(i) * 100 / n))

        if self._prefetch:
            # check the operand's outputs if any of its successor's predecessors can be prefetched
            self._prefetch_executor.submit(self._fetch_chunks, to_submit_op.outputs)
        # execute the operand and return future
        return self._operand_executor.submit(self._execute_operand, to_submit_op)

    def execute(self, retval=True):
        executed_futures = []
        while len(self._submitted_op_keys) < len(self._op_key_to_ops):
            if self._has_error.is_set():
                # something wrong happened
                break

            future = self._submit_operand_to_execute()
            if future is not None:
                executed_futures.append(future)

        # wait until all the futures completed
        for future in executed_futures:
            future.result()

        if retval:
            return [self._chunk_results[key] for key in self._keys]


class Executor(object):
    _op_runners = {}
    _op_size_estimators = {}
    _graph_execution_cls = GraphExecution

    class SyncProviderType(Enum):
        THREAD = 0
        GEVENT = 1
        MOCK = 2

    _sync_provider = {
        SyncProviderType.MOCK: MockExecutorSyncProvider,
        SyncProviderType.THREAD: ThreadExecutorSyncProvider,
        SyncProviderType.GEVENT: GeventExecutorSyncProvider,
    }

    def __init__(self, engine=None, storage=None, prefetch=False,
                 sync_provider_type=SyncProviderType.THREAD):
        self._engine = engine
        self._chunk_result = storage if storage is not None else dict()
        self._prefetch = prefetch

        # only record the executed tileable
        self.tileable_to_tiled = weakref.WeakKeyDictionary()
        # dict structure: {tileable_key -> chunk_keys, tileable_ids}
        # dict value is a tuple object which records chunk keys and tileable id
        self.stored_tileables = dict()
        # executed key to ref counts
        self.key_to_ref_counts = defaultdict(lambda: 0)
        # synchronous provider
        self._sync_provider = self._sync_provider[sync_provider_type]

        self._mock_max_memory = 0

    @property
    def chunk_result(self):
        return self._chunk_result

    @property
    def mock_max_memory(self):
        return self._mock_max_memory

    @classmethod
    def handle(cls, op, results, mock=False):
        method_name, mapper = ('execute', cls._op_runners) if not mock else \
            ('estimate_size', cls._op_size_estimators)
        try:
            runner = mapper[type(op)]
        except KeyError:
            runner = getattr(op, method_name)
        try:
            if UFuncTypeError is None:
                return runner(results, op)
            else:
                # Cast `UFuncTypeError` to `TypeError` since subclasses of the former is unpickleable.
                # The `UFuncTypeError` was introduced by numpy#12593 since v1.17.0.
                try:
                    return runner(results, op)
                except UFuncTypeError as e:
                    raise TypeError(str(e)).with_traceback(sys.exc_info()[2]) from None
        except NotImplementedError:
            for op_cls in mapper.keys():
                if isinstance(op, op_cls):
                    mapper[type(op)] = mapper[op_cls]
                    runner = mapper[op_cls]
                    return runner(results, op)
            raise KeyError('No handler found for op: %s' % op)

    def execute_graph(self, graph, keys, n_parallel=None, print_progress=False,
                      mock=False, no_intermediate=False, compose=True, retval=True,
                      chunk_result=None):
        """
        :param graph: graph to execute
        :param keys: result keys
        :param n_parallel: num of max parallelism
        :param print_progress:
        :param compose: if True. fuse nodes when possible
        :param mock: if True, only estimate data sizes without execution
        :param no_intermediate: exclude intermediate data sizes when estimating memory size
        :param retval: if True, keys specified in argument keys is returned
        :param chunk_result: dict to put chunk key to chunk data, if None, use self.chunk_result
        :return: execution result
        """
        if compose:
            Optimizer(graph, self._engine).optimize(keys=keys)
        optimized_graph = graph

        if not mock:
            # fetch_keys only useful when calculating sizes
            fetch_keys = set()
        else:
            fetch_keys = set(v.key for v in graph if isinstance(v.op, Fetch))
            for c in graph:
                if graph.count_predecessors(c) != 0:
                    continue
                fetch_keys.update(inp.key for inp in c.inputs or ())

        executed_keys = list(itertools.chain(*[v[1] for v in self.stored_tileables.values()]))
        chunk_result = self._chunk_result if chunk_result is None else chunk_result
        graph_execution = self._graph_execution_cls(
            chunk_result, optimized_graph, keys, executed_keys, self._sync_provider,
            n_parallel=n_parallel, engine=self._engine, prefetch=self._prefetch,
            print_progress=print_progress, mock=mock, mock_max_memory=self._mock_max_memory,
            fetch_keys=fetch_keys, no_intermediate=no_intermediate)
        res = graph_execution.execute(retval)
        self._mock_max_memory = max(self._mock_max_memory, graph_execution._mock_max_memory)
        if mock:
            chunk_result.clear()
        return res

    @kernel_mode
    @enter_build_mode
    def execute_tileable(self, tileable, n_parallel=None, n_thread=None, concat=False,
                         print_progress=False, mock=False, compose=True):
        result_keys = []
        tileable_data = tileable.data if hasattr(tileable, 'data') else tileable

        def _on_tile_success(before_tile_data, after_tile_data):
            if before_tile_data is tileable_data:
                if concat and len(after_tile_data.chunks) > 1:
                    after_tile_data = after_tile_data.op.concat_tileable_chunks(after_tile_data)
                result_keys.extend(c.key for c in after_tile_data.chunks)

            return after_tile_data

        # shallow copy
        chunk_result = self._chunk_result.copy()
        tileable_graph_builder = TileableGraphBuilder()
        tileable_graph = tileable_graph_builder.build([tileable])
        chunk_graph_builder = ChunkGraphBuilder(graph_cls=DirectedGraph, compose=compose,
                                                on_tile_success=_on_tile_success)
        chunk_graph = chunk_graph_builder.build([tileable], tileable_graph=tileable_graph)
        ret = self.execute_graph(chunk_graph, result_keys, n_parallel=n_parallel or n_thread,
                                 print_progress=print_progress, mock=mock,
                                 chunk_result=chunk_result)
        self._chunk_result.update(chunk_result)
        return ret

    execute_tensor = execute_tileable
    execute_dataframe = execute_tileable

    def _update_tileable_and_chunk_shape(self, tileable_graph, chunk_result, failed_ops):
        for n in tileable_graph:
            if n.op in failed_ops:
                continue
            tiled_n = get_tiled(n)
            if has_unknown_shape(tiled_n):
                if any(c.key not in chunk_result for c in tiled_n.chunks):
                    # some of the chunks has been fused
                    continue
                for c in tiled_n.chunks:
                    c.data._shape = chunk_result[c.key].shape
                new_nsplits = self.get_tileable_nsplits(n, chunk_result=chunk_result)
                for node in (n, tiled_n):
                    node._update_shape(tuple(sum(nsplit) for nsplit in new_nsplits))
                tiled_n._nsplits = new_nsplits

    @contextlib.contextmanager
    def _gen_local_context(self, chunk_result):
        if isinstance(chunk_result, LocalContext):
            with chunk_result:
                yield chunk_result
        else:
            yield chunk_result

    @kernel_mode
    @enter_build_mode
    def execute_tileables(self, tileables, fetch=True, n_parallel=None, n_thread=None,
                          print_progress=False, mock=False, compose=True):
        # shallow copy chunk_result, prevent from any chunk key decref
        chunk_result = self._chunk_result.copy()
        tileables = [tileable.data if hasattr(tileable, 'data') else tileable
                     for tileable in tileables]
        tileable_keys = [t.key for t in tileables]
        tileable_keys_set = set(tileable_keys)

        result_keys = []
        to_release_keys = set()
        tileable_data_to_concat_keys = weakref.WeakKeyDictionary()
        tileable_data_to_chunk_keys = weakref.WeakKeyDictionary()

        executed_keys = set(chunk_result)
        node_to_fetch = weakref.WeakKeyDictionary()

        def _generate_fetch_if_executed(nd):
            # node processor that if the node is executed
            # replace it with a fetch node
            _keys, _to_fetch = executed_keys, node_to_fetch  # noqa: F821
            if nd.key not in _keys:
                return nd
            if nd in _to_fetch:
                return _to_fetch[nd]
            fn = build_fetch(nd).data
            _to_fetch[nd] = fn
            return fn

        def _on_tile_success(before_tile_data, after_tile_data):
            if before_tile_data.key not in tileable_keys_set:
                return after_tile_data
            tile_chunk_keys = [c.key for c in after_tile_data.chunks]
            result_keys.extend(tile_chunk_keys)
            tileable_data_to_chunk_keys[before_tile_data] = tile_chunk_keys
            if not fetch:
                pass
            elif len(after_tile_data.chunks) > 1:
                # need to fetch data and chunks more than 1, we concatenate them into 1
                after_tile_data = after_tile_data.op.concat_tileable_chunks(after_tile_data)
                chunk = after_tile_data.chunks[0]
                result_keys.append(chunk.key)
                tileable_data_to_concat_keys[before_tile_data] = chunk.key
                # after return the data to user, we release the reference
                to_release_keys.add(chunk.key)
            else:
                tileable_data_to_concat_keys[before_tile_data] = after_tile_data.chunks[0].key
            return after_tile_data

        def _get_tileable_graph_builder(**kwargs):
            if options.optimize_tileable_graph:
                return OptimizeIntegratedTileableGraphBuilder(**kwargs)
            else:
                return TileableGraphBuilder(**kwargs)

        # As the chunk_result is copied, we cannot use the original context any more,
        # and if `chunk_result` is a LocalContext, it's copied into a LocalContext as well,
        # thus here just to make sure the new context is entered
        with self._gen_local_context(chunk_result):
            # build tileable graph
            tileable_graph_builder = _get_tileable_graph_builder()
            tileable_graph = tileable_graph_builder.build(tileables)
            chunk_graph_builder = IterativeChunkGraphBuilder(
                graph_cls=DirectedGraph, node_processor=_generate_fetch_if_executed,
                compose=compose, on_tile_success=_on_tile_success)
            intermediate_result_keys = set()
            while True:
                # build chunk graph, tile will be done during building
                chunk_graph = chunk_graph_builder.build(
                    tileables, tileable_graph=tileable_graph)
                tileable_graph = chunk_graph_builder.prev_tileable_graph
                temp_result_keys = set(result_keys)
                if not chunk_graph_builder.done:
                    # add temporary chunks keys into result keys
                    for interrupted_op in chunk_graph_builder.interrupted_ops:
                        for inp in interrupted_op.inputs:
                            if inp.op not in chunk_graph_builder.interrupted_ops:
                                for n in get_tiled(inp).chunks:
                                    temp_result_keys.add(n.key)
                # execute chunk graph
                self.execute_graph(chunk_graph, list(temp_result_keys),
                                   n_parallel=n_parallel or n_thread,
                                   print_progress=print_progress, mock=mock,
                                   chunk_result=chunk_result)
                # update shape of tileable and its chunks whatever it's successful or not
                self._update_tileable_and_chunk_shape(
                    tileable_graph, chunk_result, chunk_graph_builder.interrupted_ops)
                if chunk_graph_builder.done:
                    if len(intermediate_result_keys) > 0:
                        # failed before
                        intermediate_to_release_keys = \
                            {k for k in intermediate_result_keys
                             if k not in result_keys and k in chunk_result}
                        to_release_keys.update(intermediate_to_release_keys)
                    delattr(chunk_graph_builder, '_prev_tileable_graph')
                    break
                else:
                    executed_keys.update(temp_result_keys)
                    intermediate_result_keys.update(temp_result_keys)
                    # add the node that failed
                    to_run_tileables = list(itertools.chain(
                        *(op.outputs for op in chunk_graph_builder.interrupted_ops)))
                    to_run_tileables_set = set(to_run_tileables)
                    for op in chunk_graph_builder.interrupted_ops:
                        for inp in op.inputs:
                            if inp not in to_run_tileables_set:
                                to_run_tileables_set.add(inp)
                    tileable_graph_builder = _get_tileable_graph_builder(
                        inputs_selector=lambda inps: [inp for inp in inps
                                                      if inp in to_run_tileables_set])
                    tileable_graph = tileable_graph_builder.build(to_run_tileables_set)

            for tileable in tileables:
                if tileable.key in self.stored_tileables:
                    self.stored_tileables[tileable.key][0].add(tileable.id)
                else:
                    chunk_keys = tileable_data_to_chunk_keys[tileable_optimized.get(tileable, tileable)]
                    self.stored_tileables[tileable.key] = tuple([{tileable.id}, set(chunk_keys)])
            try:
                if fetch:
                    concat_keys = [
                        tileable_data_to_concat_keys[tileable_optimized.get(t, t)] for t in tileables]
                    return [chunk_result[k] for k in concat_keys]
                else:
                    return
            finally:
                for to_release_key in to_release_keys:
                    del chunk_result[to_release_key]
                self._chunk_result.update(
                    {k: chunk_result[k] for k in result_keys if k in chunk_result})

    execute_tensors = execute_tileables
    execute_dataframes = execute_tileables

    @classmethod
    def _check_slice_on_tileable(cls, tileable):
        from .tensor.indexing import TensorIndex
        from .dataframe.indexing.iloc import DataFrameIlocGetItem

        if isinstance(tileable.op, (TensorIndex, DataFrameIlocGetItem)):
            indexes = tileable.op.indexes
            if not all(isinstance(ind, (slice, Integral)) for ind in indexes):
                raise ValueError('Only support fetch data slices')

    @kernel_mode
    def fetch_tileables(self, tileables, **kw):
        from .tensor.indexing import TensorIndex
        from .dataframe.indexing.iloc import DataFrameIlocGetItem

        to_release_tileables = []
        for tileable in tileables:
            if tileable.key not in self.stored_tileables and \
                    isinstance(tileable.op, (TensorIndex, DataFrameIlocGetItem)):
                key = tileable.inputs[0].key
                to_release_tileables.append(tileable)
            else:
                key = tileable.key
            if key not in self.stored_tileables:
                # check if the tileable is executed before
                raise ValueError(
                    'Tileable object {} to fetch must be executed first'.format(tileable))

        try:
            # if chunk executed, fetch chunk mechanism will be triggered in execute_tileables
            return self.execute_tileables(tileables, **kw)
        finally:
            for to_release_tileable in to_release_tileables:
                for c in get_tiled(to_release_tileable).chunks:
                    del self._chunk_result[c.key]

    def get_tileable_nsplits(self, tileable, chunk_result=None):
        chunk_idx_to_shape = OrderedDict()
        tiled = get_tiled(tileable, mapping=tileable_optimized)
        chunk_result = chunk_result if chunk_result is not None else self._chunk_result
        for chunk in tiled.chunks:
            chunk_idx_to_shape[chunk.index] = chunk_result[chunk.key].shape
        return calc_nsplits(chunk_idx_to_shape)

    def decref(self, *keys):
        rs = set(self._chunk_result)
        for key in keys:
            tileable_key, tileable_id = key
            if key[0] not in self.stored_tileables:
                continue
            ids, chunk_keys = self.stored_tileables[key[0]]
            if tileable_id in ids:
                ids.remove(tileable_id)
                # for those same key tileables, do decref only when all those tileables are garbage collected
                if len(ids) != 0:
                    continue
                for chunk_key in (chunk_keys & rs):
                    self._chunk_result.pop(chunk_key, None)
                del self.stored_tileables[tileable_key]


def ignore(*_):
    pass


Executor._op_runners[Fetch] = ignore
Executor._op_runners[ShuffleProxy] = ignore


def register(op_cls, handler=None, size_estimator=None):
    if handler:
        Executor._op_runners[op_cls] = handler
    if size_estimator:
        Executor._op_size_estimators[op_cls] = size_estimator


def register_default(op_cls):
    Executor._op_runners.pop(op_cls, None)
    Executor._op_size_estimators.pop(op_cls, None)


# import to register operands
from . import tensor
from . import dataframe
from . import optimizes
from . import learn

del tensor, dataframe, optimizes, learn
