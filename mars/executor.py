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

import datetime
import itertools
import functools
import logging
import sys
import threading
import weakref
from collections import deque, defaultdict

import numpy as np
try:
    import gevent
except ImportError:  # pragma: no cover
    gevent = None

from .operands import Fetch
from .graph import DirectedGraph
from .compat import six, futures, OrderedDict, enum
from .utils import kernel_mode, calc_data_size , concat_tileable_chunks

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
        return futures.ThreadPoolExecutor(n_workers)

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


if gevent:
    import gevent.threadpool
    import gevent.event

    class GeventThreadPoolExecutor(gevent.threadpool.ThreadPoolExecutor):
        @staticmethod
        def _wrap_watch(fn):
            # Each time a function is submitted, a gevent greenlet may be created,
            # this is common especially for Mars actor,
            # but there would be no other greenlet to switch to,
            # LoopExit will be raised, in order to prevent from this,
            # we create a greenlet to watch the result

            def check(event):
                delay = 0.0005
                while not event.is_set():
                    event.wait(delay)
                    delay = min(delay * 2, .05)

            def inner(*args, **kwargs):
                event = gevent.event.Event()
                gevent.spawn(check, event)
                result = fn(*args, **kwargs)
                event.set()
                return result

            return inner

        def submit(self, fn, *args, **kwargs):
            wrapped_fn = self._wrap_watch(fn)
            return super(GeventThreadPoolExecutor, self).submit(wrapped_fn, *args, **kwargs)


class GeventExecutorSyncProvider(ExecutorSyncProvider):
    @classmethod
    def thread_pool_executor(cls, n_workers):
        return GeventThreadPoolExecutor(n_workers)

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
                six.reraise(*self._exc_info)
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


class GraphExecution(object):
    """
    Represent an execution for a specified graph.
    """
    def __init__(self, chunk_results, graph, keys, executed_keys, sync_provider,
                 n_parallel=None, prefetch=False, print_progress=False,
                 mock=False, mock_max_memory=0):
        self._chunk_results = chunk_results
        self._graph = graph
        self._keys = keys
        self._key_set = set(keys).union(executed_keys)
        self._n_parallel = n_parallel or 1
        self._prefetch = prefetch
        self._print_progress = print_progress
        self._mock = mock
        self._mock_max_memory = mock_max_memory

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
            Executor.handle(first_op.outputs[0], results, self._mock)

            # update maximal memory usage during execution
            if self._mock:
                self._mock_max_memory = max(
                    self._mock_max_memory,
                    sum(results[op_output.key][1] for op_output in first_op.outputs
                        if results.get(op_output.key) is not None))

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

        # update with the maximal memory cost during the whole execution
        if self._mock:
            avg_max_mem = self._mock_max_memory // len(self._keys)
            for key in self._keys:
                r = self._chunk_results[key]
                self._chunk_results[key] = (r[0], max(r[1], avg_max_mem))

        if retval:
            return [self._chunk_results[key] for key in self._keys]


class Executor(object):
    _op_runners = {}
    _op_size_estimators = {}

    class SyncProviderType(enum.Enum):
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

    def _preprocess(self, graph, keys):
        # TODO(xuye.qin): make an universal optimzier
        from .tensor.execution.optimizes.core import Optimizer

        Optimizer(graph, self._engine).optimize(keys=keys)
        return graph

    @staticmethod
    def _get_op_runner(chunk, mapper):
        try:
            op_cls = type(chunk.op)
            return mapper[op_cls]
        except KeyError:
            for op_cls in mapper.keys():
                if isinstance(chunk.op, op_cls):
                    mapper[type(chunk.op)] = mapper[op_cls]
                    return mapper[op_cls]

            raise KeyError('No handler found for op: %s' % chunk.op)

    @classmethod
    def handle(cls, chunk, results, mock=False):
        if not mock:
            return cls._get_op_runner(chunk, cls._op_runners)(results, chunk)
        else:
            return cls._get_op_runner(chunk, cls._op_size_estimators)(results, chunk)

    def execute_graph(self, graph, keys, n_parallel=None, print_progress=False,
                      mock=False, retval=True):
        optimized_graph = self._preprocess(graph, keys)

        executed_keys = list(itertools.chain(*[v[1] for v in self.stored_tileables.values()]))
        graph_execution = GraphExecution(self._chunk_result, optimized_graph,
                                         keys, executed_keys, self._sync_provider,
                                         n_parallel=n_parallel, prefetch=self._prefetch,
                                         print_progress=print_progress, mock=mock,
                                         mock_max_memory=self._mock_max_memory)
        res = graph_execution.execute(retval)
        self._mock_max_memory = max(self._mock_max_memory, graph_execution._mock_max_memory)
        if mock:
            self._chunk_result.clear()
        return res

    @kernel_mode
    def execute_tileable(self, tileable, n_parallel=None, n_thread=None, concat=False,
                         print_progress=False, mock=False, compose=True):
        if concat:
            # only for tests
            tileable.tiles()
            if len(tileable.chunks) > 1:
                tileable = concat_tileable_chunks(tileable)

        graph = tileable.build_graph(cls=DirectedGraph, tiled=True, compose=compose)

        return self.execute_graph(graph, [c.key for c in tileable.chunks],
                                  n_parallel=n_parallel or n_thread,
                                  print_progress=print_progress, mock=mock)

    execute_tensor = execute_tileable
    execute_dataframe = execute_tileable

    @kernel_mode
    def execute_tileables(self, tileables, fetch=True, n_parallel=None, n_thread=None,
                          print_progress=False, mock=False, compose=True):
        graph = DirectedGraph()

        result_keys = []
        to_release_keys = []
        concat_keys = []
        for tileable in tileables:
            tileable.tiles()
            chunk_keys = [c.key for c in tileable.chunks]
            result_keys.extend(chunk_keys)

            if tileable.key in self.stored_tileables:
                self.stored_tileables[tileable.key][0].add(tileable.id)
            else:
                self.stored_tileables[tileable.key] = tuple([{tileable.id}, set(chunk_keys)])
            if not fetch:
                # no need to generate concat keys
                pass
            elif len(tileable.chunks) > 1:
                # if need to fetch data and chunks more than 1, we concatenate them into 1
                tileable = concat_tileable_chunks(tileable)
                chunk = tileable.chunks[0]
                result_keys.append(chunk.key)
                # the concatenated key
                concat_keys.append(chunk.key)
                # after return the data to user, we release the reference
                to_release_keys.append(chunk.key)
            else:
                concat_keys.append(tileable.chunks[0].key)

            tileable.build_graph(graph=graph, tiled=True, compose=compose,
                                 executed_keys=list(self._chunk_result.keys()))

        self.execute_graph(graph, result_keys, n_parallel=n_parallel or n_thread,
                           print_progress=print_progress, mock=mock)

        results = self._chunk_result
        try:
            if fetch:
                return [results[k] for k in concat_keys]
            else:
                return
        finally:
            for k in to_release_keys:
                del results[k]

    execute_tensors = execute_tileables
    execute_dataframes = execute_tileables

    @kernel_mode
    def fetch_tensors(self, tensors, **kw):
        from .tensor.expressions.fetch import TensorFetch

        results = []
        to_concat_tensors = OrderedDict()

        for i, tensor in enumerate(tensors):
            if tensor.key not in self.stored_tileables:
                # check if the tensor is executed before
                raise ValueError(
                    'Tensor to fetch must be executed before, got {0}'.format(tensor))

            if len(tensor.chunks) == 1:
                result = self._chunk_result[tensor.chunks[0].key]
                results.append(result)
                continue

            # generate TensorFetch op for each chunk
            chunks = []
            for c in tensor.chunks:
                op = TensorFetch(dtype=c.dtype, sparse=c.op.sparse)
                chunk = op.new_chunk(None, shape=c.shape, index=c.index, _key=c.key)
                chunks.append(chunk)

            new_op = TensorFetch(dtype=tensor.dtype, sparse=tensor.op.sparse)
            # copy key and id to ensure that fetch tensor won't add the count of executed tensor
            tensor = new_op.new_tensor(None, tensor.shape, chunks=chunks,
                                       nsplits=tensor.nsplits, _key=tensor.key, _id=tensor.id)

            # add this concat tensor into the list which shall be executed later
            to_concat_tensors[i] = tensor
            results.append(None)

        # execute the concat tensors together
        if to_concat_tensors:
            concat_results = self.execute_tensors(list(to_concat_tensors.values()), **kw)
            for j, concat_result in zip(to_concat_tensors, concat_results):
                results[j] = concat_result

        return results

    def get_tensor_nsplits(self, tensor):
        chunk_indexes = [c.index for c in tensor.chunks]
        chunk_shapes = [r.shape for r in [self._chunk_result[c.key] for c in tensor.chunks]]

        ndim = len(chunk_shapes[0])
        tensor_nsplits = []
        for i in range(ndim):
            splits = []
            for index, shape in zip(chunk_indexes, chunk_shapes):
                if all(idx == 0 for j, idx in enumerate(index) if j != i):
                    splits.append(shape[i])
            tensor_nsplits.append(tuple(splits))

        return tuple(tensor_nsplits)

    def decref(self, *keys):
        for key in keys:
            tensor_key, tensor_id = key
            if key[0] not in self.stored_tileables:
                continue
            ids, chunk_keys = self.stored_tileables[key[0]]
            if tensor_id in ids:
                ids.remove(tensor_id)
                # for those same key tensors, do decref only when all those tensors are garbage collected
                if len(ids) != 0:
                    continue
                for chunk_key in chunk_keys:
                    if chunk_key in self.chunk_result:
                        del self.chunk_result[chunk_key]
                del self.stored_tileables[tensor_key]


def ignore(*_):
    pass


def default_size_estimator(ctx, chunk, multiplier=1):
    exec_size = 0
    for inp in chunk.inputs or ():
        try:
            exec_size += ctx[inp.key][0]
        except KeyError:
            if not chunk.is_sparse():
                inp_size = calc_data_size(inp)
                if not np.isnan(inp_size):
                    exec_size += inp_size
    exec_size = int(exec_size * multiplier)

    total_out_size = 0
    chunk_sizes = dict()
    outputs = chunk.op.outputs
    for out in outputs:
        try:
            chunk_size = calc_data_size(out) if not out.is_sparse() else exec_size
            if np.isnan(chunk_size):
                raise TypeError
            chunk_sizes[out.key] = chunk_size
            total_out_size += chunk_size
        except (AttributeError, TypeError, ValueError):
            pass
    exec_size = max(exec_size, total_out_size)
    for out in outputs:
        if out.key in ctx:
            continue
        if out.key in chunk_sizes:
            store_size = chunk_sizes[out.key]
        else:
            store_size = max(exec_size // len(outputs),
                             total_out_size // max(len(chunk_sizes), 1))
        try:
            if out.is_sparse():
                max_sparse_size = out.nbytes + np.dtype(np.int64).itemsize * np.prod(out.shape) * out.ndim
            else:
                max_sparse_size = np.nan
        except TypeError:  # pragma: no cover
            max_sparse_size = np.nan
        if not np.isnan(max_sparse_size):
            store_size = min(store_size, max_sparse_size)
        ctx[out.key] = (store_size, exec_size // len(outputs))


def size_estimator_wrapper(ctx, chunk, original_estimator=None):
    try:
        return original_estimator(ctx, chunk)
    except NotImplementedError:
        return default_size_estimator(ctx, chunk)


Executor._op_runners[Fetch] = ignore


def register(op, handler, size_estimator=None, size_multiplier=1):
    Executor._op_runners[op] = handler
    if size_estimator:
        Executor._op_size_estimators[op] = \
            functools.partial(size_estimator_wrapper, original_estimator=size_estimator)
    else:
        Executor._op_size_estimators[op] = size_estimator or \
            functools.partial(default_size_estimator, multiplier=size_multiplier)


# register tensor and dataframe execution handler
from .tensor.execution.core import register_tensor_execution_handler
register_tensor_execution_handler()
del register_tensor_execution_handler
from.dataframe.execution.core import register_dataframe_execution_handler
register_dataframe_execution_handler()
del register_dataframe_execution_handler
