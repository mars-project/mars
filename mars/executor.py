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
import threading
import weakref
import itertools
import logging
from collections import deque, defaultdict

import numpy as np
try:
    import gevent
except ImportError:  # pragma: no cover
    gevent = None

from .operands import Fetch
from .graph import DirectedGraph
from .compat import futures, OrderedDict, enum
from .utils import kernel_mode

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


class GraphExecution(object):
    """
    Represent an execution for a specified graph.
    """
    def __init__(self, chunk_results, graph, keys, executed_keys, sync_provider,
                 n_parallel=None, prefetch=False, print_progress=False,
                 mock=False, sparse_mock_percent=1.0):
        self._chunk_results = chunk_results
        self._graph = graph
        self._keys = keys
        self._key_set = set(keys).union(executed_keys)
        self._n_parallel = n_parallel or 1
        self._prefetch = prefetch
        self._print_progress = print_progress
        self._mock = mock
        self._sparse_mock_percent = sparse_mock_percent

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
            if chunk.key not in self._key_set:
                # only record ref count for those not in results
                ref_counts[chunk.key] = \
                    ref_counts.get(chunk.key, 0) + len(self._graph[chunk])


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
        try:
            ops = list(self._op_key_to_ops[op.key])
            if not self._mock:
                # do real execution
                # note that currently execution is the chunk-level
                # so we pass the first operand's first output to Executor.handle
                first_op = ops[0]
                Executor.handle(first_op.outputs[0], results)
                op_keys.add(first_op.key)
                # handle other operands
                for rest_op in ops[1:]:
                    for op_output, rest_op_output in zip(first_op.outputs, rest_op.outputs):
                        results[rest_op_output.key] = results[op_output.key]
            else:
                sparse_percent = self._sparse_mock_percent if op.sparse else 1.0
                for output in op.outputs:
                    results[output.key] = output.nbytes * sparse_percent

            with self._lock:
                for output in itertools.chain(*[op.outputs for op in ops]):
                    # in case that operand has multiple outputs
                    # and some of the output not in result keys, delete them
                    if ref_counts.get(output.key) == 0:
                        del results[output.key]

                    # clean the predecessors' results if ref counts equals 0
                    for pred_chunk in self._graph.iter_predecessors(output):
                        if pred_chunk.key in ref_counts:
                            ref_counts[pred_chunk.key] -= 1
                            if ref_counts[pred_chunk.key] == 0:
                                del results[pred_chunk.key]

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
        maximum_usage = 0
        while len(self._submitted_op_keys) < len(self._op_key_to_ops):
            if self._has_error.is_set():
                # something wrong happened
                break
            if self._mock:
                # if mock, the value in chunk_results is the data size
                # just adding up to get the maximum memory usage
                curr_usage = np.sum(list(self._chunk_results.values()))
                maximum_usage = max(maximum_usage, curr_usage)

            future = self._submit_operand_to_execute()
            if future is not None:
                executed_futures.append(future)

        # wait until all the futures completed
        for future in executed_futures:
            future.result()

        if self._mock:
            return maximum_usage
        if retval:
            return [self._chunk_results[key] for key in self._keys]


class Executor(object):
    _op_runners = {}

    class SyncProviderType(enum.Enum):
        THREAD = 0
        GEVENT = 1

    _sync_provider = {
        SyncProviderType.THREAD: ThreadExecutorSyncProvider,
        SyncProviderType.GEVENT: GeventExecutorSyncProvider,
    }

    def __init__(self, engine=None, storage=None, prefetch=False,
                 sync_provider_type=SyncProviderType.THREAD):
        self._engine = engine
        self._chunk_result = storage if storage is not None else dict()
        self._prefetch = prefetch

        # only record the executed tensor
        self.tensor_to_tiled = weakref.WeakKeyDictionary()
        # dict structure: {tensor_key -> chunk_keys, tensor_ids}
        # dict value is a tuple object which records chunk keys and tensor id
        self.stored_tensors = dict()
        # executed key to ref counts
        self.key_to_ref_counts = defaultdict(lambda: 0)
        # synchronous provider
        self._sync_provider = self._sync_provider[sync_provider_type]

    @property
    def chunk_result(self):
        return self._chunk_result

    def _preprocess(self, graph, keys):
        # TODO(xuye.qin): make an universal optimzier
        from .tensor.execution.optimizes.core import Optimizer

        Optimizer(graph, self._engine).optimize(keys=keys)
        return graph

    @classmethod
    def handle(cls, chunk, results):
        try:
            op_cls = type(chunk.op)

            return cls._op_runners[op_cls](results, chunk)
        except KeyError:
            for op_cls in cls._op_runners.keys():
                if isinstance(chunk.op, op_cls):
                    cls._op_runners[type(chunk.op)] = cls._op_runners[op_cls]
                    return cls._op_runners[op_cls](results, chunk)

            raise KeyError('No handler found for op: %s' % chunk.op)

    def execute_graph(self, graph, keys, n_parallel=None, print_progress=False,
                      mock=False, sparse_mock_percent=1.0):
        optimized_graph = self._preprocess(graph, keys)

        executed_keys = list(itertools.chain(*[v[1] for v in self.stored_tensors.values()]))
        graph_execution = GraphExecution(self._chunk_result, optimized_graph,
                                         keys, executed_keys, self._sync_provider,
                                         n_parallel=n_parallel, prefetch=self._prefetch,
                                         print_progress=print_progress, mock=mock,
                                         sparse_mock_percent=sparse_mock_percent)
        return graph_execution.execute(True)

    @kernel_mode
    def execute_tensor(self, tensor, n_parallel=None, n_thread=None, concat=False,
                       print_progress=False, mock=False, sparse_mock_percent=1.0):
        if concat:
            # only for tests
            tensor.tiles()
            if len(tensor.chunks) > 1:
                from .tensor.expressions.merge.concatenate import TensorConcatenate

                op = TensorConcatenate(dtype=tensor.op.dtype)
                chunk = TensorConcatenate(dtype=op.dtype).new_chunk(tensor.chunks, tensor.shape)
                tensor = op.new_tensor([tensor], tensor.shape, chunks=[chunk],
                                       nsplits=[(s,) for s in tensor.shape])

        graph = tensor.build_graph(cls=DirectedGraph, tiled=True)

        return self.execute_graph(graph, [c.key for c in tensor.chunks],
                                  n_parallel=n_parallel or n_thread,
                                  print_progress=print_progress, mock=mock,
                                  sparse_mock_percent=sparse_mock_percent)

    @kernel_mode
    def execute_tensors(self, tensors, fetch=True, n_parallel=None, n_thread=None,
                        print_progress=False, mock=False, sparse_mock_percent=1.0):
        graph = DirectedGraph()

        result_keys = []
        to_release_keys = []
        concat_keys = []
        for tensor in tensors:
            tensor.tiles()
            chunk_keys = [c.key for c in tensor.chunks]
            result_keys.extend(chunk_keys)

            if tensor.key in self.stored_tensors:
                self.stored_tensors[tensor.key][0].add(tensor.id)
            else:
                self.stored_tensors[tensor.key] = tuple([{tensor.id}, set(chunk_keys)])
            if not fetch:
                # no need to generate concat keys
                pass
            elif len(tensor.chunks) > 1:
                from .tensor.expressions.merge.concatenate import TensorConcatenate

                # if need to fetch data and chunks more than 1, we concatenate them into 1
                op = TensorConcatenate(dtype=tensor.op.dtype)
                chunk = TensorConcatenate(dtype=op.dtype).new_chunk(tensor.chunks, tensor.shape)
                result_keys.append(chunk.key)
                # the concatenated key
                concat_keys.append(chunk.key)
                # after return the data to user, we release the reference
                to_release_keys.append(chunk.key)
                tensor = op.new_tensor([tensor], tensor.shape, chunks=[chunk],
                                       nsplits=[(s,) for s in tensor.shape])
            else:
                concat_keys.append(tensor.chunks[0].key)

            tensor.build_graph(graph=graph, tiled=True, executed_keys=list(self._chunk_result.keys()))

        self.execute_graph(graph, result_keys, n_parallel=n_parallel or n_thread,
                           print_progress=print_progress, mock=mock,
                           sparse_mock_percent=sparse_mock_percent)

        results = self._chunk_result
        try:
            if fetch:
                return [results[k] for k in concat_keys]
            else:
                return
        finally:
            for k in to_release_keys:
                del results[k]

    @kernel_mode
    def fetch_tensors(self, tensors, **kw):
        from .tensor.expressions.datasource import TensorFetch

        results = []
        to_concat_tensors = OrderedDict()

        for i, tensor in enumerate(tensors):
            if tensor.key not in self.stored_tensors:
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
                op = TensorFetch(dtype=c.dtype)
                chunk = op.new_chunk(None, c.shape, index=c.index, _key=c.key)
                chunks.append(chunk)

            new_op = TensorFetch(dtype=tensor.dtype)
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
            if key[0] not in self.stored_tensors:
                continue
            ids, chunk_keys = self.stored_tensors[key[0]]
            if tensor_id in ids:
                ids.remove(tensor_id)
                # for those same key tensors, do decref only when all those tensors are garbage collected
                if len(ids) != 0:
                    continue
                for chunk_key in chunk_keys:
                    if chunk_key in self.chunk_result:
                        del self.chunk_result[chunk_key]
                del self.stored_tensors[tensor_key]


def ignore(*_):
    pass


Executor._op_runners[Fetch] = ignore


def register(op, handler):
    Executor._op_runners[op] = handler


# register tensor and dataframe execution handler
from .tensor.execution.core import register_tensor_execution_handler
register_tensor_execution_handler()
del register_tensor_execution_handler
from.dataframe.execution.core import register_dataframe_execution_handler
register_dataframe_execution_handler()
del register_dataframe_execution_handler
