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
from collections import deque, defaultdict

import numpy as np

from .operands import Fetch
from .graph import DirectedGraph
from .compat import futures, OrderedDict
from .core import build_mode
from .utils import kernel_mode


class Executor(object):
    _op_runners = {}

    def __init__(self, engine=None, storage=None, prefetch=False):
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

    @property
    def chunk_result(self):
        return self._chunk_result

    def _preprocess(self, graph, keys):
        # TODO(xuye.qin): make an universal optimzier
        from .tensor.execution.optimizes.core import Optimizer

        Optimizer(graph, self._engine).optimize(keys=keys)
        return graph

    def handle(self, chunk, results):
        try:
            cls = type(chunk.op)

            return self._op_runners[cls](results, chunk)
        except KeyError:
            for cls in self._op_runners.keys():
                if isinstance(chunk.op, cls):
                    self._op_runners[type(chunk.op)] = self._op_runners[cls]
                    return self._op_runners[cls](results, chunk)

            raise KeyError('No handler found for op: %s' % chunk.op)

    def execute_graph(self, graph, keys, n_parallel=None, n_thread=None, show_progress=False,
                      mock=False, sparse_mock_percent=1.0):
        with build_mode():
            optimized_graph = self._preprocess(graph, keys)

        executed_keys = list(itertools.chain(*[v[1] for v in self.stored_tensors.values()]))
        return execute_graph(optimized_graph, keys, self, executed_keys=executed_keys,
                             n_parallel=n_parallel or n_thread,
                             show_progress=show_progress, mock=mock,
                             sparse_mock_percent=sparse_mock_percent,
                             prefetch=self._prefetch, retval=True)

    @kernel_mode
    def execute_tensor(self, tensor, n_parallel=None, n_thread=None, concat=False,
                       show_progress=False, mock=False, sparse_mock_percent=1.0):
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
                                  show_progress=show_progress, mock=mock,
                                  sparse_mock_percent=sparse_mock_percent)

    @kernel_mode
    def execute_tensors(self, tensors, fetch=True, n_parallel=None, n_thread=None,
                        show_progress=False, mock=False, sparse_mock_percent=1.0):
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
                           show_progress=show_progress, mock=mock,
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


def execute_chunk(chunk, executor=None,
                  ref_counts=None, chunk_result=None,
                  finishes=None, visited=None, q=None,
                  lock=None, semaphore=None, has_error=None,
                  preds=None, succs=None, mock=False, sparse_mock_percent=1.0):
    try:
        with lock:
            if (chunk.key, chunk.id) in visited:
                return
            visited.add((chunk.key, chunk.id))
            finished = finishes.get(chunk.key)
        if not finished:
            if not mock:
                # do real execution
                if chunk.key not in chunk_result:
                    executor.handle(chunk, chunk_result)
            else:
                percent = sparse_mock_percent if chunk.op.sparse else 1.0
                # we put the estimated size of data into the chunk_result
                chunk_result[chunk.key] = np.prod(chunk.shape) * chunk.dtype.itemsize * percent
            with lock:
                for output in chunk.op.outputs:
                    finishes[output.key] = True
                    if output.key in ref_counts and ref_counts[output.key] == 0 and \
                            output.key in chunk_result:
                        # some op have more than 1 outputs,
                        # and some of the outputs are not in the result ones
                        del chunk_result[output.key]

        for pred_key in preds[chunk.key]:
            with lock:
                if pred_key not in ref_counts:
                    continue
                ref_counts[pred_key] -= 1
                if ref_counts[pred_key] == 0:
                    del chunk_result[pred_key]

        for succ in succs[chunk.key, chunk.id]:
            with lock:
                if (succ.key, succ.id) in visited:
                    continue
                if len(preds[succ.key]) == 0 or \
                        all(finishes.get(k, False) for k in preds[succ.key]):
                    q.insert(0, succ)
    except Exception:
        has_error.set()
        raise
    finally:
        semaphore.release()


def _order_starts(graph):
    visited = set()
    starts = deque(graph.iter_indep())
    stack = deque([starts.popleft()])

    while stack:
        node = stack.popleft()
        if node not in visited:
            preds = graph.predecessors(node)
            if not preds or all(pred in visited for pred in preds):
                if len(preds) == 0:
                    yield node
                visited.add(node)
                stack.extend(n for n in graph[node] if n not in visited)
            else:
                stack.appendleft(node)
                stack.extendleft(reversed(list(n for n in graph.predecessors(node)
                                               if n not in visited)))
        if not stack and starts:
            stack.appendleft(starts.popleft())


def execute_graph(graph, keys, executor, executed_keys=None, n_parallel=None, show_progress=False,
                  mock=False, sparse_mock_percent=1.0, prefetch=False, retval=True):
    pool_executor = futures.ThreadPoolExecutor(n_parallel or 1)
    prefetch_executor = futures.ThreadPoolExecutor(n_parallel or 1) if prefetch else None
    chunk_result = executor.chunk_result
    q = list()
    lock = threading.Lock()
    semaphore = threading.Semaphore(pool_executor._max_workers)
    has_error = threading.Event()

    preds = dict()
    preds.update(
        dict((t.key, [i.key for i in v]) for t, v in graph.iter_predecessor_items())
    )
    succs = dict()
    succs.update(
        dict(((t.key, t.id), v) for t, v in graph.iter_successor_items())
    )

    starts = list(_order_starts(graph)) if len(graph) > 0 else list()
    assert len(starts) == sum(1 for _ in graph.iter_indep())
    q.extend(starts)
    finishes = dict()
    visited = set()
    fs = dict()
    ref_counts = dict()
    key_set = set(keys).union(set(executed_keys or []))
    for chunk in graph:
        if chunk.key not in key_set:
            ref_counts[chunk.key] = ref_counts.get(chunk.key, 0) + len(graph[chunk])
    node_keys_set = {n.key for n in graph}
    count = itertools.count(0)
    maximum_usage = 0

    def fetch(chunk):
        with lock:
            try:
                to_fetch_chunk = next(c for c in list(succs.get((chunk.key, chunk.id), []))
                                      if all(i in finishes for i in preds[c.key] if i != chunk.key))
            except StopIteration:
                if len(q) > 0:
                    to_fetch_chunk = q[0]
                else:
                    return

        [chunk_result.get(k) for k in preds[to_fetch_chunk.key]]

    def submit_to_execute():
        if show_progress:
            c = next(count)
            if c % 30 == 0 or c >= len(graph):
                print('[{0}] {1:.2f}% percent of graph has been submitted'.format(str(datetime.datetime.now()), float(c) * 100 / len(graph)))

        with lock:
            if len(q) == 0:
                semaphore.release()
                return
            chunk = q.pop(0)

        if prefetch:
            prefetch_executor.submit(fetch, chunk)
        future = pool_executor.submit(execute_chunk, chunk, executor=executor,
                                      ref_counts=ref_counts, chunk_result=chunk_result,
                                      finishes=finishes, visited=visited, q=q,
                                      lock=lock, semaphore=semaphore, has_error=has_error,
                                      preds=preds, succs=succs,
                                      mock=mock, sparse_mock_percent=sparse_mock_percent)
        fs[chunk.key] = future

    while len(node_keys_set - set(finishes.keys())) > 0:
        if has_error.is_set():
            break
        semaphore.acquire()
        if mock:
            maximum_usage = max(maximum_usage, np.sum(list(chunk_result.values())))
            # if maximum_usage > 40 * (1024 ** 3):
            #     raise RuntimeError('memory exceed')
        submit_to_execute()

    if mock:
        [f.result() for f in fs.values()]
        return maximum_usage

    [f.result() for f in fs.values()]
    if retval:
        return [chunk_result[key] for key in keys]


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
