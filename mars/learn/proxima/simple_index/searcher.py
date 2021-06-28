# Copyright 1999-2021 Alibaba Group Holding Ltd.
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
import os
import pickle  # nosec  # pylint: disable=import_pickle
import random
from hashlib import md5
from collections import defaultdict

import numpy as np

from .... import opcodes
from .... import tensor as mt
from ....config import options
from ....core import ENTITY_TYPE, OutputType, recursive_tile
from ....core.context import get_context
from ....core.operand import OperandStage
from ....lib.filesystem import get_fs, FileSystem
from ....serialization.serializables import KeyField, StringField, Int32Field, Int64Field, \
    DictField, AnyField, BytesField, BoolField
from ....tensor.core import TensorOrder
from ....utils import has_unknown_shape, Timer, ceildiv
from ...operands import LearnOperand, LearnOperandMixin
from ..core import proxima, validate_tensor, get_proxima_type

logger = logging.getLogger(__name__)


class ProximaSearcher(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.PROXIMA_SIMPLE_SEARCHER
    _tensor = KeyField('tensor')
    _distance_metric = StringField('distance_metric')
    _dimension = Int32Field('dimension')
    _row_number = Int64Field('row_number')
    _topk = Int32Field('topk')
    _threads = Int32Field('threads')
    _index = AnyField('index')
    _index_searcher = StringField('index_searcher')
    _index_searcher_params = DictField('index_searcher_params')
    _index_reformer = StringField('index_reformer')
    _index_reformer_params = DictField('index_reformer_params')
    _download_index = BoolField('download_index')
    _storage_options = BytesField('storage_options',
                                  on_serialize=pickle.dumps,
                                  on_deserialize=pickle.loads)

    def __init__(self, tensor=None, distance_metric=None, dimension=None,
                 row_number=None, topk=None, index=None, threads=None,
                 index_searcher=None, index_searcher_params=None,
                 index_reformer=None, index_reformer_params=None,
                 download_index=None, storage_options=None, output_types=None,
                 stage=None, **kw):
        super().__init__(_tensor=tensor, _distance_metric=distance_metric,
                         _row_number=row_number, _dimension=dimension, _index=index, _threads=threads,
                         _index_searcher=index_searcher, _index_searcher_params=index_searcher_params,
                         _index_reformer=index_reformer, _index_reformer_params=index_reformer_params,
                         _download_index=download_index, _output_types=output_types, _topk=topk,
                         _storage_options=storage_options, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor, OutputType.tensor]

    @property
    def tensor(self):
        return self._tensor

    @property
    def distance_metric(self):
        return self._distance_metric

    @property
    def dimension(self):
        return self._dimension

    @property
    def row_number(self):
        return self._row_number

    @property
    def topk(self):
        return self._topk

    @property
    def threads(self):
        return self._threads

    @property
    def index(self):
        return self._index

    @property
    def index_searcher(self):
        return self._index_searcher

    @property
    def index_searcher_params(self):
        return self._index_searcher_params

    @property
    def index_reformer(self):
        return self._index_reformer

    @property
    def index_reformer_params(self):
        return self._index_reformer_params

    @property
    def download_index(self):
        return self._download_index

    @property
    def storage_options(self):
        return self._storage_options

    @property
    def output_limit(self):
        return 1 if self._download_index else 2

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self.stage != OperandStage.agg and not self._download_index:
            self._tensor = self._inputs[0]
            if isinstance(self._index, ENTITY_TYPE):
                self._index = self._inputs[-1]

    def __call__(self, tensor, index):
        kws = [
            {'dtype': np.dtype(np.uint64),
             'shape': (tensor.shape[0], self._topk),
             'order': TensorOrder.C_ORDER},
            {'dtype': np.dtype(np.float32),
             'shape': (tensor.shape[0], self._topk),
             'order': TensorOrder.C_ORDER}
        ]
        inputs = [tensor]
        if hasattr(index, 'op'):
            inputs.append(index)
        return mt.ExecutableTuple(self.new_tileables(inputs, kws=kws))

    @classmethod
    def _build_download_chunks(cls, op, indexes):
        ctx = get_context()
        workers = ctx.get_worker_addresses() or [None]
        if len(workers) < len(indexes):
            workers = [workers[i % len(workers)] for i in range(len(indexes))]
        indexes_iter = iter(itertools.cycle(indexes))

        download_chunks = defaultdict(list)
        for i, worker in enumerate(workers):
            download_op = op.copy().reset_key()
            download_op.stage = OperandStage.map
            download_op.expect_worker = worker
            download_op._download_index = True
            download_op._tensor = None
            download_op._index = next(indexes_iter)
            download_chunks[i % len(indexes)].append(
                download_op.new_chunk(None, index=(i,), shape=(),
                                      dtype=op.inputs[0].dtype))
        return download_chunks

    @classmethod
    def tile(cls, op: "ProximaSearcher"):
        tensor = op.tensor
        index = op.index
        topk = op.topk
        outs = op.outputs
        row_number = op.row_number

        ctx = get_context()

        # make sure all inputs have known chunk sizes
        if has_unknown_shape(*op.inputs):
            yield

        rechunk_size = dict()
        if tensor.chunk_shape[1] > 1:
            rechunk_size[1] = tensor.shape[1]
        if row_number is not None:
            rechunk_size[0] = tensor.shape[0] // row_number
        if len(rechunk_size) > 0:
            tensor = yield from recursive_tile(tensor.rechunk(rechunk_size))

        logger.warning(f"query chunks count: {len(tensor.chunks)} ")

        if hasattr(index, 'op'):
            built_indexes = [index.chunks] * len(tensor.chunks)
        else:
            # index path
            fs: FileSystem = get_fs(index, op.storage_options)
            index_paths = [f for f in fs.ls(index)
                           if f.rsplit('/', 1)[-1].startswith('proxima_')]
            download_chunks = cls._build_download_chunks(op, index_paths)
            iters = [iter(itertools.cycle(i)) for i in download_chunks.values()]
            built_indexes = []
            for _ in range(len(tensor.chunks)):
                built_indexes.append([next(it) for it in iters])

        if hasattr(index, 'op'):
            index_chunks_workers = [m['bands'][0][0] for m in
                                    ctx.get_chunks_meta(
                                        [c.key for c in index.chunks], fields=['bands'])]
        else:
            index_chunks_workers = [None] * len(built_indexes[0])

        out_chunks = [], []
        for i, tensor_chunk in enumerate(tensor.chunks):
            pk_chunks, distance_chunks = [], []
            for j, chunk_index, worker in \
                    zip(itertools.count(), built_indexes[i], index_chunks_workers):
                chunk_op = op.copy().reset_key()
                chunk_op.stage = OperandStage.map
                if hasattr(index, 'op'):
                    chunk_op.expect_worker = worker
                else:
                    chunk_op.expect_worker = chunk_index.op.expect_worker
                chunk_op._index = chunk_index
                chunk_op._tensor = None
                chunk_kws = [
                    {'index': (tensor_chunk.index[0], j),
                     'dtype': outs[0].dtype,
                     'shape': (tensor_chunk.shape[0], topk),
                     'order': TensorOrder.C_ORDER},
                    {'index': (tensor_chunk.index[0], j),
                     'dtype': outs[1].dtype,
                     'shape': (tensor_chunk.shape[0], topk),
                     'order': TensorOrder.C_ORDER}
                ]
                chunk_inputs = [tensor_chunk, chunk_index]
                pk_chunk, distance_chunk = chunk_op.new_chunks(
                    chunk_inputs, kws=chunk_kws)
                pk_chunks.append(pk_chunk)
                distance_chunks.append(distance_chunk)

            if len(pk_chunks) == 1:
                out_chunks[0].append(pk_chunks[0])
                out_chunks[1].append(distance_chunks[0])
                continue

            # combine topk results
            combine_size = options.combine_size

            tensor_out_chunks = [pk_chunks, distance_chunks]
            while True:
                chunk_size = ceildiv(len(tensor_out_chunks[0]), combine_size)
                cur_out_chunks = [[], []]
                for k in range(chunk_size):
                    to_combine_pks = tensor_out_chunks[0][k * combine_size: (k + 1) * combine_size]
                    to_combine_distances = tensor_out_chunks[1][k * combine_size: (k + 1) * combine_size]

                    chunk_op = op.copy().reset_key()
                    chunk_op.stage = OperandStage.agg
                    chunk_op._tensor = None
                    chunk_op._index = None
                    agg_chunk_kws = [
                        {'index': (i, 0),
                         'dtype': outs[0].dtype,
                         'shape': (tensor_chunk.shape[0], topk),
                         'order': outs[0].order},
                        {'index': (i, 0),
                         'dtype': outs[1].dtype,
                         'shape': (tensor_chunk.shape[0], topk),
                         'order': outs[1].order}
                    ]
                    pk_result_chunk, distance_result_chunk = chunk_op.new_chunks(
                        to_combine_pks + to_combine_distances,
                        kws=agg_chunk_kws)
                    cur_out_chunks[0].append(pk_result_chunk)
                    cur_out_chunks[1].append(distance_result_chunk)
                tensor_out_chunks = cur_out_chunks
                if len(tensor_out_chunks[0]) == 1:
                    break
            out_chunks[0].append(tensor_out_chunks[0][0])
            out_chunks[1].append(tensor_out_chunks[1][0])

        kws = []
        pk_params = outs[0].params
        pk_params['chunks'] = out_chunks[0]
        pk_params['nsplits'] = (tensor.nsplits[0], (topk,))
        kws.append(pk_params)
        distance_params = outs[1].params
        distance_params['chunks'] = out_chunks[1]
        distance_params['nsplits'] = (tensor.nsplits[0], (topk,))
        kws.append(distance_params)
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=kws)

    @classmethod
    def _execute_download(cls, ctx, op: "ProximaSearcher"):
        index_path = op.index
        with Timer() as timer:
            fs = get_fs(index_path, op.storage_options)

            # TODO
            dirs = os.environ.get('MARS_SPILL_DIRS')
            if dirs:
                temp_dir = random.choice(dirs.split(':'))
            else:
                temp_dir = "/tmp/proxima-index/"

            local_path = os.path.join(temp_dir, md5(str(index_path).encode('utf-8')).hexdigest())  # noqa: B303  # nosec
            exist_state = True
            if not os.path.exists(local_path):
                exist_state = False
                if not os.path.exists(local_path.rsplit("/", 1)[0]):
                    os.mkdir(local_path.rsplit("/", 1)[0])
                with open(local_path, 'wb') as out_f:
                    with fs.open(index_path, 'rb') as in_f:
                        # 32M
                        chunk_bytes = 32 * 1024 ** 2
                        while True:
                            data = in_f.read(chunk_bytes)
                            if data:
                                out_f.write(data)
                            else:
                                break

        logger.warning(f'ReadingFromVolume({op.key}), index path: {index_path}, '
                       f'local_path {local_path}'
                       f'size {os.path.getsize(local_path)}, '
                       f'already exist {exist_state}, '
                       f'costs {timer.duration} seconds '
                       f'speed {round(os.path.getsize(local_path) / (1024 ** 2) / timer.duration, 2)} MB/s')
        ctx[op.outputs[0].key] = local_path

    @classmethod
    def _execute_map(cls, ctx, op: "ProximaSearcher"):
        if op.download_index:
            cls._execute_download(ctx, op)
            return

        inp = ctx[op.tensor.key]
        index_path = ctx[op.inputs[-1].key]

        with Timer() as timer:
            flow = proxima.IndexFlow(container_name='MMapFileContainer', container_params={},
                                     searcher_name=op.index_searcher, searcher_params=op.index_searcher_params,
                                     measure_name="", measure_params={},
                                     reformer_name=op.index_reformer, reformer_params=op.index_reformer_params
                                     )

            flow.load(index_path)
            vecs = np.ascontiguousarray(inp)

        logger.warning(f'LoadIndex({op.key})  index path: {index_path}  costs {timer.duration} seconds')
        logger.warning(f"threads count:{op.threads}  vecs count:{len(vecs)}")

        with Timer() as timer:
            batch = 10000
            s_idx = 0
            e_idx = min(s_idx + batch, len(vecs))
            result_pks, result_distances = None, None
            while s_idx < len(vecs):
                with Timer() as timer_s:
                    tp = get_proxima_type(vecs.dtype)
                    result_pks_b, result_distances_b = proxima.IndexUtility.ann_search(searcher=flow,
                                                                                       type=tp,
                                                                                       query=vecs[s_idx:e_idx],
                                                                                       topk=op.topk,
                                                                                       threads=op.threads)
                    if result_pks is None:
                        result_pks = np.asarray(result_pks_b)
                        result_distances = np.asarray(result_distances_b)
                    else:
                        result_pks = np.concatenate((result_pks, np.asarray(result_pks_b)))
                        result_distances = np.concatenate((result_distances, np.asarray(result_distances_b)))

                    s_idx = e_idx
                    e_idx = min(s_idx + batch, len(vecs))
                logger.warning(
                    f'Search({op.key}) count {s_idx}/{len(vecs)}:{round(s_idx * 100 / len(vecs), 2)}%'
                    f' costs {round(timer_s.duration, 2)} seconds')
        logger.warning(f'Search({op.key}) costs {timer.duration} seconds')

        ctx[op.outputs[0].key] = np.asarray(result_pks)
        ctx[op.outputs[1].key] = np.asarray(result_distances)

    @classmethod
    def _execute_agg(cls, ctx, op: "ProximaSearcher"):
        inputs_data = [ctx[inp.key] for inp in op.inputs]

        chunk_num = len(inputs_data) // 2
        pks = np.concatenate(inputs_data[:chunk_num], axis=1)
        distances = np.concatenate(inputs_data[chunk_num:], axis=1)

        n_doc = len(pks)
        topk = op.topk

        # calculate topk on rows
        if op.distance_metric == "InnerProduct":
            inds = np.argsort(distances, axis=1)[:, -1:-topk - 1:-1]
        else:
            inds = np.argsort(distances, axis=1)[:, :topk]

        result_pks = np.empty((n_doc, topk), dtype=pks.dtype)
        result_distances = np.empty((n_doc, topk), dtype=distances.dtype)
        rng = np.arange(n_doc)
        for i in range(topk):
            ind = inds[:, i]
            result_pks[:, i] = pks[rng, ind]
            result_distances[:, i] = distances[rng, ind]
        del rng

        ctx[op.outputs[0].key] = result_pks
        ctx[op.outputs[1].key] = result_distances

    @classmethod
    def execute(cls, ctx, op: "ProximaSearcher"):
        if op.stage != OperandStage.agg:
            return cls._execute_map(ctx, op)
        else:
            return cls._execute_agg(ctx, op)


def search_index(tensor, topk, index, threads=4, row_number=None, dimension=None,
                 distance_metric=None, index_searcher=None, index_searcher_params=None,
                 index_reformer=None, index_reformer_params=None,
                 storage_options=None, run=True, session=None, run_kwargs=None):
    tensor = validate_tensor(tensor)

    if dimension is None:
        dimension = tensor.shape[1]
    if index_searcher is None:
        index_searcher = ""
    if index_searcher_params is None:
        index_searcher_params = {}
    if index_reformer is None:
        index_reformer = ""
    if index_reformer_params is None:
        index_reformer_params = {}
    if distance_metric is None:
        distance_metric = ""
    if hasattr(index, 'op') and index.op.index_path is not None:
        storage_options = storage_options or index.op.storage_options
        index = index.op.index_path

    op = ProximaSearcher(tensor=tensor, distance_metric=distance_metric, dimension=dimension,
                         row_number=row_number, topk=topk, index=index, threads=threads,
                         index_searcher=index_searcher, index_searcher_params=index_searcher_params,
                         index_reformer=index_reformer, index_reformer_params=index_reformer_params,
                         storage_options=storage_options)
    result = op(tensor, index)
    if run:
        return result.execute(session=session, **(run_kwargs or dict()))
    else:
        return result
