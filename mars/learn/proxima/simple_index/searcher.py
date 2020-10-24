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

import itertools
import logging
import tempfile

import numpy as np

from .... import opcodes
from .... import tensor as mt
from ....context import get_context, RunningMode
from ....core import Base, Entity
from ....filesystem import get_fs, FileSystem
from ....operands import OutputType, OperandStage
from ....serialize import KeyField, StringField, Int32Field, DictField, AnyField
from ....tensor.core import TensorOrder
from ....tensor.merge.concatenate import TensorConcatenate
from ....tiles import TilesError
from ....utils import check_chunks_unknown_shape
from ...operands import LearnOperand, LearnOperandMixin
from ..core import proxima, validate_tensor

logger = logging.getLogger(__name__)


class ProximaSearcher(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.PROXIMA_SIMPLE_SEARCHER
    _tensor = KeyField('tensor')
    _distance_metric = StringField('distance_metric')
    _dimension = Int32Field('dimension')
    _topk = Int32Field('topk')
    _threads = Int32Field('threads')
    _index = AnyField('index')
    _index_searcher = StringField('index_searcher')
    _index_searcher_params = DictField('index_searcher_params')
    _index_reformer = StringField('index_reformer')
    _index_reformer_params = DictField('index_reformer_params')
    _storage_options = DictField('storage_options')

    def __init__(self, tensor=None, distance_metric=None, dimension=None,
                 topk=None, index=None, threads=None,
                 index_searcher=None, index_searcher_params=None,
                 index_reformer=None, index_reformer_params=None,
                 storage_options=None, output_types=None, stage=None, **kw):
        super().__init__(_tensor=tensor, _distance_metric=distance_metric,
                         _dimension=dimension, _index=index, _threads=threads,
                         _index_searcher=index_searcher, _index_searcher_params=index_searcher_params,
                         _index_reformer=index_reformer, _index_reformer_params=index_reformer_params,
                         _output_types=output_types, _topk=topk, _stage=stage,
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
    def storage_options(self):
        return self._storage_options

    @property
    def output_limit(self):
        return 2

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self._stage != OperandStage.agg:
            self._tensor = self._inputs[0]
            if isinstance(self._index, (Base, Entity)):
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
    def tile(cls, op: "ProximaSearcher"):
        tensor = op.tensor
        index = op.index
        topk = op.topk
        outs = op.outputs

        # make sure all inputs have known chunk sizes
        check_chunks_unknown_shape(op.inputs, TilesError)

        if tensor.chunk_shape[1] > 1:
            tensor = tensor.rechunk({1: tensor.shape[1]})._inplace_tile()

        logger.warning(f"query chunks count: {len(tensor.chunks)} ")

        if hasattr(index, 'op'):
            built_indexes = index.chunks
        else:
            # index path
            fs: FileSystem = get_fs(index, op.storage_options)
            built_indexes = [f for f in fs.ls(index)
                             if f.rsplit('/', 1)[-1].startswith('proxima-')]

        if hasattr(index, 'op'):
            ctx = get_context()
            index_chunks_workers = [m.workers[0] if m.workers else None for m in
                                    ctx.get_chunk_metas([c.key for c in index.chunks])]
        else:
            index_chunks_workers = [None] * len(built_indexes)

        out_chunks = [], []
        for tensor_chunk in tensor.chunks:
            pk_chunks, distance_chunks = [], []
            for j, chunk_index, worker in \
                    zip(itertools.count(), built_indexes, index_chunks_workers):
                chunk_op = op.copy().reset_key()
                chunk_op._stage = OperandStage.map
                if hasattr(chunk_index, 'op'):
                    chunk_op._expect_worker = worker
                chunk_op._index = chunk_index
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
                chunk_inputs = [tensor_chunk]
                if hasattr(chunk_index, 'op'):
                    chunk_inputs.append(chunk_index)
                pk_chunk, distance_chunk = chunk_op.new_chunks(
                    chunk_inputs, kws=chunk_kws)
                pk_chunks.append(pk_chunk)
                distance_chunks.append(distance_chunk)

            if len(pk_chunks) == 1:
                out_chunks[0].append(pk_chunks[0])
                out_chunks[1].append(distance_chunks[0])
                continue

            shape = (tensor_chunk.shape[0], topk * len(pk_chunks))
            pk_merge_op = TensorConcatenate(axis=1)
            pk_merge_chunk = pk_merge_op.new_chunk(
                pk_chunks, index=(pk_chunks[0].index[0], 0), shape=shape,
                dtype=pk_chunks[0].dtype, order=pk_chunks[0].order)
            distance_merge_op = TensorConcatenate(axis=1)
            distance_merge_chunk = distance_merge_op.new_chunk(
                distance_chunks, index=(distance_chunks[0].index[0], 0), shape=shape,
                dtype=distance_chunks[0].dtype, order=distance_chunks[0].order)

            agg_op = ProximaSearcher(stage=OperandStage.agg,
                                     topk=op.topk,
                                     distance_metric=op.distance_metric)
            agg_chunk_kws = [
                {'index': pk_merge_chunk.index,
                 'dtype': outs[0].dtype,
                 'shape': (tensor_chunk.shape[0], topk),
                 'order': outs[0].order},
                {'index': pk_merge_chunk.index,
                 'dtype': outs[1].dtype,
                 'shape': (tensor_chunk.shape[0], topk),
                 'order': outs[1].order}
            ]
            pk_result_chunk, distance_result_chunk = agg_op.new_chunks(
                [pk_merge_chunk, distance_merge_chunk],
                kws=agg_chunk_kws)
            out_chunks[0].append(pk_result_chunk)
            out_chunks[1].append(distance_result_chunk)

        logger.warning(f"query out_chunks count: {len(out_chunks)} ")

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
    def _execute_map(cls, ctx, op: "ProximaSearcher"):
        inp = ctx[op.tensor.key]
        if hasattr(op.index, 'key'):
            check_expect_worker = True
            index_path = ctx[op.index.key]
        else:
            check_expect_worker = False
            index_path = op.index

            fs = get_fs(index_path, op.storage_options)
            local_path = tempfile.mkstemp(prefix='proxima-', suffix='.index')[1]
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

            index_path = local_path

        if hasattr(ctx, 'running_mode') and \
                ctx.running_mode == RunningMode.distributed and check_expect_worker:
            # check if the worker to execute is identical to
            # the worker where built index
            expect_worker = op.expect_worker
            curr_worker = ctx.get_local_address()
            if curr_worker:
                assert curr_worker == expect_worker, \
                    f'the worker({curr_worker}) to execute should be identical ' \
                    f'to the worker({expect_worker}) where built index'

        flow = proxima.IndexFlow(container_name='FileContainer', container_params={},
                                 searcher_name=op.index_searcher, searcher_params=op.index_searcher_params,
                                 measure_name=op.distance_metric, measure_params={},
                                 reformer_name=op.index_reformer, reformer_params=op.index_reformer_params
                                 )
        flow.load(index_path)
        vecs = np.ascontiguousarray(inp)

        logger.warning(f"threads count:{op.threads}")
        logger.warning(f"vecs count:{len(vecs)}")

        result_pks, result_distances = proxima.IndexUtility.ann_search(searcher=flow,
                                                                       query=vecs,
                                                                       topk=op.topk,
                                                                       threads=op.threads)

        ctx[op.outputs[0].key] = np.asarray(result_pks)
        ctx[op.outputs[1].key] = np.asarray(result_distances)

    @classmethod
    def _execute_agg(cls, ctx, op: "ProximaSearcher"):
        pks, distances = [ctx[inp.key] for inp in op.inputs]
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


def search_index(tensor, topk, index, threads=4, dimension=None,
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
                         topk=topk, index=index, threads=threads,
                         index_searcher=index_searcher, index_searcher_params=index_searcher_params,
                         index_reformer=index_reformer, index_reformer_params=index_reformer_params,
                         storage_options=storage_options)
    result = op(tensor, index)
    if run:
        return result.execute(session=session, **(run_kwargs or dict()))
    else:
        return result
