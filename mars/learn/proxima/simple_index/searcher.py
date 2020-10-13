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

import numpy as np

from .... import opcodes
from .... import tensor as mt
from ....context import get_context
from ....core import Base, Entity
from ....operands import OutputType, OperandStage
from ....serialize import KeyField, StringField, Int32Field, DictField
from ....tensor.core import TensorOrder
from ....tensor.merge.concatenate import TensorConcatenate
from ....tiles import TilesError
from ....tensor.utils import decide_unify_split
from ....utils import check_chunks_unknown_shape
from ...operands import LearnOperand, LearnOperandMixin
from ..core import proxima, validate_tensor


class ProximaSearcher(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.PROXIMA_SIMPLE_SEARCHER

    _tensor = KeyField('tensor')
    _pk = KeyField('pk')
    _distance_metric = StringField('distance_metric')
    _dimension = Int32Field('dimension')
    _topk = Int32Field('topk')
    _index = KeyField('index')
    _index_path = StringField('index_path')
    _index_searcher = StringField('index_searcher')
    _index_searcher_params = DictField('index_searcher_params')
    _index_reformer = StringField('index_reformer')
    _index_reformer_params = DictField('index_reformer_params')

    def __init__(self, tensor=None, pk=None, distance_metric=None, dimension=None,
                 topk=None, index=None, index_path=None,
                 index_searcher=None, index_searcher_params=None,
                 index_reformer=None, index_reformer_params=None,
                 output_types=None, stage=None, **kw):
        super().__init__(_tensor=tensor, _pk=pk, _distance_metric=distance_metric,
                         _dimension=dimension, _index_path=index_path,
                         _index_searcher=index_searcher, _index_searcher_params=index_searcher_params,
                         _index_reformer=index_reformer, _index_reformer_params=index_reformer_params,
                         _output_types=output_types, _index=index, _topk=topk, _stage=stage, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.tensor, OutputType.tensor]

    @property
    def tensor(self):
        return self._tensor

    @property
    def pk(self):
        return self._pk

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
    def index(self):
        return self._index

    @property
    def index_path(self):
        return self._index_path

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
    def output_limit(self):
        return 2

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self._stage != OperandStage.agg:
            self._tensor = self._inputs[0]
            self._pk = self._inputs[1]
            if self._index is not None:
                self._index = self._inputs[2]

    def __call__(self, tensor, pk, index):
        kws = [
            {'dtype': pk.dtype,
             'shape': (pk.shape[0], self._topk),
             'order': TensorOrder.C_ORDER},
            {'dtype': np.dtype(np.float32),
             'shape': (pk.shape[0], self._topk),
             'order': TensorOrder.C_ORDER}
        ]
        return mt.ExecutableTuple(self.new_tileables([tensor, pk, index], kws=kws))

    @classmethod
    def tile(cls, op):
        tensor = op.tensor
        pk = op.pk
        index = op.index
        topk = op.topk
        outs = op.outputs

        # make sure all inputs have known chunk sizes
        check_chunks_unknown_shape(op.inputs, TilesError)

        nsplit = decide_unify_split(tensor.nsplits[0], pk.nsplits[0])
        if tensor.chunk_shape[1] > 1:
            tensor = tensor.rechunk({0: nsplit, 1: tensor.shape[1]})._inplace_tile()
        else:
            tensor = tensor.rechunk({0: nsplit})._inplace_tile()
        pk = pk.rechunk({0: nsplit})._inplace_tile()

        ctx = get_context()
        index_chunks_workers = [m.workers[0] if m.workers else None for m in
                                ctx.get_chunk_metas([c.key for c in index.chunks])]

        out_chunks = [], []
        for tensor_chunk, pk_col_chunk in zip(tensor.chunks, pk.chunks):
            pk_chunks, distance_chunks = [], []
            for j, index_chunk, worker in \
                    zip(itertools.count(), index.chunks, index_chunks_workers):
                chunk_op = op.copy().reset_key()
                chunk_op._stage = OperandStage.map
                chunk_op._expect_worker = worker
                chunk_kws = [
                    {'index': (pk_col_chunk.index[0], j),
                     'dtype': pk_col_chunk.dtype,
                     'shape': (tensor_chunk.shape[0], topk),
                     'order': TensorOrder.C_ORDER},
                    {'index': (tensor_chunk.index[0], j),
                     'dtype': outs[1].dtype,
                     'shape': (tensor_chunk.shape[0], topk),
                     'order': TensorOrder.C_ORDER}
                ]
                pk_chunk, distance_chunk = chunk_op.new_chunks(
                    [tensor_chunk, pk_col_chunk, index_chunk],
                    kws=chunk_kws)
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
                                     topk=op.topk)
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

        kws = []
        pk_params = outs[0].params
        pk_params['chunks'] = out_chunks[0]
        pk_params['nsplits'] = (pk.nsplits[0], (topk,))
        kws.append(pk_params)
        distance_params = outs[1].params
        distance_params['chunks'] = out_chunks[1]
        distance_params['nsplits'] = (pk.nsplits[0], (topk,))
        kws.append(distance_params)
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=kws)

    @classmethod
    def _execute_map(cls, ctx, op: "ProximaSearcher"):
        inp = ctx[op.tensor.key]
        pks = ctx[op.pk.key]
        index_path = ctx[op.index.key]

        flow = proxima.IndexFlow(container_name='FileContainer', container_params={},
                                 searcher_name=op.index_searcher, searcher_params=op.index_searcher_params,
                                 # measure_name='Euclidean', measure_params={},
                                 reformer_name=op.index_reformer, reformer_params=op.index_reformer_params
                                 )
        flow.load(index_path)
        vecs = np.ascontiguousarray(inp)

        result_pks, result_distances = proxima.IndexUtility.ann_search(searcher=flow,
                                                                       query=vecs,
                                                                       topk=op.topk)

        ctx[op.outputs[0].key] = np.asarray(result_pks)
        ctx[op.outputs[1].key] = np.asarray(result_distances)

    @classmethod
    def _execute_agg(cls, ctx, op: "ProximaSearcher"):
        pks, distances = [ctx[inp.key] for inp in op.inputs]
        n_doc = len(pks)
        topk = op.topk

        # calculate topk on rows
        # inds = np.argpartition(distances, topk, axis=1)[:, :topk]
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


def search_index(tensor, pk, dimension,
                 topk, index, index_path, distance_metric="l2",
                 index_searcher="", index_searcher_params={},
                 index_reformer="", index_reformer_params={},
                 session=None, run_kwargs=None):
    tensor = validate_tensor(tensor)

    if not isinstance(pk, (Base, Entity)):
        pk = mt.tensor(pk, dtype=np.uint64)

    op = ProximaSearcher(tensor=tensor, pk=pk, distance_metric=distance_metric, dimension=dimension,
                         topk=topk, index=index, index_path=index_path,
                         index_searcher=index_searcher, index_searcher_params=index_searcher_params,
                         index_reformer=index_reformer, index_reformer_params=index_reformer_params)
    return op(tensor, pk, index).execute(session=session, **(run_kwargs or dict()))
