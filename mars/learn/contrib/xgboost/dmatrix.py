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
from typing import List, Union

import numpy as np

from .... import opcodes as OperandDef
from ....core import get_output_types, recursive_tile
from ....core.context import get_context, Context
from ....dataframe.core import DATAFRAME_TYPE
from ....serialization.serializables import KeyField, Float64Field, ListField, BoolField
from ....tensor.core import TENSOR_TYPE, TENSOR_CHUNK_TYPE
from ....tensor import tensor as astensor
from ....typing import TileableType, ChunkType
from ....utils import has_unknown_shape, ensure_own_data, build_fetch
from ...operands import LearnOperand, LearnOperandMixin
from ...utils import convert_to_tensor_or_dataframe, concat_chunks


class ToDMatrix(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.TO_DMATRIX

    data = KeyField('data')
    label = KeyField('label')
    missing = Float64Field('missing')
    weight = KeyField('weight')
    feature_names = ListField('feature_names')
    feature_types = ListField('feature_types')
    # if to collocate the data, label and weight
    _collocate = BoolField('collocate', default=False)

    @property
    def output_limit(self):
        if self._collocate:
            return 1 + (self.label is not None) + (self.weight is not None)
        return 1

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self.data is not None:
            self.data = self._inputs[0]
        has_label = self.label is not None
        if has_label:
            self.label = self._inputs[1]
        if self.weight is not None:
            i = 1 if not has_label else 2
            self.weight = self._inputs[i]

    @staticmethod
    def _get_kw(obj):
        if isinstance(obj, TENSOR_TYPE + TENSOR_CHUNK_TYPE):
            return {'shape': obj.shape,
                    'dtype': obj.dtype,
                    'order': obj.order}
        else:
            return {'shape': obj.shape,
                    'dtypes': obj.dtypes,
                    'index_value': obj.index_value,
                    'columns_value': obj.columns_value}

    def __call__(self):
        inputs = [self.data]
        kw = self._get_kw(self.data)
        if self.label is not None:
            inputs.append(self.label)
        if self.weight is not None:
            inputs.append(self.weight)

        return self.new_tileable(inputs, **kw)

    def new_tensor(self, *args, **kwargs):
        return self.new_tileable(*args, **kwargs)

    @classmethod
    def _get_collocated(cls,
                        op: "ToDMatrix",
                        data: TileableType,
                        label: TileableType,
                        weight: TileableType) -> List[TileableType]:
        nsplit = data.nsplits[0]
        out_chunkss = [[] for _ in op.inputs]
        for i in range(len(nsplit)):
            data_chunk = data.cix[i, 0]
            inps = [data_chunk]
            kws = []
            chunk_op = op.copy().reset_key()
            chunk_op._collocate = True
            chunk_op.data = data_chunk
            output_types = [get_output_types(data)[0]]
            data_kw = cls._get_kw(data_chunk)
            data_kw['index'] = data_chunk.index
            kws.append(data_kw)
            if label is not None:
                label_chunk = chunk_op.label = label.cix[i, ]
                inps.append(label_chunk)
                kw = cls._get_kw(label_chunk)
                kw['index'] = label_chunk.index
                kw['type'] = 'label'
                kws.append(kw)
                output_types.append(get_output_types(label)[0])
            if weight is not None:
                weight_chunk = chunk_op.weight = weight.cix[i, ]
                inps.append(weight_chunk)
                kw = cls._get_kw(weight_chunk)
                kw['index'] = weight_chunk.index
                kw['type'] = 'weight'
                kws.append(kw)
                output_types.append(get_output_types(weight)[0])
            chunk_op.output_types = output_types
            out_chunks = chunk_op.new_chunks(inps, kws=kws)
            for i, out_chunk in enumerate(out_chunks):
                out_chunkss[i].append(out_chunk)

        new_op = op.copy()
        new_op._collocate = True
        outs = [data, label, weight]
        params = [out.params.copy() if out is not None else None
                  for out in outs]
        types = ['data', 'label', 'weight']
        output_types = []
        for i, out in enumerate(outs):
            if out is None:
                continue
            params[i]['nsplits'] = out.nsplits
            params[i]['chunks'] = out_chunkss[i]
            params[i]['type'] = types[i]
            output_types.append(get_output_types(out)[0])
        new_op.output_types = output_types
        return new_op.new_tileables(op.inputs, kws=params)

    @staticmethod
    def _order_chunk_index(chunks: List[ChunkType]):
        ndim = chunks[0].ndim
        for i, c in enumerate(chunks):
            if ndim == 2:
                c._index = (i, 0)
            else:
                c._index = (i,)
        return chunks

    @classmethod
    def tile(cls, op: "MarsDMatrix"):
        data, label, weight = op.data, op.label, op.weight

        if has_unknown_shape(data):
            yield
        if data.chunk_shape[1] > 1:
            # make sure data's second dimension has only 1 chunk
            data = yield from recursive_tile(data.rechunk({1: data.shape[1]}))
        nsplit = data.nsplits[0]
        # rechunk label
        if label is not None:
            label = yield from recursive_tile(label.rechunk({0: nsplit}))
        # rechunk weight
        if weight is not None:
            weight = yield from recursive_tile(weight.rechunk({0: nsplit}))

        collocated = cls._get_collocated(op, data, label, weight)
        collocated_chunks = list(itertools.chain.from_iterable(
            c.chunks for c in collocated))
        yield collocated_chunks + collocated

        data = build_fetch(collocated[0])
        if label is not None:
            label = build_fetch(collocated[1])
        if weight is not None:
            weight = build_fetch(collocated[-1])

        ctx = get_context()

        # for distributed, we should concat the chunks
        # which allocated on the same worker into one
        data_chunk_metas = ctx.get_chunks_meta([c.key for c in data.chunks], fields=['bands'])
        data_chunk_workers = [m['bands'][0][0] for m in data_chunk_metas]
        worker_to_chunks = dict()
        for i, worker in enumerate(data_chunk_workers):
            size = 1 + (label is not None) + (weight is not None)
            if worker not in worker_to_chunks:
                worker_to_chunks[worker] = [[] for _ in range(size)]
            worker_to_chunks[worker][0].append(data.chunks[i])
            if label is not None:
                worker_to_chunks[worker][1].append(label.chunks[i])
            if weight is not None:
                worker_to_chunks[worker][-1].append(weight.chunks[i])
        ind = itertools.count(0)
        out_chunks = []
        for worker, chunks in worker_to_chunks.items():
            data_chunk = concat_chunks(cls._order_chunk_index(chunks[0]))
            inps = [data_chunk]
            label_chunk = None
            if label is not None:
                label_chunk = concat_chunks(cls._order_chunk_index(chunks[1]))
                inps.append(label_chunk)
            weight_chunk = None
            if weight is not None:
                weight_chunk = concat_chunks(cls._order_chunk_index(chunks[2]))
                inps.append(weight_chunk)
            chunk_op = ToDMatrix(data=data_chunk, label=label_chunk, missing=op.missing,
                                 weight=weight_chunk, feature_names=op.feature_names,
                                 feature_types=op.feature_types, _output_types=op.output_types)
            kws = data_chunk.params
            kws['index'] = (next(ind), 0)
            out_chunks.append(chunk_op.new_chunk(inps, **kws))
        nsplits = (tuple(c.shape[0] for c in out_chunks), (out_chunks[0].shape[1],))

        new_op = op.copy()
        kw = op.outputs[0].params
        kw['chunks'] = out_chunks
        kw['nsplits'] = nsplits
        return new_op.new_tileables(op.inputs, kws=[kw])

    @staticmethod
    def get_xgb_dmatrix(tup):
        from xgboost import DMatrix

        data, label, weight, missing, feature_names, feature_types = tup
        data = data.spmatrix if hasattr(data, 'spmatrix') else data
        return DMatrix(ensure_own_data(data), label=ensure_own_data(label),
                       missing=missing, weight=ensure_own_data(weight),
                       feature_names=feature_names, feature_types=feature_types,
                       nthread=-1)

    @staticmethod
    def _from_ctx_if_not_none(ctx, chunk):
        if chunk is None:
            return chunk
        return ctx[chunk.key]

    @classmethod
    def execute(cls,
                ctx: Union[dict, Context],
                op: "ToDMatrix"):
        if op._collocate:
            outs = op.outputs
            ctx[outs[0].key] = ctx[op.inputs[0].key]
            if op.label is not None:
                ctx[outs[1].key] = ctx[op.inputs[1].key]
            if op.weight is not None:
                ctx[outs[-1].key] = ctx[op.inputs[-1].key]
        else:
            out = op.outputs[0]
            data = cls._from_ctx_if_not_none(ctx, op.data)
            if data is None:
                data = np.empty((0, out.shape[1]))
            ctx[out.key] = (
                data,
                cls._from_ctx_if_not_none(ctx, op.label),
                cls._from_ctx_if_not_none(ctx, op.weight),
                op.missing,
                op.feature_names,
                op.feature_types
            )


def check_data(data):
    data = convert_to_tensor_or_dataframe(data)
    if data.ndim != 2:
        raise ValueError(f'Expecting 2-d data, got: {data.ndim}-d')

    return data


def to_dmatrix(data, label=None, missing=None, weight=None,
               feature_names=None, feature_types=None):
    data = check_data(data)
    if label is not None:
        label = convert_to_tensor_or_dataframe(label)
        if isinstance(label, DATAFRAME_TYPE):
            label = label.iloc[:, 0]
        label = astensor(label)
        if label.ndim != 1:
            raise ValueError(f'Expecting 1-d label, got: {label.ndim}-d')
    if weight is not None:
        weight = convert_to_tensor_or_dataframe(weight)
        if isinstance(weight, DATAFRAME_TYPE):
            weight = weight.iloc[:, 0]
        weight = astensor(weight)
        if weight.ndim != 1:
            raise ValueError(f'Expecting 1-d weight, got {weight.ndim}-d')

    # If not multiple outputs, try to collect the chunks on same worker into one
    # to feed the data into XGBoost for training.
    op = ToDMatrix(data=data, label=label, missing=missing, weight=weight,
                   feature_names=feature_names, feature_types=feature_types,
                   gpu=data.op.gpu, _output_types=get_output_types(data))
    return op()


MarsDMatrix = to_dmatrix
