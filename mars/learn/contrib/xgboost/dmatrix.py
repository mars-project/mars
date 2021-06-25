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

from .... import opcodes as OperandDef
from ....core import ExecutableTuple, get_output_types, recursive_tile
from ....dataframe.core import DATAFRAME_TYPE
from ....serialization.serializables import KeyField, Float64Field, ListField, BoolField
from ....tensor.core import TENSOR_TYPE, TENSOR_CHUNK_TYPE
from ....tensor import tensor as astensor
from ....utils import has_unknown_shape, ensure_own_data
from ...operands import LearnOperand, LearnOperandMixin
from ...utils import convert_to_tensor_or_dataframe, concat_chunks


class ToDMatrix(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.TO_DMATRIX

    _data = KeyField('data')
    _label = KeyField('label')
    _missing = Float64Field('missing')
    _weight = KeyField('weight')
    _feature_names = ListField('feature_names')
    _feature_types = ListField('feature_types')
    _multi_output = BoolField('multi_output')

    def __init__(self, data=None, label=None, missing=None, weight=None, feature_names=None,
                 feature_types=None, multi_output=None, gpu=None, output_types=None, **kw):
        super().__init__(_data=data, _label=label, _missing=missing, _weight=weight,
                         _feature_names=feature_names, _feature_types=feature_types,
                         _gpu=gpu, _multi_output=multi_output, _output_types=output_types,
                         **kw)

    @property
    def output_limit(self):
        if self._multi_output:
            return 1 + (self._label is not None) + (self._weight is not None)
        return 1

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def missing(self):
        return self._missing

    @property
    def weight(self):
        return self._weight

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def feature_types(self):
        return self._feature_types

    @property
    def multi_output(self):
        return self._multi_output

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._data = self._inputs[0]
        has_label = self._label is not None
        if has_label:
            self._label = self._inputs[1]
        if self._weight is not None:
            i = 1 if not has_label else 2
            self._weight = self._inputs[i]

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
        inputs = [self._data]
        kws = []
        kw = self._get_kw(self._data)
        if not self._multi_output:
            kw['type'] = 'data'
        kws.append(kw)
        if self._label is not None:
            inputs.append(self._label)
            if self._multi_output:
                kw = self._get_kw(self._label)
                kw['type'] = 'label'
                kws.append(kw)
        if self._weight is not None:
            inputs.append(self._weight)
            if self._multi_output:
                kw = self._get_kw(self._weight)
                kw['type'] = 'weight'
                kws.append(kw)
        if not self.output_types:
            self.output_types = get_output_types(*inputs)

        tileables = self.new_tileables(inputs, kws=kws)
        if not self._multi_output:
            return tileables[0]
        return tileables

    @classmethod
    def _tile_multi_output(cls, op):
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

        out_chunkss = [[] for _ in range(op.output_limit)]
        for i in range(len(nsplit)):
            data_chunk = data.cix[i, 0]
            inps = [data_chunk]
            kws = []
            chunk_op = op.copy().reset_key()
            chunk_op._data = data_chunk
            data_kw = cls._get_kw(data_chunk)
            data_kw['index'] = data_chunk.index
            kws.append(data_kw)
            if label is not None:
                label_chunk = chunk_op._label = label.cix[i, ]
                inps.append(label_chunk)
                kw = cls._get_kw(label_chunk)
                kw['index'] = label_chunk.index
                kw['type'] = 'label'
                kws.append(kw)
            if weight is not None:
                weight_chunk = chunk_op._weight = weight.cix[i, ]
                inps.append(weight_chunk)
                kw = cls._get_kw(weight_chunk)
                kw['index'] = weight_chunk.index
                kw['type'] = 'weight'
                kws.append(kw)
            out_chunks = chunk_op.new_chunks(inps, kws=kws)
            for i, out_chunk in enumerate(out_chunks):
                out_chunkss[i].append(out_chunk)

        new_op = op.copy()
        params = [out.params.copy() for out in op.outputs]
        types = ['data', 'label', 'weight']
        for i, inp in enumerate([data, label, weight]):
            if inp is None:
                continue
            params[i]['nsplits'] = inp.nsplits
            params[i]['chunks'] = out_chunkss[i]
            params[i]['type'] = types[i]
        return new_op.new_tileables(op.inputs, kws=params)

    @classmethod
    def _tile_single_output(cls, op):
        from ....core.context import get_context

        data, label, weight = op.data, op.label, op.weight

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
            data_chunk = concat_chunks(chunks[0])
            inps = [data_chunk]
            label_chunk = None
            if label is not None:
                label_chunk = concat_chunks(chunks[1])
                inps.append(label_chunk)
            weight_chunk = None
            if weight is not None:
                weight_chunk = concat_chunks(chunks[2])
                inps.append(weight_chunk)
            chunk_op = ToDMatrix(data=data_chunk, label=label_chunk, missing=op.missing,
                                 weight=weight_chunk, feature_names=op.feature_names,
                                 feature_types=op.feature_types, multi_output=False,
                                 output_types=op.output_types)
            kws = data_chunk.params
            kws['index'] = (next(ind), 0)
            out_chunks.append(chunk_op.new_chunk(inps, **kws))
        nsplits = (tuple(c.shape[0] for c in out_chunks), (out_chunks[0].shape[1],))

        new_op = op.copy()
        kw = op.outputs[0].params
        kw['chunks'] = out_chunks
        kw['nsplits'] = nsplits
        return new_op.new_tileables(op.inputs, kws=[kw])

    @classmethod
    def tile(cls, op):
        if op.multi_output:
            return (yield from cls._tile_multi_output(op))
        else:
            return cls._tile_single_output(op)

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
    def execute(cls, ctx, op):
        if op.multi_output:
            outs = op.outputs
            ctx[outs[0].key] = ctx[op.inputs[0].key]
            if op.label is not None:
                ctx[outs[1].key] = ctx[op.inputs[1].key]
            if op.weight is not None:
                ctx[outs[-1].key] = ctx[op.inputs[-1].key]
            return
        else:
            ctx[op.outputs[0].key] = (
                cls._from_ctx_if_not_none(ctx, op.data),
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
               feature_names=None, feature_types=None, session=None, run_kwargs=None):
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

    op = ToDMatrix(data=data, label=label, missing=missing, weight=weight,
                   feature_names=feature_names, feature_types=feature_types,
                   gpu=data.op.gpu, multi_output=True,
                   output_types=get_output_types(data, label, weight))
    outs = ExecutableTuple(op())
    # Execute first, to make sure the counterpart chunks of data, label and weight are co-allocated
    outs.execute(session=session, **(run_kwargs or dict()))

    data = outs[0]
    label = None if op.label is None else outs[1]
    weight = None if op.weight is None else outs[-1]
    # If not multiple outputs, try to collect the chunks on same worker into one
    # to feed the data into XGBoost for training.
    op = ToDMatrix(data=data, label=label, missing=missing, weight=weight,
                   feature_names=feature_names, feature_types=feature_types,
                   gpu=data.op.gpu, multi_output=False,
                   output_types=get_output_types(data))
    return op()


MarsDMatrix = to_dmatrix
