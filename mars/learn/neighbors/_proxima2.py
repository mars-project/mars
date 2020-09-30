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

import tempfile

import numpy as np
import pandas as pd
try:
    import pyproxima2 as pp
except ImportError:  # pragma: no cover
    pp = None

from ... import opcodes
from ... import tensor as mt
from ...core import Base, Entity
from ...operands import OutputType, OperandStage
from ...serialize import KeyField, StringField, Int32Field
from ...tiles import TilesError
from ...tensor.utils import decide_unify_split
from ...utils import check_chunks_unknown_shape
from ..operands import LearnOperand, LearnOperandMixin


if pp:
    _type_mapping = {
        np.dtype(np.float32): pp.IndexMeta.FT_FP32
    }


class Proxima2BuildIndex(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.PROXIMA2_TRAIN

    _tensor = KeyField('tensor')
    _pk = KeyField('pk')
    _metric = StringField('metric')
    _dimension = Int32Field('dimension')

    def __init__(self, tensor=None, pk=None, metric=None, dimension=None,
                 output_types=None, stage=None, **kw):
        super().__init__(_tensor=tensor, _pk=pk, _metric=metric,
                         _dimension=dimension, _output_types=output_types,
                         _stage=stage, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.object]

    @property
    def tensor(self):
        return self._tensor

    @property
    def pk(self):
        return self._pk

    @property
    def metric(self):
        return self._metric

    @property
    def dimension(self):
        return self._dimension

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._tensor = self._inputs[0]
        self._pk = self._inputs[-1]

    def __call__(self, tensor, pk):
        return self.new_tileable([tensor, pk])

    @classmethod
    def tile(cls, op):
        tensor = op.tensor
        pk = op.pk
        out = op.outputs[0]

        # make sure all inputs have known chunk sizes
        check_chunks_unknown_shape(op.inputs, TilesError)

        nsplit = decide_unify_split(tensor.nsplits[0], pk.nsplits[0])
        if tensor.chunk_shape[1] > 1:
            tensor = tensor.rechunk({0: nsplit, 1: tensor.shape[1]})._inplace_tile()
        else:
            tensor = tensor.rechunk({0: nsplit})._inplace_tile()
        pk = pk.rechunk({0: nsplit})._inplace_tile()

        out_chunks = []
        for chunk, pk_col_chunk in zip(tensor.chunks, pk.chunks):
            chunk_op = op.copy().reset_key()
            chunk_op._stage = OperandStage.map
            out_chunk = chunk_op.new_chunk([chunk, pk_col_chunk],
                                           index=pk_col_chunk.index)
            out_chunks.append(out_chunk)

        params = out.params
        params['chunks'] = out_chunks
        params['nsplits'] = ((1,) * len(out_chunks),)
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def _execute_map(cls, ctx, op: "Proxima2BuildIndex"):
        inp = ctx[op.tensor.key]
        pks = ctx[op.pk.key]

        holder = pp.IndexHolder(type=_type_mapping[inp.dtype],
                                dimension=op.dimension)
        for pk, record in zip(pks, inp):
            pk = pk.item() if hasattr(pk, 'item') else pk
            holder.emplace(pk, record.copy())

        builder = pp.IndexBuilder(name='SsgBuilder',
                                  meta=pp.IndexMeta(type=_type_mapping[inp.dtype],
                                                    dimension=op.dimension))
        builder = builder.train_and_build(holder)

        path = tempfile.mkstemp(prefix='pyproxima2-', suffix='.index')[1]
        dumper = pp.IndexDumper(name="FileDumper", path=path)
        builder.dump(dumper)

        ctx[op.outputs[0].key] = path

    @classmethod
    def _execute_agg(cls, ctx, op: "Proxima2BuildIndex"):
        paths = [ctx[inp.key] for inp in op.inputs]
        ctx[op.outputs[0].key] = paths

    @classmethod
    def execute(cls, ctx, op: "Proxima2BuildIndex"):
        if op.stage != OperandStage.agg:
            return cls._execute_map(ctx, op)
        else:
            return cls._execute_agg(ctx, op)

    @classmethod
    def concat_tileable_chunks(cls, tileable):
        assert not tileable.is_coarse()

        op = cls(stage=OperandStage.agg)
        chunk = cls(stage=OperandStage.agg).new_chunk(tileable.chunks)
        return op.new_tileable([tileable], chunks=[chunk], nsplits=((1,),))


class Proxima2SearchIndex(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.PROXIMA2_QUERY

    _tensor = KeyField('tensor')
    _pk = KeyField('pk')
    _metric = StringField('metric')
    _dimension = Int32Field('dimension')
    _topk = Int32Field('topk')
    _index = KeyField('index')

    def __init__(self, tensor=None, pk=None, metric=None, dimension=None,
                 index=None, topk=None, output_types=None, stage=None, **kw):
        super().__init__(_tensor=tensor, _pk=pk, _metric=metric,
                         _dimension=dimension, _output_types=output_types,
                         _index=index, _topk=topk, _stage=stage, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.dataframe]

    @property
    def tensor(self):
        return self._tensor

    @property
    def pk(self):
        return self._pk

    @property
    def metric(self):
        return self._metric

    @property
    def dimension(self):
        return self._dimension

    @property
    def index(self):
        return self._index

    @property
    def topk(self):
        return self._topk

    def _set_inputs(self, inputs): # to Graph Node
        super()._set_inputs(inputs)
        self._tensor = self._inputs[0]
        self._pk = self._inputs[1]
        self._index = self._inputs[2]

    def __call__(self, tensor, pk, index):
        return self.new_tileable([tensor, pk, index])

    @classmethod
    def tile(cls, op):
        # tensor = op.tensor
        # pk = op.pk
        # out = op.outputs[0]
        #
        # # make sure all inputs have known chunk sizes
        # check_chunks_unknown_shape(op.inputs, TilesError)
        #
        # nsplit = decide_unify_split(tensor.nsplits[0], pk.nsplits[0])
        # if tensor.chunk_shape[1] > 1:
        #     tensor = tensor.rechunk({0: nsplit, 1: tensor.shape[1]})._inplace_tile()
        # else:
        #     tensor = tensor.rechunk({0: nsplit})._inplace_tile()
        # pk = pk.rechunk({0: nsplit})._inplace_tile()
        #
        # out_chunks = []
        # for chunk, pk_col_chunk in zip(tensor.chunks, pk.chunks):
        #     chunk_op = op.copy().reset_key()
        #     chunk_op._stage = OperandStage.map
        #     out_chunk = chunk_op.new_chunk([chunk, pk_col_chunk],
        #                                    index=pk_col_chunk.index)
        #     out_chunks.append(out_chunk)
        #
        # params = out.params
        # params['chunks'] = out_chunks
        # params['nsplits'] = ((1,) * len(out_chunks),)
        # new_op = op.copy()
        # return new_op.new_tileables(op.inputs, kws=[params])
        pass

    @classmethod
    def _execute_map(cls, ctx, op: "Proxima2SearchIndex"):
        inp = ctx[op.tensor.key]
        pks = ctx[op.pk.key]
        index_path = ctx[op.index.key]

        container = pp.IndexContainer(name="FileContainer", path=index_path)
        searcher = pp.IndexSearcher("SsgSearcher")
        pp_ctx = searcher.load(container).create_context(topk=op.topk)

        # vec = np.random.uniform(low=0.0, high=1.0, size=self.dim).astype('f')
        vecs = np.ascontiguousarray(inp)
        results = ctx.search(query=vecs, count=len(vecs))

        result = pd.DataFrame()
        result[0] = np.empty()

        for it in results:
            for ele in it:
                ele.key(), ele.score()


        for ele in results[0]:
            print("Result: the key is: {0}\tscore is: {1}.".format(ele.key(), ele.score()))


    @classmethod
    def _execute_agg(cls, ctx, op: "Proxima2BuildIndex"):
        paths = [ctx[inp.key] for inp in op.inputs]
        ctx[op.outputs[0].key] = paths

    @classmethod
    def execute(cls, ctx, op: "Proxima2BuildIndex"):
        if op.stage != OperandStage.agg:
            return cls._execute_map(ctx, op)
        else:
            return cls._execute_agg(ctx, op)

    @classmethod
    def concat_tileable_chunks(cls, tileable):
        assert not tileable.is_coarse()

        op = cls(stage=OperandStage.agg)
        chunk = cls(stage=OperandStage.agg).new_chunk(tileable.chunks)
        return op.new_tileable([tileable], chunks=[chunk], nsplits=((1,),))


def build_proxima2_index(tensor, pk, shuffle=False, metric='l2',
                         session=None, run_kwargs=None):
    if hasattr(tensor, 'to_tensor'):
        tensor = tensor.to_tensor()
    else:
        tensor = mt.tensor(tensor)
    if tensor.ndim != 2:
        raise ValueError('Input tensor should be 2-d')
    if shuffle:
        tensor = mt.random.permutation(tensor)

    if not isinstance(pk, (Base, Entity)):
        pk = mt.tensor(pk)

    op = Proxima2BuildIndex(tensor=tensor, pk=pk,
                            metric=metric, dimension=tensor.shape[1])
    return op(tensor, pk).execute(session=session, **(run_kwargs or dict()))
