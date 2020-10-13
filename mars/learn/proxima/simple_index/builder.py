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

from .... import opcodes
from .... import tensor as mt
from ....core import Base, Entity
from ....operands import OutputType, OperandStage
from ....serialize import KeyField, StringField, Int32Field, DictField
from ....tiles import TilesError
from ....tensor.utils import decide_unify_split
from ....utils import check_chunks_unknown_shape
from ...operands import LearnOperand, LearnOperandMixin
from ..core import proxima, get_proxima_type, validate_tensor


class ProximaBuilder(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.PROXIMA_SIMPLE_BUILDER

    _tensor = KeyField('tensor')  # doc
    _pk = KeyField('pk')  # doc_pk
    _distance_metric = StringField('distance_metric')
    _dimension = Int32Field('dimension')
    _index_builder = StringField('index_builder')
    _index_builder_params = DictField('index_builder_params')
    _index_converter = StringField('index_converter')
    _index_converter_params = DictField('index_converter_params')

    def __init__(self, tensor=None, pk=None, distance_metric=None, dimension=None,
                 index_builder=None, index_builder_params=None,
                 index_converter=None, index_converter_params=None,
                 output_types=None, stage=None, **kw):
        super().__init__(_tensor=tensor, _pk=pk,
                         _distance_metric=distance_metric, _dimension=dimension,
                         _index_builder=index_builder, _index_builder_params=index_builder_params,
                         _index_converter=index_converter, _index_converter_params=index_converter_params,
                         _output_types=output_types, _stage=stage, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.object]

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
    def index_builder(self):
        return self._index_builder

    @property
    def index_builder_params(self):
        return self._index_builder_params

    @property
    def index_converter(self):
        return self._index_converter

    @property
    def index_converter_params(self):
        return self._index_converter_params

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
    def _execute_map(cls, ctx, op: "ProximaBuilder"):
        inp = ctx[op.tensor.key]
        pks = ctx[op.pk.key]

        # holder
        holder = proxima.IndexHolder(type=get_proxima_type(inp.dtype),
                                     dimension=op.dimension)
        for pk, record in zip(pks, inp):
            pk = pk.item() if hasattr(pk, 'item') else pk
            holder.emplace(pk, record.copy())

        # converter
        meta = proxima.IndexMeta(proxima.IndexMeta.FT_FP32, dimension=op.dimension)
        if op.index_converter is not None:
            converter = proxima.IndexConverter(name=op.index_converter,
                                               meta=meta, params=op.index_converter_params)
            converter.train_and_transform(holder)
            holder = converter.result()
            meta = converter.meta()

        # builder && dumper
        builder = proxima.IndexBuilder(name=op.index_builder,
                                       meta=meta,
                                       params=op.index_builder_params)
        builder = builder.train_and_build(holder)

        path = tempfile.mkstemp(prefix='proxima-', suffix='.index')[1]
        dumper = proxima.IndexDumper(name="FileDumper", path=path)
        builder.dump(dumper)

        ctx[op.outputs[0].key] = path

    @classmethod
    def _execute_agg(cls, ctx, op: "ProximaBuilder"):
        paths = [ctx[inp.key] for inp in op.inputs]
        ctx[op.outputs[0].key] = paths

    @classmethod
    def execute(cls, ctx, op: "ProximaBuilder"):
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


def build_index(tensor, pk, dimension, need_shuffle=False, distance_metric='L2',
                index_builder='SsgBuilder', index_builder_params={},
                index_converter=None, index_converter_params={},
                session=None, run_kwargs=None):
    tensor = validate_tensor(tensor)
    if need_shuffle:
        tensor = mt.random.permutation(tensor)

    if not isinstance(pk, (Base, Entity)):
        pk = mt.tensor(pk)

    op = ProximaBuilder(tensor=tensor, pk=pk,
                        distance_metric=distance_metric, dimension=dimension,
                        index_builder=index_builder, index_builder_params=index_builder_params,
                        index_converter=index_converter, index_converter_params=index_converter_params)
    return op(tensor, pk).execute(session=session, **(run_kwargs or dict()))
