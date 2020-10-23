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
import logging

from .... import opcodes
from .... import tensor as mt
from ....context import get_context, RunningMode
from ....core import Base, Entity
from ....filesystem import get_fs, LocalFileSystem
from ....operands import OutputType, OperandStage
from ....serialize import KeyField, StringField, Int32Field, DictField
from ....tiles import TilesError
from ....tensor.utils import decide_unify_split
from ....utils import check_chunks_unknown_shape
from ...operands import LearnOperand, LearnOperandMixin
from ..core import proxima, get_proxima_type, validate_tensor

logger = logging.getLogger(__name__)


class ProximaBuilder(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.PROXIMA_SIMPLE_BUILDER

    _tensor = KeyField('tensor')  # doc
    _pk = KeyField('pk')  # doc_pk
    _distance_metric = StringField('distance_metric')
    _dimension = Int32Field('dimension')
    _index_path = StringField('index_path')
    _index_builder = StringField('index_builder')
    _index_builder_params = DictField('index_builder_params')
    _index_converter = StringField('index_converter')
    _index_converter_params = DictField('index_converter_params')
    _topk = Int32Field('topk')
    _storage_options = DictField('storage_options')

    def __init__(self, tensor=None, pk=None, distance_metric=None,
                 index_path=None, dimension=None,
                 index_builder=None, index_builder_params=None,
                 index_converter=None, index_converter_params=None,
                 topk=None, storage_options=None, output_types=None, stage=None, **kw):
        super().__init__(_tensor=tensor, _pk=pk,
                         _distance_metric=distance_metric, _index_path=index_path, _dimension=dimension,
                         _index_builder=index_builder, _index_builder_params=index_builder_params,
                         _index_converter=index_converter, _index_converter_params=index_converter_params,
                         _topk=topk, _storage_options=storage_options,
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
    def index_path(self):
        return self._index_path

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

    @property
    def topk(self):
        return self._topk

    @property
    def storage_options(self):
        return self._storage_options

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._tensor = self._inputs[0]
        self._pk = self._inputs[-1]

    def __call__(self, tensor, pk):
        return self.new_tileable([tensor, pk])

    @classmethod
    def _get_atleast_topk_nsplit(cls, nsplit, topk):
        new_nsplit = []
        i = 0
        while i < len(nsplit):
            cur = nsplit[i]
            i += 1
            if cur >= topk:
                new_nsplit.append(cur)
            else:
                while i < len(nsplit):
                    cur += nsplit[i]
                    i += 1
                    if cur >= topk:
                        break
                if cur < topk and len(new_nsplit) > 0:
                    new_nsplit[-1] += cur
                elif cur >= topk:
                    new_nsplit.append(cur)
        new_nsplit = tuple(new_nsplit)
        assert sum(new_nsplit) == sum(nsplit), f'sum of nsplit not equal, ' \
                                               f'old: {nsplit}, new: {new_nsplit}'

        return new_nsplit

    @classmethod
    def tile(cls, op):
        tensor = op.tensor
        pk = op.pk
        out = op.outputs[0]
        index_path = op.index_path
        ctx = get_context()

        # check index_path for distributed
        if getattr(ctx, 'running_mode', None) == RunningMode.distributed:
            if index_path is not None:
                fs = get_fs(index_path, op.storage_options)
                if isinstance(fs, LocalFileSystem):
                    raise ValueError('`index_path` cannot be local file dir for distributed index building')

        # make sure all inputs have known chunk sizes
        check_chunks_unknown_shape(op.inputs, TilesError)

        nsplit = decide_unify_split(tensor.nsplits[0], pk.nsplits[0])
        if op.topk is not None:
            nsplit = cls._get_atleast_topk_nsplit(nsplit, op.topk)

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

        logger.warning(f"index chunks count: {len(out_chunks)} ")

        params = out.params
        params['chunks'] = out_chunks
        params['nsplits'] = ((1,) * len(out_chunks),)
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def _execute_map(cls, ctx, op: "ProximaBuilder"):
        inp = ctx[op.tensor.key]
        out = op.outputs[0]
        pks = ctx[op.pk.key]
        proxima_type = get_proxima_type(inp.dtype)

        # holder
        holder = proxima.IndexHolder(type=proxima_type,
                                     dimension=op.dimension)
        for pk, record in zip(pks, inp):
            pk = pk.item() if hasattr(pk, 'item') else pk
            holder.emplace(pk, record.copy())

        # converter
        meta = proxima.IndexMeta(proxima_type, dimension=op.dimension,
                                 measure_name=op.distance_metric)
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
        dumper.close()

        if op.index_path is None:
            ctx[out.key] = path
        else:
            # write to external file system
            fs = get_fs(op.index_path, op.storage_options)
            filename = f'proxima-{out.index[0]}.index'
            out_path = f'{op.index_path.rstrip("/")}/{filename}'
            with fs.open(out_path, 'wb') as out_f:
                with open(path, 'rb') as in_f:
                    # 32M
                    chunk_bytes = 32 * 1024 ** 2
                    while True:
                        data = in_f.read(chunk_bytes)
                        if data:
                            out_f.write(data)
                        else:
                            break

            ctx[out.key] = filename

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


def build_index(tensor, pk, dimension=None, index_path=None,
                need_shuffle=False, distance_metric='SquaredEuclidean',
                index_builder='SsgBuilder', index_builder_params=None,
                index_converter=None, index_converter_params=None,
                topk=None, storage_options=None,
                run=True, session=None, run_kwargs=None):
    tensor = validate_tensor(tensor)

    if dimension is None:
        dimension = tensor.shape[1]
    if index_builder_params is None:
        index_builder_params = {}
    if index_converter_params is None:
        index_converter_params = {}

    if need_shuffle:
        tensor = mt.random.permutation(tensor)

    if not isinstance(pk, (Base, Entity)):
        pk = mt.tensor(pk)

    op = ProximaBuilder(tensor=tensor, pk=pk, distance_metric=distance_metric,
                        index_path=index_path, dimension=dimension,
                        index_builder=index_builder,
                        index_builder_params=index_builder_params,
                        index_converter=index_converter,
                        index_converter_params=index_converter_params,
                        topk=topk, storage_options=storage_options)
    result = op(tensor, pk)
    if run:
        return result.execute(session=session, **(run_kwargs or dict()))
    else:
        return result
