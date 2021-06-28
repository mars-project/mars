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

from ..serialization.serializables import DataTypeField
from ..core import OutputType
from ..core.operand import Operand, TileableOperandMixin, HasInput, \
    ShuffleProxy, MapReduceOperand, Fuse
from ..utils import calc_nsplits


class TensorOperandMixin(TileableOperandMixin):
    __slots__ = ()
    _op_module_ = 'tensor'
    _output_type_ = OutputType.tensor

    def new_tensors(self, inputs, shape=None, dtype=None, order=None, chunks=None, nsplits=None,
                    output_limit=None, kws=None, **kw):
        return self.new_tileables(inputs, shape=shape, chunks=chunks, nsplits=nsplits,
                                  output_limit=output_limit, kws=kws, dtype=dtype, order=order, **kw)

    def new_tensor(self, inputs, shape, dtype=None, order=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new tensor with more than 1 outputs')
        return self.new_tensors(inputs, shape=shape, dtype=dtype, order=order, **kw)[0]

    @classmethod
    def concat_tileable_chunks(cls, tileable):
        from .merge.concatenate import TensorConcatenate

        tensor = tileable
        assert not tensor.is_coarse()

        op = TensorConcatenate(dtype=tensor.dtype)
        chunk = TensorConcatenate(dtype=tensor.dtype).new_chunk(
            tensor.chunks, shape=tensor.shape, index=(0,) * tileable.ndim)
        return op.new_tensor([tensor], tensor.shape, chunks=[chunk],
                             nsplits=tuple((s,) for s in tensor.shape))

    @classmethod
    def create_tileable_from_chunks(cls, chunks, inputs=None, **kw):
        chunk_idx_to_shape = {c.index: c.shape for c in chunks}
        nsplits = calc_nsplits(chunk_idx_to_shape)
        shape = tuple(sum(ns) for ns in nsplits)
        op = chunks[0].op.copy().reset_key()
        return op.new_tensor(inputs, shape=shape, chunks=chunks,
                             nsplits=nsplits, dtype=chunks[0].dtype, **kw)

    def get_fuse_op_cls(self, _):
        from .fuse import TensorFuseChunk

        return TensorFuseChunk


class TensorOperand(Operand):
    _output_type_ = OutputType.tensor

    dtype = DataTypeField('dtype', default=None)


class TensorHasInput(HasInput):
    _output_type_ = OutputType.tensor

    dtype = DataTypeField('dtype', default=None)


class TensorShuffleProxy(ShuffleProxy, TensorOperandMixin):
    _output_type_ = OutputType.tensor

    dtype = DataTypeField('dtype', default=None)

    @classmethod
    def execute(cls, ctx, op):
        pass


class TensorMapReduceOperand(MapReduceOperand):
    _output_type_ = OutputType.tensor

    dtype = DataTypeField('dtype', default=None)


class TensorFuse(Fuse):
    _output_type_ = OutputType.tensor

    dtype = DataTypeField('dtype', default=None)
