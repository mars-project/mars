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

from ..core import OutputType
from ..core.operand import Operand, TileableOperandMixin, Fuse, FuseChunkMixin, \
    ShuffleProxy
from ..tensor.core import TENSOR_TYPE, TENSOR_CHUNK_TYPE
from ..tensor.operands import TensorOperandMixin
from ..tensor.fuse import TensorFuseChunk
from ..dataframe.core import TILEABLE_TYPE as DATAFRAME_TYPE, \
    CHUNK_TYPE as DATAFRAME_CHUNK_TYPE
from ..dataframe.operands import DataFrameOperandMixin, DataFrameFuseChunk


LearnOperand = Operand


class LearnOperandMixin(TileableOperandMixin):
    __slots__ = ()
    _op_module_ = 'learn'

    @classmethod
    def concat_tileable_chunks(cls, tileable):
        if isinstance(tileable, TENSOR_TYPE):
            return TensorOperandMixin.concat_tileable_chunks(tileable)
        elif isinstance(tileable, DATAFRAME_TYPE):
            return DataFrameOperandMixin.concat_tileable_chunks(tileable)
        else:
            # op has to implement its logic of `concat_tileable_chunks`
            raise NotImplementedError

    @classmethod
    def create_tileable_from_chunks(cls, chunks, inputs=None, **kw):
        if isinstance(chunks[0], TENSOR_CHUNK_TYPE):
            return TensorOperandMixin.create_tileable_from_chunks(
                chunks, inputs=inputs, **kw)
        elif isinstance(chunks[0], DATAFRAME_CHUNK_TYPE):
            return DataFrameOperandMixin.create_tileable_from_chunks(
                chunks, inputs=inputs, **kw)
        else:
            # op has to implement its logic of `create_tileable_from_chunks`
            raise NotImplementedError

    def get_fuse_op_cls(self, obj):
        if isinstance(obj, (TENSOR_TYPE, TENSOR_CHUNK_TYPE)):
            return TensorFuseChunk
        elif isinstance(obj, (DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)):
            return DataFrameFuseChunk
        else:
            return LearnObjectFuseChunk


class LearnObjectFuseChunkMixin(FuseChunkMixin, LearnOperandMixin):
    __slots__ = ()

    _output_type_ = OutputType.object


class LearnObjectFuseChunk(LearnObjectFuseChunkMixin, Fuse):
    pass


class LearnShuffleProxy(ShuffleProxy, LearnOperandMixin):
    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)
        if not self.output_types:
            self.output_types = [OutputType.object]

    @classmethod
    def execute(cls, ctx, op):
        pass
