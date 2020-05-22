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

import numpy as np

from ..serialize import DataTypeField
from ..operands import Operand, TileableOperandMixin, HasInput, ShuffleProxy, MapReduceOperand, Fuse
from ..utils import calc_nsplits
from .core import TensorData, Tensor, SparseTensor, TensorChunkData, TensorChunk, TensorOrder


class TensorOperandMixin(TileableOperandMixin):
    __slots__ = ()
    _op_module_ = 'tensor'

    @staticmethod
    def _get_dtype(kw, i):
        dtype = kw.pop('dtype', None)
        return dtype[i] if isinstance(dtype, (list, tuple)) else dtype

    def _get_order(self, kw, i):
        inputs = self._inputs or []
        order = kw.pop('order', None)
        if order is None:
            if len(inputs) == 0:
                order = TensorOrder.C_ORDER
            elif all(hasattr(inp, 'order') and inp.order == TensorOrder.F_ORDER
                     for inp in inputs):
                order = TensorOrder.F_ORDER
            else:
                order = TensorOrder.C_ORDER

        return order[i] if isinstance(order, (list, tuple)) else order

    def _create_chunk(self, output_idx, index, **kw):
        dt = self._get_dtype(kw, output_idx)
        order = self._get_order(kw, output_idx)
        shape = kw.pop('shape', None)
        data = TensorChunkData(shape=shape, index=index, op=self, dtype=dt, order=order, **kw)
        return TensorChunk(data)

    def _create_tileable(self, output_idx, **kw):
        tensor_cls = SparseTensor if getattr(self, 'issparse')() else Tensor
        dt = self._get_dtype(kw, output_idx)
        order = self._get_order(kw, output_idx)
        nsplits = kw.pop('nsplits', None)
        shape = kw.pop('shape', None)
        chunks = kw.pop('chunks', None)
        if nsplits is not None:
            kw['nsplits'] = nsplits
            if shape is not None and any(np.isnan(s) for s in shape):
                # in the situation that `nan` in shape,
                # but not in nsplits
                shape = tuple(sum(ns) for ns in nsplits)
        data = TensorData(shape=shape, dtype=dt, order=order,
                          op=self, chunks=chunks, **kw)
        return tensor_cls(data)

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

    def get_fetch_op_cls(self, _):
        from ..operands import ShuffleProxy
        from .fetch import TensorFetchShuffle, TensorFetch

        if isinstance(self, ShuffleProxy):
            return TensorFetchShuffle
        else:
            return TensorFetch

    def get_fuse_op_cls(self, _):
        from .fuse import TensorFuseChunk

        return TensorFuseChunk


class TensorOperand(Operand):
    _dtype = DataTypeField('dtype')

    @property
    def dtype(self):
        return getattr(self, '_dtype', None)


class TensorHasInput(HasInput):
    _dtype = DataTypeField('dtype')

    @property
    def dtype(self):
        return getattr(self, '_dtype', None)


class TensorShuffleProxy(ShuffleProxy, TensorOperandMixin):
    _dtype = DataTypeField('dtype')

    def __init__(self, dtype=None, **kwargs):
        kwargs['_dtype'] = kwargs.get('_dtype', dtype)
        super().__init__(**kwargs)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None)


class TensorMapReduceOperand(MapReduceOperand):
    _dtype = DataTypeField('dtype')

    @property
    def dtype(self):
        return getattr(self, '_dtype', None)


class TensorFuse(Fuse):
    _dtype = DataTypeField('dtype')

    @property
    def dtype(self):
        return getattr(self, '_dtype', None)
