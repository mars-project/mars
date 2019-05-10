#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from __future__ import absolute_import

from ..core import TensorData, Tensor, SparseTensor, TensorChunkData, TensorChunk
from ...core import TileableOperandMixin
from ...operands import ShuffleProxy


class TensorOperandMixin(TileableOperandMixin):
    __slots__ = ()
    _op_module_ = 'tensor'

    @staticmethod
    def _get_dtype(kw, i):
        dtype = kw.pop('dtype', None)
        return dtype[i] if isinstance(dtype, (list, tuple)) else dtype

    def _create_chunk(self, output_idx, index, **kw):
        dt = self._get_dtype(kw, output_idx)
        shape = kw.pop('shape', None)
        data = TensorChunkData(_index=index, _shape=shape, _op=self,
                               _dtype=dt, **kw)
        return TensorChunk(data)

    def _create_tileable(self, output_idx, **kw):
        tensor_cls = SparseTensor if getattr(self, 'issparse')() else Tensor
        dt = self._get_dtype(kw, output_idx)
        nsplits = kw.pop('nsplits', None)
        shape = kw.pop('shape', None)
        chunks = kw.pop('chunks', None)
        if nsplits is not None:
            kw['_nsplits'] = nsplits
        data = TensorData(_shape=shape, _dtype=dt, _op=self, _chunks=chunks, **kw)
        return tensor_cls(data)

    def new_tensors(self, inputs, shape=None, dtype=None, chunks=None, nsplits=None,
                    output_limit=None, kws=None, **kw):
        return self.new_tileables(inputs, shape=shape, chunks=chunks, nsplits=nsplits,
                                  output_limit=output_limit, kws=kws, dtype=dtype, **kw)

    def new_tensor(self, inputs, shape, dtype=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new tensor with more than 1 outputs')

        return self.new_tensors(inputs, shape=shape, dtype=dtype, **kw)[0]

    def calc_shape(self, *inputs_shape):
        raise NotImplementedError


class TensorShuffleProxy(ShuffleProxy, TensorOperandMixin):
    def __init__(self, dtype=None, **kwargs):
        kwargs['_dtype'] = kwargs.get('_dtype', dtype)
        super(TensorShuffleProxy, self).__init__(**kwargs)

    def calc_shape(self, *inputs_shape):
        return ()
