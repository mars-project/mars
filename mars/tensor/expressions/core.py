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

import numpy as np

from ..core import TensorData, Tensor, SparseTensor, ChunkData, Chunk


class TensorOperandMixin(object):
    __slots__ = ()
    _op_module_ = 'tensor'

    def check_inputs(self, inputs):
        pass

    def new_chunks(self, inputs, shape, index=None, output_limit=None, kws=None, dtype=None, **kw):
        output_limit = getattr(self, 'output_limit') if output_limit is None else output_limit

        self.check_inputs(inputs)
        getattr(self, '_set_inputs')(inputs)
        if getattr(self, '_key', None) is None:
            getattr(self, '_update_key')()  # update key when inputs are set

        if isinstance(shape, (list, tuple)) and len(shape) > 0 and isinstance(shape[0], (list, tuple)):
            if len(shape) != output_limit:
                raise ValueError('shape size must be equal to output limit, expect {0}, got {1}'.format(
                    output_limit, len(shape)))
        else:
            shape = [shape] * output_limit

        if kws is not None and kw:
            raise ValueError('can only pass kws or kw')

        chunks = []
        raw_index = index
        for i, s in enumerate(shape):
            dt = None
            if kws:
                kw = kws[i]
                index = kw.pop('index', raw_index)
                dt = kw.pop('dtype', None)
            if dt is None:
                dt = dtype[i] if isinstance(dtype, (tuple, list)) else dtype
            data = ChunkData(_index=index, _shape=s, _op=self, _dtype=dt, **kw)
            chunks.append(Chunk(data))

        setattr(self, 'outputs', chunks)
        if len(chunks) > 1:
            # for each output chunk, hold the reference to the other outputs
            # so that either no one or everyone are gc collected
            for j, t in enumerate(chunks):
                t.data._siblings = [c.data for c in chunks[:j] + chunks[j + 1:]]
        return chunks

    def new_tensors(self, inputs, shape, dtype=None, chunks=None, nsplits=None,
                    output_limit=None, kws=None, **kw):
        tensor_cls = SparseTensor if getattr(self, 'issparse')() else Tensor
        output_limit = getattr(self, 'output_limit') if output_limit is None else output_limit

        self.check_inputs(inputs)
        getattr(self, '_set_inputs')(inputs)
        if getattr(self, '_key', None) is None:
            getattr(self, '_update_key')()  # update key when inputs are set

        if isinstance(shape, (list, tuple)) and len(shape) > 0 and isinstance(shape[0], (list, tuple)):
            if not np.isinf(output_limit) and len(shape) != output_limit:
                raise ValueError('shape size must be equal to output limit, expect {0}, got {1}'.format(
                    output_limit, len(shape)))
        else:
            shape = [shape] * output_limit

        if kws is not None and kw:
            raise ValueError('can only pass kws or kw')

        tensors = []
        raw_chunks = chunks
        raw_nsplits = nsplits
        for i, s in enumerate(shape):
            dt = None
            if kws:
                kw = kws[i]
                chunks = kw.pop('chunks', raw_chunks)
                nsplits = kw.pop('nsplits', raw_nsplits)
                dt = kw.pop('dtype', None)
            if nsplits is not None:
                kw['_nsplits'] = nsplits
            if dt is None:
                dt = dtype[i] if isinstance(dtype, (tuple, list)) else dtype
            data = TensorData(_shape=s, _dtype=dt, _op=self,
                              _chunks=chunks, **kw)
            tensors.append(tensor_cls(data))

        setattr(self, 'outputs', tensors)
        if len(tensors) > 1:
            # for each output tensor, hold the reference to the other outputs
            # so that either no one or everyone are gc collected
            for i, t in enumerate(tensors):
                t.data._siblings = [tensor.data for tensor in tensors[:i] + tensors[i+1:]]
        return tensors

    def new_chunk(self, inputs, shape, index=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new chunk with more than 1 outputs')

        return self.new_chunks(inputs, shape, index=index, **kw)[0]

    def new_tensor(self, inputs, shape, dtype=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new chunk with more than 1 outputs')

        return self.new_tensors(inputs, shape, dtype=dtype, **kw)[0]
