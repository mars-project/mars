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

from .... import operands
from ....tiles import NotSupportTile
from ..core import TensorOperandMixin


class TensorFuseChunk(operands.Fuse, TensorOperandMixin):
    def __init__(self, dtype=None, sparse=False, **kw):
        super(TensorFuseChunk, self).__init__(_dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def tile(cls, op):
        raise NotSupportTile('TensorFuseChunk is a chunk operand which does not support tile')


class TensorFuseChunkMixin(TensorOperandMixin):
    __slots__ = ()

    @classmethod
    def tile(cls, op):
        raise NotSupportTile('TensorFuseChunk is a chunk operand which does not support tile')

    def __call__(self, fuse_chunks):
        head_chunk = fuse_chunks[0]
        tail_chunk = fuse_chunks[-1]
        setattr(self, '_operands', [c.op for c in fuse_chunks])
        return self.new_chunk(head_chunk.inputs, tail_chunk.shape,
                              _composed=fuse_chunks, _key=tail_chunk.key)
