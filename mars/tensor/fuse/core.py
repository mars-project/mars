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

from ...tiles import NotSupportTile
from ...core import FuseChunkData, FuseChunk
from ..operands import TensorFuse, TensorOperandMixin


class TensorFuseChunkMixin(TensorOperandMixin):
    __slots__ = ()

    def _create_chunk(self, output_idx, index, **kw):
        data = FuseChunkData(_index=index, _op=self, **kw)
        return FuseChunk(data)

    @classmethod
    def tile(cls, op):
        raise NotSupportTile('TensorFuseChunk is a chunk operand which does not support tile')

    def __call__(self, fuse_chunks):
        head_chunk = fuse_chunks[0]
        tail_chunk = fuse_chunks[-1]
        setattr(self, '_operands', [c.op for c in fuse_chunks])
        return self.new_chunk(head_chunk.inputs, shape=tail_chunk.shape, order=tail_chunk.order,
                              _composed=fuse_chunks, _key=tail_chunk.key)


class TensorFuseChunk(TensorFuse, TensorFuseChunkMixin):
    def __init__(self, dtype=None, sparse=False, **kw):
        super(TensorFuseChunk, self).__init__(_dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def tile(cls, op):
        raise NotSupportTile('TensorFuseChunk is a chunk operand which does not support tile')


def estimate_fuse_size(ctx, op):
    from ...graph import DAG
    from ...executor import Executor

    chunk = op.outputs[0]
    dag = DAG()
    size_ctx = dict()
    keys = set(c.key for c in chunk.composed)
    for c in chunk.composed:
        dag.add_node(c)
        for inp in c.inputs:
            if inp.key not in keys:
                size_ctx[inp.key] = ctx[inp.key]
            if inp not in dag:
                dag.add_node(inp)
            dag.add_edge(inp, c)

    executor = Executor(storage=size_ctx)
    output_keys = [o.key for o in op.outputs]
    results = executor.execute_graph(dag, output_keys, mock=True, no_intermediate=True)
    ctx.update(zip(output_keys, results))

    # update with the maximal memory cost during the whole execution
    total_mem = sum(ctx[key][1] for key in output_keys)
    if total_mem:
        for key in output_keys:
            r = ctx[key]
            ctx[key] = (r[0], max(r[1], r[1] * executor.mock_max_memory // total_mem))
