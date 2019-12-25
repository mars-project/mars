#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from ...operands import FuseChunkMixin
from ..operands import TensorFuse, TensorOperandMixin


class TensorFuseChunkMixin(FuseChunkMixin, TensorOperandMixin):
    __slots__ = ()


class TensorFuseChunk(TensorFuse, TensorFuseChunkMixin):
    def __init__(self, dtype=None, sparse=False, **kw):
        super().__init__(_dtype=dtype, _sparse=sparse, **kw)


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
