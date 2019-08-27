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
from ...serialize import DataTypeField
from ..operands import TensorFuse
from ..array_utils import as_same_device
from .core import TensorFuseChunkMixin, estimate_fuse_size
import numpy as np

try:
    import jax  # noqa # pylint: disable=unused-import

    JAX_INSTALLED = True
except ImportError:  # pragma: no cover
    JAX_INSTALLED = False


class TensorJaxFuseChunk(TensorFuse, TensorFuseChunkMixin):
    _op_type_ = None  # no opcode, cannot be serialized
    _dtype = DataTypeField('dtype')

    # use for jax-fused operand
    def __init__(self, dtype=None, **kw):
        super(TensorJaxFuseChunk, self).__init__(_dtype=dtype, **kw)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        inputs = as_same_device([ctx[c.key] for c in op.inputs], device=op.device)

        def combine_functions(inputs):
            inputs = functions[0](*inputs)
            for f in functions[1:]:
                inputs = f(inputs)
            return inputs

        # execute the fuse operands in jax
        functions = [operand.jax_function() for operand in op.operands]
        jit_func = jax.jit(combine_functions)
        ctx[chunk.key] = np.asarray(jit_func(inputs))

    @classmethod
    def estimate_size(cls, ctx, op):
        estimate_fuse_size(ctx, op)
