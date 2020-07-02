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

from string import ascii_letters

from ...utils import tokenize
from ..operands import TensorFuse
from .. import arithmetic
from .core import TensorFuseChunkMixin, estimate_fuse_size


class TensorCpFuseChunk(TensorFuse, TensorFuseChunkMixin):
    _op_type_ = None  # no opcode, cannot be serialized

    # use for cupy-fused operand
    def __init__(self, dtype=None, **kw):
        super().__init__(_dtype=dtype, **kw)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None)

    @classmethod
    def execute(cls, ctx, op):
        import cupy as cp

        chunk = op.outputs[0]
        func = cp.ElementwiseKernel(*_evaluate(chunk))
        ctx[chunk.key] = func(*[ctx[i.key] for i in op.inputs])

    @classmethod
    def estimate_size(cls, ctx, op):
        estimate_fuse_size(ctx, op)


# execution part
CP_BINOP_TO_STRING = {
    arithmetic.TensorSubtract: '-',
    arithmetic.TensorMultiply: '*',
    arithmetic.TensorTrueDiv: '/',
}

CP_UNARYOP_TO_STRING = {
    arithmetic.TensorSqrt: 'sqrt',
}


def _evaluate(chunk):
    letters = iter(letter for letter in ascii_letters if letter not in 'ni')

    input_types = [i.dtype.name for i in chunk.op.inputs]
    input_names = {i: next(letters) for i in chunk.op.inputs}
    input_arguments = ', '.join(['{0} {1}'.format(tp, input_names[i])
                                 for i, tp in zip(chunk.op.inputs, input_types)])
    output_type = chunk.op.dtype.name
    output_name = next(letters)
    output_argument = '{0} {1}'.format(output_type, output_name)
    body = dict(input_names)

    for node in chunk.composed:
        if type(node.op) in CP_BINOP_TO_STRING:
            input_bodies = [body.get(i, repr(i)) for i in (node.op.lhs, node.op.rhs)]
            body[node] = ' {0} '.format(CP_BINOP_TO_STRING[type(node.op)]).join(input_bodies)
        elif type(node.op) in CP_UNARYOP_TO_STRING:
            body[node] = '{0}({1})'.format(CP_UNARYOP_TO_STRING[type(node.op)], body[node.op.inputs[0]])
        else:
            raise NotImplementedError

    body = '{0} = {1}'.format(output_name, body[chunk.composed[-1]])
    return input_arguments, output_argument, body, \
        '{0}_{1}'.format(chunk.op.__class__.__name__.lower(),
                         tokenize(input_arguments, output_argument, body))
