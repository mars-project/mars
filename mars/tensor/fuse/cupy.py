#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from string import ascii_letters

from ...utils import tokenize
from ..operands import TensorFuse
from .. import arithmetic
from .core import TensorFuseChunkMixin


class TensorCpFuseChunk(TensorFuse, TensorFuseChunkMixin):
    # use for cupy-fused operand
    _op_type_ = None  # no opcode, cannot be serialized

    @classmethod
    def execute(cls, ctx, op):
        import cupy as cp

        chunk = op.outputs[0]
        func = cp.ElementwiseKernel(*_evaluate(chunk))
        ctx[chunk.key] = func(*[ctx[i.key] for i in op.inputs])


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
    input_arguments = ', '.join([f'{tp} {input_names[i]}'
                                 for i, tp in zip(chunk.op.inputs, input_types)])
    output_type = chunk.op.dtype.name
    output_name = next(letters)
    output_argument = f'{output_type} {output_name}'
    body = dict(input_names)

    for node in chunk.composed:
        op_cls = type(node.op)
        if op_cls in CP_BINOP_TO_STRING:
            input_bodies = [body.get(i, repr(i)) for i in (node.op.lhs, node.op.rhs)]
            body[node] = f' {CP_BINOP_TO_STRING[op_cls]} '.join(input_bodies)
        elif op_cls in CP_UNARYOP_TO_STRING:
            input_data = body[node.op.inputs[0]]
            body[node] = f'{CP_UNARYOP_TO_STRING[op_cls]}({input_data})'
        else:
            raise NotImplementedError

    body = f'{output_name} = {body[chunk.composed[-1]]}'
    key = tokenize(input_arguments, output_argument, body)
    return input_arguments, output_argument, body, f'{type(chunk.op).__name__.lower()}_{key}'
