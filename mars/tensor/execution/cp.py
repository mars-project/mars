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

from string import ascii_letters

from .utils import estimate_fuse_size
from ..expressions import arithmetic
from ..expressions.fuse import TensorCpFuseChunk
from ...utils import tokenize


CP_BINOP_TO_STRING = {
    arithmetic.TensorSubtract: '-',
    arithmetic.TensorSubConstant: '-',
    arithmetic.TensorMultiply: '*',
    arithmetic.TensorMulConstant: '*',
    arithmetic.TensorTrueDiv: '/',
}

CP_UNARYOP_TO_STRING = {
    arithmetic.TensorSqrt: 'sqrt',
}


def _evaluate(chunk):
    letters = iter(l for l in ascii_letters if l not in 'ni')

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


def _execute_cp(ctx, chunk):
    import cupy as cp

    func = cp.ElementwiseKernel(*_evaluate(chunk))
    ctx[chunk.key] = func(*[ctx[i.key] for i in chunk.op.inputs])


def register_cp_handler():
    from .core import register

    register(TensorCpFuseChunk, _execute_cp, estimate_fuse_size)
