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

import numexpr as ne
import numpy as np

from ...compat import six, izip
from ..expressions import arithmetic, reduction
from ..expressions.fuse import TensorNeFuseChunk
from .array import as_same_device
from .utils import estimate_fuse_size

_VAR_FLAG = 'V_'

NE_UNARYOP_TO_STRING = {
    arithmetic.TensorNegative: '-',
    arithmetic.TensorAbs: 'abs',
    arithmetic.TensorConj: 'conj',
    arithmetic.TensorExp: 'exp',
    arithmetic.TensorLog: 'log',
    arithmetic.TensorLog10: 'log10',
    arithmetic.TensorExpm1: 'expm1',
    arithmetic.TensorLog1p: 'log1p',
    arithmetic.TensorSqrt: 'sqrt',

    arithmetic.TensorSin: 'sin',
    arithmetic.TensorCos: 'cos',
    arithmetic.TensorTan: 'tan',
    arithmetic.TensorArcsin: 'arcsin',
    arithmetic.TensorArccos: 'arccos',
    arithmetic.TensorArctan: 'arctan',
    arithmetic.TensorSinh: 'sinh',
    arithmetic.TensorCosh: 'cosh',
    arithmetic.TensorTanh: 'tanh',
    arithmetic.TensorArcsinh: 'arcsinh',
    arithmetic.TensorArccosh: 'arccosh',
    arithmetic.TensorArctanh: 'arctanh'
}


NE_BINOP_TO_STRING = {
    arithmetic.TensorAdd: '+',
    arithmetic.TensorAddConstant: '+',
    arithmetic.TensorTreeAdd: '+',
    arithmetic.TensorSubtract: '-',
    arithmetic.TensorSubConstant: '-',
    arithmetic.TensorMultiply: '*',
    arithmetic.TensorMulConstant: '*',
    arithmetic.TensorTreeMultiply: '*',
    arithmetic.TensorDivide: '/',
    arithmetic.TensorDivConstant: '/',
    arithmetic.TensorMod: '%',
    arithmetic.TensorModConstant: '%',
    arithmetic.TensorPower: '**',
    arithmetic.TensorPowConstant: '**',
    arithmetic.TensorLshift: '<<',
    arithmetic.TensorLshiftConstant: '<<',
    arithmetic.TensorRshift: '>>',
    arithmetic.TensorRshiftConstant: '>>',

    arithmetic.TensorEqual: '==',
    arithmetic.TensorEqConstant: '==',
    arithmetic.TensorNotEqual: '!=',
    arithmetic.TensorNeConstant: '!=',
    arithmetic.TensorLessThan: '<',
    arithmetic.TensorLtConstant: '<',
    arithmetic.TensorLessEqual: '<=',
    arithmetic.TensorLeConstant: '<=',
    arithmetic.TensorGreaterThan: '>',
    arithmetic.TensorGtConstant: '>',
    arithmetic.TensorGreaterEqual: '>=',
    arithmetic.TensorGeConstant: '>='
}


NE_REDUCTION_TO_STRING = {
    reduction.TensorSum: 'sum',
    reduction.TensorProd: 'prod',
    reduction.TensorMax: 'max',
    reduction.TensorMin: 'min'
}


def _handle_unary(chunk):
    if len(chunk.inputs) != 1:
        raise ValueError('unary operand inputs should be 1')
    data = chunk.inputs[0]
    unary_op = NE_UNARYOP_TO_STRING[type(chunk.op)]
    _expr = '{}({})'.format(unary_op, _VAR_FLAG + data.key)
    return _expr


def _decompose(chunk):
    expr = _VAR_FLAG + chunk.key
    for node, op in zip(reversed(chunk.composed), reversed(chunk.op.operands)):
        _expr = _evalute(node)
        expr = expr.replace(_VAR_FLAG + node.key, '({})'.format(_expr))
    return expr


def _handle_bin(chunk):
    const = getattr(chunk.op, 'constant', [])
    reverse = getattr(chunk.op, 'reverse', False)
    op = NE_BINOP_TO_STRING[type(chunk.op)]

    exprs = [op.join(_VAR_FLAG + c.key for c in chunk.inputs), op.join(str(c) for c in const)]

    if reverse:
        exprs = list(reversed(exprs))

    return op.join(filter(None, exprs))


def _wrap_bool(data):
    if data.dtype == np.bool_:
        return lambda x: 'where({0}, 1, 0)'.format(x)

    return lambda x: x


def _handle_reduction(chunk):
    ax = chunk.op.axis
    data = chunk.inputs[0]
    op_str = NE_REDUCTION_TO_STRING[type(chunk.op)]
    # TODO(hks): delete it if numexpr.sum fixed
    if len(ax) == data.ndim:
        _expr = '{}({})'.format(op_str, _wrap_bool(data)(_VAR_FLAG + data.key))
    elif len(ax) == 1:
        if data.shape[ax[0]] == 1:
            _expr = _VAR_FLAG + data.key
        else:
            _expr = '{}({},axis={})'.format(op_str, _wrap_bool(data)(_VAR_FLAG + data.key), ax[0])
    else:
        raise ValueError("numexpr cannot encode axis")
    return _expr


def _evalute(chunk):
    if type(chunk.op) in NE_UNARYOP_TO_STRING:
        return _handle_unary(chunk)
    elif type(chunk.op) in NE_BINOP_TO_STRING:
        return _handle_bin(chunk)
    elif type(chunk.op) in NE_REDUCTION_TO_STRING:
        return _handle_reduction(chunk)
    elif type(chunk.op) == TensorNeFuseChunk:
        return _decompose(chunk)
    else:
        raise TypeError("unsupported operator in numexpr: {}".format(chunk.op.__class__.__name__))


def _maybe_keepdims(chunk, res):
    out_chunk = chunk.composed[-1] if type(chunk.op) == TensorNeFuseChunk else chunk
    if type(out_chunk.op) in NE_REDUCTION_TO_STRING and out_chunk.op.keepdims:
        res = np.reshape(res, out_chunk.shape)
    return res


def evaluate(ctx, chunk):
    inputs = as_same_device([ctx[c.key] for c in chunk.inputs], device=chunk.op.device)
    for c, i in izip(chunk.inputs, inputs):
        six.exec_('V_' + c.key + ' = i')
    expr = _evalute(chunk)
    res = ne.evaluate(expr)
    res = _maybe_keepdims(chunk, res)
    ctx[chunk.key] = res


def register_numexpr_handler():
    from ...executor import register

    register(TensorNeFuseChunk, evaluate, estimate_fuse_size)
