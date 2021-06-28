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

import sys
import threading

try:
    import numexpr as ne
    NUMEXPR_INSTALLED = True
except ImportError:
    ne = None
    NUMEXPR_INSTALLED = False
import numpy as np

from ...serialization.serializables import DataTypeField
from ..operands import TensorFuse
from .. import arithmetic, reduction
from ..array_utils import as_same_device
from .core import TensorFuseChunkMixin


class TensorNeFuseChunk(TensorFuse, TensorFuseChunkMixin):
    _op_type_ = None  # no opcode, cannot be serialized
    _dtype = DataTypeField('dtype')

    if sys.platform == 'win32':
        # since we found thread-safe problem for ne.evaluate
        # thus add a lock for windows
        _lock = threading.Lock()
    else:
        _lock = None

    # use for numexpr-fused operand
    def __init__(self, dtype=None, **kw):
        super().__init__(_dtype=dtype, **kw)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        inputs = as_same_device([ctx[c.key] for c in op.inputs], device=op.device)
        for c, i in zip(op.inputs, inputs):
            exec('V_' + c.key + ' = i')
        expr = _evaluate(chunk)
        if cls._lock is not None:
            cls._lock.acquire()
        try:
            res = ne.evaluate(expr)
        finally:
            if cls._lock is not None:
                cls._lock.release()
        res = _maybe_keepdims(chunk, res)
        if chunk.ndim == 0 and res.ndim == 1 and res.size == 0:
            res = res.dtype.type(0)
        ctx[chunk.key] = res


# execution part
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
    arithmetic.TensorSubtract: '-',
    arithmetic.TensorMultiply: '*',
    arithmetic.TensorDivide: '/',
    arithmetic.TensorMod: '%',
    arithmetic.TensorPower: '**',
    arithmetic.TensorLshift: '<<',
    arithmetic.TensorRshift: '>>',

    arithmetic.TensorEqual: '==',
    arithmetic.TensorNotEqual: '!=',
    arithmetic.TensorLessThan: '<',
    arithmetic.TensorLessEqual: '<=',
    arithmetic.TensorGreaterThan: '>',
    arithmetic.TensorGreaterEqual: '>=',
}

NE_TREE_OP_TO_STRING = {
    arithmetic.TensorTreeAdd: '+',
    arithmetic.TensorTreeMultiply: '*',
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
    _expr = f'{unary_op}({_VAR_FLAG + data.key})'
    return _expr


def _decompose(chunk):
    expr = _VAR_FLAG + chunk.key
    for node in reversed(chunk.composed):
        _expr = _evaluate(node)
        expr = expr.replace(_VAR_FLAG + node.key, f'({_expr})')
    return expr


def _handle_bin(chunk):
    lhs = str(chunk.op.lhs) if np.isscalar(chunk.op.lhs) else _VAR_FLAG + chunk.op.lhs.key
    rhs = str(chunk.op.rhs) if np.isscalar(chunk.op.rhs) else _VAR_FLAG + chunk.op.rhs.key
    reverse = getattr(chunk.op, 'reverse', False)
    op = NE_BINOP_TO_STRING[type(chunk.op)]
    if reverse:
        exprs = [rhs, lhs]
    else:
        exprs = [lhs, rhs]

    return op.join(exprs)


def _handle_tree(chunk):
    op = NE_TREE_OP_TO_STRING[type(chunk.op)]
    return op.join(_VAR_FLAG + c.key for c in chunk.inputs)


def _wrap_bool(data):
    if data.dtype == np.bool_:
        return lambda x: f'where({x}, 1, 0)'

    return lambda x: x


def _handle_reduction(chunk):
    ax = chunk.op.axis
    data = chunk.inputs[0]
    op_str = NE_REDUCTION_TO_STRING[type(chunk.op)]
    # TODO(hks): delete it if numexpr.sum fixed
    if len(ax) == data.ndim:
        _expr = f'{op_str}({_wrap_bool(data)(_VAR_FLAG + data.key)})'
    elif len(ax) == 1:
        _expr = f'{op_str}({_wrap_bool(data)(_VAR_FLAG + data.key)},axis={ax[0]})'
    else:
        raise ValueError("numexpr cannot encode axis")
    return _expr


def _evaluate(chunk):
    if type(chunk.op) in NE_UNARYOP_TO_STRING:
        return _handle_unary(chunk)
    elif type(chunk.op) in NE_BINOP_TO_STRING:
        return _handle_bin(chunk)
    elif type(chunk.op) in NE_TREE_OP_TO_STRING:
        return _handle_tree(chunk)
    elif type(chunk.op) in NE_REDUCTION_TO_STRING:
        return _handle_reduction(chunk)
    elif type(chunk.op) == TensorNeFuseChunk:
        return _decompose(chunk)
    else:
        raise TypeError(f"unsupported operator in numexpr: {type(chunk.op).__name__}")


def _maybe_keepdims(chunk, res):
    out_chunk = chunk.composed[-1] if type(chunk.op) == TensorNeFuseChunk else chunk
    if type(out_chunk.op) in NE_REDUCTION_TO_STRING and out_chunk.op.keepdims:
        res = np.reshape(res, out_chunk.shape)
    return res
