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

from collections import defaultdict
from itertools import count

try:
    import numexpr as ne

    NUMEXPR_INSTALLED = True
except ImportError:
    ne = None
    NUMEXPR_INSTALLED = False
import numpy as np

from ..operands import TensorFuse
from .. import arithmetic, reduction
from ..array_utils import as_same_device
from .core import TensorFuseChunkMixin


class TensorNeFuseChunk(TensorFuse, TensorFuseChunkMixin):
    _op_type_ = None  # no opcode, cannot be serialized

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        inputs = as_same_device([ctx[c.key] for c in op.inputs], device=op.device)
        counter = count()
        # Unified the var names to V_0, V_1, ... for better cache hit.
        key_to_var = defaultdict(lambda: f"V_{counter.__next__()}")
        local_dict = {key_to_var[c.key]: i for c, i in zip(op.inputs, inputs)}
        expr = _evaluate(chunk).format_map(key_to_var)
        # The numexpr.evaluate is thread safe: https://github.com/pydata/numexpr/pull/200
        try:
            res = ne.evaluate(expr, local_dict=local_dict, global_dict={})
        except Exception as e:
            raise RuntimeError(
                f"Failed to evaluate numexpr {repr(expr)} on local dict {local_dict}."
            ) from e
        res = _maybe_keepdims(chunk, res)
        if chunk.ndim == 0 and res.ndim == 1 and res.size == 0:
            res = res.dtype.type(0)
        ctx[chunk.key] = res


# execution part
NE_UNARYOP_TO_STRING = {
    arithmetic.TensorNegative: "-",
    arithmetic.TensorAbs: "abs",
    arithmetic.TensorConj: "conj",
    arithmetic.TensorExp: "exp",
    arithmetic.TensorLog: "log",
    arithmetic.TensorLog10: "log10",
    arithmetic.TensorExpm1: "expm1",
    arithmetic.TensorLog1p: "log1p",
    arithmetic.TensorSqrt: "sqrt",
    arithmetic.TensorSin: "sin",
    arithmetic.TensorCos: "cos",
    arithmetic.TensorTan: "tan",
    arithmetic.TensorArcsin: "arcsin",
    arithmetic.TensorArccos: "arccos",
    arithmetic.TensorArctan: "arctan",
    arithmetic.TensorSinh: "sinh",
    arithmetic.TensorCosh: "cosh",
    arithmetic.TensorTanh: "tanh",
    arithmetic.TensorArcsinh: "arcsinh",
    arithmetic.TensorArccosh: "arccosh",
    arithmetic.TensorArctanh: "arctanh",
    arithmetic.TensorFloor: "floor",
    arithmetic.TensorCeil: "ceil",
    arithmetic.TensorNot: "~",
}


NE_BINOP_TO_STRING = {
    arithmetic.TensorAdd: "+",
    arithmetic.TensorSubtract: "-",
    arithmetic.TensorMultiply: "*",
    arithmetic.TensorDivide: "/",
    arithmetic.TensorMod: "%",
    arithmetic.TensorPower: "**",
    arithmetic.TensorLshift: "<<",
    arithmetic.TensorRshift: ">>",
    arithmetic.TensorEqual: "==",
    arithmetic.TensorNotEqual: "!=",
    arithmetic.TensorLessThan: "<",
    arithmetic.TensorLessEqual: "<=",
    arithmetic.TensorGreaterThan: ">",
    arithmetic.TensorGreaterEqual: ">=",
    arithmetic.TensorAnd: "and",
    arithmetic.TensorOr: "or",
}

NE_TREE_OP_TO_STRING = {
    arithmetic.TensorTreeAdd: "+",
    arithmetic.TensorTreeMultiply: "*",
}

NE_REDUCTION_TO_STRING = {
    reduction.TensorSum: "sum",
    reduction.TensorProd: "prod",
    reduction.TensorMax: "max",
    reduction.TensorMin: "min",
}


class _Default(dict):
    def __missing__(self, key):
        return f"{{{key}}}"


def _handle_unary(chunk):
    if len(chunk.inputs) != 1:
        raise ValueError("unary operand inputs should be 1")
    data = chunk.inputs[0]
    unary_op = NE_UNARYOP_TO_STRING[type(chunk.op)]
    return f"{unary_op}({{{data.key}}})"


def _decompose(chunk):
    expr = f"{{{chunk.key}}}"
    for node in reversed(chunk.composed):
        _expr = _evaluate(node)
        expr = expr.format_map(_Default([(node.key, f"({_expr})")]))
    return expr


def _handle_bin(chunk):
    op = chunk.op
    lhs = str(op.lhs) if np.isscalar(op.lhs) else f"{{{op.lhs.key}}}"
    rhs = str(op.rhs) if np.isscalar(op.rhs) else f"{{{op.rhs.key}}}"
    reverse = getattr(op, "reverse", False)
    op = NE_BINOP_TO_STRING[type(op)]
    if reverse:
        exprs = [rhs, lhs]
    else:
        exprs = [lhs, rhs]
    return op.join(exprs)


def _handle_tree(chunk):
    op = NE_TREE_OP_TO_STRING[type(chunk.op)]
    return op.join(f"{{{c.key}}}" for c in chunk.inputs)


def _wrap_bool(data):
    if data.dtype == np.bool_:
        return f"where({{{data.key}}}, 1, 0)"

    return f"{{{data.key}}}"


def _handle_reduction(chunk):
    ax = chunk.op.axis
    data = chunk.inputs[0]
    op_str = NE_REDUCTION_TO_STRING[type(chunk.op)]
    # TODO(hks): delete it if numexpr.sum fixed
    if len(ax) == data.ndim:
        return f"{op_str}({_wrap_bool(data)})"
    elif len(ax) == 1:
        return f"{op_str}({_wrap_bool(data)},axis={ax[0]})"
    else:
        raise ValueError("numexpr cannot encode axis")


def _evaluate(chunk):
    op_type = type(chunk.op)
    if op_type in NE_UNARYOP_TO_STRING:
        return _handle_unary(chunk)
    elif op_type in NE_BINOP_TO_STRING:
        return _handle_bin(chunk)
    elif op_type in NE_TREE_OP_TO_STRING:
        return _handle_tree(chunk)
    elif op_type in NE_REDUCTION_TO_STRING:
        return _handle_reduction(chunk)
    elif op_type is TensorNeFuseChunk:
        return _decompose(chunk)
    else:
        raise TypeError(f"unsupported operator in numexpr: {op_type.__name__}")


def _maybe_keepdims(chunk, res):
    out_chunk = chunk.composed[-1] if type(chunk.op) == TensorNeFuseChunk else chunk
    if type(out_chunk.op) in NE_REDUCTION_TO_STRING and out_chunk.op.keepdims:
        res = np.reshape(res, out_chunk.shape)
    return res
