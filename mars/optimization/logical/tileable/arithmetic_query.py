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

import weakref
from typing import Optional, Tuple

import numpy as np

from ..core import OptimizationRecord, OptimizationRecordType
from .... import dataframe as md
from ....core import Tileable
from ....dataframe.arithmetic.core import DataFrameUnaryUfunc, DataFrameBinopUfunc
from ....dataframe.base.eval import DataFrameEval
from ....dataframe.indexing.getitem import DataFrameIndex
from ....typing import OperandType
from ....utils import implements
from .core import OptimizationRule
from ..tileable.core import register_tileable_optimization_rule


def _get_binop_builder(op_str: str):
    def builder(lhs: str, rhs: str):
        return f"({lhs}) {op_str} ({rhs})"

    return builder


_func_name_to_builder = {
    "add": _get_binop_builder("+"),
    "sub": _get_binop_builder("-"),
    "mul": _get_binop_builder("*"),
    "floordiv": _get_binop_builder("//"),
    "truediv": _get_binop_builder("/"),
    "pow": _get_binop_builder("**"),
    "eq": _get_binop_builder("=="),
    "ne": _get_binop_builder("!="),
    "lt": _get_binop_builder("<"),
    "le": _get_binop_builder("<="),
    "gt": _get_binop_builder(">"),
    "ge": _get_binop_builder(">="),
    "__and__": _get_binop_builder("&"),
    "__or__": _get_binop_builder("|"),
    "__xor__": _get_binop_builder("^"),
    "negative": lambda expr: f"-({expr})",
    "__invert__": lambda expr: f"~({expr})",
}
_extract_result_cache = weakref.WeakKeyDictionary()


@register_tileable_optimization_rule([DataFrameUnaryUfunc, DataFrameBinopUfunc])
class SeriesArithmeticToEval(OptimizationRule):
    @implements(OptimizationRule.match)
    def match(self, op: OperandType) -> bool:
        _, expr = self._extract_eval_expression(op.outputs[0])
        return expr is not None

    @staticmethod
    def _is_select_dataframe_column(tileable) -> bool:
        if not isinstance(tileable, md.Series) \
                or not isinstance(tileable.op, DataFrameIndex):
            return False

        input_df = tileable.inputs[0]
        index_op: DataFrameIndex = tileable.op
        return isinstance(input_df, md.DataFrame) and input_df.dtypes is not None \
            and index_op.col_names is not None and index_op.col_names in input_df.dtypes \
            and index_op.mask is None

    @staticmethod
    def _is_eval(tileable) -> bool:
        if not isinstance(tileable, md.Series) \
                or not isinstance(tileable.op, DataFrameEval):
            return False
        return len(tileable.inputs) == 1 and not tileable.op.is_query

    @classmethod
    def _extract_eval_expression(cls, tileable) -> Tuple[Optional[Tileable], Optional[str]]:
        if np.isscalar(tileable):
            return None, repr(tileable)

        if tileable in _extract_result_cache:
            return _extract_result_cache[tileable]

        if cls._is_select_dataframe_column(tileable):
            result = tileable.inputs[0], tileable.op.col_names
        elif cls._is_eval(tileable):
            result = tileable.inputs[0], tileable.op.expr
        elif isinstance(tileable.op, DataFrameUnaryUfunc):
            result = cls._extract_unary(tileable)
        elif isinstance(tileable.op, DataFrameBinopUfunc):
            if tileable.op.fill_value is not None or tileable.op.level is not None:
                result = None, None
            else:
                result = cls._extract_binary(tileable)
        else:
            result = None, None

        _extract_result_cache[tileable] = result
        return result

    @classmethod
    def _extract_unary(cls, tileable) -> Tuple[Optional[Tileable], Optional[str]]:
        op = tileable.op
        func_name = getattr(op, '_func_name') or getattr(op, '_bin_func_name')
        if func_name not in _func_name_to_builder:
            return None, None
        in_tileable, expr = cls._extract_eval_expression(op.inputs[0])
        return in_tileable, _func_name_to_builder[func_name](expr)

    @classmethod
    def _extract_binary(cls, tileable) -> Tuple[Optional[Tileable], Optional[str]]:
        op = tileable.op
        func_name = getattr(op, '_func_name') or getattr(op, '_bin_func_name')
        if func_name not in _func_name_to_builder:
            return None, None
        lhs_tileable, lhs_expr = cls._extract_eval_expression(op.lhs)
        rhs_tileable, rhs_expr = cls._extract_eval_expression(op.rhs)
        if lhs_expr is None or rhs_expr is None:
            return None, None
        if lhs_tileable is not None and rhs_tileable is not None and lhs_tileable.key != rhs_tileable.key:
            return None, None
        in_tileable = next(t for t in [lhs_tileable, rhs_tileable] if t is not None)
        return in_tileable, _func_name_to_builder[func_name](lhs_expr, rhs_expr)

    @implements(OptimizationRule.apply)
    def apply(self, op: OperandType):
        node = op.outputs[0]
        in_tileable, expr = self._extract_eval_expression(node)
        new_node = in_tileable.eval(expr)
        self._replace_node(node, new_node)
        self._records.append_record(
            OptimizationRecord(node, new_node, OptimizationRecordType.replace)
        )

        # check node if it's in result
        try:
            i = self._graph.results.index(node)
            self._graph.results[i] = new_node
        except ValueError:
            pass


@register_tileable_optimization_rule([DataFrameIndex])
class DataFrameBoolEvalToQuery(OptimizationRule):
    def match(self, op: "DataFrameIndex") -> bool:
        if not isinstance(op.inputs[0], md.DataFrame):
            return False
        if op.col_names is not None or not isinstance(op.mask, md.Series) or op.mask.dtype != bool:
            return False
        mask_op = op.mask.op
        if not isinstance(mask_op, DataFrameEval) or mask_op.is_query:
            return False
        return True

    def apply(self, op: "DataFrameIndex"):
        node = op.outputs[0]
        in_tileable = op.inputs[0]
        new_node = in_tileable.query(op.mask.op.expr)
        self._replace_node(node, new_node)
        self._records.append_record(
            OptimizationRecord(node, new_node, OptimizationRecordType.replace)
        )

        # check node if it's in result
        try:
            i = self._graph.results.index(node)
            self._graph.results[i] = new_node
        except ValueError:
            pass
