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
from typing import NamedTuple, Optional

import numpy as np
from pandas.api.types import is_scalar

from .... import dataframe as md
from ....core import Tileable, get_output_types, ENTITY_TYPE
from ....dataframe.arithmetic.core import DataFrameUnaryUfunc, DataFrameBinopUfunc
from ....dataframe.base.eval import DataFrameEval
from ....dataframe.indexing.getitem import DataFrameIndex
from ....dataframe.indexing.setitem import DataFrameSetitem
from ....typing import OperandType
from ....utils import implements
from ..core import OptimizationRecord, OptimizationRecordType
from ..tileable.core import register_tileable_optimization_rule
from .core import OptimizationRule


class EvalExtractRecord(NamedTuple):
    tileable: Optional[Tileable] = None
    expr: Optional[str] = None
    variables: Optional[dict] = None


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
    _var_counter = 0

    @classmethod
    def _next_var_id(cls):
        cls._var_counter += 1
        return cls._var_counter

    @implements(OptimizationRule.match)
    def match(self, op: OperandType) -> bool:
        _, expr, _ = self._extract_eval_expression(op.outputs[0])
        return expr is not None

    @staticmethod
    def _is_select_dataframe_column(tileable) -> bool:
        if not isinstance(tileable, md.Series) or not isinstance(
            tileable.op, DataFrameIndex
        ):
            return False

        input_df = tileable.inputs[0]
        index_op: DataFrameIndex = tileable.op
        if (
            not isinstance(input_df, md.DataFrame)
            or input_df.dtypes is None
            or not input_df.dtypes.index.is_unique
            or any(not isinstance(v, str) for v in input_df.dtypes.keys())
        ):
            return False

        return (
            isinstance(input_df, md.DataFrame)
            and input_df.dtypes is not None
            and index_op.col_names is not None
            and index_op.col_names in input_df.dtypes
            and index_op.mask is None
        )

    @classmethod
    def _extract_eval_expression(cls, tileable) -> EvalExtractRecord:
        if is_scalar(tileable):
            if isinstance(tileable, (int, bool, str, bytes, np.integer, np.bool_)):
                return EvalExtractRecord(expr=repr(tileable))
            else:
                var_name = f"__eval_scalar_var{cls._next_var_id()}"
                var_dict = {var_name: tileable}
                return EvalExtractRecord(expr=f"@{var_name}", variables=var_dict)

        if not isinstance(tileable, ENTITY_TYPE):  # pragma: no cover
            return EvalExtractRecord()

        if tileable in _extract_result_cache:
            return _extract_result_cache[tileable]

        if cls._is_select_dataframe_column(tileable):
            result = cls._extract_column_select(tileable)
        elif isinstance(tileable.op, DataFrameUnaryUfunc):
            result = cls._extract_unary(tileable)
        elif isinstance(tileable.op, DataFrameBinopUfunc):
            if tileable.op.fill_value is not None or tileable.op.level is not None:
                result = EvalExtractRecord()
            else:
                result = cls._extract_binary(tileable)
        else:
            result = EvalExtractRecord()

        _extract_result_cache[tileable] = result
        return result

    @classmethod
    def _extract_column_select(cls, tileable) -> EvalExtractRecord:
        return EvalExtractRecord(tileable.inputs[0], f"`{tileable.op.col_names}`")

    @classmethod
    def _extract_unary(cls, tileable) -> EvalExtractRecord:
        op = tileable.op
        func_name = getattr(op, "_func_name") or getattr(op, "_bin_func_name")
        if func_name not in _func_name_to_builder:  # pragma: no cover
            return EvalExtractRecord()

        in_tileable, expr, variables = cls._extract_eval_expression(op.inputs[0])
        if in_tileable is None:
            return EvalExtractRecord()

        cls._add_collapsable_predecessor(tileable, op.inputs[0])
        return EvalExtractRecord(
            in_tileable, _func_name_to_builder[func_name](expr), variables
        )

    @classmethod
    def _extract_binary(cls, tileable) -> EvalExtractRecord:
        op = tileable.op
        func_name = getattr(op, "_func_name", None) or getattr(op, "_bit_func_name")
        if func_name not in _func_name_to_builder:  # pragma: no cover
            return EvalExtractRecord()

        lhs_tileable, lhs_expr, lhs_vars = cls._extract_eval_expression(op.lhs)
        if lhs_tileable is not None:
            cls._add_collapsable_predecessor(tileable, op.lhs)
        rhs_tileable, rhs_expr, rhs_vars = cls._extract_eval_expression(op.rhs)
        if rhs_tileable is not None:
            cls._add_collapsable_predecessor(tileable, op.rhs)

        if lhs_expr is None or rhs_expr is None:
            return EvalExtractRecord()
        if (
            lhs_tileable is not None
            and rhs_tileable is not None
            and lhs_tileable.key != rhs_tileable.key
        ):
            return EvalExtractRecord()

        variables = (lhs_vars or dict()).copy()
        variables.update(rhs_vars or dict())
        in_tileable = next(t for t in [lhs_tileable, rhs_tileable] if t is not None)
        return EvalExtractRecord(
            in_tileable, _func_name_to_builder[func_name](lhs_expr, rhs_expr), variables
        )

    @implements(OptimizationRule.apply)
    def apply(self, op: OperandType):
        node = op.outputs[0]
        in_tileable, expr, variables = self._extract_eval_expression(node)

        new_op = DataFrameEval(
            _key=node.op.key,
            _output_types=get_output_types(node),
            expr=expr,
            variables=variables or dict(),
            parser="pandas",
            is_query=False,
        )
        new_node = new_op.new_tileable([in_tileable], **node.params).data
        new_node._key = node.key
        new_node._id = node.id

        self._remove_collapsable_predecessors(node)
        self._replace_node(node, new_node)
        self._graph.add_edge(in_tileable, new_node)

        self._records.append_record(
            OptimizationRecord(node, new_node, OptimizationRecordType.replace)
        )

        # check node if it's in result
        try:
            i = self._graph.results.index(node)
            self._graph.results[i] = new_node
        except ValueError:
            pass


class _DataFrameEvalRewriteRule(OptimizationRule):
    def match(self, op: OperandType) -> bool:
        optimized_eval_op = self._get_optimized_eval_op(op)
        if (
            not isinstance(optimized_eval_op, DataFrameEval)
            or optimized_eval_op.is_query
            or optimized_eval_op.inputs[0].key != op.inputs[0].key
        ):
            return False
        return True

    def _build_new_eval_op(self, op: OperandType):
        raise NotImplementedError

    def _get_optimized_eval_op(self, op: OperandType) -> OperandType:
        in_columnar_node = self._get_input_columnar_node(op)
        optimized = self._records.get_optimization_result(in_columnar_node)
        return optimized.op if optimized is not None else in_columnar_node.op

    def _get_input_columnar_node(self, op: OperandType) -> ENTITY_TYPE:
        raise NotImplementedError

    def apply(self, op: DataFrameIndex):
        node = op.outputs[0]
        in_tileable = op.inputs[0]
        in_columnar_node = self._get_input_columnar_node(op)

        new_op = self._build_new_eval_op(op)
        new_op._key = node.op.key

        new_node = new_op.new_tileable([in_tileable], **node.params).data
        new_node._key = node.key
        new_node._id = node.id

        self._add_collapsable_predecessor(node, in_columnar_node)
        self._remove_collapsable_predecessors(node)

        self._replace_node(node, new_node)
        self._graph.add_edge(in_tileable, new_node)
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
class DataFrameBoolEvalToQuery(_DataFrameEvalRewriteRule):
    def match(self, op: DataFrameIndex) -> bool:
        if (
            op.col_names is not None
            or not isinstance(op.mask, md.Series)
            or op.mask.dtype != bool
        ):
            return False
        return super().match(op)

    def _get_input_columnar_node(self, op: OperandType) -> ENTITY_TYPE:
        return op.mask

    def _build_new_eval_op(self, op: OperandType):
        in_eval_op = self._get_optimized_eval_op(op)
        return DataFrameEval(
            _output_types=get_output_types(op.outputs[0]),
            expr=in_eval_op.expr,
            variables=in_eval_op.variables,
            parser="pandas",
            is_query=True,
        )


@register_tileable_optimization_rule([DataFrameSetitem])
class DataFrameEvalSetItemToEval(_DataFrameEvalRewriteRule):
    def match(self, op: DataFrameSetitem):
        if not isinstance(op.indexes, str) or not isinstance(op.value, md.Series):
            return False
        return super().match(op)

    def _get_input_columnar_node(self, op: DataFrameSetitem) -> ENTITY_TYPE:
        return op.value

    def _build_new_eval_op(self, op: DataFrameSetitem):
        in_eval_op = self._get_optimized_eval_op(op)
        return DataFrameEval(
            _output_types=get_output_types(op.outputs[0]),
            expr=f"`{op.indexes}` = {in_eval_op.expr}",
            variables=in_eval_op.variables,
            parser="pandas",
            is_query=False,
            self_target=True,
        )
