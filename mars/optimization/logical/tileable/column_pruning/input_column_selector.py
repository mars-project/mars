# Copyright 2022 XProbe Inc.
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
from typing import Callable, Dict, Any, Set

from .....core import TileableData
from .....dataframe import NamedAgg
from .....dataframe.arithmetic.core import DataFrameBinOp, DataFrameUnaryOp
from .....dataframe.core import (
    BaseDataFrameData,
    BaseSeriesData,
)
from .....dataframe.groupby.aggregation import DataFrameGroupByAgg
from .....dataframe.indexing.getitem import DataFrameIndex
from .....dataframe.indexing.setitem import DataFrameSetitem
from .....dataframe.merge import DataFrameMerge
from .....typing import OperandType


class InputColumnSelector:

    _OP_TO_SELECT_FUNCTION = {}

    @staticmethod
    def select_all_input_columns(
        tileable_data: TileableData, _required_cols: Set[Any]
    ) -> Dict[TileableData, Set[Any]]:
        ret = {}
        for inp in tileable_data.op.inputs:
            if isinstance(inp, BaseDataFrameData):
                ret[inp] = set(inp.dtypes.index)
            elif isinstance(inp, BaseSeriesData):
                ret[inp] = {inp.name}
        return ret

    @staticmethod
    def select_required_input_columns(
        tileable_data: TileableData, required_cols: Set[Any]
    ) -> Dict[TileableData, Set[Any]]:
        ret = {}
        for inp in tileable_data.op.inputs:
            if isinstance(inp, BaseDataFrameData):
                ret[inp] = required_cols.intersection(set(inp.dtypes.index))
            elif isinstance(inp, BaseSeriesData):
                ret[inp] = {inp.name}
        return ret

    @classmethod
    def register(
        cls,
        op_cls: OperandType,
        func: Callable[[TileableData, Set[Any]], Dict[TileableData, Set[Any]]],
    ) -> None:
        cls._OP_TO_SELECT_FUNCTION[op_cls] = func

    @classmethod
    def unregister(cls, op_cls: OperandType) -> None:
        if op_cls in cls._OP_TO_SELECT_FUNCTION:
            del cls._OP_TO_SELECT_FUNCTION[op_cls]

    @classmethod
    def select_input_columns(
        cls, tileable_data: TileableData, required_cols: Set[Any]
    ) -> Dict[TileableData, Set[Any]]:
        """
        Get the column pruning results of given tileable data.

        Parameters
        ----------
        tileable_data : TileableData
            The tileable data to be processed.
        required_cols: List[Any]
            Names of columns required by the successors of the given tileable data. If required_cols is None, all the
            input columns will be selected.
        Returns
        -------
        Dict[TileableData: List[Any]]
            A dictionary that represents the column pruning results. For every key-value pairs in the dictionary, the
            key is a predecessor of the given tileable data, and the value is a list of column names that the given
            tileable data depends on.
        """
        if required_cols is None:
            return cls.select_all_input_columns(tileable_data, set())

        op_type = type(tileable_data.op)
        if op_type in cls._OP_TO_SELECT_FUNCTION:
            return cls._OP_TO_SELECT_FUNCTION[op_type](tileable_data, required_cols)
        for op_cls in op_type.__mro__:
            if op_cls in cls._OP_TO_SELECT_FUNCTION:
                cls._OP_TO_SELECT_FUNCTION[op_type] = cls._OP_TO_SELECT_FUNCTION[op_cls]
                return cls._OP_TO_SELECT_FUNCTION[op_cls](tileable_data, required_cols)
        return cls.select_all_input_columns(tileable_data, required_cols)


def register_selector(op_type: OperandType) -> Callable:
    def wrap(selector_func: Callable):
        InputColumnSelector.register(op_type, selector_func)
        return selector_func

    return wrap


@register_selector(DataFrameMerge)
def df_merge_select_function(
    tileable_data: TileableData, required_cols: Set[Any]
) -> Dict[TileableData, Set[Any]]:
    op: DataFrameMerge = tileable_data.op
    assert len(op.inputs) == 2
    assert isinstance(op.inputs[0], BaseDataFrameData)
    assert isinstance(op.inputs[1], BaseDataFrameData)
    left_data: BaseDataFrameData = op.inputs[0]
    right_data: BaseDataFrameData = op.inputs[1]

    ret = defaultdict(set)
    for df, suffix in zip([left_data, right_data], op.suffixes):
        for col in df.dtypes.index:
            if col in required_cols:
                ret[df].add(col)
            else:
                suffix_col = str(col) + suffix
                if suffix_col in required_cols:
                    ret[df].add(col)

    if op.on is not None:
        ret[left_data].update(_get_cols_exclude_index(left_data, op.on))
        ret[right_data].update(_get_cols_exclude_index(right_data, op.on))
    if op.left_on is not None:
        ret[left_data].update(_get_cols_exclude_index(left_data, op.left_on))
    if op.right_on is not None:
        ret[right_data].update(_get_cols_exclude_index(right_data, op.right_on))

    return ret


def _get_cols_exclude_index(inp: BaseDataFrameData, cols: Any) -> Set[Any]:
    ret = set()
    if isinstance(cols, (list, tuple)):
        for col in cols:
            if col in inp.dtypes.index:
                # exclude index
                ret.add(col)
    else:
        if cols in inp.dtypes.index:
            # exclude index
            ret.add(cols)
    return ret


@register_selector(DataFrameGroupByAgg)
def df_groupby_agg_select_function(
    tileable_data: TileableData, required_cols: Set[Any]
) -> Dict[TileableData, Set[Any]]:
    op: DataFrameGroupByAgg = tileable_data.op
    assert isinstance(op.inputs[0], (BaseDataFrameData, BaseSeriesData))
    inp: BaseDataFrameData = op.inputs[0]
    by = op.groupby_params["by"]
    selection = op.groupby_params.get("selection", None)
    raw_func = op.raw_func

    ret = {}
    # group by a series
    groupby_series = False
    if isinstance(by, list) and len(by) == 1 and isinstance(by[0], BaseSeriesData):
        groupby_series = True
        ret[by[0]] = by[0].name

    if isinstance(inp, BaseSeriesData):
        ret[inp] = {inp.name}
    else:
        selected_cols = set()
        # group by keys should be included
        if not groupby_series:
            selected_cols.update(_get_cols_exclude_index(inp, by))
        # add agg columns
        if op.raw_func is not None:
            if op.raw_func == "size":
                # special for size, its return value is always series
                pass
            elif isinstance(raw_func, dict):
                selected_cols.update(set(raw_func.keys()))
            else:
                # no specified agg columns
                # required_cols should always be a subset of selection
                for col in required_cols:
                    # col is a tuple when required col is a MultiIndex
                    if isinstance(col, tuple):
                        for c in col:
                            selected_cols.add(c)
                    selected_cols.add(col)
                if selection is not None:
                    if isinstance(selection, (list, tuple)):
                        selected_cols.update(set(selection))
                    else:
                        selected_cols.add(selection)
        elif op.raw_func_kw:
            # add renamed columns
            for _, origin in op.raw_func_kw.items():
                if isinstance(origin, NamedAgg):
                    selected_cols.add(origin.column)
                else:
                    assert isinstance(origin, tuple)
                    selected_cols.add(origin[0])

        ret[inp] = selected_cols.intersection(inp.dtypes.index)
    return ret


@register_selector(DataFrameSetitem)
def df_setitem_select_function(
    tileable_data: TileableData, required_cols: Set[Any]
) -> Dict[TileableData, Set[Any]]:
    if len(tileable_data.inputs) == 1:
        # if value is not a Mars object, return required input columns
        return InputColumnSelector.select_required_input_columns(
            tileable_data, required_cols
        )
    else:
        df, value = tileable_data.inputs
        ret = {df: required_cols.intersection(set(df.dtypes.index))}
        # if value is a Mars object, return all its columns so that setitem can be executed
        if isinstance(value, BaseDataFrameData):
            value_cols = set(value.dtypes.index)
            ret[value] = value_cols
        elif isinstance(value, BaseSeriesData):
            value_cols = {value.name}
            ret[value] = value_cols
        return ret


SELECT_REQUIRED_OP_TYPES = [DataFrameBinOp, DataFrameUnaryOp, DataFrameIndex]
for op_type in SELECT_REQUIRED_OP_TYPES:
    InputColumnSelector.register(
        op_type, InputColumnSelector.select_required_input_columns
    )
