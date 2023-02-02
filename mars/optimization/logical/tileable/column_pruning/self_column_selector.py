# Copyright 2022-2023 XProbe Inc.
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

from typing import Set, Any, Callable

from .utils import get_cols_exclude_index
from .....core import TileableData
from .....dataframe.core import BaseDataFrameData, BaseSeriesData
from .....dataframe.groupby.aggregation import DataFrameGroupByAgg
from .....dataframe.indexing.getitem import DataFrameIndex
from .....dataframe.indexing.setitem import DataFrameSetitem
from .....dataframe.merge import DataFrameMerge
from .....typing import OperandType


class SelfColumnSelector:
    _OP_TO_SELECT_FUNCTION = {}

    @classmethod
    def register(
        cls,
        op_cls: OperandType,
        func: Callable[[TileableData], Set[Any]],
        replace: bool = False,
    ) -> None:
        if op_cls not in cls._OP_TO_SELECT_FUNCTION or replace:
            cls._OP_TO_SELECT_FUNCTION[op_cls] = func
        else:
            raise ValueError(f"key {op_cls} exists.")

    @classmethod
    def select(cls, tileable_data: TileableData) -> Set[Any]:
        """
        TODO: docstring
        """
        op_type = type(tileable_data.op)
        if op_type in cls._OP_TO_SELECT_FUNCTION:
            return cls._OP_TO_SELECT_FUNCTION[op_type](tileable_data)
        for op_cls in op_type.__mro__:
            if op_cls in cls._OP_TO_SELECT_FUNCTION:
                cls._OP_TO_SELECT_FUNCTION[op_type] = cls._OP_TO_SELECT_FUNCTION[op_cls]
                return cls._OP_TO_SELECT_FUNCTION[op_cls](tileable_data)
        return set()


def register_selector(op_type: OperandType) -> Callable:
    def wrap(selector_func: Callable):
        SelfColumnSelector.register(op_type, selector_func)
        return selector_func

    return wrap


@register_selector(DataFrameSetitem)
def df_setitem_select_function(tileable_data: TileableData) -> Set[Any]:
    if isinstance(tileable_data.op.indexes, list):
        return set(tileable_data.op.indexes)
    else:
        return {tileable_data.op.indexes}


@register_selector(DataFrameIndex)
def df_getitem_select_function(tileable_data: TileableData) -> Set[Any]:
    if tileable_data.op.col_names is not None:
        col_names = tileable_data.op.col_names
        if isinstance(col_names, list):
            return set(tileable_data.op.col_names)
        else:
            return {tileable_data.op.col_names}
    else:
        if isinstance(tileable_data, BaseDataFrameData):
            return set(tileable_data.dtypes.index)
        elif isinstance(tileable_data, BaseSeriesData):
            return {tileable_data.name}


@register_selector(DataFrameGroupByAgg)
def df_groupby_agg_select_function(tileable_data: TileableData) -> Set[Any]:
    """
    Make sure the "group by columns" are preserved.
    """

    op: DataFrameGroupByAgg = tileable_data.op
    by = op.groupby_params["by"]

    if isinstance(tileable_data, BaseDataFrameData):
        return get_cols_exclude_index(tileable_data, by)
    elif isinstance(tileable_data, BaseSeriesData):
        return {tileable_data.name}
    else:
        return set()


@register_selector(DataFrameMerge)
def df_merge_select_function(tileable_data: TileableData) -> Set[Any]:
    """
    Make sure the merge keys are preserved.
    """

    op: DataFrameMerge = tileable_data.op
    on = op.on
    if on is not None:
        return get_cols_exclude_index(tileable_data, on)

    ret = set()
    left_data: BaseDataFrameData = op.inputs[0]
    right_data: BaseDataFrameData = op.inputs[1]
    left_index = op.left_index
    right_index = op.right_index
    left_on = op.left_on if isinstance(op.left_on, list) else [op.left_on]
    right_on = op.right_on if isinstance(op.right_on, list) else [op.right_on]

    if left_index and right_index:
        return ret

    if left_index:
        for col in right_data.dtypes.index:
            if col in right_on:
                ret.add(col)
        return ret
    if right_index:
        for col in left_data.dtypes.index:
            if col in left_on:
                ret.add(col)
        return ret

    for data, merge_keys, suffix in zip(
        [left_data, right_data], [left_on, right_on], op.suffixes
    ):
        if merge_keys is None:
            continue
        for col in data.dtypes.index:
            if col in merge_keys:
                other_data = right_data if data is left_data else left_data
                other_merge_keys = right_on if merge_keys is left_on else left_on

                if col in other_data.dtypes.index and col not in other_merge_keys:
                    # if the merge key exists in the other dataframe but not in the other
                    # dataframe's merge keys, suffixes will be added.
                    # TODO: this does not work when col is a tuple.
                    suffix_col = str(col) + suffix
                    ret.add(suffix_col)
                else:
                    ret.add(col)
    return ret
