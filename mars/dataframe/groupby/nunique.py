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
from typing import List, Union

import pandas as pd

from ...core import OutputType
from ...utils import implements
from .aggregation import DataFrameGroupByAgg
from .custom_aggregation import (
    DataFrameCustomGroupByAggMixin,
    register_custom_groupby_agg_func,
)


@register_custom_groupby_agg_func("nunique")
class DataFrameCustomGroupByNuniqueMixin(DataFrameCustomGroupByAggMixin):
    @classmethod
    def _get_level_indexes(
        cls, op: DataFrameGroupByAgg, data: pd.DataFrame
    ) -> List[int]:
        """
        When group by level, get the level index list.
        Level can be int, level name, or sequence of such.
        This function calculates the corresponding indexes.
        Parameters
        ----------
        op
        data

        Returns
        -------

        """
        index = [data.index.name] if data.index.name else data.index.names
        index = pd.Index(index)
        level = op.groupby_params["level"]
        if isinstance(level, int):
            indexes = [level]
        elif isinstance(level, str):
            indexes = [index.get_loc(level)]
        else:
            level = list(level)
            if isinstance(level[0], int):
                indexes = level
            else:
                indexes = index.get_indexer(level).tolist()
        return indexes

    @classmethod
    def _get_selection_columns(cls, op: DataFrameGroupByAgg) -> Union[None, List]:
        """
        Get groupby selection columns from op parameters.
        If this returns None, it means all columns are required.
        Parameters
        ----------
        op

        Returns
        -------

        """
        if "selection" in op.groupby_params:
            selection = op.groupby_params["selection"]
            if isinstance(selection, (tuple, list)):
                selection = [n for n in selection]
            else:
                selection = [selection]
            return selection

    @classmethod
    def _get_execute_map_result(
        cls, op: DataFrameGroupByAgg, in_data: pd.DataFrame
    ) -> Union[pd.DataFrame, pd.Series]:
        selections = cls._get_selection_columns(op)
        by_cols = op.raw_groupby_params["by"]
        if by_cols is not None:
            cols = (
                [*selections, *by_cols] if selections is not None else in_data.columns
            )
            res = in_data[cols].drop_duplicates(subset=cols).set_index(by_cols)
        else:  # group by level
            selections = selections if selections is not None else in_data.columns
            level_indexes = cls._get_level_indexes(op, in_data)
            in_data = in_data.reset_index()
            index_names = in_data.columns[level_indexes].tolist()
            cols = [*index_names, *selections]
            res = in_data[cols].drop_duplicates().set_index(index_names)

        # if sort=True is specified， sort index when finishing drop_duplicates.
        if op.raw_groupby_params["sort"]:
            res = res.sort_index()

        if op.output_types[0] == OutputType.series:
            res = res.squeeze()

        return res

    @classmethod
    def _get_execute_combine_result(
        cls, op: DataFrameGroupByAgg, in_data: pd.DataFrame
    ) -> Union[pd.DataFrame, pd.Series]:
        # in_data.index.names means MultiIndex (groupby on multi cols)
        index_col = in_data.index.name or in_data.index.names
        res = in_data.reset_index().drop_duplicates().set_index(index_col)
        if op.output_types[0] == OutputType.series:
            res = res.squeeze()
        return res

    @classmethod
    def _get_execute_agg_result(
        cls, op: DataFrameGroupByAgg, in_data: pd.DataFrame
    ) -> Union[pd.DataFrame, pd.Series]:
        groupby_params = op.groupby_params.copy()
        cols = in_data.index.name or in_data.index.names
        by = op.raw_groupby_params["by"]

        if by is not None:
            if op.output_types[0] == OutputType.dataframe:
                groupby_params.pop("level", None)
                groupby_params["by"] = cols
                in_data = in_data.reset_index()
        else:
            # When group by multi levels, we must get the actual all levels from raw_groupby_params,
            # since level field in op.groupby_params is not correct.
            groupby_params["level"] = op.raw_groupby_params["level"]

        res = in_data.groupby(**groupby_params).nunique()
        return res

    @classmethod
    @implements(DataFrameCustomGroupByAggMixin.execute_map)
    def execute_map(cls, op, in_data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        return cls._get_execute_map_result(op, in_data)

    @classmethod
    @implements(DataFrameCustomGroupByAggMixin.execute_combine)
    def execute_combine(
        cls, op, in_data: pd.DataFrame
    ) -> Union[pd.DataFrame, pd.Series]:
        return cls._get_execute_combine_result(op, in_data)

    @classmethod
    @implements(DataFrameCustomGroupByAggMixin.execute_agg)
    def execute_agg(cls, op, in_data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        return cls._get_execute_agg_result(op, in_data)
