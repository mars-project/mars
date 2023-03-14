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
from abc import ABC, abstractmethod
from typing import Dict, Type, Union

import pandas as pd


class DataFrameCustomGroupByAggMixin(ABC):
    @classmethod
    @abstractmethod
    def execute_map(cls, op, in_data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Map stage implement.

        Parameters
        -------
        op : Any operand
            DataFrame operand.
        in_data : pd.DataFrame
            Input dataframe.

        Returns
        -------
            The result of op map stage.
        """

    @classmethod
    @abstractmethod
    def execute_combine(
        cls, op, in_data: pd.DataFrame
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Combine stage implement.

        Parameters
        ----------
        op : Any operand
            DataFrame operand.
        in_data : pd.Dataframe
            Input dataframe.

        Returns
        -------
            The result of op combine stage.
        """

    @classmethod
    @abstractmethod
    def execute_agg(cls, op, in_data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Agg stage implement.

        Parameters
        ----------
        op : Any operand
            DataFrame operand.
        in_data : pd.Dataframe
            Input dataframe.

        Returns
        -------
            The result of op agg stage.
        """


custom_agg_functions: Dict[str, Type[DataFrameCustomGroupByAggMixin]] = {}


def register_custom_groupby_agg_func(method_name: str):
    def wrap(func_type: Type[DataFrameCustomGroupByAggMixin]):
        custom_agg_functions[method_name] = func_type
        return func_type

    return wrap
