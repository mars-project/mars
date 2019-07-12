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

from ..expressions.datasource.dataframe import DataFrameDataSource
from ..expressions.datasource.series import SeriesDataSource
from ..expressions.datasource.from_tensor import DataFrameFromTensor

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pass


def _dataframe_or_series_pandas_data_source(ctx, chunk):
    ctx[chunk.key] = chunk.op.data


def _dataframe_tensor_data_source(ctx, chunk):
    tensor_data = ctx[chunk.inputs[0].key]
    ctx[chunk.key] = pd.DataFrame(tensor_data, index=chunk.index_value, columns=chunk.columns)


def register_data_source_handler():
    from ...executor import register

    register(DataFrameDataSource, _dataframe_or_series_pandas_data_source)
    register(SeriesDataSource, _dataframe_or_series_pandas_data_source)
    register(DataFrameFromTensor, _dataframe_tensor_data_source)
