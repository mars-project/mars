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

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pass

from ..tensor.core import TENSOR_TYPE
from .core import DATAFRAME_TYPE, SERIES_TYPE, DataFrame as _Frame, Series as _Series
from .datasource.dataframe import from_pandas as from_pandas_df
from .datasource.series import from_pandas as from_pandas_series
from .datasource.from_tensor import from_tensor


class DataFrame(_Frame):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False,
                 chunk_size=None, gpu=None, sparse=None):
        if isinstance(data, TENSOR_TYPE):
            df = from_tensor(data, index=index, columns=columns, gpu=gpu, sparse=sparse)
        elif isinstance(data, DATAFRAME_TYPE):
            df = data
        else:
            pdf = pd.DataFrame(data, index=index, columns=columns, dtype=dtype, copy=copy)
            df = from_pandas_df(pdf, chunk_size=chunk_size, gpu=gpu, sparse=sparse)
        super(DataFrame, self).__init__(df.data)


class Series(_Series):
    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False,
                 chunk_size=None, gpu=None, sparse=None):
        if isinstance(data, TENSOR_TYPE):
            raise NotImplementedError('Not support create Series from tensor')
        if isinstance(data, SERIES_TYPE):
            series = data
        else:
            pd_series = pd.Series(data, index=index, dtype=dtype, name=name, copy=copy)
            series = from_pandas_series(pd_series, chunk_size=chunk_size, gpu=gpu, sparse=sparse)
        super(Series, self).__init__(series.data)
