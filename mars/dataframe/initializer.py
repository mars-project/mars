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
from .core import DATAFRAME_TYPE, DataFrame as _Frame
from .expressions.datasource.dataframe import from_pandas


class DataFrame(_Frame):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False,
                 chunk_size=None, gpu=None, sparse=None):
        if isinstance(data, TENSOR_TYPE):
            raise NotImplementedError('Not support create DataFrame from tensor')
        if isinstance(data, DATAFRAME_TYPE):
            raise NotImplementedError('Not support yet')

        pdf = pd.DataFrame(data, index=index, columns=columns, dtype=dtype, copy=copy)
        df = from_pandas(pdf, chunk_size=chunk_size, gpu=gpu, sparse=sparse)
        super(DataFrame, self).__init__(df.data)
