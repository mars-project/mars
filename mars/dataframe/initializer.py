# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

import pandas as pd

from ..core import Base, Entity
from ..tensor import tensor as astensor
from ..tensor.core import TENSOR_TYPE
from .operands import ObjectType
from .core import DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE, DataFrame as _Frame, \
    Series as _Series, Index as _Index
from .datasource.dataframe import from_pandas as from_pandas_df
from .datasource.series import from_pandas as from_pandas_series
from .datasource.index import from_pandas as from_pandas_index
from .datasource.from_tensor import dataframe_from_tensor, series_from_tensor, \
    dataframe_from_1d_tileables
from .fetch import DataFrameFetch


class DataFrame(_Frame):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False,
                 chunk_size=None, gpu=None, sparse=None):
        if isinstance(data, TENSOR_TYPE):
            if chunk_size is not None:
                data = data.rechunk(chunk_size)
            df = dataframe_from_tensor(data, index=index, columns=columns, gpu=gpu, sparse=sparse)
        elif isinstance(index, (INDEX_TYPE, SERIES_TYPE)):
            df = dataframe_from_tensor(astensor(data, chunk_size=chunk_size), index=index,
                                       columns=columns, gpu=gpu, sparse=sparse)
        elif isinstance(data, DATAFRAME_TYPE):
            if not hasattr(data, 'data'):
                # DataFrameData
                df = _Frame(data)
            else:
                df = data
        elif isinstance(data, dict) and any(isinstance(v, (Base, Entity)) for v in data.values()):
            # data is a dict and some value is tensor
            df = dataframe_from_1d_tileables(
                data, index=index, columns=columns, gpu=gpu, sparse=sparse)
        else:
            pdf = pd.DataFrame(data, index=index, columns=columns, dtype=dtype, copy=copy)
            df = from_pandas_df(pdf, chunk_size=chunk_size, gpu=gpu, sparse=sparse)
        super().__init__(df.data)


class Series(_Series):
    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False,
                 chunk_size=None, gpu=None, sparse=None):
        if isinstance(data, TENSOR_TYPE):
            if chunk_size is not None:
                data = data.rechunk(chunk_size)
            series = series_from_tensor(data, index=index, name=name, gpu=gpu, sparse=sparse)
        elif isinstance(index, INDEX_TYPE):
            series = series_from_tensor(astensor(data, chunk_size=chunk_size), index=index,
                                        name=name, gpu=gpu, sparse=sparse)
        elif isinstance(data, SERIES_TYPE):
            if not hasattr(data, 'data'):
                # SeriesData
                series = _Series(data)
            else:
                series = data
        else:
            pd_series = pd.Series(data, index=index, dtype=dtype, name=name, copy=copy)
            series = from_pandas_series(pd_series, chunk_size=chunk_size, gpu=gpu, sparse=sparse)
        super().__init__(series.data)


class Index(_Index):
    def __new__(cls, data, **_):
        # just return cls always until we support other Index's initializers
        return object.__new__(cls)

    def __init__(self, data=None, dtype=None, copy=False, name=None, tupleize_cols=True,
                 chunk_size=None, gpu=None, sparse=None, names=None):
        if isinstance(data, INDEX_TYPE):
            if not hasattr(data, 'data'):
                # IndexData
                index = _Index(data)
            else:
                index = data
        else:
            if not isinstance(data, pd.Index):
                pd_index = pd.Index(data=data, dtype=dtype, copy=copy, name=name,
                                    names=names, tupleize_cols=tupleize_cols)
            else:
                pd_index = data
            index = from_pandas_index(pd_index, chunk_size=chunk_size,
                                      gpu=gpu, sparse=sparse)
        super().__init__(index.data)


def named_dataframe(name, session=None):
    from ..session import Session

    sess = session or Session.default_or_local()
    tileable_infos = sess.get_named_tileable_infos(name=name)
    fetch_op = DataFrameFetch(object_type=ObjectType.dataframe)
    return fetch_op.new_dataframe([], shape=tileable_infos.tileable_shape,
                                  _key=tileable_infos.tileable_key)


def named_series(name, session=None):
    from ..session import Session

    sess = session or Session.default_or_local()
    tileable_infos = sess.get_named_tileable_infos(name=name)
    fetch_op = DataFrameFetch(object_type=ObjectType.series)
    return fetch_op.new_series([], shape=tileable_infos.tileable_shape,
                               _key=tileable_infos.tileable_key)
