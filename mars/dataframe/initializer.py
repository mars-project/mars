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

from ..core import Base, Entity, OutputType
from ..tensor import tensor as astensor
from ..tensor.core import TENSOR_TYPE
from ..utils import ceildiv
from .core import DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE, DataFrame as _Frame, \
    Series as _Series, Index as _Index
from .datasource.dataframe import from_pandas as from_pandas_df
from .datasource.series import from_pandas as from_pandas_series
from .datasource.index import from_pandas as from_pandas_index, \
    from_tileable as from_tileable_index
from .datasource.from_tensor import dataframe_from_tensor, series_from_tensor, \
    dataframe_from_1d_tileables
from .fetch import DataFrameFetch


class InitializerMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, (cls.__base__,) + getattr(cls, '_allow_data_type_'))


class DataFrame(_Frame, metaclass=InitializerMeta):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False,
                 chunk_size=None, gpu=None, sparse=None, num_partitions=None):
        # make sure __getattr__ does not result in stack overflow
        self._data = None

        need_repart = False
        if isinstance(data, TENSOR_TYPE):
            if chunk_size is not None:
                data = data.rechunk(chunk_size)
            df = dataframe_from_tensor(data, index=index, columns=columns, gpu=gpu, sparse=sparse)
            need_repart = num_partitions is not None
        elif isinstance(data, SERIES_TYPE):
            df = data.to_frame()
            need_repart = num_partitions is not None
        elif isinstance(data, DATAFRAME_TYPE):
            if not hasattr(data, 'data'):
                # DataFrameData
                df = _Frame(data)
            else:
                df = data
            need_repart = num_partitions is not None
        elif isinstance(data, dict) and any(isinstance(v, (Base, Entity)) for v in data.values()):
            # data is a dict and some value is tensor
            df = dataframe_from_1d_tileables(
                data, index=index, columns=columns, gpu=gpu, sparse=sparse)
            need_repart = num_partitions is not None
        elif isinstance(index, (INDEX_TYPE, SERIES_TYPE)):
            if isinstance(data, dict):
                data = {k: astensor(v, chunk_size=chunk_size) for k, v in data.items()}
                df = dataframe_from_1d_tileables(data, index=index, columns=columns,
                                                 gpu=gpu, sparse=sparse)
            else:
                df = dataframe_from_tensor(astensor(data, chunk_size=chunk_size), index=index,
                                           columns=columns, gpu=gpu, sparse=sparse)
            need_repart = num_partitions is not None
        else:
            pdf = pd.DataFrame(data, index=index, columns=columns, dtype=dtype, copy=copy)
            if num_partitions is not None:
                chunk_size = ceildiv(len(pdf), num_partitions)
            df = from_pandas_df(pdf, chunk_size=chunk_size, gpu=gpu, sparse=sparse)

        if need_repart:
            df = df.rebalance(num_partitions=num_partitions)
        super().__init__(df.data)


class Series(_Series, metaclass=InitializerMeta):
    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False,
                 chunk_size=None, gpu=None, sparse=None, num_partitions=None):
        # make sure __getattr__ does not result in stack overflow
        self._data = None

        need_repart = False
        if isinstance(data, (TENSOR_TYPE, INDEX_TYPE)):
            if chunk_size is not None:
                data = data.rechunk(chunk_size)
            name = name or getattr(data, 'name', None)
            series = series_from_tensor(data, index=index, name=name, gpu=gpu, sparse=sparse)
            need_repart = num_partitions is not None
        elif isinstance(index, INDEX_TYPE):
            series = series_from_tensor(astensor(data, chunk_size=chunk_size), index=index,
                                        name=name, gpu=gpu, sparse=sparse)
            need_repart = num_partitions is not None
        elif isinstance(data, SERIES_TYPE):
            if not hasattr(data, 'data'):
                # SeriesData
                series = _Series(data)
            else:
                series = data
            need_repart = num_partitions is not None
        else:
            pd_series = pd.Series(data, index=index, dtype=dtype, name=name, copy=copy)
            if num_partitions is not None:
                chunk_size = ceildiv(len(pd_series), num_partitions)
            series = from_pandas_series(pd_series, chunk_size=chunk_size, gpu=gpu, sparse=sparse)

        if need_repart:
            series = series.rebalance(num_partitions=num_partitions)
        super().__init__(series.data)


class Index(_Index, metaclass=InitializerMeta):
    def __new__(cls, data, **_):
        # just return cls always until we support other Index's initializers
        return object.__new__(cls)

    def __init__(self, data=None, dtype=None, copy=False, name=None, tupleize_cols=True,
                 chunk_size=None, gpu=None, sparse=None, names=None, num_partitions=None,
                 store_data=False):
        # make sure __getattr__ does not result in stack overflow
        self._data = None

        need_repart = False
        if isinstance(data, INDEX_TYPE):
            if not hasattr(data, 'data'):
                # IndexData
                index = _Index(data)
            else:
                index = data
            need_repart = num_partitions is not None
        else:
            if isinstance(data, (Base, Entity)):
                name = name if name is not None else getattr(data, 'name', None)
                index = from_tileable_index(data, dtype=dtype, name=name, names=names)
                need_repart = num_partitions is not None
            else:
                if not isinstance(data, pd.Index):
                    name = name if name is not None else getattr(data, 'name', None)
                    pd_index = pd.Index(data=data, dtype=dtype, copy=copy, name=name,
                                        tupleize_cols=tupleize_cols)
                else:
                    pd_index = data

                if num_partitions is not None:
                    chunk_size = ceildiv(len(pd_index), num_partitions)
                index = from_pandas_index(pd_index, chunk_size=chunk_size, gpu=gpu,
                                          sparse=sparse, store_data=store_data)

        if need_repart:
            index = index.rebalance(num_partitions=num_partitions)
        super().__init__(index.data)


def named_dataframe(name, session=None):
    from ..session import Session
    from .utils import parse_index

    sess = session or Session.default_or_local()
    tileable_infos = sess.get_named_tileable_infos(name=name)
    fetch_op = DataFrameFetch(output_types=[OutputType.dataframe])
    extra_meta = tileable_infos.tileable_extra_meta
    for param, store in zip(['index_value', 'columns_value'], ['False', 'True']):
        if param in extra_meta:
            extra_meta[param] = parse_index(extra_meta[param],
                                            store_data=store)
    return fetch_op.new_dataframe([], shape=tileable_infos.tileable_shape,
                                  _key=tileable_infos.tileable_key,
                                  kws=[extra_meta])


def named_series(name, session=None):
    from ..session import Session
    from .utils import parse_index

    sess = session or Session.default_or_local()
    tileable_infos = sess.get_named_tileable_infos(name=name)
    fetch_op = DataFrameFetch(output_types=[OutputType.series])
    extra_meta = tileable_infos.tileable_extra_meta
    if 'index_value' in extra_meta:
        extra_meta['index_value'] = parse_index(extra_meta['index_value'])
    return fetch_op.new_series([], shape=tileable_infos.tileable_shape,
                               _key=tileable_infos.tileable_key,
                               kws=[extra_meta])
