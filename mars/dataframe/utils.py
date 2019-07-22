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

import functools

from mars.lib.mmh3 import hash as mmh_hash


def hash_index(index, size):
    def func(x, size):
        return mmh_hash(bytes(x)) % size

    f = functools.partial(func, size=size)
    idx_to_grouped = dict(index.groupby(index.map(f)).items())
    return [idx_to_grouped.get(i, list()) for i in range(size)]


def hash_dtypes(dtypes, size):
    hashed_indexes = hash_index(dtypes.index, size)
    return [dtypes[index] for index in hashed_indexes]


def concat_tileable_chunks(df):
    from .merge.concat import DataFrameConcat
    from .operands import ObjectType, DATAFRAME_TYPE, SERIES_TYPE

    assert not df.is_coarse()

    if isinstance(df, DATAFRAME_TYPE):
        chunk = DataFrameConcat(object_type=ObjectType.dataframe).new_chunk(
            df.chunks, shape=df.shape, dtypes=df.dtypes,
            index_value=df.index_value, columns_value=df.columns)
        return DataFrameConcat(object_type=ObjectType.dataframe).new_dataframe(
            [df], shape=df.shape, chunks=[chunk],
            nsplits=tuple((s,) for s in df.shape), dtypes=df.dtypes,
            index_value=df.index_value, columns_value=df.columns)
    elif isinstance(df, SERIES_TYPE):
        chunk = DataFrameConcat(object_type=ObjectType.series).new_chunk(
            df.chunks, shape=df.shape, dtype=df.dtype, index_value=df.index_value, name=df.name)
        return DataFrameConcat(object_type=ObjectType.series).new_series(
            [df], shape=df.shape, chunks=[chunk],
            nsplits=tuple((s,) for s in df.shape), dtype=df.dtype,
            index_value=df.index_value, name=df.name)
    else:
        raise NotImplementedError


def get_fetch_op_cls(op):
    from ..operands import ShuffleProxy
    from .fetch import DataFrameFetchShuffle, DataFrameFetch
    if isinstance(op, ShuffleProxy):
        cls = DataFrameFetchShuffle
    else:
        cls = DataFrameFetch

    def _inner(**kw):
        return cls(object_type=op.object_type, **kw)

    return _inner


def get_fuse_op_cls():
    from .operands import DataFrameFuseChunk

    return DataFrameFuseChunk
