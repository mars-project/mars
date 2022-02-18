# Copyright 1999-2021 Alibaba Group Holding Ltd.
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
import warnings

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...core import OutputType
from ...serialization.serializables import (
    AnyField,
    BoolField,
    ListField,
    Int64Field,
    ReferenceField,
)
from ..utils import parse_index, lazy_import, tokenize
from .core import (
    IncrementalIndexDatasource,
    IncrementalIndexDataSourceMixin,
    HeadOptimizedDataSource,
)

ray = lazy_import("ray")
# Ray Datasets is available in early preview at ray.data with Ray 1.6+
# (and ray.experimental.data in Ray 1.5)
ray_dataset = lazy_import("ray.data")
ray_exp_dataset = lazy_import("ray.experimental.data")
real_ray_dataset = ray_dataset or ray_exp_dataset


class DataFrameReadRayDataset(
    IncrementalIndexDatasource, IncrementalIndexDataSourceMixin
):
    _op_type_ = OperandDef.READ_RAYDATASET

    _refs = AnyField("refs")
    _columns = ListField("columns")
    _incremental_index = BoolField("incremental_index")
    _nrows = Int64Field("nrows")

    def __init__(
        self, refs=None, columns=None, incremental_index=None, nrows=None, **kw
    ):
        super().__init__(
            _refs=refs,
            _columns=columns,
            _incremental_index=incremental_index,
            _nrows=nrows,
            **kw,
        )

    @property
    def refs(self):
        return self._refs

    @property
    def columns(self):
        return self._columns

    @property
    def incremental_index(self):
        return self._incremental_index

    @classmethod
    def _tile_partitioned(cls, op: "DataFrameReadRayDataset"):
        out_df = op.outputs[0]
        shape = (np.nan, out_df.shape[1])
        dtypes = out_df.dtypes
        dataset = op.refs

        chunk_index = 0
        out_chunks = []
        for object_ref in dataset:
            chunk_op = op.copy().reset_key()
            chunk_op._refs = [object_ref]
            new_chunk = chunk_op.new_chunk(
                None,
                shape=shape,
                index=(chunk_index, 0),
                index_value=out_df.index_value,
                columns_value=out_df.columns_value,
                dtypes=dtypes,
            )
            out_chunks.append(new_chunk)
            chunk_index += 1

        new_op = op.copy()
        nsplits = ((np.nan,) * len(out_chunks), (out_df.shape[1],))
        return new_op.new_dataframes(
            None,
            out_df.shape,
            dtypes=dtypes,
            index_value=out_df.index_value,
            columns_value=out_df.columns_value,
            chunks=out_chunks,
            nsplits=nsplits,
        )

    @classmethod
    def _tile(cls, op):
        return cls._tile_partitioned(op)

    @classmethod
    def execute(cls, ctx, op: "DataFrameReadRayDataset"):
        out = op.outputs[0]
        ref = op.refs[0]

        df = ray.get(ref)
        ctx[out.key] = df

    def __call__(self, index_value=None, columns_value=None, dtypes=None):
        shape = (np.nan, len(dtypes))
        return self.new_dataframe(
            None,
            shape,
            dtypes=dtypes,
            index_value=index_value,
            columns_value=columns_value,
        )


def read_ray_dataset(ds, columns=None, incremental_index=False, **kwargs):
    assert isinstance(ds, real_ray_dataset.Dataset)
    refs = ds.to_pandas_refs()
    dtypes = ds.schema().empty_table().to_pandas().dtypes
    index_value = parse_index(pd.RangeIndex(-1))
    columns_value = parse_index(dtypes.index, store_data=True)

    op = DataFrameReadRayDataset(
        refs=refs, columns=columns, incremental_index=incremental_index
    )
    return op(index_value=index_value, columns_value=columns_value, dtypes=dtypes)


# keep it for back compatibility
@functools.wraps(read_ray_dataset)
def read_raydataset(*args, **kwargs):
    warnings.warn(
        "read_raydataset has been renamed to read_ray_dataset",
        DeprecationWarning,
    )
    return read_ray_dataset(*args, **kwargs)


class DataFrameReadMLDataset(HeadOptimizedDataSource):
    _op_type_ = OperandDef.READ_MLDATASET
    _mldataset = ReferenceField("mldataset", "ray.util.data.MLDataset")
    _columns = ListField("columns")

    def __init__(self, mldataset=None, columns=None, **kw):
        super().__init__(
            _mldataset=mldataset,
            _columns=columns,
            _output_types=[OutputType.dataframe],
            **kw,
        )

    @property
    def mldataset(self):
        return self._mldataset

    def _update_key(self):
        """We can't direct generate token for mldataset when we use
        ray client, so we use all mldataset's actor_id to generate
        token.
        """
        datas = []
        for value in self._values_:
            if isinstance(value, ray.util.data.MLDataset):
                actor_sets = [
                    ([str(actor) for actor in actor_set.actors], actor_set.transforms)
                    for actor_set in value.actor_sets
                ]
                datas.append(actor_sets)
                continue
            datas.append(value)
        self._obj_set("_key", tokenize(type(self).__name__, *datas))
        return self

    def __call__(self, dtypes, nrows: int):
        columns_value = parse_index(dtypes.index, store_data=True)
        index_value = parse_index(pd.RangeIndex(nrows))
        return self.new_dataframe(
            None,
            (nrows, len(dtypes)),
            dtypes=dtypes,
            index_value=index_value,
            columns_value=columns_value,
        )

    @classmethod
    def tile(cls, op: "DataFrameReadMLDataset"):
        count_iter = op.mldataset.for_each(lambda df: len(df))
        nsplits = [sum(shard) for shard in count_iter.shards()]
        nsplits_acc = np.cumsum(nsplits)
        out_df = op.outputs[0]
        out_chunks = []
        for shard_index in range(op.mldataset.num_shards()):
            chunk_op = op.copy().reset_key()
            # Make chunk key unique, otherwise all chunk will have same key.
            # See `DataFrameFromRecords#tile`
            chunk_op.extra_params["shard_index"] = shard_index
            shape = (nsplits[shard_index], out_df.shape[1])
            begin_index = nsplits_acc[shard_index] - nsplits[shard_index]
            end_index = nsplits_acc[shard_index]
            index = parse_index(pd.RangeIndex(start=begin_index, stop=end_index))
            new_chunk = chunk_op.new_chunk(
                None,
                shape=shape,
                index=(shard_index, 0),
                index_value=index,
                columns_value=out_df.columns_value,
                dtypes=out_df.dtypes,
            )
            out_chunks.append(new_chunk)
        new_op = op.copy()
        nsplits = ((np.nan,) * len(out_chunks), (out_df.shape[1],))
        return new_op.new_dataframes(
            None,
            out_df.shape,
            dtypes=out_df.dtypes,
            index_value=out_df.index_value,
            columns_value=out_df.columns_value,
            chunks=out_chunks,
            nsplits=nsplits,
        )

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        shard = op.mldataset.get_shard(chunk.index[0])
        pd_dfs = list(shard)
        pd_df = pd.concat(pd_dfs).set_index(chunk.index_value.to_pandas())
        ctx[chunk.key] = pd_df


def read_ray_mldataset(mldataset, **kwargs):
    import ray.util.data

    assert isinstance(mldataset, ray.util.data.MLDataset)
    not_empty_dfs = mldataset.filter(lambda df: len(df) > 0).take(1)
    if not not_empty_dfs:
        raise ValueError(
            f"MLDataset {mldataset} is empty, please provide an non-empty dataset."
        )
    df_record: pd.DataFrame = not_empty_dfs[0]
    columns = df_record.columns.names
    nrows = sum(mldataset.for_each(lambda df: len(df)).gather_async())
    op = DataFrameReadMLDataset(mldataset=mldataset, columns=columns, nrows=nrows)
    return op(df_record.dtypes, nrows)
