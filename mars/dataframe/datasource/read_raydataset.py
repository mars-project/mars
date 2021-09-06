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


import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...serialization.serializables import AnyField, BoolField, ListField, Int64Field
from ..utils import parse_index, lazy_import
from .core import IncrementalIndexDatasource, IncrementalIndexDataSourceMixin

ray = lazy_import('ray')
# Ray Datasets is available in early preview at ray.data with Ray 1.6+
# (and ray.experimental.data in Ray 1.5)
ray_dataset = lazy_import('ray.data')
ray_exp_dataset = lazy_import('ray.experimental.data')
real_ray_dataset = ray_dataset or ray_exp_dataset


class DataFrameReadRayDataset(IncrementalIndexDatasource,
                              IncrementalIndexDataSourceMixin):
    _op_type_ = OperandDef.READ_RAYDATASET

    _refs = AnyField('refs')
    _columns = ListField('columns')
    _incremental_index = BoolField('incremental_index')
    _nrows = Int64Field('nrows')

    def __init__(self, refs=None, columns=None,
                 incremental_index=None, nrows=None,
                 **kw):
        super().__init__(_refs=refs, _columns=columns,
                         _incremental_index=incremental_index,
                         _nrows=nrows,
                         **kw)

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
    def _tile_partitioned(cls, op: 'DataFrameReadRayDataset'):
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
                None, shape=shape, index=(chunk_index, 0),
                index_value=out_df.index_value,
                columns_value=out_df.columns_value,
                dtypes=dtypes)
            out_chunks.append(new_chunk)
            chunk_index += 1

        new_op = op.copy()
        nsplits = ((np.nan,) * len(out_chunks), (out_df.shape[1],))
        return new_op.new_dataframes(None, out_df.shape, dtypes=dtypes,
                                     index_value=out_df.index_value,
                                     columns_value=out_df.columns_value,
                                     chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def _tile(cls, op):
        return cls._tile_partitioned(op)

    @classmethod
    def execute(cls, ctx, op: 'DataFrameReadRayDataset'):
        out = op.outputs[0]
        ref = op.refs[0]

        df = ray.get(ref)
        ctx[out.key] = df

    def __call__(self, index_value=None, columns_value=None, dtypes=None):
        shape = (np.nan, len(dtypes))
        return self.new_dataframe(None, shape, dtypes=dtypes, index_value=index_value,
                                  columns_value=columns_value)


def read_raydataset(ds, columns=None,
                    incremental_index=False,
                    **kwargs):
    assert isinstance(ds, real_ray_dataset.Dataset)
    refs = ds.to_pandas()
    dtypes = ds.schema().empty_table().to_pandas().dtypes
    index_value = parse_index(pd.RangeIndex(-1))
    columns_value = parse_index(dtypes.index, store_data=True)

    op = DataFrameReadRayDataset(refs=refs, columns=columns,
                                 incremental_index=incremental_index)
    return op(index_value=index_value, columns_value=columns_value,
              dtypes=dtypes)
