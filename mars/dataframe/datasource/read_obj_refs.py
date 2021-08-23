#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from ...config import options
from ...serialization.serializables import AnyField, BoolField, ListField, StringField, Int64Field
from ..utils import parse_index, to_arrow_dtypes, lazy_import
from .core import IncrementalIndexDatasource, IncrementalIndexDataSourceMixin

ray = lazy_import('ray')


class DataFrameReadObjRefs(IncrementalIndexDatasource,
                           IncrementalIndexDataSourceMixin):
    _op_type_ = OperandDef.READ_OBJ_REF

    _refs = AnyField('refs')
    _engine = StringField('engine')
    _columns = ListField('columns')
    _use_arrow_dtype = BoolField('use_arrow_dtype')
    _incremental_index = BoolField('incremental_index')
    _nrows = Int64Field('nrows')

    def __init__(self, refs=None, engine=None, columns=None, use_arrow_dtype=None,
                 incremental_index=None, nrows=None,
                 **kw):
        super().__init__(_refs=refs, _engine=engine, _columns=columns,
                         _use_arrow_dtype=use_arrow_dtype,
                         _incremental_index=incremental_index,
                         _nrows=nrows,
                         **kw)

    @property
    def refs(self):
        return self._refs

    @property
    def engine(self):
        return self._engine

    @property
    def columns(self):
        return self._columns

    @property
    def use_arrow_dtype(self):
        return self._use_arrow_dtype

    @property
    def incremental_index(self):
        return self._incremental_index

    @classmethod
    def _to_arrow_dtypes(cls, dtypes, op):
        if op.use_arrow_dtype is None and not op.gpu and \
                options.dataframe.use_arrow_dtype:  # pragma: no cover
            # check if use_arrow_dtype set on the server side
            dtypes = to_arrow_dtypes(dtypes)
        return dtypes

    @classmethod
    def _tile_partitioned(cls, op: "DataFrameReadObjRefs"):
        out_df = op.outputs[0]
        shape = (np.nan, out_df.shape[1])
        dtypes = cls._to_arrow_dtypes(out_df.dtypes, op)
        dataset = op.refs

        chunk_index = 0
        out_chunks = []
        for piece in dataset:
            chunk_op = op.copy().reset_key()
            chunk_op._refs = piece
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
    def execute(cls, ctx, op: "DataFrameReadObjRefs"):
        out = op.outputs[0]
        refs = op.refs

        df = ray.get(refs)
        ctx[out.key] = df

    def __call__(self, index_value=None, columns_value=None, dtypes=None):
        shape = (np.nan, len(dtypes))
        return self.new_dataframe(None, shape, dtypes=dtypes, index_value=index_value,
                                  columns_value=columns_value)


def read_obj_refs(refs, columns=None,
                  incremental_index=False,
                  **kwargs):
    dtypes = ray.get(ray.remote(_get_dtypes).remote(refs[0]))
    index_value = parse_index(pd.RangeIndex(-1))
    columns_value = parse_index(dtypes.index, store_data=True)

    op = DataFrameReadObjRefs(refs=refs, columns=columns,
                              incremental_index=incremental_index)
    return op(index_value=index_value, columns_value=columns_value,
              dtypes=dtypes)


def _get_dtypes(t):
    return t.dtypes
