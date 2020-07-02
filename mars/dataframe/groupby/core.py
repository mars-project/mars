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

import itertools
from functools import partial

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...core import Entity, Base
from ...lib.groupby_wrapper import wrapped_groupby
from ...operands import OperandStage
from ...serialize import BoolField, Int32Field, AnyField
from ...utils import get_shuffle_input_keys_idxes
from ..align import align_dataframe_series, align_series_series
from ..initializer import Series as asseries
from ..core import SERIES_TYPE, SERIES_CHUNK_TYPE
from ..utils import build_concatenated_rows_frame, hash_dataframe_on, \
    build_empty_df, build_empty_series, parse_index
from ..operands import DataFrameOperandMixin, \
    DataFrameMapReduceOperand, DataFrameShuffleProxy, ObjectType


class DataFrameGroupByOperand(DataFrameMapReduceOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY

    _by = AnyField('by', on_serialize=lambda x: x.data if isinstance(x, Entity) else x)
    _level = AnyField('level')
    _as_index = BoolField('as_index')
    _sort = BoolField('sort')
    _group_keys = BoolField('group_keys')

    _shuffle_size = Int32Field('shuffle_size')

    def __init__(self, by=None, level=None, as_index=None, sort=None, group_keys=None,
                 shuffle_size=None, stage=None, shuffle_key=None, object_type=None, **kw):
        super().__init__(_by=by, _level=level, _as_index=as_index, _sort=sort,
                         _group_keys=group_keys, _shuffle_size=shuffle_size, _stage=stage,
                         _shuffle_key=shuffle_key, _object_type=object_type, **kw)
        if stage in (OperandStage.map, OperandStage.reduce):
            if object_type in (ObjectType.dataframe, ObjectType.dataframe_groupby):
                object_type = ObjectType.dataframe
            else:
                object_type = ObjectType.series
        else:
            if object_type in (ObjectType.dataframe, ObjectType.dataframe_groupby):
                object_type = ObjectType.dataframe_groupby
            elif object_type == ObjectType.series:
                object_type = ObjectType.series_groupby
        self._object_type = object_type

    @property
    def by(self):
        return self._by

    @property
    def level(self):
        return self._level

    @property
    def as_index(self):
        return self._as_index

    @property
    def sort(self):
        return self._sort

    @property
    def group_keys(self):
        return self._group_keys

    @property
    def shuffle_size(self):
        return self._shuffle_size

    @property
    def is_dataframe_obj(self):
        return self._object_type in (ObjectType.dataframe_groupby, ObjectType.dataframe)

    @property
    def groupby_params(self):
        return dict(by=self.by, level=self.level, as_index=self.as_index, sort=self.sort,
                    group_keys=self.group_keys)

    def build_mock_groupby(self, **kwargs):
        in_df = self.inputs[0]
        if self.is_dataframe_obj:
            empty_df = build_empty_df(in_df.dtypes, index=pd.RangeIndex(2))
            obj_dtypes = in_df.dtypes[in_df.dtypes == np.dtype('O')]
            empty_df[obj_dtypes.index] = 'O'
        else:
            if in_df.dtype == np.dtype('O'):
                empty_df = pd.Series('O', index=pd.RangeIndex(2), name=in_df.name, dtype=np.dtype('O'))
            else:
                empty_df = build_empty_series(in_df.dtype, index=pd.RangeIndex(2), name=in_df.name)

        new_kw = self.groupby_params
        new_kw.update(kwargs)
        if new_kw.get('level'):
            new_kw['level'] = 0
        if isinstance(new_kw['by'], list):
            new_by = []
            for v in new_kw['by']:
                if isinstance(v, (Base, Entity)):
                    new_by.append(build_empty_series(v.dtype, index=pd.RangeIndex(2), name=v.name))
                else:
                    new_by.append(v)
            new_kw['by'] = new_by
        return empty_df.groupby(**new_kw)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs[1:])
        if len(inputs) > 1:
            by = []
            for k in self._by:
                if isinstance(k, (SERIES_TYPE, SERIES_CHUNK_TYPE)):
                    by.append(next(inputs_iter))
                else:
                    by.append(k)
            self._by = by

    def __call__(self, df):
        params = df.params.copy()
        params['index_value'] = parse_index(None, df.key, df.index_value.key)
        if df.ndim == 2:
            if isinstance(self.by, list):
                index, types = [], []
                for k in self.by:
                    if isinstance(k, str):
                        if k not in df.dtypes:
                            raise KeyError(k)
                        else:
                            index.append(k)
                            types.append(df.dtypes[k])
                    else:
                        assert isinstance(k, SERIES_TYPE)
                        index.append(k.name)
                        types.append(k.dtype)
                params['key_dtypes'] = pd.Series(types, index=index)

        inputs = [df]
        if isinstance(self.by, list):
            for k in self.by:
                if isinstance(k, SERIES_TYPE):
                    inputs.append(k)

        return self.new_tileable(inputs, **params)

    @classmethod
    def _align_input_and_by(cls, op, inp, by):
        align_method = partial(align_dataframe_series, axis='index') \
            if op.is_dataframe_obj else align_series_series
        nsplits, _, inp_chunks, by_chunks = align_method(inp, by)

        inp_params = inp.params
        inp_params['chunks'] = inp_chunks
        inp_params['nsplits'] = nsplits
        inp = op.copy().new_tileable(op.inputs, kws=[inp_params])

        by_params = by.params
        by_params['chunks'] = by_chunks
        if len(nsplits) == 2:
            by_nsplits = nsplits[:1]
        else:
            by_nsplits = nsplits
        by_params['nsplits'] = by_nsplits
        by = by.op.copy().new_tileable(by.op.inputs, kws=[by_params])

        return inp, by

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        by = op.by

        series_in_by = False
        new_inputs = []
        if len(op.inputs) > 1:
            # by series
            new_by = []
            for k in by:
                if isinstance(k, SERIES_TYPE):
                    in_df, k = cls._align_input_and_by(op, in_df, k)
                    if len(new_inputs) == 0:
                        new_inputs.append(in_df)
                    new_inputs.append(k)
                    series_in_by = True
                new_by.append(k)
            by = new_by
        else:
            new_inputs = op.inputs

        is_dataframe_obj = op.is_dataframe_obj
        if is_dataframe_obj:
            in_df = build_concatenated_rows_frame(in_df)
            out_object_type = ObjectType.dataframe
            chunk_shape = (in_df.chunk_shape[0], 1)
        else:
            out_object_type = ObjectType.series
            chunk_shape = (in_df.chunk_shape[0],)

        # generate map chunks
        map_chunks = []
        for chunk in in_df.chunks:
            map_op = op.copy().reset_key()
            map_op._stage = OperandStage.map
            map_op._shuffle_size = chunk_shape[0]
            chunk_inputs = [chunk]
            if len(op.inputs) > 1:
                chunk_by = []
                for k in by:
                    if isinstance(k, SERIES_TYPE):
                        by_chunk = k.cix[chunk.index[0], ]
                        chunk_by.append(by_chunk)
                        chunk_inputs.append(by_chunk)
                    else:
                        chunk_by.append(k)
                map_op._by = chunk_by
            map_chunks.append(map_op.new_chunk(chunk_inputs, shape=(np.nan, np.nan), index=chunk.index))

        proxy_chunk = DataFrameShuffleProxy(object_type=out_object_type).new_chunk(map_chunks, shape=())

        # generate reduce chunks
        reduce_chunks = []
        for out_idx in itertools.product(*(range(s) for s in chunk_shape)):
            shuffle_key = ','.join(str(idx) for idx in out_idx)
            reduce_op = op.copy().reset_key()
            reduce_op._by = None
            reduce_op._stage = OperandStage.reduce
            reduce_op._shuffle_key = shuffle_key
            reduce_chunks.append(
                reduce_op.new_chunk([proxy_chunk], shape=(np.nan, np.nan), index=out_idx))

        # generate groupby chunks
        out_chunks = []
        for chunk in reduce_chunks:
            groupby_op = op.copy().reset_key()
            if series_in_by:
                # set by to None, cuz data of by will be passed from map to reduce to groupby
                groupby_op._by = None
            if is_dataframe_obj:
                new_shape = (np.nan, in_df.shape[1])
            else:
                new_shape = (np.nan,)
            params = dict(shape=new_shape, index=chunk.index)
            if op.is_dataframe_obj:
                params.update(dict(dtypes=in_df.dtypes, columns_value=in_df.columns_value,
                                   index_value=parse_index(None, chunk.key, proxy_chunk.key)))
            else:
                params.update(dict(name=in_df.name, dtype=in_df.dtype,
                                   index_value=parse_index(None, chunk.key, proxy_chunk.key)))
            out_chunks.append(groupby_op.new_chunk([chunk], **params))

        new_op = op.copy()
        params = op.outputs[0].params.copy()
        if is_dataframe_obj:
            params['nsplits'] = ((np.nan,) * len(out_chunks), (in_df.shape[1],))
        else:
            params['nsplits'] = ((np.nan,) * len(out_chunks),)
        params['chunks'] = out_chunks
        return new_op.new_tileables(new_inputs, **params)

    @classmethod
    def execute_map(cls, ctx, op):
        is_dataframe_obj = op.is_dataframe_obj
        by = op.by
        chunk = op.outputs[0]
        df = ctx[op.inputs[0].key]

        deliver_by = False  # output by for the upcoming process
        if isinstance(by, list):
            new_by = []
            for v in by:
                if isinstance(v, Base):
                    deliver_by = True
                    new_by.append(ctx[v.key])
                else:
                    new_by.append(v)
            by = new_by

        if isinstance(by, list) or callable(by):
            on = by
        else:
            on = None
        filters = hash_dataframe_on(df, on, op.shuffle_size, level=op.level)

        for index_idx, index_filter in enumerate(filters):
            if is_dataframe_obj:
                group_key = ','.join([str(index_idx), str(chunk.index[1])])
            else:
                group_key = str(index_idx)

            if deliver_by:
                filtered_by = []
                for v in by:
                    if isinstance(v, pd.Series):
                        filtered_by.append(v.loc[index_filter])
                    else:
                        filtered_by.append(v)
                ctx[(chunk.key, group_key)] = (df.loc[index_filter], filtered_by)
            else:
                ctx[(chunk.key, group_key)] = df.loc[index_filter]

    @classmethod
    def execute_reduce(cls, ctx, op):
        is_dataframe_obj = op.inputs[0].op.object_type == ObjectType.dataframe
        chunk = op.outputs[0]
        input_keys, input_idxes = get_shuffle_input_keys_idxes(op.inputs[0])
        input_idx_to_df = {idx: ctx[inp_key, ','.join(str(ix) for ix in chunk.index)]
                           for inp_key, idx in zip(input_keys, input_idxes)}
        row_idxes = sorted({idx[0] for idx in input_idx_to_df})

        res = []
        for row_idx in row_idxes:
            if is_dataframe_obj:
                row_df = input_idx_to_df.get((row_idx, 0), None)
            else:
                row_df = input_idx_to_df.get((row_idx,), None)
            if row_df is not None:
                res.append(row_df)
        by = None
        if isinstance(res[0], tuple):
            # By is series
            r = pd.concat([it[0] for it in res], axis=0)
            by = [None] * len(res[0][1])
            for it in res:
                for i, v in enumerate(it[1]):
                    if isinstance(v, pd.Series):
                        if by[i] is None:
                            by[i] = v
                        else:
                            by[i] = pd.concat([by[i], v], axis=0)
                    else:
                        by[i] = v
        else:
            r = pd.concat(res, axis=0)
        if chunk.index_value is not None:
            r.index.name = chunk.index_value.name
        if by is None:
            ctx[chunk.key] = r
        else:
            ctx[chunk.key] = (r, by)

    @classmethod
    def execute(cls, ctx, op: "DataFrameGroupByOperand"):
        if op.stage == OperandStage.map:
            cls.execute_map(ctx, op)
        elif op.stage == OperandStage.reduce:
            cls.execute_reduce(ctx, op)
        else:
            inp = ctx[op.inputs[0].key]
            if isinstance(inp, tuple):
                # df, by
                df, by = inp
            else:
                df = inp
                by = op.by
            ctx[op.outputs[0].key] = wrapped_groupby(
                df, by=by, level=op.level, as_index=op.as_index, sort=op.sort,
                group_keys=op.group_keys)


def groupby(df, by=None, level=None, as_index=True, sort=True, group_keys=True):
    if not as_index and df.op.object_type == ObjectType.series:
        raise TypeError('as_index=False only valid with DataFrame')

    object_type = ObjectType.dataframe_groupby if df.ndim == 2 else ObjectType.series_groupby
    if isinstance(by, (str, SERIES_TYPE, pd.Series)):
        if isinstance(by, pd.Series):
            by = asseries(by)
        by = [by]
    op = DataFrameGroupByOperand(by=by, level=level, as_index=as_index, sort=sort,
                                 group_keys=group_keys, object_type=object_type)
    return op(df)
