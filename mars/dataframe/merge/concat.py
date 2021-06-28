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

import pandas as pd
import numpy as np

from ... import opcodes as OperandDef
from ...core import ENTITY_TYPE, OutputType, recursive_tile
from ...serialization.serializables import FieldTypes, ListField, StringField, \
    BoolField, AnyField
from ...utils import lazy_import, has_unknown_shape
from ..operands import DataFrameOperand, DataFrameOperandMixin, SERIES_TYPE
from ..utils import parse_index, build_empty_df, build_empty_series, \
    standardize_range_index, validate_axis

cudf = lazy_import('cudf', globals=globals())


class DataFrameConcat(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.CONCATENATE

    _axis = AnyField('axis')
    _join = StringField('join')
    _ignore_index = BoolField('ignore_index')
    _keys = ListField('keys')
    _levels = ListField('levels')
    _names = ListField('names')
    _verify_integrity = BoolField('verify_integrity')
    _sort = BoolField('sort')
    _copy = BoolField('copy')

    def __init__(self, axis=None, join=None, ignore_index=None,
                 keys=None, levels=None, names=None, verify_integrity=None,
                 sort=None, copy=None, sparse=None, output_types=None, **kw):
        super().__init__(
            _axis=axis, _join=join, _ignore_index=ignore_index,
            _keys=keys, _levels=levels, _names=names,
            _verify_integrity=verify_integrity, _sort=sort, _copy=copy,
            _sparse=sparse, _output_types=output_types, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def join(self):
        return self._join

    @property
    def ignore_index(self):
        return self._ignore_index

    @property
    def keys(self):
        return self._keys

    @property
    def level(self):
        return self._levels

    @property
    def name(self):
        return self._names

    @property
    def verify_integrity(self):
        return self._verify_integrity

    @property
    def sort(self):
        return self._sort

    @property
    def copy_(self):
        return self._copy

    @classmethod
    def _tile_dataframe(cls, op):
        from ..indexing.iloc import DataFrameIlocGetItem

        out_df = op.outputs[0]
        inputs = op.inputs
        axis = op.axis

        if not all(inputs[i].nsplits[1 - axis] == inputs[i + 1].nsplits[1 - axis]
                   for i in range(len(inputs) - 1)):
            # need rechunk
            if has_unknown_shape(*inputs):
                yield
            normalized_nsplits = {1 - axis: inputs[0].nsplits[1 - axis]}
            new_inputs = []
            for inp in inputs:
                new_inputs.append(
                    (yield from recursive_tile(inp.rechunk(normalized_nsplits))))
            inputs = new_inputs

        out_chunks = []
        nsplits = []
        cum_index = 0
        for df in inputs:
            for c in df.chunks:
                if op.axis == 0:
                    index = (c.index[0] + cum_index, c.index[1])
                else:
                    index = (c.index[0], c.index[1] + cum_index)

                iloc_op = DataFrameIlocGetItem(indexes=[slice(None)] * 2)
                out_chunks.append(iloc_op.new_chunk([c], shape=c.shape, index=index,
                                                    dtypes=c.dtypes,
                                                    index_value=c.index_value,
                                                    columns_value=c.columns_value))
            nsplits.extend(df.nsplits[op.axis])
            cum_index += len(df.nsplits[op.axis])
        out_nsplits = (tuple(nsplits), inputs[0].nsplits[1]) \
            if op.axis == 0 else (inputs[0].nsplits[0], tuple(nsplits))

        if op.ignore_index:
            out_chunks = standardize_range_index(out_chunks)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, out_df.shape,
                                     nsplits=out_nsplits, chunks=out_chunks,
                                     dtypes=out_df.dtypes,
                                     index_value=out_df.index_value,
                                     columns_value=out_df.columns_value)

    @classmethod
    def _tile_series(cls, op):
        from ..datasource.from_tensor import DataFrameFromTensor
        from ..indexing.iloc import SeriesIlocGetItem, DataFrameIlocGetItem

        out = op.outputs[0]
        inputs = op.inputs
        out_chunks = []

        if op.axis == 1:
            if has_unknown_shape(*inputs):
                yield
            new_inputs = []
            for inp in inputs:
                new_inputs.append(
                    (yield from recursive_tile(inp.rechunk(op.inputs[0].nsplits))))
            inputs = new_inputs

        cum_index = 0
        offset = 0
        nsplits = []
        for series in inputs:
            for c in series.chunks:
                if op.axis == 0:
                    index = (c.index[0] + cum_index,)
                    shape = c.shape
                    iloc_op = SeriesIlocGetItem(indexes=(slice(None),))
                    out_chunks.append(iloc_op.new_chunk([c], shape=shape, index=index,
                                                        index_value=c.index_value,
                                                        dtype=c.dtype,
                                                        name=c.name))
                else:
                    index = (c.index[0], cum_index)
                    shape = (c.shape[0], 1)
                    to_frame_op = DataFrameFromTensor(input_=c)
                    if c.name:
                        dtypes = pd.Series([c.dtype], index=[c.name])
                    else:
                        dtypes = pd.Series([c.dtype], index=pd.RangeIndex(offset, offset + 1))
                    df_chunk = to_frame_op.new_chunk(
                        [c], shape=shape, index=index, index_value=c.index_value,
                        columns_value=parse_index(dtypes.index, store_data=True),
                        dtypes=dtypes)
                    iloc_op = DataFrameIlocGetItem(indexes=[slice(None)] * 2)
                    out_chunks.append(iloc_op.new_chunk([df_chunk], shape=df_chunk.shape,
                                                        index=index,
                                                        dtypes=df_chunk.dtypes,
                                                        index_value=df_chunk.index_value,
                                                        columns_value=df_chunk.columns_value))

            if op.axis == 0:
                nsplits.extend(series.nsplits[0])
                cum_index += len(series.nsplits[op.axis])
            else:
                nsplits.append(1)
                cum_index += 1
                offset += 1

        if op.ignore_index:
            out_chunks = standardize_range_index(out_chunks)

        new_op = op.copy()
        if op.axis == 0:
            nsplits = (tuple(nsplits),)
            return new_op.new_seriess(op.inputs, out.shape,
                                      nsplits=nsplits, chunks=out_chunks,
                                      dtype=out.dtype,
                                      index_value=out.index_value,
                                      name=out.name)
        else:
            nsplits = (inputs[0].nsplits[0], tuple(nsplits))
            return new_op.new_dataframes(op.inputs, out.shape,
                                         nsplits=nsplits, chunks=out_chunks,
                                         dtypes=out.dtypes,
                                         index_value=out.index_value,
                                         columns_value=out.columns_value)

    @classmethod
    def tile(cls, op):
        if isinstance(op.inputs[0], SERIES_TYPE):
            return (yield from cls._tile_series(op))
        else:
            return (yield from cls._tile_dataframe(op))

    @classmethod
    def execute(cls, ctx, op):
        def _base_concat(chunk, inputs):
            # auto generated concat when executing a DataFrame, Series or Index
            if chunk.op.output_types[0] == OutputType.dataframe:
                return _auto_concat_dataframe_chunks(chunk, inputs)
            elif chunk.op.output_types[0] == OutputType.series:
                return _auto_concat_series_chunks(chunk, inputs)
            elif chunk.op.output_types[0] == OutputType.index:
                return _auto_concat_index_chunks(chunk, inputs)
            elif chunk.op.output_types[0] == OutputType.categorical:
                return _auto_concat_categorical_chunks(chunk, inputs)
            else:  # pragma: no cover
                raise TypeError('Only DataFrameChunk, SeriesChunk, IndexChunk, '
                                'and CategoricalChunk can be automatically concatenated')

        def _auto_concat_dataframe_chunks(chunk, inputs):
            xdf = pd if isinstance(inputs[0], (pd.DataFrame, pd.Series)) or cudf is None else cudf

            if chunk.op.axis is not None:
                return xdf.concat(inputs, axis=op.axis)

            # auto generated concat when executing a DataFrame
            if len(inputs) == 1:
                ret = inputs[0]
            else:
                n_rows = len(set(inp.index[0] for inp in chunk.inputs))
                n_cols = int(len(inputs) // n_rows)
                assert n_rows * n_cols == len(inputs)

                concats = []
                for i in range(n_rows):
                    if n_cols == 1:
                        concats.append(inputs[i])
                    else:
                        concat = xdf.concat([inputs[i * n_cols + j] for j in range(n_cols)], axis=1)
                        concats.append(concat)

                if xdf is pd:
                    # The `sort=False` is to suppress a `FutureWarning` of pandas,
                    # when the index or column of chunks to concatenate is not aligned,
                    # which may happens for certain ops.
                    #
                    # See also Note [Columns of Left Join] in test_merge_execution.py.
                    ret = xdf.concat(concats, sort=False)
                else:
                    ret = xdf.concat(concats)
                    # cuDF will lost index name when concat two seriess.
                    ret.index.name = concats[0].index.name

            if getattr(chunk.index_value, 'should_be_monotonic', False):
                ret.sort_index(inplace=True)
            if getattr(chunk.columns_value, 'should_be_monotonic', False):
                ret.sort_index(axis=1, inplace=True)
            return ret

        def _auto_concat_series_chunks(chunk, inputs):
            # auto generated concat when executing a Series
            if len(inputs) == 1:
                concat = inputs[0]
            else:
                xdf = pd if isinstance(inputs[0], pd.Series) or cudf is None else cudf
                if chunk.op.axis is not None:
                    concat = xdf.concat(inputs, axis=chunk.op.axis)
                else:
                    concat = xdf.concat(inputs)
            if getattr(chunk.index_value, 'should_be_monotonic', False):
                concat.sort_index(inplace=True)
            return concat

        def _auto_concat_index_chunks(chunk, inputs):
            if len(inputs) == 1:
                xdf = pd if isinstance(inputs[0], pd.Index) or cudf is None else cudf
                concat_df = xdf.DataFrame(index=inputs[0])
            else:
                xdf = pd if isinstance(inputs[0], pd.Index) or cudf is None else cudf
                empty_dfs = [xdf.DataFrame(index=inp) for inp in inputs]
                concat_df = xdf.concat(empty_dfs, axis=0)
            if getattr(chunk.index_value, 'should_be_monotonic', False):
                concat_df.sort_index(inplace=True)
            return concat_df.index

        def _auto_concat_categorical_chunks(_, inputs):
            if len(inputs) == 1:  # pragma: no cover
                return inputs[0]
            else:
                # convert categorical into array
                arrays = [np.asarray(inp) for inp in inputs]
                array = np.concatenate(arrays)
                return pd.Categorical(array, categories=inputs[0].categories,
                                      ordered=inputs[0].ordered)

        chunk = op.outputs[0]
        inputs = [ctx[input.key] for input in op.inputs]

        if isinstance(inputs[0], tuple):
            ctx[chunk.key] = tuple(_base_concat(chunk, [input[i] for input in inputs])
                                   for i in range(len(inputs[0])))
        else:
            ctx[chunk.key] = _base_concat(chunk, inputs)

    @classmethod
    def _concat_index(cls, prev_index: pd.Index, cur_index: pd.Index):
        if isinstance(prev_index, pd.RangeIndex) and \
                isinstance(cur_index, pd.RangeIndex):
            # handle RangeIndex that append may generate huge amount of data
            # e.g. pd.RangeIndex(10_000) and pd.RangeIndex(10_000)
            # will generate a Int64Index full of data
            # for details see GH#1647
            prev_stop = prev_index.start + prev_index.size * prev_index.step
            cur_start = cur_index.start
            if prev_stop == cur_start and prev_index.step == cur_index.step:
                # continuous RangeIndex, still return RangeIndex
                return prev_index.append(cur_index)
            else:
                # otherwise, return an empty index
                return pd.Index([], dtype=prev_index.dtype)
        elif isinstance(prev_index, pd.RangeIndex):
            return pd.Index([], prev_index.dtype).append(cur_index)
        elif isinstance(cur_index, pd.RangeIndex):
            return prev_index.append(pd.Index([], cur_index.dtype))
        return prev_index.append(cur_index)

    def _call_series(self, objs):
        if self.axis == 0:
            row_length = 0
            index = None
            for series in objs:
                if index is None:
                    index = series.index_value.to_pandas()
                else:
                    index = self._concat_index(index, series.index_value.to_pandas())
                row_length += series.shape[0]
            if self.ignore_index:  # pragma: no cover
                index_value = parse_index(pd.RangeIndex(row_length))
            else:
                index_value = parse_index(index, objs)
            return self.new_series(objs, shape=(row_length,), dtype=objs[0].dtype,
                                   index_value=index_value, name=objs[0].name)
        else:
            col_length = 0
            columns = []
            dtypes = dict()
            undefined_name = 0
            for series in objs:
                if series.name is None:
                    dtypes[undefined_name] = series.dtype
                    undefined_name += 1
                    columns.append(undefined_name)
                else:
                    dtypes[series.name] = series.dtype
                    columns.append(series.name)
                col_length += 1
            if self.ignore_index or undefined_name == len(objs):
                columns_value = parse_index(pd.RangeIndex(col_length))
            else:
                columns_value = parse_index(pd.Index(columns), store_data=True)

            shape = (objs[0].shape[0], col_length)
            return self.new_dataframe(objs, shape=shape, dtypes=pd.Series(dtypes),
                                      index_value=objs[0].index_value,
                                      columns_value=columns_value)

    def _call_dataframes(self, objs):
        if self.axis == 0:
            row_length = 0
            index = None
            empty_dfs = []
            for df in objs:
                if index is None:
                    index = df.index_value.to_pandas()
                else:
                    index = self._concat_index(index, df.index_value.to_pandas())
                row_length += df.shape[0]
                if df.ndim == 2:
                    empty_dfs.append(build_empty_df(df.dtypes))
                else:
                    empty_dfs.append(build_empty_series(df.dtype, name=df.name))

            emtpy_result = pd.concat(empty_dfs, join=self.join, sort=True)
            shape = (row_length, emtpy_result.shape[1])
            columns_value = parse_index(emtpy_result.columns, store_data=True)

            if self.join == 'inner':
                objs = [o[list(emtpy_result.columns)] for o in objs]

            if self.ignore_index:  # pragma: no cover
                index_value = parse_index(pd.RangeIndex(row_length))
            else:
                index_value = parse_index(index, objs)

            new_objs = []
            for obj in objs:
                if obj.ndim != 2:
                    # series
                    new_obj = obj.to_frame().reindex(columns=emtpy_result.dtypes.index)
                else:
                    # dataframe
                    if list(obj.dtypes.index) != list(emtpy_result.dtypes.index):
                        new_obj = obj.reindex(columns=emtpy_result.dtypes.index)
                    else:
                        new_obj = obj
                new_objs.append(new_obj)

            return self.new_dataframe(new_objs, shape=shape, dtypes=emtpy_result.dtypes,
                                      index_value=index_value, columns_value=columns_value)
        else:
            col_length = 0
            empty_dfs = []
            for df in objs:
                if df.ndim == 2:
                    # DataFrame
                    col_length += df.shape[1]
                    empty_dfs.append(build_empty_df(df.dtypes))
                else:
                    # Series
                    col_length += 1
                    empty_dfs.append(build_empty_series(df.dtype, name=df.name))

            emtpy_result = pd.concat(empty_dfs, join=self.join, axis=1, sort=True)
            if self.ignore_index:
                columns_value = parse_index(pd.RangeIndex(col_length))
            else:
                columns_value = parse_index(pd.Index(emtpy_result.columns), store_data=True)

            if self.ignore_index or len({o.index_value.key for o in objs}) == 1:
                new_objs = [obj if obj.ndim == 2 else obj.to_frame()
                            for obj in objs]
            else:  # pragma: no cover
                raise NotImplementedError('Does not support concat dataframes '
                                          'which has different index')

            shape = (objs[0].shape[0], col_length)
            return self.new_dataframe(new_objs, shape=shape, dtypes=emtpy_result.dtypes,
                                      index_value=objs[0].index_value,
                                      columns_value=columns_value)

    def __call__(self, objs):
        if all(isinstance(obj, SERIES_TYPE) for obj in objs):
            self.output_types = [OutputType.series]
            return self._call_series(objs)
        else:
            self.output_types = [OutputType.dataframe]
            return self._call_dataframes(objs)


class GroupByConcat(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY_CONCAT

    _groups = ListField('groups', FieldTypes.key)
    _groupby_params = AnyField('groupby_params')

    def __init__(self, groups=None, groupby_params=None, output_types=None, **kw):
        super().__init__(_groups=groups, _groupby_params=groupby_params,
                         _output_types=output_types, **kw)

    @property
    def groups(self):
        return self._groups

    @property
    def groupby_params(self):
        return self._groupby_params

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)

        new_groups = []
        for _ in self._groups:
            new_groups.append(next(inputs_iter))
        self._groups = new_groups

        if isinstance(self._groupby_params['by'], list):
            by = []
            for v in self._groupby_params['by']:
                if isinstance(v, ENTITY_TYPE):
                    by.append(next(inputs_iter))
                else:
                    by.append(v)
            self._groupby_params['by'] = by

    @classmethod
    def execute(cls, ctx, op):
        input_data = [ctx[input.key] for input in op.groups]
        obj = pd.concat([d.obj for d in input_data])

        params = op.groupby_params.copy()
        if isinstance(params['by'], list):
            by = []
            for v in params['by']:
                if isinstance(v, ENTITY_TYPE):
                    by.append(ctx[v.key])
                else:
                    by.append(v)
            params['by'] = by
        selection = params.pop('selection', None)

        result = obj.groupby(**params)
        if selection:
            result = result[selection]

        ctx[op.outputs[0].key] = result


def concat(objs, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None,
           verify_integrity=False, sort=False, copy=True):
    if not isinstance(objs, (list, tuple)):  # pragma: no cover
        raise TypeError('first argument must be an iterable of dataframe or series objects')
    axis = validate_axis(axis)
    if isinstance(objs, dict):  # pragma: no cover
        keys = objs.keys()
        objs = objs.values()
    if axis == 1 and join == 'inner':  # pragma: no cover
        raise NotImplementedError('inner join is not support when specify `axis=1`')
    if verify_integrity or sort or keys:  # pragma: no cover
        raise NotImplementedError('verify_integrity, sort, keys arguments are not supported now')
    op = DataFrameConcat(axis=axis, join=join, ignore_index=ignore_index, keys=keys, levels=levels,
                         names=names, verify_integrity=verify_integrity, sort=sort, copy=copy)

    return op(objs)
