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

from ...serialize import ListField, BoolField
from ... import opcodes as OperandDef
from ...tensor.base.sort import _validate_sort_psrs_kinds
from ..utils import parse_index, validate_axis, build_concatenated_rows_frame, standardize_range_index
from ..operands import DATAFRAME_TYPE, ObjectType
from .core import DataFrameSortOperand
from .psrs import DataFramePSRSOperandMixin, execute_sort_index


class DataFrameSortIndex(DataFrameSortOperand, DataFramePSRSOperandMixin):
    _op_type_ = OperandDef.SORT_INDEX

    _level = ListField('level')
    _sort_remaining = BoolField('sort_remaining')

    def __init__(self, level=None, sort_remaining=None, **kw):
        super(DataFrameSortIndex, self).__init__(_level=level, _sort_remaining=sort_remaining, **kw)

    @property
    def level(self):
        return self._level

    @property
    def sort_remaining(self):
        return self._sort_remaining

    @classmethod
    def tile(cls, op):
        df = op.inputs[0]

        if op.axis == 0:
            if df.chunk_shape[op.axis] == 1:
                if op.object_type == ObjectType.dataframe:
                    df = build_concatenated_rows_frame(df)
                    out_chunks = []
                    for chunk in df.chunks:
                        chunk_op = op.copy().reset_key()
                        out_chunks.append(chunk_op.new_chunk(
                            [chunk], shape=chunk.shape, index=chunk.index, index_value=chunk.index_value,
                            columns_value=chunk.columns_value, dtypes=chunk.dtypes))
                    new_op = op.copy()
                    kws = op.outputs[0].params.copy()
                    kws['nsplits'] = df.nsplits
                    kws['chunks'] = out_chunks
                    return new_op.new_dataframes(op.inputs, **kws)
                else:
                    out_chunks = []
                    for chunk in df.chunks:
                        chunk_op = op.copy().reset_key()
                        out_chunks.append(chunk_op.new_chunk(
                            [chunk], shape=chunk.shape, index=chunk.index, index_value=chunk.index_value,
                            name=chunk.name, dtype=chunk.dtype))
                    new_op = op.copy()
                    kws = op.outputs[0].params.copy()
                    kws['nsplits'] = df.nsplits
                    kws['chunks'] = out_chunks
                    return new_op.new_seriess(op.inputs, **kws)
            else:
                if op.object_type == ObjectType.dataframe:
                    df = build_concatenated_rows_frame(df)
                if op.na_position != 'last':  # pragma: no cover
                    raise NotImplementedError('Only support puts NaNs at the end.')
                # use parallel sorting by regular sampling
                return cls._tile_psrs(op, df)
        else:
            assert op.axis == 1

            sorted_columns = list(df.columns_value.to_pandas().sort_values(ascending=op.ascending))
            r = [df[sorted_columns]._inplace_tile()]
            if op.ignore_index:
                out = op.outputs[0]
                chunks = standardize_range_index(r[0].chunks, axis=0)
                new_op = op.copy()
                return new_op.new_dataframes(op.inputs, shape=out.shape, chunks=chunks,
                                             nsplits=r[0].nsplits, index_value=out.index_value,
                                             columns_value=out.columns_value, dtypes=out.dtypes)
            return r

    @classmethod
    def execute(cls, ctx, op):
        in_data = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = execute_sort_index(in_data, op)

    def _call_dataframe(self, df):
        if self.ignore_index:
            index_value = parse_index(pd.RangeIndex(df.shape[0]))
        else:
            index_value = df.index_value
        if self.axis == 0:
            return self.new_dataframe([df], shape=df.shape, dtypes=df.dtypes,
                                      index_value=index_value,
                                      columns_value=df.columns_value)
        else:
            dtypes = df.dtypes.sort_index(ascending=self.ascending)
            columns_value = parse_index(dtypes.index, store_data=True)
            return self.new_dataframe([df], shape=df.shape, dtypes=dtypes,
                                      index_value=index_value,
                                      columns_value=columns_value)

    def _call_series(self, series):
        if self.axis != 0:  # pragma: no cover
            raise TypeError('Invalid axis: {}'.format(self.axis))
        if self.ignore_index:
            index_value = parse_index(pd.RangeIndex(series.shape[0]))
        else:
            index_value = series.index_value

        return self.new_series([series], shape=series.shape, dtype=series.dtype,
                               index_value=index_value, name=series.name)

    def __call__(self, a):
        if isinstance(a, DATAFRAME_TYPE):
            self._object_type = ObjectType.dataframe
            return self._call_dataframe(a)
        else:
            self._object_type = ObjectType.series
            return self._call_series(a)


def sort_index(a, axis=0, level=None, ascending=True, inplace=False, kind='quicksort',
               na_position='last', sort_remaining=True, ignore_index: bool = False,
               parallel_kind='PSRS', psrs_kinds=None):
    """
    Sort object by labels (along an axis).

    Parameters
    ----------
    a : Input DataFrame or Series.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        The axis along which to sort.  The value 0 identifies the rows,
        and 1 identifies the columns.
    level : int or level name or list of ints or list of level names
        If not None, sort on values in specified index level(s).
    ascending : bool, default True
        Sort ascending vs. descending.
    inplace : bool, default False
        If True, perform operation in-place.
    kind : {'quicksort', 'mergesort', 'heapsort'}, default 'quicksort'
        Choice of sorting algorithm. See also ndarray.np.sort for more
        information.  `mergesort` is the only stable algorithm. For
        DataFrames, this option is only applied when sorting on a single
        column or label.
    na_position : {'first', 'last'}, default 'last'
        Puts NaNs at the beginning if `first`; `last` puts NaNs at the end.
        Not implemented for MultiIndex.
    sort_remaining : bool, default True
        If True and sorting by level and index is multilevel, sort by other
        levels too (in order) after sorting by specified level.
    ignore_index : bool, default False
        If True, the resulting axis will be labeled 0, 1, …, n - 1.
    parallel_kind: {'PSRS'}, optional.
        Parallel sorting algorithm, for the details, refer to:
        http://csweb.cs.wfu.edu/bigiron/LittleFE-PSRS/build/html/PSRSalgorithm.html
    psrs_kinds: Sorting algorithms during PSRS algorithm.

    Returns
    -------
    sorted_obj : DataFrame or None
        DataFrame with sorted index if inplace=False, None otherwise.
    """
    if na_position not in ['last', 'first']:  # pragma: no cover
        raise TypeError('Invalid na_position: {}'.format(na_position))
    psrs_kinds = _validate_sort_psrs_kinds(psrs_kinds)
    axis = validate_axis(axis, a)
    level = level if isinstance(level, (list, tuple)) else [level]
    op = DataFrameSortIndex(level=level, axis=axis, ascending=ascending, inplace=inplace,
                            kind=kind, na_position=na_position, sort_remaining=sort_remaining,
                            ignore_index=ignore_index, parallel_kind=parallel_kind,
                            psrs_kinds=psrs_kinds)
    sorted_a = op(a)
    if inplace:
        a.data = sorted_a.data
    else:
        return sorted_a
