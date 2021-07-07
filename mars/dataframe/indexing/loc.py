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

from numbers import Integral
from typing import Dict

import numpy as np
import pandas as pd
from pandas.core.dtypes.cast import find_common_type
from pandas.core.indexing import IndexingError

from ... import opcodes as OperandDef
from ...core import ENTITY_TYPE
from ...core.operand import OperandStage
from ...serialization.serializables import KeyField, ListField
from ...tensor.datasource import asarray
from ...tensor.utils import calc_sliced_size, filter_inputs
from ...utils import lazy_import
from ..core import IndexValue, DATAFRAME_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index
from .index_lib import DataFrameLocIndexesHandler


cudf = lazy_import('cudf', globals=globals())


def process_loc_indexes(inp, indexes):
    ndim = inp.ndim

    if not isinstance(indexes, tuple):
        indexes = (indexes,)
    if len(indexes) < ndim:
        indexes += (slice(None),) * (ndim - len(indexes))
    if len(indexes) > ndim:
        raise IndexingError('Too many indexers')

    new_indexes = []
    for ax, index in enumerate(indexes):
        if isinstance(index, (list, np.ndarray, pd.Series, ENTITY_TYPE)):
            if not isinstance(index, ENTITY_TYPE):
                index = np.asarray(index)
            else:
                index = asarray(index)
                if ax == 1:
                    # do not support tensor index on axis 1
                    # because if so, the dtypes and columns_value would be unknown
                    try:
                        index = index.fetch()
                    except (RuntimeError, ValueError):
                        raise NotImplementedError('indexer on axis columns cannot be '
                                                  'non-executed tensor')
        new_indexes.append(index)

    return new_indexes


class DataFrameLoc:
    def __init__(self, obj):
        self._obj = obj

    def _use_iloc(self, indexes):
        # for RangeIndex from 0, use iloc instead of loc
        index_value = self._obj.index_value.value
        if len(indexes) == 2:
            if not isinstance(indexes[1], slice):
                return False
            elif indexes[1] != slice(None):
                return False
        if not isinstance(index_value, IndexValue.RangeIndex):
            return False
        if index_value.slice.start != 0 and index_value.slice.start is not None:
            return False
        if not isinstance(indexes[0], (Integral, slice)):
            return False
        if isinstance(indexes[0], Integral):
            if indexes[0] < 0:
                return False
        else:
            for v in (indexes[0].start, indexes[0].stop, indexes[0].step):
                if v is None:
                    continue
                if not isinstance(v, Integral):
                    return False
                if v < 0:
                    return False
        return True

    def __getitem__(self, indexes):
        indexes = process_loc_indexes(self._obj, indexes)

        if self._use_iloc(indexes):
            # use iloc instead
            return self._obj.iloc[tuple(indexes)]

        op = DataFrameLocGetItem(indexes=indexes)
        return op(self._obj)


class DataFrameLocGetItem(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_LOC_GETITEM

    _input = KeyField('input')
    _indexes = ListField('indexes')

    def __init__(self, indexes=None, gpu=False, sparse=False, output_types=None, **kw):
        super().__init__(_indexes=indexes, _gpu=gpu, _sparse=sparse,
                         _output_types=output_types, **kw)

    @property
    def input(self):
        return self._input

    @property
    def indexes(self):
        return self._indexes

    @property
    def can_index_miss(self):
        return False

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        self._input = next(inputs_iter)
        indexes = []
        for index in self._indexes:
            if isinstance(index, ENTITY_TYPE):
                indexes.append(next(inputs_iter))
            else:
                indexes.append(index)
        self._indexes = list(indexes)

    @classmethod
    def _calc_slice_param(cls,
                          input_index_value: IndexValue,
                          pd_index: pd.Index,
                          inp,
                          index: slice,
                          axis: int) -> Dict:
        param = dict()
        if input_index_value.has_value():
            start, end = pd_index.slice_locs(
                index.start, index.stop, index.step, kind='loc')
            slc = slice(start, end, index.step)
            size = calc_sliced_size(inp.shape[axis], slc)
            param['shape'] = size
            out_index = pd_index[slc]
            param['index_value'] = parse_index(out_index,
                                               store_data=axis == 1)
            if axis == 1:
                param['dtypes'] = inp.dtypes[slc]
        else:
            assert axis == 0
            if index.start is None and index.stop is None:
                param['shape'] = calc_sliced_size(inp.shape[axis], index)
            else:
                param['shape'] = np.nan
            param['index_value'] = parse_index(pd_index, inp, index)

        return param

    @classmethod
    def _calc_bool_index_param(cls,
                               input_index_value: IndexValue,
                               pd_index: pd.Index,
                               inp,
                               index,
                               axis: int) -> Dict:
        param = dict()
        if input_index_value.has_value():
            if isinstance(index, np.ndarray):
                filtered_index = pd_index[index]
                param['shape'] = len(filtered_index)
                param['index_value'] = parse_index(filtered_index,
                                                   store_data=axis == 1)
                if axis == 1:
                    param['dtypes'] = inp.dtypes[index]
            else:
                # tensor, cannot be indexer on axis 1
                assert axis == 0
                param['shape'] = np.nan
                param['index_value'] = parse_index(pd.Index([], dtype=pd_index.dtype),
                                                   inp, index, store_data=False)
        else:
            assert axis == 0
            if isinstance(index, np.ndarray):
                param['shape'] = int(index.sum())
            else:
                param['shape'] = np.nan
            param['index_value'] = parse_index(pd_index, inp, index,
                                               store_data=False)

        return param

    @classmethod
    def _calc_fancy_index_param(cls,
                                input_index_value: IndexValue,
                                pd_index: pd.Index,
                                inp,
                                index,
                                axis: int) -> Dict:
        param = dict()
        if input_index_value.has_value():
            if isinstance(index, np.ndarray):
                if not pd_index.is_unique:
                    assert axis == 1
                    # as there's no direct method in pandas to handle fancy indexes
                    # we creates a empty
                    new_dtypes = inp.dtypes.loc[index]
                    param['shape'] = len(new_dtypes)
                    param['index_value'] = parse_index(new_dtypes.index, store_data=True)
                    param['dtypes'] = new_dtypes
                else:
                    for it in index:
                        if it not in pd_index:
                            axis_name = 'index' if axis == 0 else 'columns'
                            raise KeyError(f'Label [{it}] not found in the [{axis_name}]')
                    param['shape'] = len(index)
                    param['index_value'] = parse_index(pd.Index(index), store_data=True)
                    if axis == 1:
                        param['dtypes'] = inp.dtypes[index]
            else:
                assert axis == 0
                param['shape'] = index.shape[0]
                param['index_value'] = parse_index(pd.Index([], dtype=pd_index.dtype),
                                                   inp, index)
        else:
            assert axis == 0
            param['shape'] = index.shape[0]
            param['index_value'] = parse_index(pd_index, inp, index)

        return param

    @classmethod
    def _calc_param(cls, inp, axis: int, index) -> Dict:
        input_index_value = inp.index_value if axis == 0 else inp.columns_value
        pd_index = input_index_value.to_pandas()

        if isinstance(index, slice):
            return cls._calc_slice_param(input_index_value,
                                         pd_index, inp, index, axis)
        elif hasattr(index, 'dtype') and index.ndim == 1:
            if index.dtype == np.bool:
                # bool indexing
                return cls._calc_bool_index_param(input_index_value,
                                                  pd_index, inp, index, axis)
            else:
                # fancy indexing
                return cls._calc_fancy_index_param(input_index_value,
                                                   pd_index, inp, index, axis)
        else:
            param = dict()
            if input_index_value.has_value():
                loc = pd_index.get_loc(index)
                if isinstance(loc, (slice, np.ndarray)):
                    assert axis == 1
                    new_dtypes = inp.dtypes[loc]
                    param['shape'] = len(new_dtypes)
                    param['index_value'] = parse_index(new_dtypes.index,
                                                       store_data=True)
                    param['dtypes'] = new_dtypes
                else:
                    # append None to indicate returning Series
                    param['shape'] = None
            else:
                param['shape'] = None
            return param

    def __call__(self, inp):
        inputs = [inp] + filter_inputs(self._indexes)

        shape = []
        sizes = []
        index_value = columns_value = dtypes = None
        for ax, index in enumerate(self._indexes):
            param = self._calc_param(inp, ax, index)

            size = param.get('shape')
            sizes.append(size)
            if size is not None:
                shape.append(size)

            if ax == 0:
                index_value = param.get('index_value')
            else:
                columns_value = param.get('index_value')
                dtypes = param.get('dtypes')

        shape = tuple(shape)
        if len(shape) == 0:
            # scalar
            if isinstance(inp, DATAFRAME_TYPE):
                dtype = inp.dtypes[self._indexes[1]]
            else:
                dtype = inp.dtype
            return self.new_scalar(inputs, dtype=dtype)
        elif len(shape) == 1:
            # series
            if isinstance(inp, DATAFRAME_TYPE):
                if sizes[0] is None:
                    # label on axis 0
                    dtype = find_common_type(list(dtypes))
                    return self.new_series(inputs, shape=shape, dtype=dtype,
                                           index_value=columns_value, name=self._indexes[0])
                else:
                    # label on axis 1
                    dtype = inp.dtypes[self._indexes[1]]
                    return self.new_series(inputs, shape=shape, dtype=dtype,
                                           index_value=index_value, name=self._indexes[1])
            else:
                return self.new_series(inputs, shape=shape, dtype=inp.dtype,
                                       index_value=index_value, name=inp.name)
        else:
            # dataframe
            return self.new_dataframe(inputs, shape=shape, dtypes=dtypes,
                                      index_value=index_value, columns_value=columns_value)

    @classmethod
    def tile(cls, op):
        handler = DataFrameLocIndexesHandler()
        return [(yield from handler.handle(op))]

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        df = ctx[op.input.key]
        if len(op.inputs) > 1:
            indexes = tuple(ctx[index.key] if hasattr(index, 'key') else index
                            for index in op.indexes)
        else:
            indexes = tuple(op.indexes)
        xdf = pd if isinstance(df, (pd.Series, pd.DataFrame)) or cudf is None else cudf

        if op.stage != OperandStage.map:
            try:
                r = df.loc[indexes]
            except AttributeError:
                # workaround for error when calling series.loc[(index,)]
                r = df.loc[indexes[0]]
        else:
            # for map stage, and when some index is fancy index
            # ignore keys that do not exist
            new_indexes = []
            str_loc_on_datetime_index = False
            for ax, index in enumerate(indexes):
                if ax == 0:
                    if isinstance(index, np.ndarray) and index.dtype != np.bool_:
                        new_indexes.append(df.index.intersection(index))
                    elif isinstance(df.index, pd.DatetimeIndex) and isinstance(index, str):
                        # special process for datetime index
                        str_loc_on_datetime_index = True
                        new_indexes.append(index)
                    else:
                        new_indexes.append(index)
                else:
                    new_indexes.append(index)

            try:
                r = df.loc[tuple(new_indexes)]
                if str_loc_on_datetime_index:
                    # convert back to DataFrame or Series
                    if r.ndim == 0:
                        index = df.index[df.index.get_loc(new_indexes[0])]
                        r = xdf.Series([r], index=[index])
                    elif r.ndim == 1:
                        rdf = xdf.DataFrame(columns=r.index)
                        rdf.loc[r.name] = r
                        r = rdf
            except KeyError:
                if str_loc_on_datetime_index:
                    new_indexes[0] = []
                    r = df.loc[tuple(new_indexes)]
                else:  # pragma: no cover
                    raise

        if isinstance(r, pd.Series) and r.dtype != chunk.dtype:
            r = r.astype(chunk.dtype)
        ctx[chunk.key] = r


def loc(a):
    return DataFrameLoc(a)
