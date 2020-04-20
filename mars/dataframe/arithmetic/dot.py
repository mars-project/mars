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

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...serialize import KeyField
from ...tiles import TilesError
from ...utils import recursive_tile
from ...tensor import tensor as astensor
from ...tensor.core import TENSOR_TYPE
from ...tensor.utils import decide_unify_split, validate_axis
from ..core import DATAFRAME_TYPE, SERIES_TYPE, IndexValue
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import parse_index


class DataFrameDot(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DOT

    _lhs = KeyField('lhs')
    _rhs = KeyField('rhs')

    def __init__(self, object_type=None, lhs=None, rhs=None, **kw):
        super().__init__(_object_type=object_type, _lhs=lhs, _rhs=rhs, **kw)

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._lhs = self._inputs[0]
        self._rhs = self._inputs[1]

    def __call__(self, lhs, rhs):
        if not isinstance(rhs, (DATAFRAME_TYPE, SERIES_TYPE)):
            rhs = astensor(rhs)
            test_rhs = rhs
        else:
            test_rhs = rhs.to_tensor()

        test_ret = lhs.to_tensor().dot(test_rhs)
        if test_ret.ndim == 0:
            if isinstance(lhs, SERIES_TYPE) and isinstance(rhs, TENSOR_TYPE):
                # return tensor
                return test_ret
            self._object_type = ObjectType.scalar
            return self.new_scalar([lhs, rhs], dtype=test_ret.dtype)
        elif test_ret.ndim == 1:
            self._object_type = ObjectType.series
            if lhs.ndim == 1:
                if hasattr(rhs, 'columns_value'):
                    index_value = rhs.columns_value
                else:
                    # tensor
                    length = -1 if np.isnan(rhs.shape[1]) else rhs.shape[1]
                    pd_index = pd.RangeIndex(length)
                    index_value = parse_index(pd_index, store_data=True)
            else:
                assert rhs.ndim == 1
                index_value = lhs.index_value
            return self.new_series([lhs, rhs], shape=test_ret.shape,
                                   dtype=test_ret.dtype, index_value=index_value)
        else:
            self._object_type = ObjectType.dataframe
            if isinstance(rhs, TENSOR_TYPE):
                dtypes = pd.Series(np.repeat(test_ret.dtype, test_ret.shape[1]),
                                   index=pd.RangeIndex(test_ret.shape[1]))
                columns_value = parse_index(dtypes.index, store_data=True)
            else:
                dtypes = pd.Series(np.repeat(test_ret.dtype, test_ret.shape[1]),
                                   index=rhs.columns_value.to_pandas())
                columns_value = rhs.columns_value
            return self.new_dataframe([lhs, rhs], shape=test_ret.shape,
                                      index_value=lhs.index_value,
                                      columns_value=columns_value,
                                      dtypes=dtypes)

    @classmethod
    def _align(cls, lhs, rhs):
        if isinstance(rhs, TENSOR_TYPE):
            # no need to align when rhs is a tensor
            return lhs, rhs

        is_lhs_range_index = False
        if isinstance(lhs, DATAFRAME_TYPE) and \
                isinstance(lhs.columns_value.value, IndexValue.RangeIndex):
            is_lhs_range_index = True
        if isinstance(lhs, SERIES_TYPE) and \
                isinstance(lhs.index_value.value, IndexValue.RangeIndex):
            is_lhs_range_index = True

        is_rhs_range_index = False
        if isinstance(rhs.index_value.value, IndexValue.RangeIndex):
            is_rhs_range_index = True

        if not is_lhs_range_index or not is_rhs_range_index:
            # TODO: e.g. use rhs.loc[lhs.columns_value.to_pandas()]
            # when lhs is a DataFrame and lhs.columns is not a RangeIndex,
            # so does Series
            raise NotImplementedError

        return lhs, rhs

    @classmethod
    def tile(cls, op):
        from ..datasource.from_tensor import dataframe_from_tensor, series_from_tensor

        lhs, rhs = op.lhs, op.rhs
        lhs, rhs = cls._align(lhs, rhs)
        out = op.outputs[0]

        if any(np.isnan(ns) for ns in lhs.nsplits[-1]):
            raise TilesError('lhs has unknown chunk shape on last axis')
        if any(np.isnan(ns) for ns in rhs.nsplits[0]):
            raise TilesError('rhs has unknown chunk shape on first axis')

        nsplit = decide_unify_split(lhs.nsplits[-1], rhs.nsplits[0])
        lhs_axis = validate_axis(lhs.ndim, -1)
        lhs = lhs.rechunk({lhs_axis: nsplit})._inplace_tile()
        rhs = rhs.rechunk({0: nsplit})._inplace_tile()

        # delegate computation to tensor
        lhs_tensor = lhs if isinstance(lhs, TENSOR_TYPE) else lhs.to_tensor()
        rhs_tensor = rhs if isinstance(rhs, TENSOR_TYPE) else rhs.to_tensor()
        ret = lhs_tensor.dot(rhs_tensor)

        if isinstance(out, TENSOR_TYPE):
            pass
        elif ret.ndim == 1:
            ret = series_from_tensor(ret)
            if isinstance(lhs, DATAFRAME_TYPE):
                # lhs DataFrame
                ret._index_value = lhs.index_value
            elif isinstance(rhs, DATAFRAME_TYPE):
                # lhs Series, rhs DataFrame
                ret._index_value = rhs.columns_value
        elif ret.ndim == 2:
            ret = dataframe_from_tensor(ret)
            ret._index_value = lhs.index_value
            if isinstance(rhs, DATAFRAME_TYPE):
                ret._columns_value = rhs.columns_value
                ret._dtypes = rhs.dtypes

        return [recursive_tile(ret)]


def dot(df_or_seris, other):
    op = DataFrameDot(lhs=df_or_seris, rhs=other)
    return op(df_or_seris, other)
