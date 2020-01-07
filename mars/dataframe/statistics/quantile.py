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

from collections import OrderedDict

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...core import Base, Entity
from ...serialize import KeyField, AnyField, StringField, DataTypeField, \
    BoolField, Int32Field
from ...tensor.core import TENSOR_TYPE
from ...tensor.datasource import empty, tensor as astensor, \
    from_series as tensor_from_series, from_dataframe as tensor_from_dataframe
from ...tensor.merge import stack
from ...tensor.statistics.quantile import quantile as tensor_quantile
from ...tensor.utils import recursive_tile
from ...utils import tokenize
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..core import DATAFRAME_TYPE
from ..datasource.from_tensor import series_from_tensor, dataframe_from_tensor
from ..initializer import DataFrame as create_df
from ..utils import parse_index, build_empty_df, find_common_type


class DataFrameQuantile(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.QUANTILE

    _input = KeyField('input')
    _q = AnyField('q')
    _axis = Int32Field('axis')
    _numeric_only = BoolField('numeric_only')
    _interpolation = StringField('interpolation')

    _dtype = DataTypeField('dtype')

    def __init__(self, q=None, interpolation=None, axis=None, numeric_only=None,
                 dtype=None, gpu=None, object_type=None, **kw):
        super().__init__(_q=q, _interpolation=interpolation, _axis=axis,
                         _numeric_only=numeric_only, _dtype=dtype, _gpu=gpu,
                         _object_type=object_type, **kw)

    @property
    def input(self):
        return self._input

    @property
    def q(self):
        return self._q

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def axis(self):
        return self._axis

    @property
    def numeric_only(self):
        return self._numeric_only

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if isinstance(self._q, TENSOR_TYPE):
            self._q = self._inputs[-1]

    def _calc_dtype_on_axis_1(self, a, dtypes):
        quantile_dtypes = []
        for name in dtypes.index:
            dt = tensor_quantile(tensor_from_series(a[name]), self._q,
                                 interpolation=self._interpolation).dtype
            quantile_dtypes.append(dt)
        return find_common_type(quantile_dtypes)

    def _call_dataframe(self, a, inputs):
        if self._numeric_only:
            empty_df = build_empty_df(a.dtypes)
            dtypes = empty_df._get_numeric_data().dtypes
        else:
            dtypes = a.dtypes
        if isinstance(self._q, TENSOR_TYPE):
            q_val = self._q
            pd_index = pd.Index([], dtype=q_val.dtype)
        else:
            q_val = np.asanyarray(self._q)
            pd_index = pd.Index(q_val)
        q_key = tokenize(a, q_val, self._interpolation,
                         type(self).__name__)

        if q_val.ndim == 0 and self._axis == 0:
            self._object_type = ObjectType.series
            index_value = parse_index(dtypes.index, store_data=True)
            shape = (len(dtypes),)
            # calc dtype
            dtype = self._calc_dtype_on_axis_1(a, dtypes)
            return self.new_series(inputs, shape=shape, dtype=dtype,
                                   index_value=index_value, name=dtypes.index.name)
        elif q_val.ndim == 0 and self._axis == 1:
            self._object_type = ObjectType.series
            index_value = a.index_value
            shape = (len(a),)
            # calc dtype
            dt = tensor_quantile(empty(a.shape[1], dtype=find_common_type(dtypes)),
                                 self._q, interpolation=self._interpolation).dtype
            return self.new_series(inputs, shape=shape, dtype=dt,
                                   index_value=index_value, name=index_value.name)
        elif q_val.ndim == 1 and self._axis == 0:
            self._object_type = ObjectType.dataframe
            shape = (len(q_val), len(dtypes))
            index_value = parse_index(pd_index, key=q_key, store_data=True)
            dtype_list = []
            for name in dtypes.index:
                dtype_list.append(
                    tensor_quantile(tensor_from_series(a[name]), self._q,
                                    interpolation=self._interpolation).dtype)
            dtypes = pd.Series(dtype_list, index=dtypes.index)
            return self.new_dataframe(inputs, shape=shape, dtypes=dtypes,
                                      index_value=index_value,
                                      columns_value=parse_index(dtypes.index, store_data=True))
        else:
            assert q_val.ndim == 1 and self._axis == 1
            self._object_type = ObjectType.dataframe
            shape = (len(q_val), a.shape[0])
            index_value = parse_index(pd_index, key=q_key, store_data=True)
            pd_columns = a.index_value.to_pandas()
            dtype_list = np.full(len(pd_columns), self._calc_dtype_on_axis_1(a, dtypes))
            dtypes = pd.Series(dtype_list, index=pd_columns)
            return self.new_dataframe(inputs, shape=shape,
                                      dtypes=dtypes,
                                      index_value=index_value,
                                      columns_value=parse_index(dtypes.index, store_data=True,
                                                                key=a.index_value.key))

    def _call_series(self, a, inputs):
        if isinstance(self._q, TENSOR_TYPE):
            q_val = self._q
            index_val = pd.Index([], dtype=q_val.dtype)
        else:
            q_val = np.asanyarray(self._q)
            index_val = pd.Index(q_val)
        index_key = tokenize(a, q_val, self._interpolation,
                             type(self).__name__)

        # get dtype by tensor
        a_t = astensor(a)
        self._dtype = dtype = tensor_quantile(
            a_t, self._q, interpolation=self._interpolation).dtype

        if q_val.ndim == 0:
            self._object_type = ObjectType.scalar
            return self.new_scalar(inputs, dtype=dtype)
        else:
            self._object_type = ObjectType.series
            return self.new_series(
                inputs, shape=q_val.shape, dtype=dtype,
                index_value=parse_index(index_val, store_data=True, key=index_key),
                name=a.name)

    def __call__(self, a, q_input=None):
        inputs = [a]
        if q_input is not None:
            inputs.append(q_input)
        if isinstance(a, DATAFRAME_TYPE):
            return self._call_dataframe(a, inputs)
        else:
            return self._call_series(a, inputs)

    @classmethod
    def _tile_dataframe(cls, op):
        df = op.outputs[0]
        if op.object_type == ObjectType.series:
            if op.axis == 0:
                ts = []
                for name in df.index_value.to_pandas():
                    a = tensor_from_series(op.input[name])
                    t = tensor_quantile(a, op.q, interpolation=op.interpolation)
                    ts.append(t)
                tr = stack(ts)
                r = series_from_tensor(tr, index=op.input.columns_value.to_pandas(),
                                       name=np.asscalar(ts[0].op.q))
            else:
                assert op.axis == 1
                empty_df = build_empty_df(op.input.dtypes)
                fields = empty_df._get_numeric_data().columns.tolist()
                t = tensor_from_dataframe(op.input[fields])
                tr = tensor_quantile(t, op.q, axis=1, interpolation=op.interpolation)
                r = series_from_tensor(tr, name=np.asscalar(tr.op.q))
                r._index_value = op.input.index_value
        else:
            assert op.object_type == ObjectType.dataframe
            if op.axis == 0:
                d = OrderedDict()
                for name in df.dtypes.index:
                    a = tensor_from_series(op.input[name])
                    t = tensor_quantile(a, op.q, interpolation=op.interpolation)
                    d[name] = t
                r = create_df(d, index=op.q)
            else:
                assert op.axis == 1
                empty_df = build_empty_df(op.input.dtypes)
                fields = empty_df._get_numeric_data().columns.tolist()
                t = tensor_from_dataframe(op.input[fields])
                tr = tensor_quantile(t, op.q, axis=1, interpolation=op.interpolation)
                if not op.input.index_value.has_value():
                    raise NotImplementedError
                # TODO(xuye.qin): use index=op.input.index when we support DataFrame.index
                r = dataframe_from_tensor(tr, index=op.q,
                                          columns=op.input.index_value.to_pandas())

        return [recursive_tile(r)]

    @classmethod
    def _tile_series(cls, op):
        a = tensor_from_series(op.input)
        t = tensor_quantile(a, op.q, interpolation=op.interpolation)
        if op.object_type == ObjectType.scalar:
            r = t
        else:
            r = series_from_tensor(t, index=op.q, name=op.outputs[0].name)
        return [recursive_tile(r)]

    @classmethod
    def tile(cls, op):
        if isinstance(op.input, DATAFRAME_TYPE):
            return cls._tile_dataframe(op)
        else:
            return cls._tile_series(op)


def quantile_series(series, q=0.5, interpolation='linear'):
    if isinstance(q, (Base, Entity)):
        q = astensor(q)
        q_input = q
    else:
        q_input = None

    op = DataFrameQuantile(q=q, interpolation=interpolation,
                           gpu=series.op.gpu)
    return op(series, q_input=q_input)


def quantile_dataframe(df, q=0.5, axis=0, numeric_only=True,
                       interpolation='linear'):
    if isinstance(q, (Base, Entity)):
        q = astensor(q)
        q_input = q
    else:
        q_input = None

    op = DataFrameQuantile(q=q, interpolation=interpolation,
                           axis=axis, numeric_only=numeric_only,
                           gpu=df.op.gpu)
    return op(df, q_input=q_input)
