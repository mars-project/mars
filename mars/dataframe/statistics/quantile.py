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
from ...tensor.statistics.quantile import quantile as tensor_quantile
from ...utils import recursive_tile
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..core import DATAFRAME_TYPE
from ..datasource.from_tensor import series_from_tensor, dataframe_from_tensor
from ..initializer import DataFrame as create_df
from ..utils import parse_index, build_empty_df, find_common_type, validate_axis


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
                                 interpolation=self._interpolation,
                                 handle_non_numeric=not self._numeric_only).dtype
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
            name = None
            store_index_value = False
        else:
            q_val = np.asanyarray(self._q)
            pd_index = pd.Index(q_val)
            name = self._q if q_val.size == 1 else None
            store_index_value = True
        tokenize_objects = (a, q_val, self._interpolation, type(self).__name__)

        if q_val.ndim == 0 and self._axis == 0:
            self._object_type = ObjectType.series
            index_value = parse_index(dtypes.index, store_data=store_index_value)
            shape = (len(dtypes),)
            # calc dtype
            dtype = self._calc_dtype_on_axis_1(a, dtypes)
            return self.new_series(inputs, shape=shape, dtype=dtype,
                                   index_value=index_value, name=name or dtypes.index.name)
        elif q_val.ndim == 0 and self._axis == 1:
            self._object_type = ObjectType.series
            index_value = a.index_value
            shape = (len(a),)
            # calc dtype
            dt = tensor_quantile(empty(a.shape[1], dtype=find_common_type(dtypes)),
                                 self._q, interpolation=self._interpolation,
                                 handle_non_numeric=not self._numeric_only).dtype
            return self.new_series(inputs, shape=shape, dtype=dt,
                                   index_value=index_value, name=name or index_value.name)
        elif q_val.ndim == 1 and self._axis == 0:
            self._object_type = ObjectType.dataframe
            shape = (len(q_val), len(dtypes))
            index_value = parse_index(pd_index, *tokenize_objects, store_data=store_index_value)
            dtype_list = []
            for name in dtypes.index:
                dtype_list.append(
                    tensor_quantile(tensor_from_series(a[name]), self._q,
                                    interpolation=self._interpolation,
                                    handle_non_numeric=not self._numeric_only).dtype)
            dtypes = pd.Series(dtype_list, index=dtypes.index)
            return self.new_dataframe(inputs, shape=shape, dtypes=dtypes,
                                      index_value=index_value,
                                      columns_value=parse_index(dtypes.index, store_data=True))
        else:
            assert q_val.ndim == 1 and self._axis == 1
            self._object_type = ObjectType.dataframe
            shape = (len(q_val), a.shape[0])
            index_value = parse_index(pd_index, *tokenize_objects, store_data=store_index_value)
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
            store_index_value = False
        else:
            q_val = np.asanyarray(self._q)
            index_val = pd.Index(q_val)
            store_index_value = True

        # get dtype by tensor
        a_t = astensor(a)
        self._dtype = dtype = tensor_quantile(
            a_t, self._q, interpolation=self._interpolation,
            handle_non_numeric=not self._numeric_only).dtype

        if q_val.ndim == 0:
            self._object_type = ObjectType.scalar
            return self.new_scalar(inputs, dtype=dtype)
        else:
            self._object_type = ObjectType.series
            return self.new_series(
                inputs, shape=q_val.shape, dtype=dtype,
                index_value=parse_index(index_val, a, q_val, self._interpolation,
                                        type(self).__name__, store_data=store_index_value),
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
        from ...tensor.merge.stack import TensorStack

        df = op.outputs[0]
        if op.object_type == ObjectType.series:
            if op.axis == 0:
                ts = []
                for name in df.index_value.to_pandas():
                    a = tensor_from_series(op.input[name])
                    t = tensor_quantile(a, op.q, interpolation=op.interpolation,
                                        handle_non_numeric=not op.numeric_only)
                    ts.append(t)
                try:
                    dtype = np.result_type(*[it.dtype for it in ts])
                except TypeError:
                    dtype = np.dtype(object)
                stack_op = TensorStack(axis=0, dtype=dtype)
                tr = stack_op(ts)
                r = series_from_tensor(tr, index=df.index_value.to_pandas(),
                                       name=np.asscalar(ts[0].op.q))
            else:
                assert op.axis == 1
                empty_df = build_empty_df(op.input.dtypes)
                fields = empty_df._get_numeric_data().columns.tolist()
                t = tensor_from_dataframe(op.input[fields])
                tr = tensor_quantile(t, op.q, axis=1, interpolation=op.interpolation,
                                     handle_non_numeric=not op.numeric_only)
                r = series_from_tensor(tr, name=np.asscalar(tr.op.q))
                r._index_value = op.input.index_value
        else:
            assert op.object_type == ObjectType.dataframe
            if op.axis == 0:
                d = OrderedDict()
                for name in df.dtypes.index:
                    a = tensor_from_series(op.input[name])
                    t = tensor_quantile(a, op.q, interpolation=op.interpolation,
                                        handle_non_numeric=not op.numeric_only)
                    d[name] = t
                r = create_df(d, index=op.q)
            else:
                assert op.axis == 1
                empty_df = build_empty_df(op.input.dtypes)
                fields = empty_df._get_numeric_data().columns.tolist()
                t = tensor_from_dataframe(op.input[fields])
                tr = tensor_quantile(t, op.q, axis=1, interpolation=op.interpolation,
                                     handle_non_numeric=not op.numeric_only)
                if not op.input.index_value.has_value():
                    raise NotImplementedError
                # TODO(xuye.qin): use index=op.input.index when we support DataFrame.index
                r = dataframe_from_tensor(tr, index=op.q,
                                          columns=op.input.index_value.to_pandas())

        return [recursive_tile(r)]

    @classmethod
    def _tile_series(cls, op):
        a = tensor_from_series(op.input)
        t = tensor_quantile(a, op.q, interpolation=op.interpolation,
                            handle_non_numeric=not op.numeric_only)
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
    """
    Return value at the given quantile.

    Parameters
    ----------
    q : float or array-like, default 0.5 (50% quantile)
        0 <= q <= 1, the quantile(s) to compute.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        .. versionadded:: 0.18.0

        This optional parameter specifies the interpolation method to use,
        when the desired quantile lies between two data points `i` and `j`:

            * linear: `i + (j - i) * fraction`, where `fraction` is the
              fractional part of the index surrounded by `i` and `j`.
            * lower: `i`.
            * higher: `j`.
            * nearest: `i` or `j` whichever is nearest.
            * midpoint: (`i` + `j`) / 2.

    Returns
    -------
    float or Series
        If ``q`` is an array or a tensor, a Series will be returned where the
        index is ``q`` and the values are the quantiles, otherwise
        a float will be returned.

    See Also
    --------
    core.window.Rolling.quantile
    numpy.percentile

    Examples
    --------
    >>> import mars.dataframe as md
    >>> s = md.Series([1, 2, 3, 4])
    >>> s.quantile(.5).execute()
    2.5
    >>> s.quantile([.25, .5, .75]).execute()
    0.25    1.75
    0.50    2.50
    0.75    3.25
    dtype: float64
    """

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
    """
    Return values at the given quantile over requested axis.

    Parameters
    ----------
    q : float or array-like, default 0.5 (50% quantile)
        Value between 0 <= q <= 1, the quantile(s) to compute.
    axis : {0, 1, 'index', 'columns'} (default 0)
        Equals 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
    numeric_only : bool, default True
        If False, the quantile of datetime and timedelta data will be
        computed as well.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to use,
        when the desired quantile lies between two data points `i` and `j`:
        * linear: `i + (j - i) * fraction`, where `fraction` is the
          fractional part of the index surrounded by `i` and `j`.
        * lower: `i`.
        * higher: `j`.
        * nearest: `i` or `j` whichever is nearest.
        * midpoint: (`i` + `j`) / 2.
        .. versionadded:: 0.18.0
    Returns
    -------
    Series or DataFrame
        If ``q`` is an array or a tensor, a DataFrame will be returned where the
          index is ``q``, the columns are the columns of self, and the
          values are the quantiles.
        If ``q`` is a float, a Series will be returned where the
          index is the columns of self and the values are the quantiles.

    See Also
    --------
    core.window.Rolling.quantile: Rolling quantile.
    numpy.percentile: Numpy function to compute the percentile.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100]]),
    ...                   columns=['a', 'b'])
    >>> df.quantile(.1).execute()
    a    1.3
    b    3.7
    Name: 0.1, dtype: float64

    >>> df.quantile([.1, .5]).execute()
           a     b
    0.1  1.3   3.7
    0.5  2.5  55.0

    Specifying `numeric_only=False` will also compute the quantile of
    datetime and timedelta data.
    >>> df = md.DataFrame({'A': [1, 2],
    ...                    'B': [md.Timestamp('2010'),
    ...                          md.Timestamp('2011')],
    ...                    'C': [md.Timedelta('1 days'),
    ...                          md.Timedelta('2 days')]})
    >>> df.quantile(0.5, numeric_only=False).execute()
    A                    1.5
    B    2010-07-02 12:00:00
    C        1 days 12:00:00
    Name: 0.5, dtype: object
    """
    if isinstance(q, (Base, Entity)):
        q = astensor(q)
        q_input = q
    else:
        q_input = None
    axis = validate_axis(axis, df)

    op = DataFrameQuantile(q=q, interpolation=interpolation,
                           axis=axis, numeric_only=numeric_only,
                           gpu=df.op.gpu)
    return op(df, q_input=q_input)
