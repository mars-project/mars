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

import inspect

import numpy as np
import pandas as pd

from ... import opcodes
from ...config import options
from ...core import OutputType, recursive_tile
from ...core.custom_log import redirect_custom_log
from ...serialization.serializables import StringField, AnyField, BoolField, \
    TupleField, DictField, FunctionField
from ...utils import enter_current_session, quiet_stdio
from ..operands import DataFrameOperandMixin, DataFrameOperand
from ..utils import build_df, build_series, parse_index, validate_axis, \
    validate_output_types, make_dtypes, make_dtype


class ApplyOperand(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.APPLY

    _func = FunctionField('func')
    _axis = AnyField('axis')
    _convert_dtype = BoolField('convert_dtype')
    _raw = BoolField('raw')
    _result_type = StringField('result_type')
    _elementwise = BoolField('elementwise')
    _args = TupleField('args')
    _kwds = DictField('kwds')

    def __init__(self, func=None, axis=None, convert_dtype=None, raw=None, result_type=None,
                 args=None, kwds=None, output_type=None, elementwise=None, **kw):
        if output_type:
            kw['_output_types'] = [output_type]
        super().__init__(_func=func, _axis=axis, _convert_dtype=convert_dtype, _raw=raw,
                         _result_type=result_type, _args=args, _kwds=kwds,
                         _elementwise=elementwise, **kw)

    @property
    def func(self):
        return self._func

    @property
    def axis(self):
        return self._axis

    @property
    def convert_dtype(self):
        return self._convert_dtype

    @property
    def raw(self):
        return self._raw

    @property
    def result_type(self):
        return self._result_type

    @property
    def elementwise(self):
        return self._elementwise

    @property
    def args(self):
        return getattr(self, '_args', None) or ()

    @property
    def kwds(self):
        return getattr(self, '_kwds', None) or dict()

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx, op):
        input_data = ctx[op.inputs[0].key]
        if isinstance(input_data, pd.DataFrame):
            result = input_data.apply(op.func, axis=op.axis, raw=op.raw, result_type=op.result_type,
                                      args=op.args, **op.kwds)
        else:
            result = input_data.apply(op.func, convert_dtype=op.convert_dtype, args=op.args,
                                      **op.kwds)
        ctx[op.outputs[0].key] = result

    @classmethod
    def _tile_df(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]
        axis = op.axis
        elementwise = op.elementwise

        if not elementwise and in_df.chunk_shape[axis] > 1:
            chunk_size = (
                in_df.shape[axis],
                max(1, options.chunk_store_limit // in_df.shape[axis]),
            )
            if axis == 1:
                chunk_size = chunk_size[::-1]
            in_df = yield from recursive_tile(in_df.rechunk(chunk_size))

        chunks = []
        if out_df.ndim == 2:
            for c in in_df.chunks:
                if elementwise:
                    new_shape = c.shape
                    new_index_value, new_columns_value = c.index_value, c.columns_value
                else:
                    new_shape = [np.nan, np.nan]
                    new_shape[1 - axis] = c.shape[1 - axis]
                    if axis == 0:
                        new_index_value = out_df.index_value
                        new_columns_value = c.columns_value
                    else:
                        new_index_value = c.index_value
                        new_columns_value = out_df.columns_value

                if op.axis == 0:
                    new_dtypes = out_df.dtypes[c.dtypes.keys()]
                else:
                    new_dtypes = out_df.dtypes

                new_op = op.copy().reset_key()
                new_op.tileable_op_key = op.key
                chunks.append(new_op.new_chunk([c], shape=tuple(new_shape), index=c.index, dtypes=new_dtypes,
                                               index_value=new_index_value, columns_value=new_columns_value))

            new_nsplits = list(in_df.nsplits)
            if not elementwise:
                new_nsplits[axis] = (np.nan,) * len(new_nsplits[axis])
        else:
            for c in in_df.chunks:
                shape_len = c.shape[1 - axis]
                new_index_value = c.index_value if axis == 1 else c.columns_value
                new_index = (c.index[1 - axis],)
                new_op = op.copy().reset_key()
                new_op.tileable_op_key = op.key
                chunks.append(new_op.new_chunk([c], shape=(shape_len,), index=new_index, dtype=out_df.dtype,
                                               index_value=new_index_value))
            new_nsplits = (in_df.nsplits[1 - axis],)

        new_op = op.copy()
        kw = out_df.params.copy()
        kw.update(dict(chunks=chunks, nsplits=tuple(new_nsplits)))
        return new_op.new_tileables(op.inputs, **kw)

    @classmethod
    def _tile_series(cls, op):
        in_series = op.inputs[0]
        out_series = op.outputs[0]

        chunks = []
        for c in in_series.chunks:
            new_op = op.copy().reset_key()
            new_op.tileable_op_key = op.key
            kw = c.params.copy()
            if out_series.ndim == 1:
                kw['dtype'] = out_series.dtype
            else:
                kw['index'] = (c.index[0], 0)
                kw['shape'] = (c.shape[0], out_series.shape[1])
                kw['dtypes'] = out_series.dtypes
                kw['columns_value'] = out_series.columns_value
            chunks.append(new_op.new_chunk([c], **kw))

        new_op = op.copy()
        kw = out_series.params.copy()
        kw.update(dict(chunks=chunks, nsplits=in_series.nsplits))
        if out_series.ndim == 2:
            kw['nsplits'] = (in_series.nsplits[0], (out_series.shape[1],))
            kw['columns_value'] = out_series.columns_value
        return new_op.new_tileables(op.inputs, **kw)

    @classmethod
    def tile(cls, op):
        if op.inputs[0].ndim == 2:
            return (yield from cls._tile_df(op))
        else:
            return cls._tile_series(op)

    def _infer_df_func_returns(self, df, dtypes, dtype=None, name=None, index=None):
        if isinstance(self._func, np.ufunc):
            output_type, new_dtypes, index_value, new_elementwise = \
                OutputType.dataframe, None, 'inherit', True
        else:
            output_type, new_dtypes, index_value, new_elementwise = None, None, None, False

        try:
            empty_df = build_df(df, size=2)
            with np.errstate(all='ignore'), quiet_stdio():
                infer_df = empty_df.apply(self._func, axis=self._axis, raw=self._raw,
                                          result_type=self._result_type, args=self.args, **self.kwds)
            if index_value is None:
                if infer_df.index is empty_df.index:
                    index_value = 'inherit'
                else:
                    index_value = parse_index(pd.RangeIndex(-1))

            if isinstance(infer_df, pd.DataFrame):
                output_type = output_type or OutputType.dataframe
                new_dtypes = new_dtypes or infer_df.dtypes
            else:
                output_type = output_type or OutputType.series
                new_dtypes = (name or infer_df.name, dtype or infer_df.dtype)
            new_elementwise = False if new_elementwise is None else new_elementwise
        except:  # noqa: E722  # nosec
            pass

        self.output_types = [output_type] if not self.output_types else self.output_types
        dtypes = new_dtypes if dtypes is None else dtypes
        index_value = index_value if index is None else parse_index(index)
        self._elementwise = new_elementwise if self._elementwise is None else self._elementwise
        return dtypes, index_value

    def _call_dataframe(self, df, dtypes=None, dtype=None, name=None, index=None):
        # for backward compatibility
        dtype = dtype if dtype is not None else dtypes
        dtypes, index_value = self._infer_df_func_returns(
            df, dtypes, dtype=dtype, name=name, index=index)
        if index_value is None:
            index_value = parse_index(None, (df.key, df.index_value.key))
        for arg, desc in zip((self.output_types, dtypes), ('output_types', 'dtypes')):
            if arg is None:
                raise TypeError(f'Cannot determine {desc} by calculating with enumerate data, '
                                'please specify it as arguments')

        if index_value == 'inherit':
            index_value = df.index_value

        if self._elementwise:
            shape = df.shape
        elif self.output_types[0] == OutputType.dataframe:
            shape = [np.nan, np.nan]
            shape[1 - self.axis] = df.shape[1 - self.axis]
            shape = tuple(shape)
        else:
            shape = (df.shape[1 - self.axis],)

        if self.output_types[0] == OutputType.dataframe:
            if self.axis == 0:
                return self.new_dataframe([df], shape=shape, dtypes=dtypes, index_value=index_value,
                                          columns_value=parse_index(dtypes.index, store_data=True))
            else:
                return self.new_dataframe([df], shape=shape, dtypes=dtypes, index_value=df.index_value,
                                          columns_value=parse_index(dtypes.index, store_data=True))
        else:
            name, dtype = dtypes
            return self.new_series([df], shape=shape, name=name, dtype=dtype, index_value=index_value)

    def _call_series(self, series, dtypes=None, dtype=None, name=None, index=None):
        # for backward compatibility
        dtype = dtype if dtype is not None else dtypes
        if self._convert_dtype:
            try:
                test_series = build_series(series, size=2, name=series.name)
                with np.errstate(all='ignore'), quiet_stdio():
                    infer_series = test_series.apply(self._func, args=self.args, **self.kwds)
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                infer_series = None

            output_type = self._output_types[0]

            if index is not None:
                index_value = parse_index(index)
            elif infer_series is not None:
                index_value = parse_index(infer_series.index)
            else:
                index_value = parse_index(None, series)

            if output_type == OutputType.dataframe:
                if dtypes is None:
                    if infer_series is not None and infer_series.ndim == 2:
                        dtypes = infer_series.dtypes
                    else:
                        raise TypeError('Cannot determine dtypes, '
                                        'please specify `dtypes` as argument')

                columns_value = parse_index(dtypes.index, store_data=True)

                return self.new_dataframe([series], shape=(series.shape[0], len(dtypes)),
                                          index_value=index_value, columns_value=columns_value,
                                          dtypes=dtypes)
            else:
                if dtype is None and infer_series is not None and infer_series.ndim == 1:
                    dtype = infer_series.dtype
                else:
                    dtype = dtype if dtype is not None else np.dtype(object)
                if infer_series is not None and infer_series.ndim == 1:
                    name = name or infer_series.name
                return self.new_series([series], dtype=dtype, shape=series.shape,
                                       index_value=index_value, name=name)
        else:
            dtype = dtype if dtype is not None else np.dtype('object')
            return self.new_series([series], dtype=dtype, shape=series.shape,
                                   index_value=series.index_value, name=name)

    def __call__(self, df_or_series, dtypes=None, dtype=None, name=None, index=None):
        axis = getattr(self, 'axis', None) or 0
        dtypes = make_dtypes(dtypes)
        dtype = make_dtype(dtype)
        self._axis = validate_axis(axis, df_or_series)

        if df_or_series.op.output_types[0] == OutputType.dataframe:
            return self._call_dataframe(df_or_series, dtypes=dtypes, index=index)
        else:
            return self._call_series(df_or_series, dtypes=dtypes, dtype=dtype, name=name, index=index)


def df_apply(df, func, axis=0, raw=False, result_type=None, args=(), dtypes=None,
             dtype=None, name=None, output_type=None, index=None, elementwise=None,
             **kwds):
    """
    Apply a function along an axis of the DataFrame.

    Objects passed to the function are Series objects whose index is
    either the DataFrame's index (``axis=0``) or the DataFrame's columns
    (``axis=1``). By default (``result_type=None``), the final return type
    is inferred from the return type of the applied function. Otherwise,
    it depends on the `result_type` argument.

    Parameters
    ----------
    func : function
        Function to apply to each column or row.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Axis along which the function is applied:

        * 0 or 'index': apply function to each column.
        * 1 or 'columns': apply function to each row.

    raw : bool, default False
        Determines if row or column is passed as a Series or ndarray object:

        * ``False`` : passes each row or column as a Series to the
          function.
        * ``True`` : the passed function will receive ndarray objects
          instead.
          If you are just applying a NumPy reduction function this will
          achieve much better performance.

    result_type : {'expand', 'reduce', 'broadcast', None}, default None
        These only act when ``axis=1`` (columns):

        * 'expand' : list-like results will be turned into columns.
        * 'reduce' : returns a Series if possible rather than expanding
          list-like results. This is the opposite of 'expand'.
        * 'broadcast' : results will be broadcast to the original shape
          of the DataFrame, the original index and columns will be
          retained.

        The default behaviour (None) depends on the return value of the
        applied function: list-like results will be returned as a Series
        of those. However if the apply function returns a Series these
        are expanded to columns.

    output_type : {'dataframe', 'series'}, default None
        Specify type of returned object. See `Notes` for more details.

    dtypes : Series, default None
        Specify dtypes of returned DataFrames. See `Notes` for more details.

    dtype : numpy.dtype, default None
        Specify dtype of returned Series. See `Notes` for more details.

    name : str, default None
        Specify name of returned Series. See `Notes` for more details.

    index : Index, default None
        Specify index of returned object. See `Notes` for more details.

    elementwise : bool, default False
        Specify whether ``func`` is an elementwise function:

        * ``False`` : The function is not elementwise. Mars will try
          concatenating chunks in rows (when ``axis=0``) or in columns
          (when ``axis=1``) and then apply ``func`` onto the concatenated
          chunk. The concatenation step can cause extra latency.
        * ``True`` : The function is elementwise. Mars will apply
          ``func`` to original chunks. This will not introduce extra
          concatenation step and reduce overhead.

    args : tuple
        Positional arguments to pass to `func` in addition to the
        array/series.

    **kwds
        Additional keyword arguments to pass as keywords arguments to
        `func`.

    Returns
    -------
    Series or DataFrame
        Result of applying ``func`` along the given axis of the
        DataFrame.

    See Also
    --------
    DataFrame.applymap: For elementwise operations.
    DataFrame.aggregate: Only perform aggregating type operations.
    DataFrame.transform: Only perform transforming type operations.

    Notes
    -----
    When deciding output dtypes and shape of the return value, Mars will
    try applying ``func`` onto a mock DataFrame,  and the apply call may
    fail. When this happens, you need to specify the type of apply call
    (DataFrame or Series) in output_type.

    * For DataFrame output, you need to specify a list or a pandas Series
      as ``dtypes`` of output DataFrame. ``index`` of output can also be
      specified.
    * For Series output, you need to specify ``dtype`` and ``name`` of
      output Series.

    Examples
    --------
    >>> import numpy as np
    >>> import mars.tensor as mt
    >>> import mars.dataframe as md
    >>> df = md.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
    >>> df.execute()
       A  B
    0  4  9
    1  4  9
    2  4  9

    Using a reducing function on either axis

    >>> df.apply(np.sum, axis=0).execute()
    A    12
    B    27
    dtype: int64

    >>> df.apply(np.sum, axis=1).execute()
    0    13
    1    13
    2    13
    dtype: int64

    Returning a list-like will result in a Series

    >>> df.apply(lambda x: [1, 2], axis=1).execute()
    0    [1, 2]
    1    [1, 2]
    2    [1, 2]
    dtype: object

    Passing ``result_type='expand'`` will expand list-like results
    to columns of a Dataframe

    >>> df.apply(lambda x: [1, 2], axis=1, result_type='expand').execute()
       0  1
    0  1  2
    1  1  2
    2  1  2

    Returning a Series inside the function is similar to passing
    ``result_type='expand'``. The resulting column names
    will be the Series index.

    >>> df.apply(lambda x: md.Series([1, 2], index=['foo', 'bar']), axis=1).execute()
       foo  bar
    0    1    2
    1    1    2
    2    1    2

    Passing ``result_type='broadcast'`` will ensure the same shape
    result, whether list-like or scalar is returned by the function,
    and broadcast it along the axis. The resulting column names will
    be the originals.

    >>> df.apply(lambda x: [1, 2], axis=1, result_type='broadcast').execute()
       A  B
    0  1  2
    1  1  2
    2  1  2
    """
    if isinstance(func, (list, dict)):
        return df.aggregate(func)

    output_types = kwds.pop('output_types', None)
    object_type = kwds.pop('object_type', None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type)
    output_type = output_types[0] if output_types else None

    # calling member function
    if isinstance(func, str):
        func = getattr(df, func)
        sig = inspect.getfullargspec(func)
        if "axis" in sig.args:
            kwds["axis"] = axis
        return func(*args, **kwds)

    op = ApplyOperand(func=func, axis=axis, raw=raw, result_type=result_type, args=args, kwds=kwds,
                      output_type=output_type, elementwise=elementwise)
    return op(df, dtypes=dtypes, dtype=dtype, name=name, index=index)


def series_apply(series, func, convert_dtype=True, output_type=None, args=(),
                 dtypes=None, dtype=None, name=None, index=None, **kwds):
    """
    Invoke function on values of Series.

    Can be ufunc (a NumPy function that applies to the entire Series)
    or a Python function that only works on single values.

    Parameters
    ----------
    func : function
        Python function or NumPy ufunc to apply.

    convert_dtype : bool, default True
        Try to find better dtype for elementwise function results. If
        False, leave as dtype=object.

    output_type : {'dataframe', 'series'}, default None
        Specify type of returned object. See `Notes` for more details.

    dtypes : Series, default None
        Specify dtypes of returned DataFrames. See `Notes` for more details.

    dtype : numpy.dtype, default None
        Specify dtype of returned Series. See `Notes` for more details.

    name : str, default None
        Specify name of returned Series. See `Notes` for more details.

    index : Index, default None
        Specify index of returned object. See `Notes` for more details.

    args : tuple
        Positional arguments passed to func after the series value.

    **kwds
        Additional keyword arguments passed to func.

    Returns
    -------
    Series or DataFrame
        If func returns a Series object the result will be a DataFrame.

    See Also
    --------
    Series.map: For element-wise operations.
    Series.agg: Only perform aggregating type operations.
    Series.transform: Only perform transforming type operations.

    Notes
    -----
    When deciding output dtypes and shape of the return value, Mars will
    try applying ``func`` onto a mock Series, and the apply call may fail.
    When this happens, you need to specify the type of apply call
    (DataFrame or Series) in output_type.

    * For DataFrame output, you need to specify a list or a pandas Series
      as ``dtypes`` of output DataFrame. ``index`` of output can also be
      specified.
    * For Series output, you need to specify ``dtype`` and ``name`` of
      output Series.

    Examples
    --------
    Create a series with typical summer temperatures for each city.

    >>> import mars.tensor as mt
    >>> import mars.dataframe as md
    >>> s = md.Series([20, 21, 12],
    ...               index=['London', 'New York', 'Helsinki'])
    >>> s.execute()
    London      20
    New York    21
    Helsinki    12
    dtype: int64

    Square the values by defining a function and passing it as an
    argument to ``apply()``.

    >>> def square(x):
    ...     return x ** 2
    >>> s.apply(square).execute()
    London      400
    New York    441
    Helsinki    144
    dtype: int64

    Square the values by passing an anonymous function as an
    argument to ``apply()``.

    >>> s.apply(lambda x: x ** 2).execute()
    London      400
    New York    441
    Helsinki    144
    dtype: int64

    Define a custom function that needs additional positional
    arguments and pass these additional arguments using the
    ``args`` keyword.

    >>> def subtract_custom_value(x, custom_value):
    ...     return x - custom_value

    >>> s.apply(subtract_custom_value, args=(5,)).execute()
    London      15
    New York    16
    Helsinki     7
    dtype: int64

    Define a custom function that takes keyword arguments
    and pass these arguments to ``apply``.

    >>> def add_custom_values(x, **kwargs):
    ...     for month in kwargs:
    ...         x += kwargs[month]
    ...     return x

    >>> s.apply(add_custom_values, june=30, july=20, august=25).execute()
    London      95
    New York    96
    Helsinki    87
    dtype: int64
    """
    if isinstance(func, (list, dict)):
        return series.aggregate(func)

    # calling member function
    if isinstance(func, str):
        func_body = getattr(series, func, None)
        if func_body is not None:
            return func_body(*args, **kwds)
        func_str = func
        func = getattr(np, func_str, None)
        if func is None:
            raise AttributeError(f"'{func_str!r}' is not a valid function "
                                 f"for '{type(series).__name__}' object")

    output_types = kwds.pop('output_types', None)
    object_type = kwds.pop('object_type', None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type)
    output_type = output_types[0] if output_types else OutputType.series

    op = ApplyOperand(func=func, convert_dtype=convert_dtype, args=args, kwds=kwds,
                      output_type=output_type)
    return op(series, dtypes=dtypes, dtype=dtype, name=name, index=index)
