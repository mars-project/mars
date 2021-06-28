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
from collections.abc import MutableMapping

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...core import OutputType, recursive_tile
from ...core.custom_log import redirect_custom_log
from ...serialization.serializables import KeyField, AnyField, StringField
from ...utils import has_unknown_shape, enter_current_session, quiet_stdio
from ..core import SERIES_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_series


class DataFrameMap(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.MAP

    _input = KeyField('input')
    _arg = AnyField('arg')
    _na_action = StringField('na_action')

    def __init__(self, arg=None, na_action=None, output_types=None,
                 memory_scale=None, **kw):
        super().__init__(_arg=arg, _na_action=na_action, _output_types=output_types,
                         _memory_scale=memory_scale, **kw)
        if not self.output_types:
            self.output_types = [OutputType.series]

    @property
    def input(self):
        return self._input

    @property
    def arg(self):
        return self._arg

    @property
    def na_action(self):
        return self._na_action

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if len(inputs) == 2:
            self._arg = self._inputs[1]

    def __call__(self, series, dtype):
        if dtype is None:
            inferred_dtype = None
            if callable(self._arg):
                # arg is a function, try to inspect the signature
                sig = inspect.signature(self._arg)
                return_type = sig.return_annotation
                if return_type is not inspect._empty:
                    inferred_dtype = np.dtype(return_type)
                else:
                    try:
                        with quiet_stdio():
                            # try to infer dtype by calling the function
                            inferred_dtype = build_series(series).map(
                                self._arg, na_action=self._na_action).dtype
                    except:  # noqa: E722  # nosec
                        pass
            else:
                if isinstance(self._arg, MutableMapping):
                    inferred_dtype = pd.Series(self._arg).dtype
                else:
                    inferred_dtype = self._arg.dtype
            if inferred_dtype is not None and np.issubdtype(inferred_dtype, np.number):
                if np.issubdtype(inferred_dtype, np.inexact):
                    # for the inexact e.g. float
                    # we can make the decision,
                    # but for int, due to the nan which may occur,
                    # we cannot infer the dtype
                    dtype = inferred_dtype
            else:
                dtype = inferred_dtype

        if dtype is None:
            raise ValueError('cannot infer dtype, '
                             'it needs to be specified manually for `map`')
        else:
            dtype = np.int64 if dtype is int else dtype
            dtype = np.dtype(dtype)

        inputs = [series]
        if isinstance(self._arg, SERIES_TYPE):
            inputs.append(self._arg)

        if isinstance(series, SERIES_TYPE):
            return self.new_series(inputs, shape=series.shape, dtype=dtype,
                                   index_value=series.index_value, name=series.name)
        else:
            return self.new_index(inputs, shape=series.shape, dtype=dtype,
                                  index_value=series.index_value, name=series.name)

    @classmethod
    def tile(cls, op):
        in_series = op.input
        out_series = op.outputs[0]

        arg = op.arg
        if len(op.inputs) == 2:
            # make sure arg has known shape when it's a md.Series
            if has_unknown_shape(op.arg):
                yield
            arg = yield from recursive_tile(op.arg.rechunk(op.arg.shape))

        out_chunks = []
        for chunk in in_series.chunks:
            chunk_op = op.copy().reset_key()
            chunk_op.tileable_op_key = op.key
            chunk_inputs = [chunk]
            if len(op.inputs) == 2:
                chunk_inputs.append(arg.chunks[0])
            out_chunk = chunk_op.new_chunk(chunk_inputs, shape=chunk.shape,
                                           dtype=out_series.dtype,
                                           index_value=chunk.index_value,
                                           name=out_series.name,
                                           index=chunk.index)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        params = out_series.params
        params['chunks'] = out_chunks
        params['nsplits'] = in_series.nsplits
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx, op):
        series = ctx[op.inputs[0].key]
        out = op.outputs[0]
        if len(op.inputs) == 2:
            arg = ctx[op.inputs[1].key]
        else:
            arg = op.arg

        ret = series.map(arg, na_action=op.na_action)
        if ret.dtype != out.dtype:
            ret = ret.astype(out.dtype)
        ctx[out.key] = ret


def series_map(series, arg, na_action=None, dtype=None, memory_scale=None):
    """
    Map values of Series according to input correspondence.

    Used for substituting each value in a Series with another value,
    that may be derived from a function, a ``dict`` or
    a :class:`Series`.

    Parameters
    ----------
    arg : function, collections.abc.Mapping subclass or Series
        Mapping correspondence.
    na_action : {None, 'ignore'}, default None
        If 'ignore', propagate NaN values, without passing them to the
        mapping correspondence.
    dtype : np.dtype, default None
        Specify return type of the function. Must be specified when
        we cannot decide the return type of the function.
    memory_scale : float
        Specify the scale of memory uses in the function versus
        input size.

    Returns
    -------
    Series
        Same index as caller.

    See Also
    --------
    Series.apply : For applying more complex functions on a Series.
    DataFrame.apply : Apply a function row-/column-wise.
    DataFrame.applymap : Apply a function elementwise on a whole DataFrame.

    Notes
    -----
    When ``arg`` is a dictionary, values in Series that are not in the
    dictionary (as keys) are converted to ``NaN``. However, if the
    dictionary is a ``dict`` subclass that defines ``__missing__`` (i.e.
    provides a method for default values), then this default is used
    rather than ``NaN``.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> import mars.dataframe as md
    >>> s = md.Series(['cat', 'dog', mt.nan, 'rabbit'])
    >>> s.execute()
    0      cat
    1      dog
    2      NaN
    3   rabbit
    dtype: object

    ``map`` accepts a ``dict`` or a ``Series``. Values that are not found
    in the ``dict`` are converted to ``NaN``, unless the dict has a default
    value (e.g. ``defaultdict``):

    >>> s.map({'cat': 'kitten', 'dog': 'puppy'}).execute()
    0   kitten
    1    puppy
    2      NaN
    3      NaN
    dtype: object

    It also accepts a function:

    >>> s.map('I am a {}'.format).execute()
    0       I am a cat
    1       I am a dog
    2       I am a nan
    3    I am a rabbit
    dtype: object

    To avoid applying the function to missing values (and keep them as
    ``NaN``) ``na_action='ignore'`` can be used:

    >>> s.map('I am a {}'.format, na_action='ignore').execute()
    0     I am a cat
    1     I am a dog
    2            NaN
    3  I am a rabbit
    dtype: object
    """
    op = DataFrameMap(arg=arg, na_action=na_action, memory_scale=memory_scale)
    return op(series, dtype=dtype)


def index_map(idx, mapper, na_action=None, dtype=None, memory_scale=None):
    """
    Map values using input correspondence (a dict, Series, or function).

    Parameters
    ----------
    mapper : function, dict, or Series
        Mapping correspondence.
    na_action : {None, 'ignore'}
        If 'ignore', propagate NA values, without passing them to the
        mapping correspondence.
    dtype : np.dtype, default None
        Specify return type of the function. Must be specified when
        we cannot decide the return type of the function.
    memory_scale : float
        Specify the scale of memory uses in the function versus
        input size.

    Returns
    -------
    applied : Union[Index, MultiIndex], inferred
        The output of the mapping function applied to the index.
        If the function returns a tuple with more than one element
        a MultiIndex will be returned.
    """
    op = DataFrameMap(arg=mapper, na_action=na_action, memory_scale=memory_scale)
    return op(idx, dtype=dtype)
