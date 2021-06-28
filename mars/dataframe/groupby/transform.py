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

from ... import opcodes
from ...core import OutputType
from ...core.custom_log import redirect_custom_log
from ...serialization.serializables import BoolField, TupleField, DictField, AnyField
from ...utils import enter_current_session, quiet_stdio
from ..operands import DataFrameOperandMixin, DataFrameOperand
from ..utils import build_empty_df, build_empty_series, parse_index


class GroupByTransform(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.TRANSFORM
    _op_module_ = 'dataframe.groupby'

    _func = AnyField('func')
    _args = TupleField('args')
    _kwds = DictField('kwds')

    _call_agg = BoolField('call_agg')

    def __init__(self, func=None, args=None, kwds=None, call_agg=None, output_types=None,
                 **kw):
        super().__init__(_func=func, _args=args, _kwds=kwds, _call_agg=call_agg,
                         _output_types=output_types, **kw)

    @property
    def func(self):
        return self._func

    @property
    def args(self):
        return getattr(self, '_args', None) or ()

    @property
    def kwds(self):
        return getattr(self, '_kwds', None) or dict()

    @property
    def call_agg(self):
        return self._call_agg

    def _infer_df_func_returns(self, in_groupby, dtypes, index):
        index_value, output_types, new_dtypes = None, None, None

        output_types = [OutputType.dataframe] \
            if in_groupby.op.output_types[0] == OutputType.dataframe_groupby else [OutputType.series]

        try:
            mock_groupby = in_groupby.op.build_mock_groupby()
            with np.errstate(all='ignore'), quiet_stdio():
                if self.call_agg:
                    infer_df = mock_groupby.agg(self.func, *self.args, **self.kwds)
                else:
                    infer_df = mock_groupby.transform(self.func, *self.args, **self.kwds)

            # todo return proper index when sort=True is implemented
            index_value = parse_index(None, in_groupby.key, self.func)

            if isinstance(infer_df, pd.DataFrame):
                output_types = [OutputType.dataframe]
                new_dtypes = new_dtypes or infer_df.dtypes
            else:
                output_types = [OutputType.series]
                new_dtypes = new_dtypes or (infer_df.name, infer_df.dtype)
        except:  # noqa: E722  # nosec
            pass

        self.output_types = output_types if not self.output_types else self.output_types
        dtypes = new_dtypes if dtypes is None else dtypes
        index_value = index_value if index is None else parse_index(index)
        return dtypes, index_value

    def __call__(self, groupby, dtypes=None, dtype=None, name=None, index=None):
        in_df = groupby.inputs[0]

        if dtypes is None and dtype is not None:
            dtypes = (name, dtype)
        dtypes, index_value = self._infer_df_func_returns(groupby, dtypes, index)
        if index_value is None:
            index_value = parse_index(None, (in_df.key, in_df.index_value.key))
        for arg, desc in zip((self.output_types, dtypes), ('output_types', 'dtypes')):
            if arg is None:
                raise TypeError(f'Cannot determine {desc} by calculating with enumerate data, '
                                'please specify it as arguments')

        if self.output_types[0] == OutputType.dataframe:
            new_shape = (np.nan if self.call_agg else in_df.shape[0], len(dtypes))
            return self.new_dataframe([groupby], shape=new_shape, dtypes=dtypes, index_value=index_value,
                                      columns_value=parse_index(dtypes.index, store_data=True))
        else:
            name, dtype = dtypes
            new_shape = (np.nan,) if self.call_agg else groupby.shape
            return self.new_series([groupby], name=name, shape=new_shape, dtype=dtype,
                                   index_value=index_value)

    @classmethod
    def tile(cls, op):
        in_groupby = op.inputs[0]
        out_df = op.outputs[0]

        chunks = []
        for c in in_groupby.chunks:
            inp_chunks = [c]

            new_op = op.copy().reset_key()
            new_op.tileable_op_key = op.key
            if op.output_types[0] == OutputType.dataframe:
                new_index = c.index if c.ndim == 2 else c.index + (0,)
                chunks.append(new_op.new_chunk(
                    inp_chunks, index=new_index, shape=(np.nan, len(out_df.dtypes)), dtypes=out_df.dtypes,
                    columns_value=out_df.columns_value, index_value=out_df.index_value))
            else:
                chunks.append(new_op.new_chunk(
                    inp_chunks, name=out_df.name, index=(c.index[0],), shape=(np.nan,), dtype=out_df.dtype,
                    index_value=out_df.index_value))

        new_op = op.copy()
        kw = out_df.params.copy()
        kw['chunks'] = chunks
        if op.output_types[0] == OutputType.dataframe:
            kw['nsplits'] = ((np.nan,) * len(chunks), (len(out_df.dtypes),))
        else:
            kw['nsplits'] = ((np.nan,) * len(chunks),)
        return new_op.new_tileables([in_groupby], **kw)

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx, op):
        in_data = ctx[op.inputs[0].key]
        out_chunk = op.outputs[0]

        if not in_data:
            if op.output_types[0] == OutputType.dataframe:
                ctx[op.outputs[0].key] = build_empty_df(out_chunk.dtypes)
            else:
                ctx[op.outputs[0].key] = build_empty_series(
                    out_chunk.dtype, name=out_chunk.name)
            return

        if op.call_agg:
            result = in_data.agg(op.func, *op.args, **op.kwds)
        elif in_data.shape[0] > 0:
            # cannot perform groupby-transform over empty dataframe
            result = in_data.transform(op.func, *op.args, **op.kwds)
        else:
            if out_chunk.ndim == 2:
                result = pd.DataFrame(columns=out_chunk.dtypes.index)
            else:
                result = pd.Series([], name=out_chunk.name, dtype=out_chunk.dtype)

        if result.ndim == 2:
            result = result.astype(out_chunk.dtypes, copy=False)
        else:
            result = result.astype(out_chunk.dtype, copy=False)
        ctx[op.outputs[0].key] = result


def groupby_transform(groupby, f, *args, dtypes=None, dtype=None, name=None, index=None,
                      output_types=None, **kwargs):
    """
    Call function producing a like-indexed DataFrame on each group and
    return a DataFrame having the same indexes as the original object
    filled with the transformed values

    Parameters
    ----------
    f : function
        Function to apply to each group.

    dtypes : Series, default None
        Specify dtypes of returned DataFrames. See `Notes` for more details.

    dtype : numpy.dtype, default None
        Specify dtype of returned Series. See `Notes` for more details.

    name : str, default None
        Specify name of returned Series. See `Notes` for more details.

    *args
        Positional arguments to pass to func

    **kwargs
        Keyword arguments to be passed into func.

    Returns
    -------
    DataFrame

    See Also
    --------
    DataFrame.groupby.apply
    DataFrame.groupby.aggregate
    DataFrame.transform

    Notes
    -----
    Each group is endowed the attribute 'name' in case you need to know
    which group you are working on.

    The current implementation imposes three requirements on f:

    * f must return a value that either has the same shape as the input
      subframe or can be broadcast to the shape of the input subframe.
      For example, if `f` returns a scalar it will be broadcast to have the
      same shape as the input subframe.
    * if this is a DataFrame, f must support application column-by-column
      in the subframe. If f also supports application to the entire subframe,
      then a fast path is used starting from the second chunk.
    * f must not mutate groups. Mutation is not supported and may
      produce unexpected results.

    Notes
    -----
    When deciding output dtypes and shape of the return value, Mars will
    try applying ``func`` onto a mock grouped object, and the transform call
    may fail.

    * For DataFrame output, you need to specify a list or a pandas Series
      as ``dtypes`` of output DataFrame. ``index`` of output can also be
      specified.
    * For Series output, you need to specify ``dtype`` and ``name`` of
      output Series.

    Examples
    --------

    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
    ...                           'foo', 'bar'],
    ...                    'B' : ['one', 'one', 'two', 'three',
    ...                           'two', 'two'],
    ...                    'C' : [1, 5, 5, 2, 5, 5],
    ...                    'D' : [2.0, 5., 8., 1., 2., 9.]})
    >>> grouped = df.groupby('A')
    >>> grouped.transform(lambda x: (x - x.mean()) / x.std()).execute()
              C         D
    0 -1.154701 -0.577350
    1  0.577350  0.000000
    2  0.577350  1.154701
    3 -1.154701 -1.000000
    4  0.577350 -0.577350
    5  0.577350  1.000000

    Broadcast result of the transformation

    >>> grouped.transform(lambda x: x.max() - x.min()).execute()
       C    D
    0  4  6.0
    1  3  8.0
    2  4  6.0
    3  3  8.0
    4  4  6.0
    5  3  8.0
    """
    call_agg = kwargs.pop('_call_agg', False)
    if not call_agg and isinstance(f, (dict, list)):
        raise TypeError(f'Does not support transform with {type(f)}')

    op = GroupByTransform(func=f, args=args, kwds=kwargs, output_types=output_types,
                          call_agg=call_agg)
    return op(groupby, dtypes=dtypes, dtype=dtype, name=name, index=index)
