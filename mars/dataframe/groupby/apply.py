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
from ...serialization.serializables import TupleField, DictField, FunctionField
from ...utils import enter_current_session, quiet_stdio
from ..operands import DataFrameOperandMixin, DataFrameOperand
from ..utils import build_empty_df, build_empty_series, parse_index, \
    validate_output_types, make_dtypes, make_dtype


class GroupByApply(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.APPLY
    _op_module_ = 'dataframe.groupby'

    _func = FunctionField('func')
    _args = TupleField('args')
    _kwds = DictField('kwds')

    def __init__(self, func=None, args=None, kwds=None, output_types=None, **kw):
        super().__init__(_func=func, _args=args, _kwds=kwds, _output_types=output_types, **kw)

    @property
    def func(self):
        return self._func

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
        in_data = ctx[op.inputs[0].key]
        out = op.outputs[0]
        if not in_data:
            if op.output_types[0] == OutputType.dataframe:
                ctx[op.outputs[0].key] = build_empty_df(op.outputs[0].dtypes)
            else:
                ctx[op.outputs[0].key] = build_empty_series(
                    op.outputs[0].dtype, name=out.name)
            return

        applied = in_data.apply(op.func, *op.args, **op.kwds)

        if isinstance(applied, pd.DataFrame):
            # when there is only one group, pandas tend to return a DataFrame, while
            # we need to convert it into a compatible series
            if op.output_types[0] == OutputType.series:
                assert len(applied.index) == 1
                applied_idx = pd.MultiIndex.from_arrays(
                    [[applied.index[0]] * len(applied.columns), applied.columns.tolist()])
                applied_idx.names = [applied.index.name, None]
                applied = pd.Series(np.array(applied.iloc[0]), applied_idx,
                                    name=applied.columns.name)
            else:
                applied.columns.name = None
        else:
            applied.name = out.name
        ctx[out.key] = applied

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
                chunks.append(new_op.new_chunk(
                    inp_chunks, index=c.index, shape=(np.nan, len(out_df.dtypes)), dtypes=out_df.dtypes,
                    columns_value=out_df.columns_value, index_value=out_df.index_value))
            else:
                chunks.append(new_op.new_chunk(
                    inp_chunks, name=out_df.name, index=(c.index[0],), shape=(np.nan,), dtype=out_df.dtype,
                    index_value=out_df.index_value))

        new_op = op.copy()
        kw = out_df.params.copy()
        kw['chunks'] = chunks
        if op.output_types[0] == OutputType.dataframe:
            kw['nsplits'] = ((np.nan,) * len(chunks), (out_df.shape[1],))
        else:
            kw['nsplits'] = ((np.nan,) * len(chunks),)
        return new_op.new_tileables([in_groupby], **kw)

    def _infer_df_func_returns(self, in_groupby, in_df, dtypes, dtype=None,
                               name=None, index=None):
        index_value, output_type, new_dtypes = None, None, None

        try:
            infer_df = in_groupby.op.build_mock_groupby().apply(self.func, *self.args, **self.kwds)

            # todo return proper index when sort=True is implemented
            index_value = parse_index(infer_df.index[:0], in_df.key, self.func)

            # for backward compatibility
            dtype = dtype if dtype is not None else dtypes
            if isinstance(infer_df, pd.DataFrame):
                output_type = output_type or OutputType.dataframe
                new_dtypes = new_dtypes or infer_df.dtypes
            elif isinstance(infer_df, pd.Series):
                output_type = output_type or OutputType.series
                new_dtypes = new_dtypes or (name or infer_df.name, dtype or infer_df.dtype)
            else:
                output_type = OutputType.series
                new_dtypes = (name, dtype or pd.Series(infer_df).dtype)
        except:  # noqa: E722  # nosec
            pass

        self.output_types = [output_type] if not self.output_types else self.output_types
        dtypes = new_dtypes if dtypes is None else dtypes
        index_value = index_value if index is None else parse_index(index)
        return dtypes, index_value

    def __call__(self, groupby, dtypes=None, dtype=None, name=None, index=None):
        in_df = groupby
        while in_df.op.output_types[0] not in (OutputType.dataframe, OutputType.series):
            in_df = in_df.inputs[0]

        with quiet_stdio():
            dtypes, index_value = self._infer_df_func_returns(
                groupby, in_df, dtypes, dtype=dtype, name=name, index=index)
        if index_value is None:
            index_value = parse_index(None, (in_df.key, in_df.index_value.key))
        for arg, desc in zip((self.output_types, dtypes), ('output_types', 'dtypes')):
            if arg is None:
                raise TypeError(f'Cannot determine {desc} by calculating with enumerate data, '
                                'please specify it as arguments')

        if self.output_types[0] == OutputType.dataframe:
            new_shape = (np.nan, len(dtypes))
            return self.new_dataframe([groupby], shape=new_shape, dtypes=dtypes, index_value=index_value,
                                      columns_value=parse_index(dtypes.index, store_data=True))
        else:
            name = name or dtypes[0]
            dtype = dtype or dtypes[1]
            new_shape = (np.nan,)
            return self.new_series([groupby], name=name, shape=new_shape, dtype=dtype,
                                   index_value=index_value)


def groupby_apply(groupby, func, *args, output_type=None, dtypes=None, dtype=None,
                  name=None, index=None, **kwargs):
    """
    Apply function `func` group-wise and combine the results together.

    The function passed to `apply` must take a dataframe as its first
    argument and return a DataFrame, Series or scalar. `apply` will
    then take care of combining the results back together into a single
    dataframe or series. `apply` is therefore a highly flexible
    grouping method.

    While `apply` is a very flexible method, its downside is that
    using it can be quite a bit slower than using more specific methods
    like `agg` or `transform`. Pandas offers a wide range of method that will
    be much faster than using `apply` for their specific purposes, so try to
    use them before reaching for `apply`.

    Parameters
    ----------
    func : callable
        A callable that takes a dataframe as its first argument, and
        returns a dataframe, a series or a scalar. In addition the
        callable may take positional and keyword arguments.

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

    args, kwargs : tuple and dict
        Optional positional and keyword arguments to pass to `func`.

    Returns
    -------
    applied : Series or DataFrame

    See Also
    --------
    pipe : Apply function to the full GroupBy object instead of to each
        group.
    aggregate : Apply aggregate function to the GroupBy object.
    transform : Apply function column-by-column to the GroupBy object.
    Series.apply : Apply a function to a Series.
    DataFrame.apply : Apply a function to each row or column of a DataFrame.

    Notes
    -----
    When deciding output dtypes and shape of the return value, Mars will
    try applying ``func`` onto a mock grouped object, and the apply call
    may fail. When this happens, you need to specify the type of apply
    call (DataFrame or Series) in output_type.

    * For DataFrame output, you need to specify a list or a pandas Series
      as ``dtypes`` of output DataFrame. ``index`` of output can also be
      specified.
    * For Series output, you need to specify ``dtype`` and ``name`` of
      output Series.
    """
    output_types = kwargs.pop('output_types', None)
    object_type = kwargs.pop('object_type', None)
    output_types = validate_output_types(
        output_types=output_types, output_type=output_type, object_type=object_type)

    dtypes = make_dtypes(dtypes)
    dtype = make_dtype(dtype)
    op = GroupByApply(func=func, args=args, kwds=kwargs, output_types=output_types)
    return op(groupby, dtypes=dtypes, dtype=dtype, name=name, index=index)
