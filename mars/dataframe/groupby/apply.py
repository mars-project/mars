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

from ... import opcodes
from ...core import OutputType, get_output_types
from ...custom_log import redirect_custom_log
from ...serialize import TupleField, DictField, FunctionField, StringField
from ...utils import enter_current_session
from ..operands import DataFrameOperandMixin, DataFrameOperand
from ..utils import build_df, build_empty_df, build_series, build_empty_series, \
    parse_index, validate_output_types


class GroupByApply(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.APPLY
    _op_module_ = 'dataframe.groupby'

    _func = FunctionField('func')
    _args = TupleField('args')
    _kwds = DictField('kwds')
    # for chunk
    _tileable_op_key = StringField('tileable_op_key')

    def __init__(self, func=None, args=None, kwds=None, output_types=None,
                 tileable_op_key=None, **kw):
        super().__init__(_func=func, _args=args, _kwds=kwds, _output_types=output_types,
                         _tileable_op_key=tileable_op_key, **kw)

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
    def tileable_op_key(self):
        return self._tileable_op_key

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx, op):
        in_data = ctx[op.inputs[0].key]
        if not in_data:
            if op.output_types[0] == OutputType.dataframe:
                ctx[op.outputs[0].key] = build_empty_df(op.outputs[0].dtypes)
            else:
                ctx[op.outputs[0].key] = build_empty_series(op.outputs[0].dtype)
            return

        applied = in_data.apply(op.func, *op.args, **op.kwds)

        # when there is only one group, pandas tend to return a DataFrame, while
        # we need to convert it into a compatible series
        if op.output_types[0] == OutputType.series and isinstance(applied, pd.DataFrame):
            assert len(applied.index) == 1
            applied_idx = pd.MultiIndex.from_arrays(
                [[applied.index[0]] * len(applied.columns), applied.columns.tolist()])
            applied_idx.names = [applied.index.name, None]
            applied = pd.Series(np.array(applied.iloc[0]), applied_idx, name=applied.columns.name)
        ctx[op.outputs[0].key] = applied

    @classmethod
    def tile(cls, op):
        in_groupby = op.inputs[0]
        out_df = op.outputs[0]

        chunks = []
        for c in in_groupby.chunks:
            inp_chunks = [c]

            new_op = op.copy().reset_key()
            new_op._tileable_op_key = op.key
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

    def _infer_df_func_returns(self, in_groupby, in_df, dtypes, index):
        index_value, output_type, new_dtypes = None, None, None

        try:
            if in_df.op.output_types[0] == OutputType.dataframe:
                test_df = build_df(in_df, size=2)
            else:
                test_df = build_series(in_df, size=2, name=in_df.name)

            selection = getattr(in_groupby.op, 'selection', None)
            if selection:
                test_df = test_df[selection]

            with np.errstate(all='ignore'):
                infer_df = self.func(test_df, *self.args, **self.kwds)

            # todo return proper index when sort=True is implemented
            index_value = parse_index(None, in_df.key, self.func)

            if infer_df is None:
                output_type = get_output_types(in_df)[0]
                index_value = parse_index(pd.Index([], dtype=np.object))
                if output_type == OutputType.dataframe:
                    new_dtypes = pd.Series([], index=pd.Index([]))
                else:
                    new_dtypes = (None, np.dtype('O'))
            elif isinstance(infer_df, pd.DataFrame):
                output_type = output_type or OutputType.dataframe
                new_dtypes = new_dtypes or infer_df.dtypes
            elif isinstance(infer_df, pd.Series):
                output_type = output_type or OutputType.series
                new_dtypes = new_dtypes or (infer_df.name, infer_df.dtype)
            else:
                output_type = OutputType.series
                new_dtypes = (None, pd.Series(infer_df).dtype)
        except:  # noqa: E722  # nosec
            pass

        self.output_types = [output_type] if not self.output_types else self.output_types
        dtypes = new_dtypes if dtypes is None else dtypes
        index_value = index_value if index is None else parse_index(index)
        return dtypes, index_value

    def __call__(self, groupby, dtypes=None, index=None):
        in_df = groupby
        while in_df.op.output_types[0] not in (OutputType.dataframe, OutputType.series):
            in_df = in_df.inputs[0]

        dtypes, index_value = self._infer_df_func_returns(groupby, in_df, dtypes, index)
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
            name, dtype = dtypes
            new_shape = (np.nan,)
            return self.new_series([groupby], name=name, shape=new_shape, dtype=dtype,
                                   index_value=index_value)


def groupby_apply(groupby, func, *args, dtypes=None, index=None, output_type=None, **kwargs):
    # todo this can be done with sort_index implemented
    if not groupby.op.groupby_params.get('as_index', True):
        raise NotImplementedError('apply when set_index == False is not supported')

    output_types = kwargs.pop('output_types', None)
    object_type = kwargs.pop('object_type', None)
    output_types = validate_output_types(
        output_types=output_types, output_type=output_type, object_type=object_type)

    op = GroupByApply(func=func, args=args, kwds=kwargs, output_types=output_types)
    return op(groupby, dtypes=dtypes, index=index)
