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

import cloudpickle
import numpy as np
import pandas as pd

from ... import opcodes
from ...serialize import BytesField, TupleField, DictField
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType
from ..utils import build_empty_df, build_empty_series, parse_index


class GroupByApply(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.GROUPBY_APPLY

    _func = BytesField('func', on_serialize=cloudpickle.dumps,
                       on_deserialize=cloudpickle.loads)
    _args = TupleField('args')
    _kwds = DictField('kwds')

    def __init__(self, func=None, args=None, kwds=None, object_type=None, **kw):
        super().__init__(_func=func, _args=args, _kwds=kwds, _object_type=object_type,
                         **kw)

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
    def tile(cls, op):
        pass

    def _infer_df_func_returns(self, in_object_type, in_dtypes, dtypes, index):
        index_value, object_type, new_dtypes = None, None, None
        try:
            if in_object_type == ObjectType.dataframe:
                empty_df = build_empty_df(in_dtypes, index=pd.RangeIndex(0, 10))
            else:
                empty_df = build_empty_series(in_dtypes, index=pd.RangeIndex(0, 10))

            with np.errstate(all='ignore'):
                infer_df = self._func(empty_df, *self.args, **self.kwds)
            index_value = parse_index(infer_df.index)

            if isinstance(infer_df, pd.DataFrame):
                object_type = object_type or ObjectType.dataframe
                new_dtypes = new_dtypes or infer_df.dtypes
            elif isinstance(infer_df, pd.Series):
                object_type = object_type or ObjectType.series
                new_dtypes = new_dtypes or infer_df.dtype
            else:
                object_type = ObjectType.series
                new_dtypes = pd.Series(infer_df).dtype
        except:  # noqa: E722
            pass

        self._object_type = object_type if self._object_type is None else self._object_type
        dtypes = new_dtypes if dtypes is None else dtypes
        index_value = index_value if index is None else parse_index(index)
        return dtypes, index_value

    def __call__(self, groupby, dtypes=None, index=None):
        in_df = groupby.inputs[0]
        dtypes, index_value = self._infer_df_func_returns(
            in_df.op.object_type, in_df.dtypes, dtypes, index)
        for arg, desc in zip((self._object_type, dtypes, index_value),
                             ('object_type', 'dtypes', 'index')):
            if arg is None:
                raise TypeError('Cannot determine %s by calculating with enumerate data, '
                                'please specify it as arguments' % desc)

        if self.object_type == ObjectType.dataframe:
            return self.new_dataframe([groupby], shape=(np.nan, np.nan), dtypes=dtypes,
                                      index_value=index_value, columns_value=in_df.columns_value)
        else:
            return self.new_series([groupby], shape=(np.nan,), dtype=dtypes, index_value=index_value)


def groupby_apply(groupby, func, *args, dtypes=None, index=None, object_type=None, **kwargs):
    op = GroupByApply(func=func, args=args, kwds=kwargs, object_type=object_type)
    return op(groupby, dtypes=dtypes, index=index)
