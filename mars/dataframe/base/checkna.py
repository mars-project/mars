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
from ...config import options
from ...serialize import BoolField
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType, \
    DATAFRAME_TYPE


class DataFrameCheckNA(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.CHECK_NA

    _positive = BoolField('positive')
    _use_inf_as_na = BoolField('use_inf_as_na')

    def __init__(self, positive=None, use_inf_as_na=None, sparse=None, object_type=None, **kw):
        super().__init__(_positive=positive, _use_inf_as_na=use_inf_as_na, _sparse=sparse,
                         _object_type=object_type, **kw)

    @property
    def positive(self) -> bool:
        return self._positive

    @property
    def use_inf_as_na(self) -> bool:
        return self._use_inf_as_na

    def __call__(self, df):
        if isinstance(df, DATAFRAME_TYPE):
            self._object_type = ObjectType.dataframe
        else:
            self._object_type = ObjectType.series

        params = df.params.copy()
        if self.object_type == ObjectType.dataframe:
            params['dtypes'] = pd.Series([np.dtype('bool')] * len(df.dtypes),
                                         index=df.columns_value.to_pandas())
        else:
            params['dtype'] = np.dtype('bool')
        return self.new_tileable([df], **params)

    @classmethod
    def tile(cls, op: "DataFrameCheckNA"):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        chunks = []
        for c in in_df.chunks:
            params = c.params.copy()
            if op.object_type == ObjectType.dataframe:
                params['dtypes'] = pd.Series([np.dtype('bool')] * len(c.dtypes),
                                             index=c.columns_value.to_pandas())
            else:
                params['dtype'] = np.dtype('bool')
            new_op = op.copy().reset_key()
            chunks.append(new_op.new_chunk([c], **params))

        new_op = op.copy().reset_key()
        params = out_df.params.copy()
        params.update(dict(chunks=chunks, nsplits=in_df.nsplits))
        return new_op.new_tileables([in_df], **params)

    @classmethod
    def execute(cls, ctx, op: "DataFrameCheckNA"):
        in_data = ctx[op.inputs[0].key]
        try:
            pd.set_option('mode.use_inf_as_na', op.use_inf_as_na)
            if op.positive:
                ctx[op.outputs[0].key] = in_data.isna()
            else:
                ctx[op.outputs[0].key] = in_data.notna()
        finally:
            pd.reset_option('mode.use_inf_as_na')


def isna(df):
    op = DataFrameCheckNA(positive=True, use_inf_as_na=options.dataframe.mode.use_inf_as_na)
    return op(df)


def notna(df):
    op = DataFrameCheckNA(positive=False, use_inf_as_na=options.dataframe.mode.use_inf_as_na)
    return op(df)
