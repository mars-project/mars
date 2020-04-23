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
    """
    Detect missing values.

    Return a boolean same-sized object indicating if the values are NA.
    NA values, such as None or :attr:`numpy.NaN`, gets mapped to True
    values.

    Everything else gets mapped to False values. Characters such as empty
    strings ``''`` or :attr:`numpy.inf` are not considered NA values
    (unless you set ``pandas.options.mode.use_inf_as_na = True``).

    Returns
    -------
    DataFrame
        Mask of bool values for each element in DataFrame that
        indicates whether an element is not an NA value.

    See Also
    --------
    DataFrame.isnull : Alias of isna.
    DataFrame.notna : Boolean inverse of isna.
    DataFrame.dropna : Omit axes labels with missing values.
    isna : Top-level isna.

    Examples
    --------
    Show which entries in a DataFrame are NA.

    >>> import numpy as np
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'age': [5, 6, np.NaN],
    ...                    'born': [md.NaT, md.Timestamp('1939-05-27'),
    ...                             md.Timestamp('1940-04-25')],
    ...                    'name': ['Alfred', 'Batman', ''],
    ...                    'toy': [None, 'Batmobile', 'Joker']})
    >>> df.execute()
       age       born    name        toy
    0  5.0        NaT  Alfred       None
    1  6.0 1939-05-27  Batman  Batmobile
    2  NaN 1940-04-25              Joker

    >>> df.isna().execute()
         age   born   name    toy
    0  False   True  False   True
    1  False  False  False  False
    2   True  False  False  False

    Show which entries in a Series are NA.

    >>> ser = md.Series([5, 6, np.NaN])
    >>> ser.execute()
    0    5.0
    1    6.0
    2    NaN
    dtype: float64

    >>> ser.isna().execute()
    0    False
    1    False
    2     True
    dtype: bool
    """
    op = DataFrameCheckNA(positive=True, use_inf_as_na=options.dataframe.mode.use_inf_as_na)
    return op(df)


def notna(df):
    """
    Detect existing (non-missing) values.

    Return a boolean same-sized object indicating if the values are not NA.
    Non-missing values get mapped to True. Characters such as empty
    strings ``''`` or :attr:`numpy.inf` are not considered NA values
    (unless you set ``pandas.options.mode.use_inf_as_na = True``).
    NA values, such as None or :attr:`numpy.NaN`, get mapped to False
    values.

    Returns
    -------
    DataFrame
        Mask of bool values for each element in DataFrame that
        indicates whether an element is not an NA value.

    See Also
    --------
    DataFrame.notnull : Alias of notna.
    DataFrame.isna : Boolean inverse of notna.
    DataFrame.dropna : Omit axes labels with missing values.
    notna : Top-level notna.

    Examples
    --------
    Show which entries in a DataFrame are not NA.

    >>> import numpy as np
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'age': [5, 6, np.NaN],
    ...                    'born': [md.NaT, md.Timestamp('1939-05-27'),
    ...                             md.Timestamp('1940-04-25')],
    ...                    'name': ['Alfred', 'Batman', ''],
    ...                    'toy': [None, 'Batmobile', 'Joker']})
    >>> df.execute()
       age       born    name        toy
    0  5.0        NaT  Alfred       None
    1  6.0 1939-05-27  Batman  Batmobile
    2  NaN 1940-04-25              Joker

    >>> df.notna().execute()
         age   born  name    toy
    0   True  False  True  False
    1   True   True  True   True
    2  False   True  True   True

    Show which entries in a Series are not NA.

    >>> ser = md.Series([5, 6, np.NaN])
    >>> ser.execute()
    0    5.0
    1    6.0
    2    NaN
    dtype: float64

    >>> ser.notna().execute()
    0     True
    1     True
    2    False
    dtype: bool
    """
    op = DataFrameCheckNA(positive=False, use_inf_as_na=options.dataframe.mode.use_inf_as_na)
    return op(df)


isnull = isna
notnull = notna
