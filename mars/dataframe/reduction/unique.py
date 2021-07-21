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


import pandas as pd
import numpy as np

from ... import opcodes as OperandDef
from ...core import OutputType
from ...tensor.core import TensorOrder
from ...utils import lazy_import
from .core import DataFrameReductionOperand, DataFrameReductionMixin, CustomReduction

cudf = lazy_import('cudf', globals=globals())


class UniqueReduction(CustomReduction):
    def agg(self, data):  # noqa: W0221  # pylint: disable=arguments-differ
        xdf = cudf if self.is_gpu() else pd
        # convert to series data
        return xdf.Series(data.unique())

    def post(self, data):  # noqa: W0221  # pylint: disable=arguments-differ
        return data.unique()


class DataFrameUnique(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = OperandDef.UNIQUE
    _func_name = 'unique'

    @classmethod
    def get_reduction_callable(cls, op):
        return UniqueReduction(name=cls._func_name, is_gpu=op.is_gpu())

    @classmethod
    def tile(cls, op):
        if op.method == 'tree':
            return (yield from super().tile(op))
        else:
            raise NotImplementedError(f"Method {op.method} hasn't been supported")

    def __call__(self, a):
        self.output_types = [OutputType.tensor]
        return self.new_tileables([a], shape=(np.nan,), dtype=a.dtype, order=TensorOrder.C_ORDER)[0]


def unique(values, method='tree'):
    """
    Uniques are returned in order of appearance. This does NOT sort.

    Parameters
    ----------
    values : 1d array-like
    method : 'shuffle' or 'tree', 'tree' method provide a better performance, 'shuffle'
    is recommended if the number of unique values is very large.
    See Also
    --------
    Index.unique
    Series.unique

    Examples
    --------
    >>> import mars.dataframe as md
    >>> import pandas as pd
    >>> md.unique(md.Series([2, 1, 3, 3])).execute()
    array([2, 1, 3])

    >>> md.unique(md.Series([2] + [1] * 5)).execute()
    array([2, 1])

    >>> md.unique(md.Series([pd.Timestamp('20160101'),
    ...                     pd.Timestamp('20160101')])).execute()
    array(['2016-01-01T00:00:00.000000000'], dtype='datetime64[ns]')

    >>> md.unique(md.Series([pd.Timestamp('20160101', tz='US/Eastern'),
    ...                      pd.Timestamp('20160101', tz='US/Eastern')])).execute()
    array([Timestamp('2016-01-01 00:00:00-0500', tz='US/Eastern')],
          dtype=object)
    """
    op = DataFrameUnique(method=method)
    return op(values)
