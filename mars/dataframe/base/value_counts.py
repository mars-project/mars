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
from ...core import OutputType, recursive_tile
from ...core.operand import OperandStage
from ...lib.version import parse as parse_version
from ...serialization.serializables import KeyField, BoolField, \
    Int64Field, StringField
from ...utils import has_unknown_shape
from ..core import Series
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_series, parse_index

_keep_original_order = parse_version(pd.__version__) >= parse_version('1.3.0')


class DataFrameValueCounts(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.VALUE_COUNTS

    _input = KeyField('input')
    _normalize = BoolField('normalize')
    _sort = BoolField('sort')
    _ascending = BoolField('ascending')
    _bins = Int64Field('bins')
    _dropna = BoolField('dropna')
    _method = StringField('method')
    _convert_index_to_interval = BoolField('convert_index_to_interval')
    _nrows = Int64Field('nrows')

    def __init__(self, normalize=None, sort=None, ascending=None,
                 bins=None, dropna=None, method=None,
                 convert_index_to_interval=None, nrows=None, **kw):
        super().__init__(_normalize=normalize, _sort=sort, _ascending=ascending,
                         _bins=bins, _dropna=dropna, _method=method,
                         _convert_index_to_interval=convert_index_to_interval,
                         _nrows=nrows, **kw)
        self.output_types = [OutputType.series]

    @property
    def input(self):
        return self._input

    @property
    def normalize(self):
        return self._normalize

    @property
    def sort(self):
        return self._sort

    @property
    def ascending(self):
        return self._ascending

    @property
    def bins(self):
        return self._bins

    @property
    def dropna(self):
        return self._dropna

    @property
    def method(self):
        return self._method

    @property
    def convert_index_to_interval(self):
        return self._convert_index_to_interval

    @property
    def nrows(self):
        return self._nrows

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, inp):
        test_series = build_series(inp).value_counts(normalize=self.normalize)
        if self._bins is not None:
            from .cut import cut

            # cut
            try:
                inp = cut(inp, self._bins, include_lowest=True)
            except TypeError:  # pragma: no cover
                raise TypeError("bins argument only works with numeric data.")

            self._bins = None
            self._convert_index_to_interval = True
            return self.new_series([inp], shape=(np.nan,),
                                   index_value=parse_index(pd.CategoricalIndex([]),
                                                           inp, store_data=False),
                                   name=inp.name, dtype=test_series.dtype)
        else:
            return self.new_series([inp], shape=(np.nan,),
                                   index_value=parse_index(test_series.index, store_data=False),
                                   name=inp.name, dtype=test_series.dtype)

    @classmethod
    def tile(cls, op: "DataFrameValueCounts"):
        inp = op.input
        out = op.outputs[0]

        if len(inp.chunks) == 1:
            chunk_op = op.copy().reset_key()
            chunk_param = out.params
            chunk_param['index'] = (0,)
            chunk = chunk_op.new_chunk(inp.chunks, kws=[chunk_param])

            new_op = op.copy()
            param = out.params
            param['chunks'] = [chunk]
            param['nsplits'] = ((np.nan,),)
            return new_op.new_seriess(op.inputs, kws=[param])

        inp = Series(inp)

        if op.dropna:
            inp = inp.dropna()

        inp = inp.groupby(inp, sort=not _keep_original_order).count(method=op.method)

        if op.normalize:
            if op.convert_index_to_interval:
                if has_unknown_shape(op.input):
                    yield
                inp = inp.truediv(op.input.shape[0], axis=0)
            else:
                inp = inp.truediv(inp.sum(), axis=0)

        if op.sort:
            inp = inp.sort_values(ascending=op.ascending,
                                  kind='mergesort' if _keep_original_order else 'quicksort')

            if op.nrows:
                # set to sort_values
                inp.op._nrows = op.nrows
        elif op.nrows:
            inp = inp.iloc[:op.nrows]

        ret = yield from recursive_tile(inp)

        chunks = []
        for c in ret.chunks:
            chunk_op = DataFrameValueCounts(
                convert_index_to_interval=op.convert_index_to_interval,
                stage=OperandStage.map)
            chunk_params = c.params
            if op.convert_index_to_interval:
                # convert index to IntervalDtype
                chunk_params['index_value'] = parse_index(pd.IntervalIndex([]),
                                                          c, store_data=False)
            chunks.append(chunk_op.new_chunk([c], kws=[chunk_params]))

        new_op = op.copy()
        params = out.params
        params['chunks'] = chunks
        params['nsplits'] = ret.nsplits
        return new_op.new_seriess(out.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx, op: "DataFrameValueCounts"):
        if op.stage != OperandStage.map:
            in_data = ctx[op.input.key]
            if op.convert_index_to_interval:
                result = in_data.value_counts(
                    normalize=False, sort=op.sort, ascending=op.ascending,
                    bins=op.bins, dropna=op.dropna)
                if op.normalize:
                    result /= in_data.shape[0]
            else:
                try:
                    result = in_data.value_counts(
                        normalize=op.normalize, sort=op.sort, ascending=op.ascending,
                        bins=op.bins, dropna=op.dropna)
                except ValueError:
                    in_data = in_data.copy()
                    result = in_data.value_counts(
                        normalize=op.normalize, sort=op.sort, ascending=op.ascending,
                        bins=op.bins, dropna=op.dropna)
        else:
            result = ctx[op.input.key]
            # set index name to None to keep consistency with pandas
            result.index.name = None
        if op.convert_index_to_interval:
            # convert CategoricalDtype which generated in `cut`
            # to IntervalDtype
            result.index = result.index.astype('interval')
        if op.nrows:
            result = result.head(op.nrows)
        ctx[op.outputs[0].key] = result


def value_counts(series, normalize=False, sort=True, ascending=False,
                 bins=None, dropna=True, method='auto'):
    """
    Return a Series containing counts of unique values.

    The resulting object will be in descending order so that the
    first element is the most frequently-occurring element.
    Excludes NA values by default.

    Parameters
    ----------
    normalize : bool, default False
        If True then the object returned will contain the relative
        frequencies of the unique values.
    sort : bool, default True
        Sort by frequencies.
    ascending : bool, default False
        Sort in ascending order.
    bins : int, optional
        Rather than count values, group them into half-open bins,
        a convenience for ``pd.cut``, only works with numeric data.
    dropna : bool, default True
        Don't include counts of NaN.
    method : str, default 'auto'
        'auto', 'shuffle', or 'tree', 'tree' method provide
        a better performance, while 'shuffle' is recommended
        if aggregated result is very large, 'auto' will use
        'shuffle' method in distributed mode and use 'tree'
        in local mode.

    Returns
    -------
    Series

    See Also
    --------
    Series.count: Number of non-NA elements in a Series.
    DataFrame.count: Number of non-NA elements in a DataFrame.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> import mars.tensor as mt

    >>> s = md.Series([3, 1, 2, 3, 4, mt.nan])
    >>> s.value_counts().execute()
    3.0    2
    4.0    1
    2.0    1
    1.0    1
    dtype: int64

    With `normalize` set to `True`, returns the relative frequency by
    dividing all values by the sum of values.

    >>> s = md.Series([3, 1, 2, 3, 4, mt.nan])
    >>> s.value_counts(normalize=True).execute()
    3.0    0.4
    4.0    0.2
    2.0    0.2
    1.0    0.2
    dtype: float64

    **bins**

    Bins can be useful for going from a continuous variable to a
    categorical variable; instead of counting unique
    apparitions of values, divide the index in the specified
    number of half-open bins.

    >>> s.value_counts(bins=3).execute()
    (2.0, 3.0]      2
    (0.996, 2.0]    2
    (3.0, 4.0]      1
    dtype: int64

    **dropna**

    With `dropna` set to `False` we can also see NaN index values.

    >>> s.value_counts(dropna=False).execute()
    3.0    2
    NaN    1
    4.0    1
    2.0    1
    1.0    1
    dtype: int64
    """
    op = DataFrameValueCounts(normalize=normalize, sort=sort,
                              ascending=ascending, bins=bins,
                              dropna=dropna, method=method)
    return op(series)
