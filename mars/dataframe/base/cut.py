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

from functools import partial
from numbers import Integral

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...core import Base, Entity, ExecutableTuple
from ...context import get_context
from ...serialize import KeyField, AnyField, BoolField, Int32Field, StringField
from ...tensor import tensor as astensor
from ...tensor.core import TENSOR_TYPE
from ...tensor.operands import TensorOrder
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape
from ..core import SERIES_TYPE, INDEX_TYPE
from ..datasource.index import from_pandas as asindex
from ..initializer import Series as asseries
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import parse_index


class DataFrameCut(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.CUT

    _input = KeyField('input')
    _bins = AnyField('bins')
    _right = BoolField('right')
    _labels = AnyField('labels')
    _retbins = BoolField('retbins')
    _precision = Int32Field('precision')
    _include_lowest = BoolField('include_lowest')
    _duplicates = StringField('duplicates')
    _input_min = KeyField('input_min')
    _input_max = KeyField('input_max')

    def __init__(self, bins=None, right=None, labels=None, retbins=None,
                 precision=None, include_lowest=None, duplicates=None,
                 input_min=None, input_max=None, **kw):
        super().__init__(_bins=bins, _right=right, _labels=labels,
                         _retbins=retbins, _precision=precision,
                         _include_lowest=include_lowest, _duplicates=duplicates,
                         _input_min=input_min, _input_max=input_max, **kw)

    @property
    def input(self):
        return self._input

    @property
    def bins(self):
        return self._bins

    @property
    def right(self):
        return self._right

    @property
    def labels(self):
        return self._labels

    @property
    def retbins(self):
        return self._retbins

    @property
    def precision(self):
        return self._precision

    @property
    def include_lowest(self):
        return self._include_lowest

    @property
    def duplicates(self):
        return self._duplicates

    @property
    def input_min(self):
        return self._input_min

    @property
    def input_max(self):
        return self._input_max

    @property
    def output_limit(self):
        return 1 if not self._retbins else 2

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        self._input = next(inputs_iter)
        if self._input_min is not None:
            self._input_min = next(inputs_iter)
        if self._input_max is not None:
            self._input_max = next(inputs_iter)
        if isinstance(self._bins, (Base, Entity)):
            self._bins = next(inputs_iter)
        if isinstance(self._labels, (Base, Entity)):
            self._labels = next(inputs_iter)

    def __call__(self, x):
        if isinstance(x, pd.Series):
            x = asseries(x)
        elif not isinstance(x, (Base, Entity)):
            x = astensor(x)
        if x.ndim != 1:
            raise ValueError('Input array must be 1 dimensional')
        if x.size == 0:
            raise ValueError('Cannot cut empty array')

        inputs = [x]
        if isinstance(self.bins, Integral):
            self._input_min = input_min = x.min()
            self._input_max = input_max = x.max()
            inputs.extend([input_min, input_max])

        if self._labels is not None and \
                not isinstance(self._labels, (bool, Base, Entity)):
            self._labels = np.asarray(self._labels)

        # infer dtype
        x_empty = pd.Series([1], dtype=x.dtype) if isinstance(x, SERIES_TYPE) else \
            np.asarray([1], dtype=x.dtype)
        if isinstance(self._bins, INDEX_TYPE):
            bins = self._bins.index_value.to_pandas()
            inputs.append(self._bins)
            bins_unknown = True
        elif isinstance(self._bins, (Base, Entity)):
            bins = np.asarray([2], dtype=self._bins.dtype)
            inputs.append(self._bins)
            bins_unknown = True
        else:
            bins = self._bins
            bins_unknown = isinstance(self._bins, Integral)
        if isinstance(self._labels, (Base, Entity)):
            bins_unknown = True
            labels = None
            inputs.append(self._labels)
        else:
            if self._labels is False or not bins_unknown:
                labels = self._labels
            else:
                labels = None
        ret = pd.cut(x_empty, bins, right=self._right, labels=labels,
                     retbins=True, include_lowest=self._include_lowest,
                     duplicates=self._duplicates)

        kws = []
        object_types = []
        if bins_unknown and isinstance(ret[0].dtype, pd.CategoricalDtype):
            # inaccurate dtype, just create an empty one
            out_dtype = pd.CategoricalDtype()
        else:
            out_dtype = ret[0].dtype
        if isinstance(ret[0], pd.Series):
            object_types.append(ObjectType.series)
            kws.append({
                'dtype': out_dtype,
                'shape': x.shape,
                'index_value': x.index_value,
                'name': x.name
            })
        elif isinstance(ret[0], np.ndarray):
            object_types.append(ObjectType.tensor)
            kws.append({
                'dtype': out_dtype,
                'shape': x.shape,
                'order': TensorOrder.C_ORDER
            })
        else:
            assert isinstance(ret[0], pd.Categorical)
            object_types.append(ObjectType.categorical)
            kws.append({
                'dtype': out_dtype,
                'shape': x.shape,
                'categories_value': parse_index(out_dtype.categories,
                                                store_data=True)
            })

        if self._retbins:
            if isinstance(self._bins, (pd.IntervalIndex, INDEX_TYPE)):
                object_types.append(ObjectType.index)
                kws.append({
                    'dtype': self._bins.dtype,
                    'shape': self._bins.shape,
                    'index_value': self._bins.index_value
                        if isinstance(self._bins, INDEX_TYPE) else
                        parse_index(self._bins, store_data=False),
                    'name': self._bins.name
                })
            else:
                object_types.append(ObjectType.tensor)
                kws.append({
                    'dtype': ret[1].dtype,
                    'shape': ret[1].shape if ret[1].size > 0 else (np.nan,),
                    'order': TensorOrder.C_ORDER
                })

        self._object_type = object_types
        return ExecutableTuple(self.new_tileables(inputs, kws=kws))

    @classmethod
    def tile(cls, op):
        if isinstance(op.bins, (Base, Entity)):
            # check op.bins chunk shapes
            check_chunks_unknown_shape([op.bins], TilesError)
            bins = op.bins.rechunk(op.bins.shape)._inplace_tile()
        else:
            bins = op.bins

        if isinstance(op.labels, (Base, Entity)):
            # check op.labels chunk shapes
            check_chunks_unknown_shape([op.labels], TilesError)
            labels = op.labels.rechunk(op.labels.shape)._inplace_tile()
        else:
            labels = op.labels

        if op.input_min is not None:
            # check if input_min and input_max executed
            input_min_chunk = op.input_min.chunks[0]
            input_max_chunk = op.input_max.chunks[0]
            ctx = get_context()
            keys = [input_min_chunk.key, input_max_chunk.key]
            metas = ctx.get_chunk_metas(keys)
            if any(meta is None for meta in metas):
                # not executed before
                raise TilesError('min and max of x have not been executed yet')

            # get min and max of x
            min_val, max_val = ctx.get_chunk_results(keys)
            # calculate bins
            if np.isinf(min_val) or np.isinf(max_val):
                raise ValueError('cannot specify integer `bins` '
                                 'when input data contains infinity')
            elif min_val == max_val:  # adjust end points before binning
                min_val -= 0.001 * abs(min_val) if min_val != 0 else 0.001
                max_val += 0.001 * abs(max_val) if max_val != 0 else 0.001
                bins = np.linspace(min_val, max_val, bins + 1, endpoint=True)
            else:  # adjust end points before binning
                bins = np.linspace(min_val, max_val, bins + 1, endpoint=True)
                adj = (max_val - min_val) * 0.001  # 0.1% of the range
                if op.right:
                    bins[0] -= adj
                else:
                    bins[-1] += adj

        outs = op.outputs

        out_chunks = []
        for c in op.input.chunks:
            chunk_op = op.copy().reset_key()
            chunk_inputs = [c]
            # make sure input_min and input_max reset
            chunk_op._input_min = chunk_op._input_max = None
            chunk_op._bins = bins
            # do not return bins always for chunk
            chunk_op._retbins = False
            if isinstance(bins, (Base, Entity)):
                chunk_inputs.append(bins.chunks[0])
            chunk_op._labels = labels
            if isinstance(labels, (Base, Entity)):
                chunk_inputs.append(labels.chunks[0])

            chunk_kws = []
            if isinstance(outs[0], SERIES_TYPE):
                chunk_kws.append({
                    'dtype': outs[0].dtype,
                    'shape': c.shape,
                    'index_value': c.index_value,
                    'name': c.name,
                    'index': c.index,
                })
            elif isinstance(outs[0], TENSOR_TYPE):
                chunk_kws.append({
                    'dtype': outs[0].dtype,
                    'shape': c.shape,
                    'order': TensorOrder.C_ORDER,
                    'index': c.index,
                })
            else:
                chunk_kws.append({
                    'dtype': outs[0].dtype,
                    'shape': c.shape,
                    'categories_value': outs[0].categories_value,
                    'index': c.index,
                })

            out_chunks.append(chunk_op.new_chunk(chunk_inputs, kws=chunk_kws))

        kws = []
        out_kw = outs[0].params
        out_kw['chunks'] = out_chunks
        out_kw['nsplits'] = op.input.nsplits
        kws.append(out_kw)
        if len(outs) == 2:
            bins_kw = outs[1].params
            bins_kw['chunks'] = bins_chunks = []
            if isinstance(bins, (Base, Entity)):
                bins_chunks.append(bins.chunks[0])
            else:
                if op.duplicates == 'drop':
                    if isinstance(bins, (np.ndarray, list, tuple)):
                        bins = np.unique(bins)
                    else:
                        bins = bins.unique()
                    bins = bins.astype(outs[1].dtype, copy=False)
                convert = \
                    astensor if not isinstance(bins, pd.IntervalIndex) else asindex
                bins_chunks.append(
                    convert(bins, chunk_size=len(bins))._inplace_tile().chunks[0])
            bins_kw['nsplits'] = ((len(bins),),)
            kws.append(bins_kw)
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=kws)

    @classmethod
    def execute(cls, ctx, op):
        x = ctx[op.input.key]
        bins = ctx[op.bins.key] if isinstance(op.bins, (Base, Entity)) else op.bins
        labels = ctx[op.labels.key] if isinstance(op.labels, (Base, Entity)) else op.labels

        cut = partial(pd.cut, right=op.right, retbins=op.retbins, precision=op.precision,
                      include_lowest=op.include_lowest, duplicates=op.duplicates)
        try:
            ret = cut(x, bins, labels=labels)
        except ValueError:
            # fail due to buffer source array is read-only
            ret = cut(x.copy(), bins, labels=labels)
        if op.retbins:  # pragma: no cover
            ctx[op.outputs[0].key] = ret[0]
            ctx[op.outputs[1].key] = ret[1]
        else:
            ctx[op.outputs[0].key] = ret


def cut(x, bins, right: bool = True, labels=None, retbins: bool = False,
        precision: int = 3, include_lowest: bool = False, duplicates: str = 'raise'):
    """
    Bin values into discrete intervals.

    Use `cut` when you need to segment and sort data values into bins. This
    function is also useful for going from a continuous variable to a
    categorical variable. For example, `cut` could convert ages to groups of
    age ranges. Supports binning into an equal number of bins, or a
    pre-specified array of bins.

    Parameters
    ----------
    x : array-like
        The input array to be binned. Must be 1-dimensional.
    bins : int, sequence of scalars, or IntervalIndex
        The criteria to bin by.

        * int : Defines the number of equal-width bins in the range of `x`. The
          range of `x` is extended by .1% on each side to include the minimum
          and maximum values of `x`.
        * sequence of scalars : Defines the bin edges allowing for non-uniform
          width. No extension of the range of `x` is done.
        * IntervalIndex : Defines the exact bins to be used. Note that
          IntervalIndex for `bins` must be non-overlapping.

    right : bool, default True
        Indicates whether `bins` includes the rightmost edge or not. If
        ``right == True`` (the default), then the `bins` ``[1, 2, 3, 4]``
        indicate (1,2], (2,3], (3,4]. This argument is ignored when
        `bins` is an IntervalIndex.
    labels : array or False, default None
        Specifies the labels for the returned bins. Must be the same length as
        the resulting bins. If False, returns only integer indicators of the
        bins. This affects the type of the output container (see below).
        This argument is ignored when `bins` is an IntervalIndex. If True,
        raises an error.
    retbins : bool, default False
        Whether to return the bins or not. Useful when bins is provided
        as a scalar.
    precision : int, default 3
        The precision at which to store and display the bins labels.
    include_lowest : bool, default False
        Whether the first interval should be left-inclusive or not.
    duplicates : {default 'raise', 'drop'}, optional
        If bin edges are not unique, raise ValueError or drop non-uniques.

    Returns
    -------
    out : Categorical, Series, or Tensor
        An array-like object representing the respective bin for each value
        of `x`. The type depends on the value of `labels`.

        * True (default) : returns a Series for Series `x` or a
          Categorical for all other inputs. The values stored within
          are Interval dtype.

        * sequence of scalars : returns a Series for Series `x` or a
          Categorical for all other inputs. The values stored within
          are whatever the type in the sequence is.

        * False : returns a tensor of integers.

    bins : Tensor or IntervalIndex.
        The computed or specified bins. Only returned when `retbins=True`.
        For scalar or sequence `bins`, this is a tensor with the computed
        bins. If set `duplicates=drop`, `bins` will drop non-unique bin. For
        an IntervalIndex `bins`, this is equal to `bins`.

    See Also
    --------
    qcut : Discretize variable into equal-sized buckets based on rank
        or based on sample quantiles.
    Categorical : Array type for storing data that come from a
        fixed set of values.
    Series : One-dimensional array with axis labels (including time series).
    IntervalIndex : Immutable Index implementing an ordered, sliceable set.

    Notes
    -----
    Any NA values will be NA in the result. Out of bounds values will be NA in
    the resulting Series or Categorical object.

    Examples
    --------
    Discretize into three equal-sized bins.

    >>> import mars.tensor as mt
    >>> import mars.dataframe as md

    >>> md.cut(mt.array([1, 7, 5, 4, 6, 3]), 3).execute()
    ... # doctest: +ELLIPSIS
    [(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], ...
    Categories (3, interval[float64]): [(0.994, 3.0] < (3.0, 5.0] ...

    >>> md.cut(mt.array([1, 7, 5, 4, 6, 3]), 3, retbins=True).execute()
    ... # doctest: +ELLIPSIS
    ([(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], ...
    Categories (3, interval[float64]): [(0.994, 3.0] < (3.0, 5.0] ...
    array([0.994, 3.   , 5.   , 7.   ]))

    Discovers the same bins, but assign them specific labels. Notice that
    the returned Categorical's categories are `labels` and is ordered.

    >>> md.cut(mt.array([1, 7, 5, 4, 6, 3]),
    ...        3, labels=["bad", "medium", "good"]).execute()
    [bad, good, medium, medium, good, bad]
    Categories (3, object): [bad < medium < good]

    ``labels=False`` implies you just want the bins back.

    >>> md.cut([0, 1, 1, 2], bins=4, labels=False).execute()
    array([0, 1, 1, 3])

    Passing a Series as an input returns a Series with categorical dtype:

    >>> s = md.Series(mt.array([2, 4, 6, 8, 10]),
    ...               index=['a', 'b', 'c', 'd', 'e'])
    >>> md.cut(s, 3).execute()
    ... # doctest: +ELLIPSIS
    a    (1.992, 4.667]
    b    (1.992, 4.667]
    c    (4.667, 7.333]
    d     (7.333, 10.0]
    e     (7.333, 10.0]
    dtype: category
    Categories (3, interval[float64]): [(1.992, 4.667] < (4.667, ...

    Passing a Series as an input returns a Series with mapping value.
    It is used to map numerically to intervals based on bins.

    >>> s = md.Series(mt.array([2, 4, 6, 8, 10]),
    ...               index=['a', 'b', 'c', 'd', 'e'])
    >>> md.cut(s, [0, 2, 4, 6, 8, 10], labels=False, retbins=True, right=False).execute()
    ... # doctest: +ELLIPSIS
    (a    0.0
     b    1.0
     c    2.0
     d    3.0
     e    NaN
     dtype: float64, array([0, 2, 4, 6, 8, 10]))

    Use `drop` optional when bins is not unique

    >>> md.cut(s, [0, 2, 4, 6, 10, 10], labels=False, retbins=True,
    ...        right=False, duplicates='drop').execute()
    ... # doctest: +ELLIPSIS
    (a    0.0
     b    1.0
     c    2.0
     d    3.0
     e    NaN
     dtype: float64, array([0, 2, 4, 6, 10]))

    Passing an IntervalIndex for `bins` results in those categories exactly.
    Notice that values not covered by the IntervalIndex are set to NaN. 0
    is to the left of the first bin (which is closed on the right), and 1.5
    falls between two bins.

    >>> bins = md.Index(pd.IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)]))
    >>> md.cut([0, 0.5, 1.5, 2.5, 4.5], bins).execute()
    [NaN, (0, 1], NaN, (2, 3], (4, 5]]
    Categories (3, interval[int64]): [(0, 1] < (2, 3] < (4, 5]]
    """

    if isinstance(bins, Integral) and bins < 1:
        raise ValueError('`bins` should be a positive integer')

    op = DataFrameCut(bins=bins, right=right, labels=labels,
                      retbins=retbins, precision=precision,
                      include_lowest=include_lowest, duplicates=duplicates)
    ret = op(x)
    if not retbins:
        return ret[0]
    else:
        return ret
