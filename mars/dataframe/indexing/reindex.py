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
try:
    import scipy.sparse as sps
except ImportError:  # pragma: no cover
    sps = None

from ... import opcodes
from ...core import Base, Entity
from ...operands import OperandStage
from ...serialize import KeyField, AnyField, StringField, Int64Field, BoolField
from ...tensor import tensor as astensor
from ...utils import lazy_import, recursive_tile
from ..core import Index as DataFrameIndexType
from ..initializer import Index as asindex
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import validate_axis_style_args, parse_index
from .index_lib import DataFrameReindexHandler


cudf = lazy_import('cudf', globals=globals())


class DataFrameReindex(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.REINDEX

    _input = KeyField('input')
    _index = AnyField('index')
    _index_freq = AnyField('index_freq')
    _columns = AnyField('columns')
    _method = StringField('method')
    _level = AnyField('level')
    _fill_value = AnyField('fill_value')
    _limit = Int64Field('limit')
    _enable_sparse = BoolField('enable_sparse')

    def __init__(self, index=None, index_freq=None, columns=None, method=None, level=None,
                 fill_value=None, limit=None, enable_sparse=None, stage=None, **kw):
        super().__init__(_index=index, _index_freq=index_freq, _columns=columns,
                         _method=method, _level=level, _fill_value=fill_value,
                         _limit=limit, _enable_sparse=enable_sparse, _stage=stage, **kw)

    @property
    def input(self):
        return self._input

    @property
    def index(self):
        return self._index

    @property
    def index_freq(self):
        return self._index_freq

    @property
    def columns(self):
        return self._columns

    @property
    def method(self):
        return self._method

    @property
    def level(self):
        return self._level

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def limit(self):
        return self._limit

    @property
    def enable_sparse(self):
        return self._enable_sparse

    @property
    def _indexes(self):
        # used for index_lib
        indexes = []
        names = ('index', 'columns')
        for ax in range(self.input.ndim):
            index = names[ax]
            val = getattr(self, index)
            if val is not None:
                indexes.append(val)
            else:
                indexes.append(slice(None))
        return indexes

    @_indexes.setter
    def _indexes(self, new_indexes):
        for index_field, new_index in zip(['_index', '_columns'],
                                          new_indexes):
            setattr(self, index_field, new_index)

    @property
    def indexes(self):
        return self._indexes

    @property
    def can_index_miss(self):
        return True

    def _new_chunks(self, inputs, kws=None, **kw):
        if self._stage == OperandStage.map and len(inputs) < len(self._inputs):
            assert len(inputs) == len(self._inputs) - 1
            inputs.append(self._fill_value.chunks[0])

        if self._stage == OperandStage.agg and self._fill_value is not None:
            # fill_value is not required
            self._fill_value = None

        return super()._new_chunks(inputs, kws=kws, **kw)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        self._input = next(inputs_iter)
        if self._index is not None and \
                isinstance(self._index, (Base, Entity)):
            self._index = next(inputs_iter)
        if self._fill_value is not None and \
                isinstance(self._fill_value, (Base, Entity)):
            self._fill_value = next(inputs_iter)

    def __call__(self, df_or_series):
        inputs = [df_or_series]
        shape = list(df_or_series.shape)
        index_value = df_or_series.index_value
        columns_value = dtypes = None
        if df_or_series.ndim == 2:
            columns_value = df_or_series.columns_value
            dtypes = df_or_series.dtypes

        if self._index is not None:
            shape[0] = len(self._index)
            index_value = asindex(self._index).index_value
            if isinstance(self._index, (Base, Entity)):
                inputs.append(self._index)
        if self._columns is not None:
            shape[1] = len(self._columns)
            dtypes = df_or_series.dtypes.reindex(index=self._columns).fillna(
                np.dtype(np.float64))
            columns_value = parse_index(dtypes.index, store_data=True)
        if self._fill_value is not None and \
                isinstance(self._fill_value, (Base, Entity)):
            inputs.append(self._fill_value)

        if df_or_series.ndim == 1:
            return self.new_series(inputs, shape=shape, dtype=df_or_series.dtype,
                                   index_value=index_value, name=df_or_series.name)
        else:
            return self.new_dataframe(inputs, shape=shape, dtypes=dtypes,
                                      index_value=index_value,
                                      columns_value=columns_value)

    @classmethod
    def tile(cls, op):
        if all(len(inp.chunks) == 1 for inp in op.inputs):
            # tile one chunk
            out = op.outputs[0]

            chunk_op = op.copy().reset_key()
            chunk_params = out.params.copy()
            chunk_params['index'] = (0,) * out.ndim
            out_chunk = chunk_op.new_chunk([inp.chunks[0] for inp in op.inputs],
                                           kws=[chunk_params])

            params = out.params.copy()
            params['nsplits'] = ((s,) for s in out.shape)
            params['chunks'] = [out_chunk]
            new_op = op.copy()
            return new_op.new_tileables(op.inputs, kws=[params])

        handler = DataFrameReindexHandler()
        result = handler.handle(op)
        if op.method is None and op.fill_value is None:
            return [result]
        else:
            axis = 1 if op.columns is not None and op.index is None else 0
            result = result.fillna(value=op.fill_value, method=op.method,
                                   axis=axis, limit=op.limit)
            return [recursive_tile(result)]

    @classmethod
    def _get_value(cls, ctx, obj):
        if obj is not None and hasattr(obj, 'key'):
            return ctx[obj.key]
        return obj

    @classmethod
    def _convert_to_writable(cls, obj):
        if isinstance(obj, np.ndarray) and not obj.flags.writeable:
            return obj.copy()
        return obj

    @classmethod
    def _sparse_reindex(cls, inp, index=None, columns=None):
        if inp.ndim == 2:
            columns = inp.columns if columns is None else columns
            index_shape = len(index) if index is not None else len(inp)
            i_to_columns = dict()

            for i, col in enumerate(columns):
                if col in inp.dtypes:
                    if index is None:
                        i_to_columns[i] = inp[col]
                    else:
                        indexer = inp.index.reindex(index)[1]
                        spmatrix = sps.lil_matrix((index_shape, 1), dtype=inp[col].dtype)
                        cond = indexer >= 0
                        available_indexer = indexer[cond]
                        del indexer
                        spmatrix[cond, 0] = inp[col].iloc[available_indexer].to_numpy()
                        i_to_columns[i] = pd.DataFrame.sparse.from_spmatrix(
                            spmatrix, index=index).iloc[:, 0]
                else:
                    ind = index if index is not None else inp.index
                    i_to_columns[i] = pd.DataFrame.sparse.from_spmatrix(
                        sps.coo_matrix((index_shape, 1), dtype=np.float64),
                        index=ind).iloc[:, 0]

            df = pd.DataFrame(i_to_columns)
            df.columns = columns
            return df
        else:
            indexer = inp.index.reindex(index)[1]
            spmatrix = sps.lil_matrix((len(index), 1), dtype=inp.dtype)
            cond = indexer >= 0
            available_indexer = indexer[cond]
            del indexer
            spmatrix[cond, 0] = inp.iloc[available_indexer].to_numpy()
            series = pd.DataFrame.sparse.from_spmatrix(spmatrix, index=index).iloc[:, 0]
            series.name = inp.name
            return series

    @classmethod
    def _reindex(cls, ctx, op, fill=True, try_sparse=None):
        inp = cls._convert_to_writable(ctx[op.input.key])
        index = cls._get_value(ctx, op.index)
        if op.index_freq is not None:
            index = pd.Index(index, freq=op.index_freq)
        columns = cls._get_value(ctx, op.columns)
        kw = {'level': op.level}
        if index is not None and not isinstance(index, slice):
            kw['index'] = cls._convert_to_writable(index)
        if columns is not None and not isinstance(columns, slice):
            kw['columns'] = cls._convert_to_writable(columns)
        if fill:
            kw['method'] = op.method
            kw['fill_value'] = cls._get_value(ctx, op.fill_value)
            kw['limit'] = op.limit

        if try_sparse and not fill and op.level is None and \
                isinstance(inp, (pd.DataFrame, pd.Series)) and \
                sps is not None:
            # 1. sparse is used in map only
            # 2. for MultiIndex, sparse is not needed as well
            # 3. only consider cpu
            # 4. scipy is installed

            if op.enable_sparse is None:
                # try to use sparse if estimated size > 2 * input_size
                cur_size = inp.memory_usage(deep=True)
                if inp.ndim == 2:
                    cur_size = cur_size.sum()
                element_size = cur_size / inp.size
                shape = list(inp.shape)
                if 'index' in kw:
                    shape[0] = len(kw['index'])
                if 'columns' in kw:
                    shape[1] = len(kw['columns'])
                estimate_size = np.prod(shape) * element_size

                fitted = estimate_size > cur_size * 2
            else:
                # specified when op.enable_sparse == True
                fitted = True

            if fitted:
                # use sparse instead
                return cls._sparse_reindex(inp,
                                           index=kw.get('index'),
                                           columns=kw.get('columns'))

        return inp.reindex(**kw)

    @classmethod
    def _execute_reindex(cls, ctx, op):
        ctx[op.outputs[0].key] = cls._reindex(ctx, op)

    @classmethod
    def _execute_map(cls, ctx, op):
        if op.enable_sparse is not None:
            try_sparse = op.enable_sparse
        else:
            try_sparse = True
        ctx[op.outputs[0].key] = cls._reindex(ctx, op, fill=False,
                                              try_sparse=try_sparse)

    @classmethod
    def _convert_to_dense(cls, series):
        if isinstance(series.dtype, pd.SparseDtype):
            return series.astype(pd.SparseDtype(series.dtype.subtype, np.nan)).sparse.to_dense()
        return series

    @classmethod
    def _merge_chunks(cls, inputs):
        xdf = cls._get_xdf(inputs[0])

        ndim = inputs[0].ndim
        if ndim == 2:
            columns = inputs[0].columns
            result = xdf.DataFrame(np.full(inputs[0].shape, np.nan),
                                   columns=columns,
                                   index=inputs[0].index)
        else:
            columns = [inputs[0].name]
            result = None

        for i in range(len(columns)):
            if ndim == 1:
                curr = cls._convert_to_dense(inputs[0]).copy()
            else:
                curr = cls._convert_to_dense(inputs[0].iloc[:, i]).copy()
            for j in range(len(inputs) - 1):
                if ndim == 2:
                    left = cls._convert_to_dense(inputs[j].iloc[:, i])
                    right = cls._convert_to_dense(inputs[j + 1].iloc[:, i])
                else:
                    left = cls._convert_to_dense(inputs[j])
                    right = cls._convert_to_dense(inputs[j + 1])
                if ((~left.isna()) & (~right.isna())).sum() > 0:
                    raise ValueError('cannot reindex from a duplicate axis')
                curr.loc[~left.isna()] = left
                curr.loc[~right.isna()] = right
            if ndim == 1:
                result = curr
            else:
                result.iloc[:, i] = curr

        return result

    @classmethod
    def _get_xdf(cls, obj):
        return pd if isinstance(obj, (pd.DataFrame, pd.Series)) or \
                     cudf is None else cudf

    @classmethod
    def _execute_agg(cls, ctx, op):
        out = op.outputs[0]

        if op.index is None and op.columns is None:
            # index is tensor
            inputs = [ctx[inp.key] for inp in op.inputs]

            xdf = cls._get_xdf(inputs[0])

            if inputs[0].index.nlevels > 1 and op.level is not None:
                # multi index
                result = xdf.concat(inputs)
            else:
                result = cls._merge_chunks(inputs) if len(inputs) > 1 else inputs[0]

            ctx[out.key] = result

        else:
            # ndarray index or columns
            if isinstance(op.index, slice) and op.index == slice(None):
                axis = 1
                labels = op.columns
            else:
                assert op.columns is None or \
                       (isinstance(op.columns, slice) and
                        op.columns == slice(None))
                axis = 0
                labels = op.index

            inp = ctx[op.inputs[0].key]
            if inp.index.nlevels > 1 and op.level is not None:
                new_inp = inp
            else:
                # split input
                size = out.shape[axis]
                assert inp.shape[axis] % size == 0
                inputs = []
                for i in range(inp.shape[axis] // size):
                    slc = [slice(None)] * inp.ndim
                    slc[axis] = slice(size * i, size * (i + 1))
                    inputs.append(inp.iloc[tuple(slc)])
                new_inp = cls._merge_chunks(inputs)

            labels = cls._convert_to_writable(labels)
            if out.ndim == 2:
                result = new_inp.reindex(labels=labels, axis=axis,
                                         level=op.level)
            else:
                result = new_inp.reindex(index=labels, level=op.level)
            ctx[out.key] = result

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            return cls._execute_map(ctx, op)
        elif op.stage == OperandStage.agg:
            return cls._execute_agg(ctx, op)
        else:
            assert op.stage is None
            return cls._execute_reindex(ctx, op)


def reindex(df_or_series, *args, **kwargs):
    """
    Conform Series/DataFrame to new index with optional filling logic.

    Places NA/NaN in locations having no value in the previous index. A new object
    is produced unless the new index is equivalent to the current one and
    ``copy=False``.

    Parameters
    ----------
    labels : array-like, optional
        New labels / index to conform the axis specified by 'axis' to.
    index, columns : array-like, optional
        New labels / index to conform to, should be specified using
        keywords. Preferably an Index object to avoid duplicating data.
    axis : int or str, optional
        Axis to target. Can be either the axis name ('index', 'columns')
        or number (0, 1).
    method : {None, 'backfill'/'bfill', 'pad'/'ffill', 'nearest'}
        Method to use for filling holes in reindexed DataFrame.
        Please note: this is only applicable to DataFrames/Series with a
        monotonically increasing/decreasing index.

        * None (default): don't fill gaps
        * pad / ffill: Propagate last valid observation forward to next
          valid.
        * backfill / bfill: Use next valid observation to fill gap.
        * nearest: Use nearest valid observations to fill gap.

    copy : bool, default True
        Return a new object, even if the passed indexes are the same.
    level : int or name
        Broadcast across a level, matching Index values on the
        passed MultiIndex level.
    fill_value : scalar, default np.NaN
        Value to use for missing values. Defaults to NaN, but can be any
        "compatible" value.
    limit : int, default None
        Maximum number of consecutive elements to forward or backward fill.
    tolerance : optional
        Maximum distance between original and new labels for inexact
        matches. The values of the index at the matching locations most
        satisfy the equation ``abs(index[indexer] - target) <= tolerance``.

        Tolerance may be a scalar value, which applies the same tolerance
        to all values, or list-like, which applies variable tolerance per
        element. List-like includes list, tuple, array, Series, and must be
        the same size as the index and its dtype must exactly match the
        index's type.

    Returns
    -------
    Series/DataFrame with changed index.

    See Also
    --------
    DataFrame.set_index : Set row labels.
    DataFrame.reset_index : Remove row labels or move them to new columns.
    DataFrame.reindex_like : Change to same indices as other DataFrame.

    Examples
    --------

    ``DataFrame.reindex`` supports two calling conventions

    * ``(index=index_labels, columns=column_labels, ...)``
    * ``(labels, axis={'index', 'columns'}, ...)``

    We *highly* recommend using keyword arguments to clarify your
    intent.

    Create a dataframe with some fictional data.

    >>> import mars.dataframe as md
    >>> index = ['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror']
    >>> df = md.DataFrame({'http_status': [200, 200, 404, 404, 301],
    ...                   'response_time': [0.04, 0.02, 0.07, 0.08, 1.0]},
    ...                   index=index)
    >>> df.execute()
               http_status  response_time
    Firefox            200           0.04
    Chrome             200           0.02
    Safari             404           0.07
    IE10               404           0.08
    Konqueror          301           1.00

    Create a new index and reindex the dataframe. By default
    values in the new index that do not have corresponding
    records in the dataframe are assigned ``NaN``.

    >>> new_index = ['Safari', 'Iceweasel', 'Comodo Dragon', 'IE10',
    ...              'Chrome']
    >>> df.reindex(new_index).execute()
                   http_status  response_time
    Safari               404.0           0.07
    Iceweasel              NaN            NaN
    Comodo Dragon          NaN            NaN
    IE10                 404.0           0.08
    Chrome               200.0           0.02

    We can fill in the missing values by passing a value to
    the keyword ``fill_value``. Because the index is not monotonically
    increasing or decreasing, we cannot use arguments to the keyword
    ``method`` to fill the ``NaN`` values.

    >>> df.reindex(new_index, fill_value=0).execute()
                   http_status  response_time
    Safari                 404           0.07
    Iceweasel                0           0.00
    Comodo Dragon            0           0.00
    IE10                   404           0.08
    Chrome                 200           0.02

    >>> df.reindex(new_index, fill_value='missing').execute()
                  http_status response_time
    Safari                404          0.07
    Iceweasel         missing       missing
    Comodo Dragon     missing       missing
    IE10                  404          0.08
    Chrome                200          0.02

    We can also reindex the columns.

    >>> df.reindex(columns=['http_status', 'user_agent']).execute()
               http_status  user_agent
    Firefox            200         NaN
    Chrome             200         NaN
    Safari             404         NaN
    IE10               404         NaN
    Konqueror          301         NaN

    Or we can use "axis-style" keyword arguments

    >>> df.reindex(['http_status', 'user_agent'], axis="columns").execute()
               http_status  user_agent
    Firefox            200         NaN
    Chrome             200         NaN
    Safari             404         NaN
    IE10               404         NaN
    Konqueror          301         NaN

    To further illustrate the filling functionality in
    ``reindex``, we will create a dataframe with a
    monotonically increasing index (for example, a sequence
    of dates).

    >>> date_index = md.date_range('1/1/2010', periods=6, freq='D')
    >>> df2 = md.DataFrame({"prices": [100, 101, np.nan, 100, 89, 88]},
    ...                    index=date_index)
    >>> df2.execute()
                prices
    2010-01-01   100.0
    2010-01-02   101.0
    2010-01-03     NaN
    2010-01-04   100.0
    2010-01-05    89.0
    2010-01-06    88.0

    Suppose we decide to expand the dataframe to cover a wider
    date range.

    >>> date_index2 = md.date_range('12/29/2009', periods=10, freq='D')
    >>> df2.reindex(date_index2).execute()
                prices
    2009-12-29     NaN
    2009-12-30     NaN
    2009-12-31     NaN
    2010-01-01   100.0
    2010-01-02   101.0
    2010-01-03     NaN
    2010-01-04   100.0
    2010-01-05    89.0
    2010-01-06    88.0
    2010-01-07     NaN

    The index entries that did not have a value in the original data frame
    (for example, '2009-12-29') are by default filled with ``NaN``.
    If desired, we can fill in the missing values using one of several
    options.

    For example, to back-propagate the last valid value to fill the ``NaN``
    values, pass ``bfill`` as an argument to the ``method`` keyword.

    >>> df2.reindex(date_index2, method='bfill').execute()
                prices
    2009-12-29   100.0
    2009-12-30   100.0
    2009-12-31   100.0
    2010-01-01   100.0
    2010-01-02   101.0
    2010-01-03     NaN
    2010-01-04   100.0
    2010-01-05    89.0
    2010-01-06    88.0
    2010-01-07     NaN

    Please note that the ``NaN`` value present in the original dataframe
    (at index value 2010-01-03) will not be filled by any of the
    value propagation schemes. This is because filling while reindexing
    does not look at dataframe values, but only compares the original and
    desired indexes. If you do want to fill in the ``NaN`` values present
    in the original dataframe, use the ``fillna()`` method.

    See the :ref:`user guide <basics.reindexing>` for more.
    """
    axes = validate_axis_style_args(df_or_series, args, kwargs, "labels", "reindex")
    # Pop these, since the values are in `kwargs` under different names
    kwargs.pop('index', None)
    if df_or_series.ndim > 1:
        kwargs.pop('columns', None)
        kwargs.pop("axis", None)
        kwargs.pop("labels", None)
    method = kwargs.pop("method", None)
    level = kwargs.pop("level", None)
    copy = kwargs.pop("copy", True)
    limit = kwargs.pop("limit", None)
    tolerance = kwargs.pop("tolerance", None)
    fill_value = kwargs.pop("fill_value", None)
    enable_sparse = kwargs.pop("enable_sparse", None)

    if kwargs:
        raise TypeError(
            "reindex() got an unexpected keyword "
            f'argument "{list(kwargs.keys())[0]}"'
        )

    if tolerance is not None:  # pragma: no cover
        raise NotImplementedError('`tolerance` is not supported yet')

    if method == 'nearest':  # pragma: no cover
        raise NotImplementedError('method=nearest is not supported yet')

    index = axes.get('index')
    index_freq = None
    if isinstance(index, (Base, Entity)):
        if isinstance(index, DataFrameIndexType):
            index_freq = getattr(index.index_value.value, 'freq', None)
        index = astensor(index)
    elif index is not None:
        index = np.asarray(index)
        index_freq = getattr(index, 'freq', None)

    columns = axes.get('columns')
    if isinstance(columns, (Base, Entity)):  # pragma: no cover
        try:
            columns = columns.fetch()
        except ValueError:
            raise NotImplementedError("`columns` need to be executed first "
                                      "if it's a Mars object")
    elif columns is not None:
        columns = np.asarray(columns)

    if isinstance(fill_value, (Base, Entity)) and getattr(fill_value, 'ndim', 0) != 0:
        raise ValueError('fill_value must be a scalar')

    op = DataFrameReindex(index=index, index_freq=index_freq, columns=columns,
                          method=method, level=level, fill_value=fill_value,
                          limit=limit, enable_sparse=enable_sparse)
    ret = op(df_or_series)

    if copy:
        return ret.copy()
    return ret
