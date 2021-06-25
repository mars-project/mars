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

import warnings
from collections import OrderedDict

import numpy as np

from ... import opcodes
from ...core import Entity, Chunk, CHUNK_TYPE, OutputType, recursive_tile
from ...serialization.serializables import AnyField, StringField
from ..core import IndexValue, DATAFRAME_TYPE, SERIES_TYPE, INDEX_CHUNK_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index, validate_axis


class DataFrameDrop(DataFrameOperandMixin, DataFrameOperand):
    _op_type_ = opcodes.DATAFRAME_DROP

    _index = AnyField('index')
    _columns = AnyField('columns')
    _level = AnyField('level')
    _errors = StringField('errors')

    def __init__(self, index=None, columns=None, level=None, errors=None, **kw):
        super().__init__(_index=index, _columns=columns, _level=level, _errors=errors,
                         **kw)

    @property
    def index(self):
        return self._index

    @property
    def columns(self):
        return self._columns

    @property
    def level(self):
        return self._level

    @property
    def errors(self):
        return self._errors

    def _filter_dtypes(self, dtypes, ignore_errors=False):
        if self._columns:
            return dtypes.drop(index=self._columns, level=self._level,
                               errors='ignore' if ignore_errors else self._errors)
        else:
            return dtypes

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs[1:])
        if len(self._inputs) > 1:
            self._index = next(inputs_iter)

    def __call__(self, df_or_series):
        params = df_or_series.params.copy()
        shape_list = list(df_or_series.shape)

        if self._index is not None:
            if isinstance(df_or_series.index_value.value, IndexValue.RangeIndex):
                params['index_value'] = parse_index(None, (df_or_series.key, df_or_series.index_value.key))
            shape_list[0] = np.nan

        if isinstance(df_or_series, DATAFRAME_TYPE):
            new_dtypes = self._filter_dtypes(df_or_series.dtypes)
            params['columns_value'] = parse_index(new_dtypes.index, store_data=True)
            params['dtypes'] = new_dtypes
            shape_list[1] = len(new_dtypes)
            self.output_types = [OutputType.dataframe]
        elif isinstance(df_or_series, SERIES_TYPE):
            self.output_types = [OutputType.series]
        else:
            self.output_types = [OutputType.index]

        params['shape'] = tuple(shape_list)

        inputs = [df_or_series]
        if isinstance(self._index, Entity):
            inputs.append(self._index)
        return self.new_tileable(inputs, **params)

    @classmethod
    def tile(cls, op: 'DataFrameDrop'):
        inp = op.inputs[0]
        out = op.outputs[0]
        if len(op.inputs) > 1:
            rechunked = yield from recursive_tile(
                op.index.rechunk({0: (op.index.shape[0],)}))
            index_chunk = rechunked.chunks[0]
        else:
            index_chunk = op.index

        col_to_args = OrderedDict()
        chunks = []
        for c in inp.chunks:
            params = c.params.copy()
            if isinstance(inp, DATAFRAME_TYPE):
                new_dtypes, new_col_id = col_to_args.get(c.index[1], (None, None))

                if new_dtypes is None:
                    new_col_id = len(col_to_args)
                    new_dtypes = op._filter_dtypes(c.dtypes, ignore_errors=True)
                    if len(new_dtypes) == 0:
                        continue
                    col_to_args[c.index[1]] = (new_dtypes, new_col_id)

                params.update(dict(dtypes=new_dtypes, index=(c.index[0], new_col_id),
                                   index_value=c.index_value,
                                   columns_value=parse_index(new_dtypes.index, store_data=True)))
                if op.index is not None:
                    params.update(dict(shape=(np.nan, len(new_dtypes)),
                                       index_value=parse_index(None, (c.key, c.index_value.key))))
                else:
                    params['shape'] = (c.shape[0], len(new_dtypes))
            elif op.index is not None:
                params.update(dict(shape=(np.nan,), index_value=parse_index(None, (c.key, c.index_value.key))))

            chunk_inputs = [c]
            if isinstance(index_chunk, Chunk):
                chunk_inputs.append(index_chunk)

            new_op = op.copy().reset_key()
            new_op._index = index_chunk
            chunks.append(new_op.new_chunk(chunk_inputs, **params))

        new_op = op.copy().reset_key()
        params = out.params.copy()
        if op.index is not None:
            nsplits_list = [(np.nan,) * inp.chunk_shape[0]]
        else:
            nsplits_list = [inp.nsplits[0]]
        if isinstance(inp, DATAFRAME_TYPE):
            nsplits_list.append(tuple(len(dt) for dt, _ in col_to_args.values()))
        params.update(dict(chunks=chunks, nsplits=tuple(nsplits_list)))
        return new_op.new_tileables(op.inputs, **params)

    @classmethod
    def execute(cls, ctx, op: 'DataFrameDrop'):
        inp = op.inputs[0]
        if isinstance(op.index, CHUNK_TYPE):
            index_val = ctx[op.index.key]
        else:
            index_val = op.index

        if isinstance(inp, INDEX_CHUNK_TYPE):
            ctx[op.outputs[0].key] = ctx[inp.key].drop(index_val, errors='ignore')
        else:
            ctx[op.outputs[0].key] = ctx[inp.key].drop(
                index=index_val, columns=op.columns, level=op.level, errors='ignore')


def _drop(df_or_series, labels=None, axis=0, index=None, columns=None, level=None,
          inplace=False, errors='raise'):
    axis = validate_axis(axis, df_or_series)
    if labels is not None:
        if axis == 0:
            index = labels
        else:
            columns = labels

    if index is not None and errors == 'raise':
        warnings.warn('Errors will not raise for non-existing indices')
    if isinstance(columns, Entity):
        raise NotImplementedError('Columns cannot be Mars objects')

    op = DataFrameDrop(index=index, columns=columns, level=level, errors=errors)
    df = op(df_or_series)
    if inplace:
        df_or_series.data = df.data
    else:
        return df


def df_drop(df, labels=None, axis=0, index=None, columns=None, level=None,
            inplace=False, errors='raise'):
    """
    Drop specified labels from rows or columns.

    Remove rows or columns by specifying label names and corresponding
    axis, or by specifying directly index or column names. When using a
    multi-index, labels on different levels can be removed by specifying
    the level.

    Parameters
    ----------
    labels : single label or list-like
        Index or column labels to drop.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Whether to drop labels from the index (0 or 'index') or
        columns (1 or 'columns').
    index : single label or list-like
        Alternative to specifying axis (``labels, axis=0``
        is equivalent to ``index=labels``).
    columns : single label or list-like
        Alternative to specifying axis (``labels, axis=1``
        is equivalent to ``columns=labels``).
    level : int or level name, optional
        For MultiIndex, level from which the labels will be removed.
    inplace : bool, default False
        If True, do operation inplace and return None.
    errors : {'ignore', 'raise'}, default 'raise'
        If 'ignore', suppress error and only existing labels are
        dropped. Note that errors for missing indices will not raise.

    Returns
    -------
    DataFrame
        DataFrame without the removed index or column labels.

    Raises
    ------
    KeyError
        If any of the labels is not found in the selected axis.

    See Also
    --------
    DataFrame.loc : Label-location based indexer for selection by label.
    DataFrame.dropna : Return DataFrame with labels on given axis omitted
        where (all or any) data are missing.
    DataFrame.drop_duplicates : Return DataFrame with duplicate rows
        removed, optionally only considering certain columns.
    Series.drop : Return Series with specified index labels removed.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import mars.dataframe as md
    >>> df = md.DataFrame(np.arange(12).reshape(3, 4),
    ...                   columns=['A', 'B', 'C', 'D'])
    >>> df.execute()
       A  B   C   D
    0  0  1   2   3
    1  4  5   6   7
    2  8  9  10  11

    Drop columns

    >>> df.drop(['B', 'C'], axis=1).execute()
       A   D
    0  0   3
    1  4   7
    2  8  11

    >>> df.drop(columns=['B', 'C']).execute()
       A   D
    0  0   3
    1  4   7
    2  8  11

    Drop a row by index

    >>> df.drop([0, 1]).execute()
       A  B   C   D
    2  8  9  10  11

    Drop columns and/or rows of MultiIndex DataFrame

    >>> midx = pd.MultiIndex(levels=[['lama', 'cow', 'falcon'],
    ...                              ['speed', 'weight', 'length']],
    ...                      codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],
    ...                             [0, 1, 2, 0, 1, 2, 0, 1, 2]])
    >>> df = md.DataFrame(index=midx, columns=['big', 'small'],
    ...                   data=[[45, 30], [200, 100], [1.5, 1], [30, 20],
    ...                         [250, 150], [1.5, 0.8], [320, 250],
    ...                         [1, 0.8], [0.3, 0.2]])
    >>> df.execute()
                    big     small
    lama    speed   45.0    30.0
            weight  200.0   100.0
            length  1.5     1.0
    cow     speed   30.0    20.0
            weight  250.0   150.0
            length  1.5     0.8
    falcon  speed   320.0   250.0
            weight  1.0     0.8
            length  0.3     0.2

    >>> df.drop(index='cow', columns='small').execute()
                    big
    lama    speed   45.0
            weight  200.0
            length  1.5
    falcon  speed   320.0
            weight  1.0
            length  0.3

    >>> df.drop(index='length', level=1).execute()
                    big     small
    lama    speed   45.0    30.0
            weight  200.0   100.0
    cow     speed   30.0    20.0
            weight  250.0   150.0
    falcon  speed   320.0   250.0
            weight  1.0     0.8
    """
    return _drop(df, labels=labels, axis=axis, index=index, columns=columns,
                 level=level, inplace=inplace, errors=errors)


def df_pop(df, item):
    """
    Return item and drop from frame. Raise KeyError if not found.

    Parameters
    ----------
    item : str
        Label of column to be popped.

    Returns
    -------
    Series

    Examples
    --------
    >>> import numpy as np
    >>> import mars.dataframe as md
    >>> df = md.DataFrame([('falcon', 'bird', 389.0),
    ...                    ('parrot', 'bird', 24.0),
    ...                    ('lion', 'mammal', 80.5),
    ...                    ('monkey', 'mammal', np.nan)],
    ...                   columns=('name', 'class', 'max_speed'))
    >>> df.execute()
         name   class  max_speed
    0  falcon    bird      389.0
    1  parrot    bird       24.0
    2    lion  mammal       80.5
    3  monkey  mammal        NaN

    >>> df.pop('class').execute()
    0      bird
    1      bird
    2    mammal
    3    mammal
    Name: class, dtype: object

    >>> df.execute()
         name  max_speed
    0  falcon      389.0
    1  parrot       24.0
    2    lion       80.5
    3  monkey        NaN
    """
    series = df.data[item]
    df_drop(df, item, axis=1, inplace=True)
    return series


def series_drop(series, labels=None, axis=0, index=None, columns=None, level=None,
                inplace=False, errors='raise'):
    """
    Return Series with specified index labels removed.

    Remove elements of a Series based on specifying the index labels.
    When using a multi-index, labels on different levels can be removed
    by specifying the level.

    Parameters
    ----------
    labels : single label or list-like
        Index labels to drop.
    axis : 0, default 0
        Redundant for application on Series.
    index : single label or list-like
        Redundant for application on Series, but 'index' can be used instead
        of 'labels'.

        .. versionadded:: 0.21.0
    columns : single label or list-like
        No change is made to the Series; use 'index' or 'labels' instead.

        .. versionadded:: 0.21.0
    level : int or level name, optional
        For MultiIndex, level for which the labels will be removed.
    inplace : bool, default False
        If True, do operation inplace and return None.
    errors : {'ignore', 'raise'}, default 'raise'
        Note that this argument is kept only for compatibility, and errors
        will not raise even if ``errors=='raise'``.

    Returns
    -------
    Series
        Series with specified index labels removed.

    Raises
    ------
    KeyError
        If none of the labels are found in the index.

    See Also
    --------
    Series.reindex : Return only specified index labels of Series.
    Series.dropna : Return series without null values.
    Series.drop_duplicates : Return Series with duplicate values removed.
    DataFrame.drop : Drop specified labels from rows or columns.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import mars.dataframe as md
    >>> s = md.Series(data=np.arange(3), index=['A', 'B', 'C'])
    >>> s.execute()
    A  0
    B  1
    C  2
    dtype: int64

    Drop labels B en C

    >>> s.drop(labels=['B', 'C']).execute()
    A  0
    dtype: int64

    Drop 2nd level label in MultiIndex Series

    >>> midx = pd.MultiIndex(levels=[['lama', 'cow', 'falcon'],
    ...                              ['speed', 'weight', 'length']],
    ...                      codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],
    ...                             [0, 1, 2, 0, 1, 2, 0, 1, 2]])
    >>> s = md.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
    ...               index=midx)
    >>> s.execute()
    lama    speed      45.0
            weight    200.0
            length      1.2
    cow     speed      30.0
            weight    250.0
            length      1.5
    falcon  speed     320.0
            weight      1.0
            length      0.3
    dtype: float64

    >>> s.drop(labels='weight', level=1).execute()
    lama    speed      45.0
            length      1.2
    cow     speed      30.0
            length      1.5
    falcon  speed     320.0
            length      0.3
    dtype: float64
    """
    return _drop(series, labels=labels, axis=axis, index=index, columns=columns,
                 level=level, inplace=inplace, errors=errors)


def index_drop(index, labels, errors='raise'):
    """
    Make new Index with passed list of labels deleted.

    Parameters
    ----------
    labels : array-like
    errors : {'ignore', 'raise'}, default 'raise'
        Note that this argument is kept only for compatibility, and errors
        will not raise even if ``errors=='raise'``.

    Returns
    -------
    dropped : Index

    Raises
    ------
    KeyError
        If not all of the labels are found in the selected axis
    """
    return _drop(index, labels=labels, errors=errors)
