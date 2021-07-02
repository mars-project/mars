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

from ... import opcodes
from ...core import get_output_types, OutputType
from ...serialization.serializables import AnyField, StringField
from ..core import SERIES_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_df, build_series, validate_axis, parse_index


class DataFrameRename(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.RENAME

    _columns_mapper = AnyField('columns_mapper')
    _index_mapper = AnyField('index_mapper')
    _new_name = AnyField('new_name')
    _level = AnyField('level')
    _errors = StringField('errors')

    def __init__(self, columns_mapper=None, index_mapper=None, new_name=None, level=None,
                 errors=None, output_types=None, **kw):
        super().__init__(_columns_mapper=columns_mapper, _index_mapper=index_mapper,
                         _new_name=new_name, _level=level, _errors=errors,
                         _output_types=output_types, **kw)

    @property
    def columns_mapper(self):
        return self._columns_mapper

    @property
    def index_mapper(self):
        return self._index_mapper

    @property
    def new_name(self):
        return self._new_name

    @property
    def level(self):
        return self._level

    @property
    def errors(self) -> str:
        return self._errors

    def _calc_renamed_df(self, df, errors='ignore'):
        empty_df = build_df(df)
        return empty_df.rename(columns=self._columns_mapper, index=self._index_mapper,
                               level=self._level, errors=errors)

    def _calc_renamed_series(self, df, errors='ignore'):
        empty_series = build_series(df, name=df.name)
        new_series = empty_series.rename(index=self._index_mapper, level=self._level, errors=errors)
        if self._new_name:
            new_series.name = self._new_name
        return new_series

    def __call__(self, df):
        params = df.params
        raw_index = df.index_value.to_pandas()
        if df.ndim == 2:
            new_df = self._calc_renamed_df(df, errors=self.errors)
            new_index = new_df.index
        elif isinstance(df, SERIES_TYPE):
            new_df = self._calc_renamed_series(df, errors=self.errors)
            new_index = new_df.index
        else:
            new_df = new_index = raw_index.set_names(self._index_mapper or self._new_name,
                                                     level=self._level)

        if self._columns_mapper is not None:
            params['columns_value'] = parse_index(new_df.columns, store_data=True)
            params['dtypes'] = new_df.dtypes
        if self._index_mapper is not None:
            params['index_value'] = parse_index(new_index)
        if df.ndim == 1:
            params['name'] = new_df.name
        return self.new_tileable([df], **params)

    @classmethod
    def tile(cls, op: 'DataFrameRename'):
        inp = op.inputs[0]
        out = op.outputs[0]
        chunks = []

        dtypes_cache = dict()
        for c in inp.chunks:
            params = c.params
            new_op = op.copy().reset_key()

            if op.columns_mapper is not None:
                try:
                    new_dtypes = dtypes_cache[c.index[1]]
                except KeyError:
                    new_dtypes = dtypes_cache[c.index[1]] = op._calc_renamed_df(c).dtypes

                params['columns_value'] = parse_index(new_dtypes.index, store_data=True)
                params['dtypes'] = new_dtypes
            if op.index_mapper is not None:
                params['index_value'] = out.index_value
            if out.ndim == 1:
                params['name'] = out.name

            if isinstance(op.columns_mapper, dict):
                idx = params['dtypes'].index
                if op._level is not None:
                    idx = idx.get_level_values(op._level)
                new_op._columns_mapper = {k: v for k, v in op.columns_mapper.items()
                                          if v in idx}
            chunks.append(new_op.new_chunk([c], **params))

        new_op = op.copy().reset_key()
        return new_op.new_tileables([inp], chunks=chunks, nsplits=inp.nsplits, **out.params)

    @classmethod
    def execute(cls, ctx, op: 'DataFrameRename'):
        input_ = ctx[op.inputs[0].key]
        if input_.ndim == 2:
            ctx[op.outputs[0].key] = input_.rename(index=op.index_mapper, columns=op.columns_mapper,
                                                   level=op.level)
        elif op.output_types[0] == OutputType.series:
            ctx[op.outputs[0].key] = input_.rename(index=op.index_mapper or op.new_name, level=op.level)
        else:
            ctx[op.outputs[0].key] = input_.set_names(op.index_mapper or op.new_name,
                                                      level=op.level)


def _rename(df_obj, index_mapper=None, columns_mapper=None, copy=True, inplace=False,
            level=None, errors='ignore'):
    if not copy:
        raise NotImplementedError('`copy=False` not implemented')

    if index_mapper is not None and errors == 'raise' and not inplace:
        warnings.warn('Errors will not raise for non-existing indices')

    op = DataFrameRename(columns_mapper=columns_mapper, index_mapper=index_mapper,
                         level=level, errors=errors, output_types=get_output_types(df_obj))
    ret = op(df_obj)
    if inplace:
        df_obj.data = ret.data
    else:
        return ret


def df_rename(df, mapper=None, index=None, columns=None, axis='index', copy=True,
              inplace=False, level=None, errors='ignore'):
    """
    Alter axes labels.

    Function / dict values must be unique (1-to-1). Labels not contained in
    a dict / Series will be left as-is. Extra labels listed don't throw an
    error.

    Parameters
    ----------
    mapper : dict-like or function
        Dict-like or functions transformations to apply to
        that axis' values. Use either ``mapper`` and ``axis`` to
        specify the axis to target with ``mapper``, or ``index`` and
        ``columns``.
    index : dict-like or function
        Alternative to specifying axis (``mapper, axis=0``
        is equivalent to ``index=mapper``).
    columns : dict-like or function
        Alternative to specifying axis (``mapper, axis=1``
        is equivalent to ``columns=mapper``).
    axis : int or str
        Axis to target with ``mapper``. Can be either the axis name
        ('index', 'columns') or number (0, 1). The default is 'index'.
    copy : bool, default True
        Also copy underlying data.
    inplace : bool, default False
        Whether to return a new DataFrame. If True then value of copy is
        ignored.
    level : int or level name, default None
        In case of a MultiIndex, only rename labels in the specified
        level.
    errors : {'ignore', 'raise'}, default 'ignore'
        If 'raise', raise a `KeyError` when a dict-like `mapper`, `index`,
        or `columns` contains labels that are not present in the Index
        being transformed.
        If 'ignore', existing keys will be renamed and extra keys will be
        ignored.

    Returns
    -------
    DataFrame
        DataFrame with the renamed axis labels.

    Raises
    ------
    KeyError
        If any of the labels is not found in the selected axis and
        "errors='raise'".

    See Also
    --------
    DataFrame.rename_axis : Set the name of the axis.

    Examples
    --------

    ``DataFrame.rename`` supports two calling conventions

    * ``(index=index_mapper, columns=columns_mapper, ...)``
    * ``(mapper, axis={'index', 'columns'}, ...)``

    We *highly* recommend using keyword arguments to clarify your
    intent.

    Rename columns using a mapping:

    >>> import mars.dataframe as md
    >>> df = md.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    >>> df.rename(columns={"A": "a", "B": "c"}).execute()
       a  c
    0  1  4
    1  2  5
    2  3  6

    Rename index using a mapping:

    >>> df.rename(index={0: "x", 1: "y", 2: "z"}).execute()
       A  B
    x  1  4
    y  2  5
    z  3  6

    Cast index labels to a different type:

    >>> df.index.execute()
    RangeIndex(start=0, stop=3, step=1)
    >>> df.rename(index=str).index.execute()
    Index(['0', '1', '2'], dtype='object')

    >>> df.rename(columns={"A": "a", "B": "b", "C": "c"}, errors="raise").execute()
    Traceback (most recent call last):
    KeyError: ['C'] not found in axis

    Using axis-style parameters

    >>> df.rename(str.lower, axis='columns').execute()
       a  b
    0  1  4
    1  2  5
    2  3  6

    >>> df.rename({1: 2, 2: 4}, axis='index').execute()
       A  B
    0  1  4
    2  2  5
    4  3  6

    """
    axis = validate_axis(axis, df)
    if axis == 0:
        index_mapper = index if index is not None else mapper
        columns_mapper = columns
    else:
        columns_mapper = columns if columns is not None else mapper
        index_mapper = index

    if index_mapper is not None and errors == 'raise' and not inplace:
        warnings.warn('Errors will not raise for non-existing indices')

    return _rename(df, index_mapper=index_mapper, columns_mapper=columns_mapper, copy=copy,
                   inplace=inplace, level=level, errors=errors)


def series_rename(series, index=None, *, axis='index', copy=True, inplace=False, level=None,
                  errors='ignore'):
    """
    Alter Series index labels or name.

    Function / dict values must be unique (1-to-1). Labels not contained in
    a dict / Series will be left as-is. Extra labels listed don't throw an
    error.

    Alternatively, change ``Series.name`` with a scalar value.

    Parameters
    ----------
    axis : {0 or "index"}
        Unused. Accepted for compatibility with DataFrame method only.
    index : scalar, hashable sequence, dict-like or function, optional
        Functions or dict-like are transformations to apply to
        the index.
        Scalar or hashable sequence-like will alter the ``Series.name``
        attribute.

    **kwargs
        Additional keyword arguments passed to the function. Only the
        "inplace" keyword is used.

    Returns
    -------
    Series
        Series with index labels or name altered.

    See Also
    --------
    DataFrame.rename : Corresponding DataFrame method.
    Series.rename_axis : Set the name of the axis.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> s = md.Series([1, 2, 3])
    >>> s.execute()
    0    1
    1    2
    2    3
    dtype: int64
    >>> s.rename("my_name").execute()  # scalar, changes Series.name.execute()
    0    1
    1    2
    2    3
    Name: my_name, dtype: int64
    >>> s.rename(lambda x: x ** 2).execute()  # function, changes labels.execute()
    0    1
    1    2
    4    3
    dtype: int64
    >>> s.rename({1: 3, 2: 5}).execute()  # mapping, changes labels.execute()
    0    1
    3    2
    5    3
    dtype: int64
    """
    validate_axis(axis)
    return _rename(series, index_mapper=index, copy=copy, inplace=inplace, level=level,
                   errors=errors)


def index_rename(index, name, inplace=False):
    """
    Alter Index or MultiIndex name.

    Able to set new names without level. Defaults to returning new index.
    Length of names must match number of levels in MultiIndex.

    Parameters
    ----------
    name : label or list of labels
        Name(s) to set.
    inplace : bool, default False
        Modifies the object directly, instead of creating a new Index or
        MultiIndex.

    Returns
    -------
    Index
        The same type as the caller or None if inplace is True.

    See Also
    --------
    Index.set_names : Able to set new names partially and by level.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> idx = md.Index(['A', 'C', 'A', 'B'], name='score')
    >>> idx.rename('grade').execute()
    Index(['A', 'C', 'A', 'B'], dtype='object', name='grade')

    >>> idx = md.Index([('python', 2018),
    ...                 ('python', 2019),
    ...                 ('cobra', 2018),
    ...                 ('cobra', 2019)],
    ...                names=['kind', 'year'])
    >>> idx.execute()
    MultiIndex([('python', 2018),
                ('python', 2019),
                ( 'cobra', 2018),
                ( 'cobra', 2019)],
               names=['kind', 'year'])
    >>> idx.rename(['species', 'year']).execute()
    MultiIndex([('python', 2018),
                ('python', 2019),
                ( 'cobra', 2018),
                ( 'cobra', 2019)],
               names=['species', 'year'])
    >>> idx.rename('species').execute()
    Traceback (most recent call last):
    TypeError: Must pass list-like as `names`.
    """
    op = DataFrameRename(index_mapper=name, output_types=get_output_types(index))
    ret = op(index)
    if inplace:
        index.data = ret.data
    else:
        return ret


def index_set_names(index, names, level=None, inplace=False):
    """
    Set Index or MultiIndex name.

    Able to set new names partially and by level.

    Parameters
    ----------
    names : label or list of label
        Name(s) to set.
    level : int, label or list of int or label, optional
        If the index is a MultiIndex, level(s) to set (None for all
        levels). Otherwise level must be None.
    inplace : bool, default False
        Modifies the object directly, instead of creating a new Index or
        MultiIndex.

    Returns
    -------
    Index
        The same type as the caller or None if inplace is True.

    See Also
    --------
    Index.rename : Able to set new names without level.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> idx = md.Index([1, 2, 3, 4])
    >>> idx.execute()
    Int64Index([1, 2, 3, 4], dtype='int64')
    >>> idx.set_names('quarter').execute()
    Int64Index([1, 2, 3, 4], dtype='int64', name='quarter')

    >>> idx = md.MultiIndex.from_product([['python', 'cobra'],
    ...                                   [2018, 2019]])
    >>> idx.execute()
    MultiIndex([('python', 2018),
                ('python', 2019),
                ( 'cobra', 2018),
                ( 'cobra', 2019)],
               )
    >>> idx.set_names(['kind', 'year'], inplace=True)
    >>> idx.execute()
    MultiIndex([('python', 2018),
                ('python', 2019),
                ( 'cobra', 2018),
                ( 'cobra', 2019)],
               names=['kind', 'year'])
    >>> idx.set_names('species', level=0).execute()
    MultiIndex([('python', 2018),
                ('python', 2019),
                ( 'cobra', 2018),
                ( 'cobra', 2019)],
               names=['species', 'year'])
    """
    op = DataFrameRename(index_mapper=names, level=level,
                         output_types=get_output_types(index))
    ret = op(index)

    if inplace:
        df_or_series = getattr(index, '_get_df_or_series', lambda: None)()
        if df_or_series is not None:
            from .rename_axis import rename_axis_with_level
            rename_axis_with_level(df_or_series, names, axis=index._axis,
                                   level=level, inplace=True)
            index.data = df_or_series.axes[index._axis].data
        else:
            index.data = ret.data
    else:
        return ret
