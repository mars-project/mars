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

import warnings

from ... import opcodes
from ...serialize import AnyField, Int64Field, StringField
from ..operands import DataFrameOperand, DataFrameOperandMixin, OutputType
from ..utils import build_empty_df, validate_axis, parse_index


class DataFrameRename(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.RENAME

    _columns_mapper = AnyField('columns_mapper')
    _index_mapper = AnyField('index_mapper')
    _level = Int64Field('level')
    _errors = StringField('errors')

    def __init__(self, columns_mapper=None, index_mapper=None, level=None, errors=None,
                 output_types=None, **kw):
        super().__init__(_columns_mapper=columns_mapper, _index_mapper=index_mapper,
                         _level=level, _errors=errors, _output_types=output_types, **kw)

    @property
    def columns_mapper(self):
        return self._columns_mapper

    @property
    def index_mapper(self):
        return self._index_mapper

    @property
    def level(self):
        return self._level

    @property
    def errors(self) -> str:
        return self._errors

    def _calc_renamed_df(self, dtypes, index, errors='ignore'):
        empty_df = build_empty_df(dtypes, index=index)
        return empty_df.rename(columns=self._columns_mapper, index=self._index_mapper,
                               level=self._level, errors=errors)

    def __call__(self, df):
        params = df.params
        new_df = self._calc_renamed_df(
            df.dtypes, df.index_value.to_pandas(), errors=self.errors)

        if self._columns_mapper is not None:
            params['columns_value'] = parse_index(new_df.columns, store_data=True)
            params['dtypes'] = new_df.dtypes
        if self._index_mapper is not None:
            params['index_value'] = parse_index(new_df.index)
        return self.new_dataframe([df], **params)

    @classmethod
    def tile(cls, op: 'DataFrameRename'):
        inp = op.inputs[0]
        out = op.outputs[0]
        chunks = []

        dtypes_cache = dict()
        for c in inp.chunks:
            params = c.params
            new_op = op.copy().reset_key()

            try:
                new_dtypes = dtypes_cache[c.index[0]]
            except KeyError:
                new_dtypes = dtypes_cache[c.index[0]] = \
                    op._calc_renamed_df(c.dtypes, c.index_value.to_pandas()).dtypes

            if op.columns_mapper is not None:
                params['columns_value'] = parse_index(new_dtypes.index, store_data=True)
                params['dtypes'] = new_dtypes
            if op.index_mapper is not None:
                params['index_value'] = out.index_value

            if isinstance(op.columns_mapper, dict):
                idx = new_dtypes.index
                if op._level is not None:
                    idx = idx.get_level_values(op._level)
                new_op._columns_mapper = {k: v for k, v in op.columns_mapper.items()
                                          if v in idx}
            chunks.append(new_op.new_chunk([c], **params))

        new_op = op.copy().reset_key()
        return new_op.new_dataframes([inp], chunks=chunks, nsplits=inp.nsplits, **out.params)

    @classmethod
    def execute(cls, ctx, op: 'DataFrameRename'):
        input_ = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = input_.rename(index=op.index_mapper, columns=op.columns_mapper,
                                               level=op.level)


def rename(df, mapper=None, index=None, columns=None, axis='index', copy=True, inplace=False,
           level=None, errors='ignore'):
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
    if not copy:
        raise NotImplementedError('`copy=False` not implemented')

    axis = validate_axis(axis, df)
    if axis == 0:
        index_mapper = index if index is not None else mapper
        columns_mapper = columns
    else:
        columns_mapper = columns if columns is not None else mapper
        index_mapper = index

    if index_mapper is not None and errors == 'raise' and not inplace:
        warnings.warn('Errors will not raise for non-existing indices')

    op = DataFrameRename(columns_mapper=columns_mapper, index_mapper=index_mapper,
                         level=level, errors=errors, output_types=[OutputType.dataframe])
    ret = op(df)
    if inplace:
        df.data = ret.data
    else:
        return ret
