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

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import AnyField, BoolField
from ..core import DATAFRAME_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import validate_axis, parse_index, build_df, build_series


class DataFrameRenameAxis(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.RENAME_AXIS

    _index = AnyField('index')
    _columns = AnyField('columns')
    _copy_value = BoolField('copy_value')
    _level = AnyField('level')

    def __init__(self, index=None, columns=None, copy_value=None, level=None, **kw):
        super().__init__(_index=index, _columns=columns, _copy_value=copy_value,
                         _level=level, **kw)

    @property
    def index(self):
        return self._index

    @property
    def columns(self):
        return self._columns

    @property
    def copy_value(self):
        return self._copy_value

    @property
    def level(self):
        return self._level

    @staticmethod
    def _update_params(params, obj, mapper, axis, level):
        if obj.ndim == 2:
            test_obj = build_df(obj)
        else:
            test_obj = build_series(obj)

        if level is None:
            test_obj = test_obj.rename_axis(mapper, axis=axis)
        else:
            test_obj.axes[axis].set_names(mapper, level=level, inplace=True)

        if axis == 0:
            params['index_value'] = parse_index(test_obj.index, store_data=False)
        else:
            params['dtypes'] = test_obj.dtypes
            params['columns_value'] = parse_index(test_obj.columns, store_data=True)

    def __call__(self, df_or_series):
        params = df_or_series.params

        if isinstance(df_or_series, DATAFRAME_TYPE):
            self._output_types = [OutputType.dataframe]
        else:
            self._output_types = [OutputType.series]

        if self.index is not None:
            self._update_params(params, df_or_series, self.index, axis=0, level=self.level)
        else:
            self._update_params(params, df_or_series, self.columns, axis=1, level=self.level)

        return self.new_tileable([df_or_series], **params)

    @classmethod
    def tile(cls, op: 'DataFrameRenameAxis'):
        in_obj = op.inputs[0]
        out_obj = op.outputs[0]

        chunks = []
        idx_cache = dict()
        for c in in_obj.chunks:
            params = c.params
            if op.index is not None:
                try:
                    params['index_value'] = idx_cache[c.index[0]]
                except KeyError:
                    cls._update_params(params, c, op.index, axis=0, level=op.level)
                    idx_cache[c.index[0]] = params['index_value']
            else:
                try:
                    params['columns_value'], params['dtypes'] = idx_cache[c.index[1]]
                except KeyError:
                    cls._update_params(params, c, op.columns, axis=1, level=op.level)
                    idx_cache[c.index[1]] = params['columns_value'], params['dtypes']

            new_op = op.copy().reset_key()
            chunks.append(new_op.new_chunk([c], **params))

        new_op = op.copy().reset_key()
        return new_op.new_tileables([in_obj], chunks=chunks, nsplits=in_obj.nsplits,
                                    **out_obj.params)

    @classmethod
    def execute(cls, ctx, op: 'DataFrameRenameAxis'):
        in_data = ctx[op.inputs[0].key]
        if op.index is not None:
            val, axis = op.index, 0
        else:
            val, axis = op.columns, 1

        if op.level is None:
            ctx[op.outputs[0].key] = in_data.rename_axis(val, axis=axis, copy=op.copy_value)
        else:
            ret = in_data.copy() if op.copy_value else in_data
            ret.axes[axis].set_names(val, level=op.level, inplace=True)
            ctx[op.outputs[0].key] = ret


def rename_axis_with_level(df_or_series, mapper=None, index=None, columns=None,
                           axis=0, copy=True, level=None, inplace=False):
    axis = validate_axis(axis, df_or_series)
    if mapper is not None:
        if axis == 0:
            index = mapper
        else:
            columns = mapper
    op = DataFrameRenameAxis(index=index, columns=columns, copy_value=copy,
                             level=level)
    result = op(df_or_series)
    if not inplace:
        return result
    else:
        df_or_series.data = result.data


def rename_axis(df_or_series, mapper=None, index=None, columns=None, axis=0,
                copy=True, inplace=False):
    """
    Set the name of the axis for the index or columns.

    Parameters
    ----------
    mapper : scalar, list-like, optional
        Value to set the axis name attribute.
    index, columns : scalar, list-like, dict-like or function, optional
        A scalar, list-like, dict-like or functions transformations to
        apply to that axis' values.
        Note that the ``columns`` parameter is not allowed if the
        object is a Series. This parameter only apply for DataFrame
        type objects.

        Use either ``mapper`` and ``axis`` to
        specify the axis to target with ``mapper``, or ``index``
        and/or ``columns``.

    axis : {0 or 'index', 1 or 'columns'}, default 0
        The axis to rename.
    copy : bool, default True
        Also copy underlying data.
    inplace : bool, default False
        Modifies the object directly, instead of creating a new Series
        or DataFrame.

    Returns
    -------
    Series, DataFrame, or None
        The same type as the caller or None if `inplace` is True.

    See Also
    --------
    Series.rename : Alter Series index labels or name.
    DataFrame.rename : Alter DataFrame index labels or name.
    Index.rename : Set new names on index.

    Notes
    -----
    ``DataFrame.rename_axis`` supports two calling conventions

    * ``(index=index_mapper, columns=columns_mapper, ...)``
    * ``(mapper, axis={'index', 'columns'}, ...)``

    The first calling convention will only modify the names of
    the index and/or the names of the Index object that is the columns.
    In this case, the parameter ``copy`` is ignored.

    The second calling convention will modify the names of the
    the corresponding index if mapper is a list or a scalar.
    However, if mapper is dict-like or a function, it will use the
    deprecated behavior of modifying the axis *labels*.

    We *highly* recommend using keyword arguments to clarify your
    intent.

    Examples
    --------
    **Series**

    >>> import mars.dataframe as md
    >>> s = md.Series(["dog", "cat", "monkey"])
    >>> s.execute()
    0       dog
    1       cat
    2    monkey
    dtype: object
    >>> s.rename_axis("animal").execute()
    animal
    0    dog
    1    cat
    2    monkey
    dtype: object

    **DataFrame**

    >>> df = md.DataFrame({"num_legs": [4, 4, 2],
    ...                    "num_arms": [0, 0, 2]},
    ...                   ["dog", "cat", "monkey"])
    >>> df.execute()
            num_legs  num_arms
    dog            4         0
    cat            4         0
    monkey         2         2
    >>> df = df.rename_axis("animal")
    >>> df.execute()
            num_legs  num_arms
    animal
    dog            4         0
    cat            4         0
    monkey         2         2
    >>> df = df.rename_axis("limbs", axis="columns")
    >>> df.execute()
    limbs   num_legs  num_arms
    animal
    dog            4         0
    cat            4         0
    monkey         2         2
    """
    return rename_axis_with_level(
        df_or_series, mapper=mapper, index=index, columns=columns, axis=axis,
        copy=copy, inplace=inplace)
