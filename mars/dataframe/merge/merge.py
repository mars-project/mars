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

import itertools
from collections import namedtuple
from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...core import OutputType
from ...core.context import get_context
from ...core.operand import OperandStage, MapReduceOperand
from ...serialization.serializables import (
    AnyField,
    BoolField,
    StringField,
    TupleField,
    KeyField,
    Int32Field,
    NamedTupleField,
)
from ..core import DataFrame, Series
from ..operands import DataFrameOperand, DataFrameOperandMixin, DataFrameShuffleProxy
from ..utils import (
    auto_merge_chunks,
    build_concatenated_rows_frame,
    build_df,
    parse_index,
    hash_dataframe_on,
    infer_index_value,
)

import logging

logger = logging.getLogger(__name__)


class DataFrameMergeAlign(MapReduceOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_SHUFFLE_MERGE_ALIGN

    _index_shuffle_size = Int32Field("index_shuffle_size")
    _shuffle_on = AnyField("shuffle_on")

    _input = KeyField("input")

    def __init__(self, index_shuffle_size=None, shuffle_on=None, **kw):
        super().__init__(
            _index_shuffle_size=index_shuffle_size,
            _shuffle_on=shuffle_on,
            _output_types=[OutputType.dataframe],
            **kw,
        )

    @property
    def index_shuffle_size(self):
        return self._index_shuffle_size

    @property
    def shuffle_on(self):
        return self._shuffle_on

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    @classmethod
    def execute_map(cls, ctx, op):
        chunk = op.outputs[0]
        df = ctx[op.inputs[0].key]
        shuffle_on = op.shuffle_on

        if shuffle_on is not None:
            # shuffle on field may be resident in index
            to_reset_index_names = []
            if not isinstance(shuffle_on, (list, tuple)):
                if shuffle_on not in df.dtypes:
                    to_reset_index_names.append(shuffle_on)
            else:
                for son in shuffle_on:
                    if son not in df.dtypes:
                        to_reset_index_names.append(shuffle_on)
            if len(to_reset_index_names) > 0:
                df = df.reset_index(to_reset_index_names)

        filters = hash_dataframe_on(df, shuffle_on, op.index_shuffle_size)

        # shuffle on index
        for index_idx, index_filter in enumerate(filters):
            reducer_index = (index_idx, chunk.index[1])
            if index_filter is not None and index_filter is not list():
                ctx[chunk.key, reducer_index] = df.iloc[index_filter]
            else:
                ctx[chunk.key, reducer_index] = None

    @classmethod
    def execute_reduce(cls, ctx, op: "DataFrameMergeAlign"):
        chunk = op.outputs[0]
        input_idx_to_df = dict(op.iter_mapper_data_with_index(ctx))
        row_idxes = sorted({idx[0] for idx in input_idx_to_df})

        res = []
        for row_idx in row_idxes:
            row_df = input_idx_to_df.get((row_idx, 0), None)
            if row_df is not None:
                res.append(row_df)
        ctx[chunk.key] = pd.concat(res, axis=0)

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls.execute_map(ctx, op)
        else:
            cls.execute_reduce(ctx, op)


MergeSplitInfo = namedtuple("MergeSplitInfo", "split_side, split_index, nsplits")


class DataFrameMerge(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_MERGE

    how = StringField("how")
    on = AnyField("on")
    left_on = AnyField("left_on")
    right_on = AnyField("right_on")
    left_index = BoolField("left_index")
    right_index = BoolField("right_index")
    sort = BoolField("sort")
    suffixes = TupleField("suffixes")
    copy_ = BoolField("copy_")
    indicator = BoolField("indicator")
    validate = AnyField("validate")
    method = StringField("method")
    auto_merge_threshold = Int32Field("auto_merge_threshold")

    # only for broadcast merge
    split_info = NamedTupleField("split_info")

    def __init__(self, copy=None, **kwargs):
        super().__init__(copy_=copy, **kwargs)

    def __call__(self, left, right):
        empty_left, empty_right = build_df(left), build_df(right)
        # this `merge` will check whether the combination of those arguments is valid
        merged = empty_left.merge(
            empty_right,
            how=self.how,
            on=self.on,
            left_on=self.left_on,
            right_on=self.right_on,
            left_index=self.left_index,
            right_index=self.right_index,
            sort=self.sort,
            suffixes=self.suffixes,
            copy=self.copy_,
            indicator=self.indicator,
            validate=self.validate,
        )

        # the `index_value` doesn't matter.
        index_tokenize_objects = [
            left,
            right,
            self.how,
            self.left_on,
            self.right_on,
            self.left_index,
            self.right_index,
        ]
        return self.new_dataframe(
            [left, right],
            shape=(np.nan, merged.shape[1]),
            dtypes=merged.dtypes,
            index_value=parse_index(merged.index, *index_tokenize_objects),
            columns_value=parse_index(merged.columns, store_data=True),
        )

    @classmethod
    def _gen_shuffle_chunks(
        cls,
        out_shape: Tuple,
        shuffle_on: Union[List, str],
        df: Union[DataFrame, Series],
    ):
        # gen map chunks
        map_chunks = []
        for chunk in df.chunks:
            map_op = DataFrameMergeAlign(
                stage=OperandStage.map,
                shuffle_on=shuffle_on,
                sparse=chunk.issparse(),
                index_shuffle_size=out_shape[0],
            )
            map_chunks.append(
                map_op.new_chunk(
                    [chunk],
                    shape=(np.nan, np.nan),
                    dtypes=chunk.dtypes,
                    index=chunk.index,
                    index_value=chunk.index_value,
                    columns_value=chunk.columns_value,
                )
            )

        proxy_chunk = DataFrameShuffleProxy(
            output_types=[OutputType.dataframe]
        ).new_chunk(
            map_chunks,
            shape=(),
            dtypes=df.dtypes,
            index_value=df.index_value,
            columns_value=df.columns_value,
        )

        # gen reduce chunks
        reduce_chunks = []
        for out_idx in itertools.product(*(range(s) for s in out_shape)):
            reduce_op = DataFrameMergeAlign(
                stage=OperandStage.reduce, sparse=proxy_chunk.issparse()
            )
            reduce_chunks.append(
                reduce_op.new_chunk(
                    [proxy_chunk],
                    shape=(np.nan, np.nan),
                    dtypes=proxy_chunk.dtypes,
                    index=out_idx,
                    index_value=proxy_chunk.index_value,
                    columns_value=proxy_chunk.columns_value,
                )
            )
        return reduce_chunks

    @classmethod
    def _tile_one_chunk(
        cls,
        op: "DataFrameMerge",
        left: Union[DataFrame, Series],
        right: Union[DataFrame, Series],
    ):
        df = op.outputs[0]
        if len(left.chunks) == 1 and len(right.chunks) == 1:
            merge_op = op.copy().reset_key()
            out_chunk = merge_op.new_chunk(
                [left.chunks[0], right.chunks[0]],
                shape=df.shape,
                index=left.chunks[0].index,
                index_value=df.index_value,
                dtypes=df.dtypes,
                columns_value=df.columns_value,
            )
            out_chunks = [out_chunk]
            nsplits = ((np.nan,), (df.shape[1],))
        elif len(left.chunks) == 1:
            out_chunks = []
            left_chunk = left.chunks[0]
            for c in right.chunks:
                merge_op = op.copy().reset_key()
                out_chunk = merge_op.new_chunk(
                    [left_chunk, c],
                    shape=(np.nan, df.shape[1]),
                    index=c.index,
                    index_value=infer_index_value(
                        left_chunk.index_value, c.index_value
                    ),
                    dtypes=df.dtypes,
                    columns_value=df.columns_value,
                )
                out_chunks.append(out_chunk)
            nsplits = ((np.nan,) * len(right.chunks), (df.shape[1],))
        else:
            out_chunks = []
            right_chunk = right.chunks[0]
            for c in left.chunks:
                merge_op = op.copy().reset_key()
                out_chunk = merge_op.new_chunk(
                    [c, right_chunk],
                    shape=(np.nan, df.shape[1]),
                    index=c.index,
                    index_value=infer_index_value(
                        right_chunk.index_value, c.index_value
                    ),
                    dtypes=df.dtypes,
                    columns_value=df.columns_value,
                )
                out_chunks.append(out_chunk)
            nsplits = ((np.nan,) * len(left.chunks), (df.shape[1],))

        new_op = op.copy()
        return new_op.new_dataframes(
            op.inputs,
            df.shape,
            nsplits=nsplits,
            chunks=out_chunks,
            dtypes=df.dtypes,
            index_value=df.index_value,
            columns_value=df.columns_value,
        )

    @classmethod
    def _tile_shuffle(
        cls,
        op: "DataFrameMerge",
        left: Union[DataFrame, Series],
        right: Union[DataFrame, Series],
    ):
        df = op.outputs[0]
        left_row_chunk_size = left.chunk_shape[0]
        right_row_chunk_size = right.chunk_shape[0]
        out_row_chunk_size = max(left_row_chunk_size, right_row_chunk_size)

        out_chunk_shape = (out_row_chunk_size, 1)
        nsplits = [[np.nan for _ in range(out_row_chunk_size)], [df.shape[1]]]

        left_on = _prepare_shuffle_on(op.left_index, op.left_on, op.on)
        right_on = _prepare_shuffle_on(op.right_index, op.right_on, op.on)

        # do shuffle
        left_chunks = cls._gen_shuffle_chunks(out_chunk_shape, left_on, left)
        right_chunks = cls._gen_shuffle_chunks(out_chunk_shape, right_on, right)

        out_chunks = []
        for left_chunk, right_chunk in zip(left_chunks, right_chunks):
            merge_op = op.copy().reset_key()
            out_chunk = merge_op.new_chunk(
                [left_chunk, right_chunk],
                shape=(np.nan, df.shape[1]),
                index=left_chunk.index,
                index_value=infer_index_value(
                    left_chunk.index_value, right_chunk.index_value
                ),
                dtypes=df.dtypes,
                columns_value=df.columns_value,
            )
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(
            op.inputs,
            df.shape,
            nsplits=tuple(tuple(ns) for ns in nsplits),
            chunks=out_chunks,
            dtypes=df.dtypes,
            index_value=df.index_value,
            columns_value=df.columns_value,
        )

    @classmethod
    def _tile_broadcast(
        cls,
        op: "DataFrameMerge",
        left: Union[DataFrame, Series],
        right: Union[DataFrame, Series],
    ):
        from .concat import DataFrameConcat

        out_df = op.outputs[0]
        out_chunks = []
        if left.chunk_shape[0] < right.chunk_shape[0]:
            # broadcast left
            if op.how == "inner":
                left_chunks = left.chunks
                need_split = False
            else:
                left_on = _prepare_shuffle_on(op.left_index, op.left_on, op.on)
                left_chunks = cls._gen_shuffle_chunks(left.chunk_shape, left_on, left)
                need_split = True
            right_chunks = right.chunks
            for right_chunk in right_chunks:
                merged_chunks = []
                # concat all merged results
                for j, left_chunk in enumerate(left_chunks):
                    merge_op = op.copy().reset_key()
                    if need_split:
                        merge_op.split_info = MergeSplitInfo(
                            "right", j, len(left_chunks)
                        )
                    merged_chunks.append(
                        merge_op.new_chunk(
                            [left_chunk, right_chunk],
                            index=(j, 0),
                            shape=(np.nan, out_df.shape[1]),
                            columns_value=out_df.columns_value,
                        )
                    )
                concat_op = DataFrameConcat(output_types=[OutputType.dataframe])
                out_chunks.append(
                    concat_op.new_chunk(
                        merged_chunks,
                        shape=(np.nan, out_df.shape[1]),
                        dtypes=out_df.dtypes,
                        index=right_chunk.index,
                        index_value=infer_index_value(
                            left_chunks[0].index_value, right_chunk.index_value
                        ),
                        columns_value=out_df.columns_value,
                    )
                )
            nsplits = ((np.nan,) * len(right.chunks), (out_df.shape[1],))
        else:
            # broadcast right
            if op.how == "inner":
                need_split = False
                right_chunks = right.chunks
            else:
                need_split = True
                right_on = _prepare_shuffle_on(op.right_index, op.right_on, op.on)
                right_chunks = cls._gen_shuffle_chunks(
                    right.chunk_shape, right_on, right
                )
            left_chunks = left.chunks
            for left_chunk in left_chunks:
                merged_chunks = []
                # concat all merged results
                for j, right_chunk in enumerate(right_chunks):
                    merge_op = op.copy().reset_key()
                    if need_split:
                        merge_op.split_info = MergeSplitInfo(
                            "left", j, len(right_chunks)
                        )
                    merged_chunks.append(
                        merge_op.new_chunk(
                            [left_chunk, right_chunk],
                            shape=(np.nan, out_df.shape[1]),
                            index=(j, 0),
                            columns_value=out_df.columns_value,
                        )
                    )
                concat_op = DataFrameConcat(output_types=[OutputType.dataframe])
                out_chunks.append(
                    concat_op.new_chunk(
                        merged_chunks,
                        shape=(np.nan, out_df.shape[1]),
                        dtypes=out_df.dtypes,
                        index=left_chunk.index,
                        index_value=infer_index_value(
                            left_chunk.index_value, right_chunks[0].index_value
                        ),
                        columns_value=out_df.columns_value,
                    )
                )
            nsplits = ((np.nan,) * len(left.chunks), (out_df.shape[1],))

        new_op = op.copy()
        return new_op.new_dataframes(
            op.inputs,
            out_df.shape,
            nsplits=tuple(tuple(ns) for ns in nsplits),
            chunks=out_chunks,
            dtypes=out_df.dtypes,
            index_value=out_df.index_value,
            columns_value=out_df.columns_value,
        )

    @classmethod
    def tile(cls, op: "DataFrameMerge"):
        left = build_concatenated_rows_frame(op.inputs[0])
        right = build_concatenated_rows_frame(op.inputs[1])
        how = op.how
        left_row_chunk_size = left.chunk_shape[0]
        right_row_chunk_size = right.chunk_shape[0]
        if left_row_chunk_size > right_row_chunk_size:
            big_side = "left"
            big_chunk_size = left_row_chunk_size
            small_chunk_size = right_row_chunk_size
        else:
            big_side = "right"
            big_chunk_size = right_row_chunk_size
            small_chunk_size = left_row_chunk_size

        if op.method != "shuffle" and (
            (len(left.chunks) == 1 and op.how in ["right", "inner"])
            or (len(right.chunks) == 1 and op.how in ["left", "inner"])
        ):
            ret = cls._tile_one_chunk(op, left, right)
        elif op.method == "broadcast" or (
            how in [big_side, "inner"] and np.log2(big_chunk_size) > small_chunk_size
        ):
            ret = cls._tile_broadcast(op, left, right)
        else:
            ret = cls._tile_shuffle(op, left, right)
        if op.how == "inner" and len(ret[0].chunks) > op.auto_merge_threshold:
            # if how=="inner", output data size will reduce greatly with high probability，
            # use auto_merge_chunks to combine small chunks.
            yield ret[0].chunks  # trigger execution for chunks
            return [auto_merge_chunks(get_context(), ret[0])]
        else:
            return ret

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        left, right = ctx[op.inputs[0].key], ctx[op.inputs[1].key]

        if getattr(op, "split_info", None) is not None:
            split_info = op.split_info
            if split_info.split_side == "left":
                index = hash_dataframe_on(left, on=op.on, size=split_info.nsplits)[
                    split_info.split_index
                ]
                left = left.iloc[index]
            else:
                index = hash_dataframe_on(right, on=op.on, size=split_info.nsplits)[
                    split_info.split_index
                ]
                right = right.iloc[index]

        def execute_merge(x, y):
            if not op.gpu:
                kwargs = dict(
                    copy=op.copy, validate=op.validate, indicator=op.indicator
                )
            else:  # pragma: no cover
                # cudf doesn't support 'validate' and 'copy'
                kwargs = dict(indicator=op.indicator)
            return x.merge(
                y,
                how=op.how,
                on=op.on,
                left_on=op.left_on,
                right_on=op.right_on,
                left_index=op.left_index,
                right_index=op.right_index,
                sort=op.sort,
                suffixes=op.suffixes,
                **kwargs,
            )

        # workaround for: https://github.com/pandas-dev/pandas/issues/27943
        try:
            r = execute_merge(left, right)
        except ValueError:
            r = execute_merge(left.copy(deep=True), right.copy(deep=True))

        # make sure column's order
        if not all(
            n1 == n2 for n1, n2 in zip(chunk.columns_value.to_pandas(), r.columns)
        ):
            r = r[list(chunk.columns_value.to_pandas())]
        ctx[chunk.key] = r


def _prepare_shuffle_on(use_index, side_on, on):
    # consistent with pandas: `left_index` precedes `left_on` and `right_index` precedes `right_on`
    if use_index:
        # `None` means we will shuffle on df.index.
        return None
    elif side_on is not None:
        return side_on
    else:
        return on


def merge(
    df: Union[DataFrame, Series],
    right: Union[DataFrame, Series],
    how: str = "inner",
    on: str = None,
    left_on: str = None,
    right_on: str = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: Tuple[Optional[str], Optional[str]] = ("_x", "_y"),
    copy: bool = True,
    indicator: bool = False,
    validate: str = None,
    method: str = None,
    auto_merge_threshold: int = 8,
) -> DataFrame:
    """
    Merge DataFrame or named Series objects with a database-style join.

    A named Series object is treated as a DataFrame with a single named column.

    The join is done on columns or indexes. If joining columns on
    columns, the DataFrame indexes *will be ignored*. Otherwise if joining indexes
    on indexes or indexes on a column or columns, the index will be passed on.
    When performing a cross merge, no column specifications to merge on are
    allowed.

    Parameters
    ----------
    right : DataFrame or named Series
        Object to merge with.
    how : {'left', 'right', 'outer', 'inner'}, default 'inner'
        Type of merge to be performed.

        * left: use only keys from left frame, similar to a SQL left outer join;
          preserve key order.
        * right: use only keys from right frame, similar to a SQL right outer join;
          preserve key order.
        * outer: use union of keys from both frames, similar to a SQL full outer
          join; sort keys lexicographically.
        * inner: use intersection of keys from both frames, similar to a SQL inner
          join; preserve the order of the left keys.

    on : label or list
        Column or index level names to join on. These must be found in both
        DataFrames. If `on` is None and not merging on indexes then this defaults
        to the intersection of the columns in both DataFrames.
    left_on : label or list, or array-like
        Column or index level names to join on in the left DataFrame. Can also
        be an array or list of arrays of the length of the left DataFrame.
        These arrays are treated as if they are columns.
    right_on : label or list, or array-like
        Column or index level names to join on in the right DataFrame. Can also
        be an array or list of arrays of the length of the right DataFrame.
        These arrays are treated as if they are columns.
    left_index : bool, default False
        Use the index from the left DataFrame as the join key(s). If it is a
        MultiIndex, the number of keys in the other DataFrame (either the index
        or a number of columns) must match the number of levels.
    right_index : bool, default False
        Use the index from the right DataFrame as the join key. Same caveats as
        left_index.
    sort : bool, default False
        Sort the join keys lexicographically in the result DataFrame. If False,
        the order of the join keys depends on the join type (how keyword).
    suffixes : list-like, default is ("_x", "_y")
        A length-2 sequence where each element is optionally a string
        indicating the suffix to add to overlapping column names in
        `left` and `right` respectively. Pass a value of `None` instead
        of a string to indicate that the column name from `left` or
        `right` should be left as-is, with no suffix. At least one of the
        values must not be None.
    copy : bool, default True
        If False, avoid copy if possible.
    indicator : bool or str, default False
        If True, adds a column to the output DataFrame called "_merge" with
        information on the source of each row. The column can be given a different
        name by providing a string argument. The column will have a Categorical
        type with the value of "left_only" for observations whose merge key only
        appears in the left DataFrame, "right_only" for observations
        whose merge key only appears in the right DataFrame, and "both"
        if the observation's merge key is found in both DataFrames.
    validate : str, optional
        If specified, checks if merge is of specified type.

        * "one_to_one" or "1:1": check if merge keys are unique in both
          left and right datasets.
        * "one_to_many" or "1:m": check if merge keys are unique in left
          dataset.
        * "many_to_one" or "m:1": check if merge keys are unique in right
          dataset.
        * "many_to_many" or "m:m": allowed, but does not result in checks.
    method : {"shuffle", "broadcast"}, default None
        "broadcast" is recommended when one DataFrame is much smaller than the other,
        otherwise, "shuffle" will be a better choice. By default, we choose method
        according to actual data size.
    auto_merge_threshold : int, default 8
        When how is "inner", merged result could be much smaller than original DataFrame,
        if the number of chunks is greater than the threshold,
        it will merge small chunks automatically.

    Returns
    -------
    DataFrame
        A DataFrame of the two merged objects.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df1 = md.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
    ...                     'value': [1, 2, 3, 5]})
    >>> df2 = md.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],
    ...                     'value': [5, 6, 7, 8]})
    >>> df1.execute()
        lkey value
    0   foo      1
    1   bar      2
    2   baz      3
    3   foo      5
    >>> df2.execute()
        rkey value
    0   foo      5
    1   bar      6
    2   baz      7
    3   foo      8

    Merge df1 and df2 on the lkey and rkey columns. The value columns have
    the default suffixes, _x and _y, appended.

    >>> df1.merge(df2, left_on='lkey', right_on='rkey').execute()
      lkey  value_x rkey  value_y
    0  foo        1  foo        5
    1  foo        1  foo        8
    2  foo        5  foo        5
    3  foo        5  foo        8
    4  bar        2  bar        6
    5  baz        3  baz        7

    Merge DataFrames df1 and df2 with specified left and right suffixes
    appended to any overlapping columns.

    >>> df1.merge(df2, left_on='lkey', right_on='rkey',
    ...           suffixes=('_left', '_right')).execute()
      lkey  value_left rkey  value_right
    0  foo           1  foo            5
    1  foo           1  foo            8
    2  foo           5  foo            5
    3  foo           5  foo            8
    4  bar           2  bar            6
    5  baz           3  baz            7

    Merge DataFrames df1 and df2, but raise an exception if the DataFrames have
    any overlapping columns.

    >>> df1.merge(df2, left_on='lkey', right_on='rkey', suffixes=(False, False)).execute()
    Traceback (most recent call last):
    ...
    ValueError: columns overlap but no suffix specified:
        Index(['value'], dtype='object')

    >>> df1 = md.DataFrame({'a': ['foo', 'bar'], 'b': [1, 2]})
    >>> df2 = md.DataFrame({'a': ['foo', 'baz'], 'c': [3, 4]})
    >>> df1.execute()
          a  b
    0   foo  1
    1   bar  2
    >>> df2.execute()
          a  c
    0   foo  3
    1   baz  4

    >>> df1.merge(df2, how='inner', on='a').execute()
          a  b  c
    0   foo  1  3

    >>> df1.merge(df2, how='left', on='a').execute()
          a  b  c
    0   foo  1  3.0
    1   bar  2  NaN
    """
    if method is not None and method not in [
        "shuffle",
        "broadcast",
    ]:  # pragma: no cover
        raise NotImplementedError(f"{method} merge is not supported")
    op = DataFrameMerge(
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        sort=sort,
        suffixes=suffixes,
        copy=copy,
        indicator=indicator,
        validate=validate,
        method=method,
        auto_merge_threshold=auto_merge_threshold,
        output_types=[OutputType.dataframe],
    )
    return op(df, right)


def join(
    df: Union[DataFrame, Series],
    other: Union[DataFrame, Series],
    on: str = None,
    how: str = "left",
    lsuffix: str = "",
    rsuffix: str = "",
    sort: bool = False,
    method: str = None,
    auto_merge_threshold: int = 8,
) -> DataFrame:
    """
    Join columns of another DataFrame.

    Join columns with `other` DataFrame either on index or on a key
    column. Efficiently join multiple DataFrame objects by index at once by
    passing a list.

    Parameters
    ----------
    other : DataFrame, Series, or list of DataFrame
        Index should be similar to one of the columns in this one. If a
        Series is passed, its name attribute must be set, and that will be
        used as the column name in the resulting joined DataFrame.
    on : str, list of str, or array-like, optional
        Column or index level name(s) in the caller to join on the index
        in `other`, otherwise joins index-on-index. If multiple
        values given, the `other` DataFrame must have a MultiIndex. Can
        pass an array as the join key if it is not already contained in
        the calling DataFrame. Like an Excel VLOOKUP operation.
    how : {'left', 'right', 'outer', 'inner'}, default 'left'
        How to handle the operation of the two objects.

        * left: use calling frame's index (or column if on is specified)
        * right: use `other`'s index.
        * outer: form union of calling frame's index (or column if on is
          specified) with `other`'s index, and sort it.
          lexicographically.
        * inner: form intersection of calling frame's index (or column if
          on is specified) with `other`'s index, preserving the order
          of the calling's one.

    lsuffix : str, default ''
        Suffix to use from left frame's overlapping columns.
    rsuffix : str, default ''
        Suffix to use from right frame's overlapping columns.
    sort : bool, default False
        Order result DataFrame lexicographically by the join key. If False,
        the order of the join key depends on the join type (how keyword).
    method : {"shuffle", "broadcast"}, default None
        "broadcast" is recommended when one DataFrame is much smaller than the other,
        otherwise, "shuffle" will be a better choice. By default, we choose method
        according to actual data size.
    auto_merge_threshold : int, default 8
        When how is "inner", merged result could be much smaller than original DataFrame,
        if the number of chunks is greater than the threshold,
        it will merge small chunks automatically.

    Returns
    -------
    DataFrame
        A dataframe containing columns from both the caller and `other`.

    See Also
    --------
    DataFrame.merge : For column(s)-on-column(s) operations.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
    ...                    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})

    >>> df.execute()
      key   A
    0  K0  A0
    1  K1  A1
    2  K2  A2
    3  K3  A3
    4  K4  A4
    5  K5  A5

    >>> other = md.DataFrame({'key': ['K0', 'K1', 'K2'],
    ...                       'B': ['B0', 'B1', 'B2']})

    >>> other.execute()
      key   B
    0  K0  B0
    1  K1  B1
    2  K2  B2

    Join DataFrames using their indexes.

    >>> df.join(other, lsuffix='_caller', rsuffix='_other').execute()
      key_caller   A key_other    B
    0         K0  A0        K0   B0
    1         K1  A1        K1   B1
    2         K2  A2        K2   B2
    3         K3  A3       NaN  NaN
    4         K4  A4       NaN  NaN
    5         K5  A5       NaN  NaN

    If we want to join using the key columns, we need to set key to be
    the index in both `df` and `other`. The joined DataFrame will have
    key as its index.

    >>> df.set_index('key').join(other.set_index('key')).execute()
          A    B
    key
    K0   A0   B0
    K1   A1   B1
    K2   A2   B2
    K3   A3  NaN
    K4   A4  NaN
    K5   A5  NaN

    Another option to join using the key columns is to use the `on`
    parameter. DataFrame.join always uses `other`'s index but we can use
    any column in `df`. This method preserves the original DataFrame's
    index in the result.

    >>> df.join(other.set_index('key'), on='key').execute()
      key   A    B
    0  K0  A0   B0
    1  K1  A1   B1
    2  K2  A2   B2
    3  K3  A3  NaN
    4  K4  A4  NaN
    5  K5  A5  NaN

    Using non-unique key values shows how they are matched.

    >>> df = md.DataFrame({'key': ['K0', 'K1', 'K1', 'K3', 'K0', 'K1'],
    ...                    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})

    >>> df.execute()
      key   A
    0  K0  A0
    1  K1  A1
    2  K1  A2
    3  K3  A3
    4  K0  A4
    5  K1  A5

    >>> df.join(other.set_index('key'), on='key').execute()
      key   A    B
    0  K0  A0   B0
    1  K1  A1   B1
    2  K1  A2   B1
    3  K3  A3  NaN
    4  K0  A4   B0
    5  K1  A5   B1
    """
    return merge(
        df,
        other,
        left_on=on,
        how=how,
        left_index=on is None,
        right_index=True,
        suffixes=(lsuffix, rsuffix),
        sort=sort,
        method=method,
        auto_merge_threshold=auto_merge_threshold,
    )
