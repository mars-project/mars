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

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...core import OutputType
from ...core.operand import OperandStage, MapReduceOperand
from ...serialization.serializables import (
    AnyField,
    BoolField,
    StringField,
    TupleField,
    KeyField,
    Int32Field,
)
from ..operands import DataFrameOperand, DataFrameOperandMixin, DataFrameShuffleProxy
from ..utils import (
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
    strategy = StringField("strategy")

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
    def _gen_shuffle_chunks(cls, out_shape, shuffle_on, df):
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
    def _tile_one_chunk(cls, op, left, right):
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
    def _tile_shuffle(cls, op, left, right):
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
    def _tile_broadcast(cls, op, left, right):
        from .concat import DataFrameConcat

        out_df = op.outputs[0]
        out_chunks = []
        if left.chunk_shape[0] < right.chunk_shape[0]:
            # broadcast left
            left_on = _prepare_shuffle_on(op.left_index, op.left_on, op.on)
            left_chunks = cls._gen_shuffle_chunks(left.chunk_shape, left_on, left)
            right_chunks = right.chunks
            for right_chunk in right_chunks:
                merged_chunks = []
                # concat all merged results
                for j, left_chunk in enumerate(left_chunks):
                    merge_op = op.copy().reset_key()
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
            right_on = _prepare_shuffle_on(op.right_index, op.right_on, op.on)
            right_chunks = cls._gen_shuffle_chunks(right.chunk_shape, right_on, right)
            left_chunks = left.chunks
            for left_chunk in left_chunks:
                merged_chunks = []
                # concat all merged results
                for j, right_chunk in enumerate(right_chunks):
                    merge_op = op.copy().reset_key()
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
    def tile(cls, op):
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

        if op.strategy != "shuffle" and (
                (len(left.chunks) == 1
            and op.how in ["right", "inner"])
            or (len(right.chunks) == 1
            and op.how in ["left", "inner"])
        ):
            return cls._tile_one_chunk(op, left, right)
        elif op.strategy == "broadcast" or (
            how in [big_side, "inner"] and np.log2(big_chunk_size) > small_chunk_size
        ):
            return cls._tile_broadcast(op, left, right)
        else:
            return cls._tile_shuffle(op, left, right)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        left, right = ctx[op.inputs[0].key], ctx[op.inputs[1].key]

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
    df,
    right,
    how="inner",
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    sort=False,
    suffixes=("_x", "_y"),
    copy=True,
    indicator=False,
    strategy=None,
    validate=None,
):
    if strategy is not None and strategy not in [
        "shuffle",
        "broadcast",
    ]:  # pragma: no cover
        raise NotImplementedError(f"{strategy} merge is not supported")
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
        strategy=strategy,
        output_types=[OutputType.dataframe],
    )
    return op(df, right)


def join(
    df, other, on=None, how="left", lsuffix="", rsuffix="", sort=False, strategy=None
):
    return merge(
        df,
        other,
        left_on=on,
        how=how,
        left_index=on is None,
        right_index=True,
        suffixes=(lsuffix, rsuffix),
        sort=sort,
        strategy=strategy,
    )
