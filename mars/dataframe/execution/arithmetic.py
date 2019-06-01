# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import operator
import itertools

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pass

from ...utils import get_shuffle_input_keys_idxes
from ..utils import hash_index
from ..expressions.arithmetic.core import DataFrameIndexAlignMap, DataFrameIndexAlignReduce
from ..expressions.arithmetic import DataFrameAdd, DataFrameAbs


def _index_align_map(ctx, chunk):
    # TODO(QIN): add GPU support here
    df = ctx[chunk.inputs[0].key]

    filters = [[], []]

    if chunk.op.index_shuffle_size is None:
        # no shuffle on index
        op = operator.ge if chunk.op.index_min_close else operator.gt
        index_cond = op(df.index, chunk.op.index_min)
        op = operator.le if chunk.op.index_max_close else operator.lt
        index_cond = index_cond & op(df.index, chunk.op.index_max)
        filters[0].append(index_cond)
    else:
        # shuffle on index
        shuffle_size = chunk.op.index_shuffle_size
        filters[0].extend(hash_index(df.index, shuffle_size))

    if chunk.op.column_shuffle_size is None:
        # no shuffle on columns
        op = operator.ge if chunk.op.column_min_close else operator.gt
        columns_cond = op(df.columns, chunk.op.column_min)
        op = operator.le if chunk.op.column_max_close else operator.lt
        columns_cond = columns_cond & op(df.columns, chunk.op.column_max)
        filters[1].append(columns_cond)
    else:
        # shuffle on columns
        shuffle_size = chunk.op.column_shuffle_size
        filters[1].extend(hash_index(df.columns, shuffle_size))

    if all(len(it) == 1 for it in filters):
        # no shuffle
        ctx[chunk.key] = df.loc[filters[0][0], filters[1][0]]
        return
    elif len(filters[0]) == 1:
        # shuffle on columns
        for column_idx, column_filter in enumerate(filters[1]):
            group_key = ','.join([str(chunk.index[0]), str(column_idx)])
            ctx[(chunk.key, group_key)] = df.loc[filters[0][0], column_filter]
    elif len(filters[1]) == 1:
        # shuffle on index
        for index_idx, index_filter in enumerate(filters[0]):
            group_key = ','.join([str(index_idx), str(chunk.index[1])])
            ctx[(chunk.key, group_key)] = df.loc[index_filter, filters[1][0]]
    else:
        # full shuffle
        shuffle_index_size = chunk.op.index_shuffle_size
        shuffle_column_size = chunk.op.column_shuffle_size
        out_idxes = itertools.product(range(shuffle_index_size), range(shuffle_column_size))
        out_index_columns = itertools.product(*filters)
        for out_idx, out_index_column in zip(out_idxes, out_index_columns):
            index_filter, column_filter = out_index_column
            group_key = ','.join(str(i) for i in out_idx)
            ctx[(chunk.key, group_key)] = df.loc[index_filter, column_filter]


def _index_align_reduce(ctx, chunk):
    input_keys, input_idxes = get_shuffle_input_keys_idxes(chunk.inputs[0])
    input_idx_to_df = {idx: ctx[inp_key, ','.join(str(ix) for ix in chunk.index)]
                       for inp_key, idx in zip(input_keys, input_idxes)}
    row_idxes = sorted({idx[0] for idx in input_idx_to_df})
    col_idxes = sorted({idx[1] for idx in input_idx_to_df})

    res = None
    for row_idx in row_idxes:
        row_df = None
        for col_idx in col_idxes:
            df = input_idx_to_df[row_idx, col_idx]
            if row_df is None:
                row_df = df
            else:
                row_df = pd.concat([row_df, df], axis=1)

        if res is None:
            res = row_df
        else:
            res = pd.concat([res, row_df], axis=0)

    ctx[chunk.key] = res


def _add(ctx, chunk):
    left, right = ctx[chunk.inputs[0].key], ctx[chunk.inputs[1].key]
    ctx[chunk.key] = left.add(right, axis=chunk.op.axis,
                              level=chunk.op.level, fill_value=chunk.op.fill_value)

def _abs(ctx, chunk):
    df = ctx[chunk.inputs[0].key]
    ctx[chunk.key] = df.abs()

def register_arithmetic_handler():
    from ...executor import register

    register(DataFrameIndexAlignMap, _index_align_map)
    register(DataFrameIndexAlignReduce, _index_align_reduce)
    register(DataFrameAdd, _add)
    register(DataFrameAbs, _abs)
