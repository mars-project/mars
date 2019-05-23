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

import itertools
import operator

import numpy as np
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pass

from ...tensor.expressions.utils import dictify_chunk_size, normalize_chunk_sizes
from ...utils import tokenize
from ..core import IndexValue


def is_pd_range_empty(pd_range_index):
    return (pd_range_index._start >= pd_range_index._stop and pd_range_index._step >= 0) or \
           (pd_range_index._start <= pd_range_index._stop and pd_range_index._step < 0)


def decide_chunk_sizes(shape, chunk_size, memory_usage):
    """
    Decide how a given DataFrame can be split into chunk.

    :param shape: DataFrame's shape
    :param chunk_size: if dict provided, it's dimension id to chunk size;
                       if provided, it's the chunk size for each dimension.
    :param memory_usage: pandas Series in which each column's memory usage
    :type memory_usage: pandas.Series
    :return: the calculated chunk size for each dimension
    :rtype: tuple
    """
    from ...config import options

    chunk_size = dictify_chunk_size(shape, chunk_size)
    average_memory_usage = memory_usage / shape[0]

    nleft = len(shape) - len(chunk_size)
    if nleft < 0:
        raise ValueError("chunks have more than two dimensions")
    if nleft == 0:
        return normalize_chunk_sizes(shape, tuple(chunk_size[j] for j in range(len(shape))))

    max_chunk_size = options.tensor.chunk_store_limit

    # for the row side, along axis 0
    if 0 not in chunk_size:
        row_chunk_size = []
        row_left_size = shape[0]
    else:
        row_chunk_size = normalize_chunk_sizes((shape[0],), (chunk_size[0],))[0]
        row_left_size = 0
    # for the column side, along axis 1
    if 1 not in chunk_size:
        col_chunk_size = []
        col_chunk_store = []
        col_left_size = shape[1]
    else:
        col_chunk_size = normalize_chunk_sizes((shape[1],), (chunk_size[1],))[0]
        acc = [0] + np.cumsum(col_chunk_size).tolist()
        col_chunk_store = [average_memory_usage[acc[i]: acc[i+1]].sum()
                           for i in range(len(col_chunk_size))]
        col_left_size = 0

    while True:
        nbytes_occupied = np.prod([max(it) for it in (row_chunk_size, col_chunk_store) if it])
        dim_size = np.maximum(int(np.power(max_chunk_size / nbytes_occupied, 1 / float(nleft))), 1)
        # check col first
        if col_left_size > 0:
            cs = min(col_left_size, dim_size)
            col_chunk_size.append(cs)
            start = int(np.sum(col_chunk_size[:-1]))
            col_chunk_store.append(average_memory_usage[start: start+cs].sum())
            col_left_size -= cs
        if row_left_size > 0:
            max_col_chunk_store = max(col_chunk_store)
            cs = min(row_left_size, int(max_chunk_size / max_col_chunk_store))
            row_chunk_size.append(cs)
            row_left_size -= cs

        if col_left_size == 0 and row_left_size == 0:
            break

    return tuple(row_chunk_size), tuple(col_chunk_size)


def parse_index(index_value, store_data=False, key=None):
    import pandas as pd

    def _extract_property(index, ret_data):
        kw = {
            '_is_monotonic_increasing': index.is_monotonic_increasing,
            '_is_monotonic_decreasing': index.is_monotonic_decreasing,
            '_is_unique': index.is_unique,
            '_min_val': index.min(),
            '_max_val': index.max(),
            '_min_val_close': True,
            '_max_val_close': True,
            '_key': key or tokenize(index),
        }
        if ret_data:
            kw['_data'] = index.values
        return kw

    def _serialize_index(index):
        properties = _extract_property(index, store_data)
        return getattr(IndexValue, type(index).__name__)(_name=index.name, **properties)

    def _serialize_range_index(index):
        if is_pd_range_empty(index):
            properties = {
                '_is_monotonic_increasing': True,
                '_is_monotonic_decreasing': False,
                '_is_unique': True,
                '_min_val': index._start,
                '_max_val': index._stop,
                '_min_val_close': True,
                '_max_val_close': False,
                '_key': key or tokenize(index),
            }
        else:
            properties = _extract_property(index, False)
        return IndexValue.RangeIndex(_slice=slice(index._start, index._stop, index._step),
                                     _name=index.name, **properties)

    def _serialize_multi_index(index):
        kw = _extract_property(index, store_data)
        kw['_sortorder'] = index.sortorder
        return IndexValue.MultiIndex(_names=index.names, **kw)

    if isinstance(index_value, pd.RangeIndex):
        return IndexValue(_index_value=_serialize_range_index(index_value))
    elif isinstance(index_value, pd.MultiIndex):
        return IndexValue(_index_value=_serialize_multi_index(index_value))
    else:
        return IndexValue(_index_value=_serialize_index(index_value))


def split_monotonic_index_min_max(left_min_max, left_increase, right_min_max, right_increase):
    """
    Split the original two min_max into new min_max. Each min_max should be a list
    in which each item should be a 4-tuple indicates that this chunk's min value,
    whether the min value is close, the max value, and whether the max value is close.
    The return value would be a nested list, each item is a list
    indicates that how this chunk should be split into.

    :param left_min_max: the left min_max
    :param left_increase: if the original data of left is increased
    :param right_min_max: the right min_max
    :param right_increase: if the original data of right is increased
    :return: nested list in which each item indicates how min_max is split

    >>> left_min_max = [(0, True, 3, True), (4, True, 8, True), (12, True, 18, True),
    >>>                 (20, True, 22, True)]
    >>> right_min_max = [(2, True, 6, True), (7, True, 9, True), (10, True, 14, True),
    >>>                  (18, True, 19, True)]
    >>> l, r = split_monotonic_index_min_max(left_min_max, True, right_min_max, True)
    >>> l
    [[(0, True, 2, False), (2, True, 3, True)], [(3, False, 4, False), (4, True, 6, True), (6, False, 7, False),
    (7, True, 8, True)], [(8, False, 9, True), (10, True, 12, False), (12, True, 14, True), (14, False, 18, False),
    (18, True, 18, True)], [(18, False, 19, True), [20, True, 22, True]]]
    >>> r
    [[(0, True, 2, False), (2, True, 3, True), (3, False, 4, False), (4, True, 6, True)],
    [(6, False, 7, False), (7, True, 8, True), (8, False, 9, True)], [(10, True, 12, False), (12, True, 14, True)],
    [(14, False, 18, False), (18, True, 18, True), (18, False, 19, True), [20, True, 22, True]]]
    """
    left_idx_to_min_max = [[] for _ in left_min_max]
    right_idx_to_min_max = [[] for _ in right_min_max]
    left_curr_min_max = list(left_min_max[0])
    right_curr_min_max = list(right_min_max[0])
    left_curr_idx = right_curr_idx = 0
    left_terminate = right_terminate = False

    while not left_terminate or not right_terminate:
        if left_terminate:
            left_idx_to_min_max[left_curr_idx].append(tuple(right_curr_min_max))
            right_idx_to_min_max[right_curr_idx].append(tuple(right_curr_min_max))
            if right_curr_idx + 1 >= len(right_min_max):
                right_terminate = True
            else:
                right_curr_idx += 1
                right_curr_min_max = list(right_min_max[right_curr_idx])
        elif right_terminate:
            right_idx_to_min_max[right_curr_idx].append(tuple(left_curr_min_max))
            left_idx_to_min_max[left_curr_idx].append(tuple(left_curr_min_max))
            if left_curr_idx + 1 >= len(left_min_max):
                left_terminate = True
            else:
                left_curr_idx += 1
                left_curr_min_max = list(left_min_max[left_curr_idx])
        elif left_curr_min_max[0] < right_curr_min_max[0]:
            # left min < right min
            right_min = [right_curr_min_max[0], not right_curr_min_max[1]]
            max_val = min(left_curr_min_max[2:], right_min)
            assert len(max_val) == 2
            min_max = (left_curr_min_max[0], left_curr_min_max[1],
                       max_val[0], max_val[1])
            left_idx_to_min_max[left_curr_idx].append(min_max)
            right_idx_to_min_max[right_curr_idx].append(min_max)
            if left_curr_min_max[2:] == max_val:
                # left max < right min
                if left_curr_idx + 1 >= len(left_min_max):
                    left_terminate = True
                else:
                    left_curr_idx += 1
                    left_curr_min_max = list(left_min_max[left_curr_idx])
            else:
                # from left min(left min close) to right min(exclude right min close)
                left_curr_min_max[:2] = right_curr_min_max[:2]
        elif left_curr_min_max[0] > right_curr_min_max[0]:
            # left min > right min
            left_min = [left_curr_min_max[0], not left_curr_min_max[1]]
            max_val = min(right_curr_min_max[2:], left_min)
            min_max = (right_curr_min_max[0], right_curr_min_max[1],
                       max_val[0], max_val[1])
            left_idx_to_min_max[left_curr_idx].append(min_max)
            right_idx_to_min_max[right_curr_idx].append(min_max)
            if right_curr_min_max[2:] == max_val:
                # right max < left min
                if right_curr_idx + 1 >= len(right_min_max):
                    right_terminate = True
                else:
                    right_curr_idx += 1
                    right_curr_min_max = list(right_min_max[right_curr_idx])
            else:
                # from left min(left min close) to right min(exclude right min close)
                right_curr_min_max[:2] = left_curr_min_max[:2]
        else:
            # left min == right min
            max_val = min(left_curr_min_max[2:], right_curr_min_max[2:])
            assert len(max_val) == 2
            min_max = (left_curr_min_max[0], left_curr_min_max[1], max_val[0], max_val[1])
            left_idx_to_min_max[left_curr_idx].append(min_max)
            right_idx_to_min_max[right_curr_idx].append(min_max)
            if max_val == left_curr_min_max[2:]:
                if left_curr_idx + 1 >= len(left_min_max):
                    left_terminate = True
                else:
                    left_curr_idx += 1
                    left_curr_min_max = list(left_min_max[left_curr_idx])
            else:
                left_curr_min_max[:2] = max_val[0], not max_val[1]
            if max_val == right_curr_min_max[2:]:
                if right_curr_idx + 1 >= len(right_min_max):
                    right_terminate = True
                else:
                    right_curr_idx += 1
                    right_curr_min_max = list(right_min_max[right_curr_idx])
            else:
                right_curr_min_max[:2] = max_val[0], not max_val[1]

    if not left_increase:
        left_idx_to_min_max = list(reversed(left_idx_to_min_max))
    if not right_increase:
        right_idx_to_min_max = list(reversed(right_idx_to_min_max))

    return left_idx_to_min_max, right_idx_to_min_max


def build_split_idx_to_origin_idx(splits, increase=True):
    # splits' len is equal to the original chunk size on a specified axis,
    # splits is sth like [[(0, True, 2, True), (2, False, 3, True)]]
    # which means there is one input chunk, and will be split into 2 out chunks
    # in this function, we want to build a new dict from the out chunk index to
    # the original chunk index and the inner position, like {0: (0, 0), 1: (0, 1)}
    if not increase:
        splits = list(reversed(splits))
    out_idx = itertools.count(0)
    res = dict()
    for origin_idx, _ in enumerate(splits):
        for pos in range(len(splits[origin_idx])):
            if increase:
                o_idx = origin_idx
            else:
                o_idx = len(splits) - origin_idx - 1
            res[next(out_idx)] = o_idx, pos
    return res


def build_empty_df(dtypes):
    columns = dtypes.index
    df = pd.DataFrame(columns=columns)
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df


def _filter_range_index(pd_range_index, min_val, min_val_close, max_val, max_val_close):
    if is_pd_range_empty(pd_range_index):
        return pd_range_index

    raw_min, raw_max, step = pd_range_index.min(), pd_range_index.max(), pd_range_index._step

    # seek min range
    greater_func = operator.gt if min_val_close else operator.ge
    actual_min = raw_min
    while greater_func(min_val, actual_min):
        actual_min += abs(step)
    if step < 0:
        actual_min += step  # on the right side

    # seek max range
    less_func = operator.lt if max_val_close else operator.le
    actual_max = raw_max
    while less_func(max_val, actual_max):
        actual_max -= abs(step)
    if step > 0:
        actual_max += step  # on the right side

    if step > 0:
        return pd.RangeIndex(actual_min, actual_max, step)
    return pd.RangeIndex(actual_max, actual_min, step)


def filter_index_value(index_value, min_max, store_data=False):
    min_val, min_val_close, max_val, max_val_close = min_max

    pd_index = index_value.to_pandas()

    if isinstance(index_value.value, IndexValue.RangeIndex):
        pd_filtered_index = _filter_range_index(pd_index, min_val, min_val_close,
                                                max_val, max_val_close)
        return parse_index(pd_filtered_index, store_data=store_data)

    if min_val_close:
        f = pd_index >= min_val
    else:
        f = pd_index > min_val
    if max_val_close:
        f = f & (pd_index <= max_val)
    else:
        f = f & (pd_index < max_val)

    return parse_index(pd_index[f], store_data=store_data)


def concat_tileable_chunks(df):
    from .merge.concat import DataFrameConcat

    assert not df.is_coarse()

    op = DataFrameConcat()
    chunk = DataFrameConcat().new_chunk(df.chunks, shape=df.shape, dtypes=df.dtypes,
                                        index_value=df.index_value,
                                        columns_value=df.columns)
    return op.new_dataframe([df], shape=df.shape, chunks=[chunk],
                            nsplits=tuple((s,) for s in df.shape), dtypes=df.dtypes,
                            index_value=df.index_value, columns_value=df.columns)


def get_fetch_op_cls(op):
    from ...operands import ShuffleProxy
    from .fetch import DataFrameFetchShuffle, DataFrameFetch
    if isinstance(op, ShuffleProxy):
        return DataFrameFetchShuffle
    else:
        return DataFrameFetch


def get_fuse_op_cls():
    from .core import DataFrameFuseChunk

    return DataFrameFuseChunk
