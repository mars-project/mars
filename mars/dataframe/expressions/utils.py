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

import numpy as np

from ...tensor.expressions.utils import dictify_chunk_size, normalize_chunk_sizes
from ...utils import tokenize
from ..core import IndexValue


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


def parse_index(index_value, store_data=False):
    import pandas as pd

    def _parse_property(index, ret_data):
        kw = {
            '_is_monotonic_increasing': index.is_monotonic_increasing,
            '_is_monotonic_decreasing': index.is_monotonic_decreasing,
            '_is_unique': index.is_unique,
            '_min_val': index.min(),
            '_max_val': index.max(),
            '_min_val_close': True,
            '_max_val_close': True,
            '_key': tokenize(index),
        }
        if ret_data:
            kw['_data'] = index.values
        return kw

    def _serialize_index(index):
        params = _parse_property(index, store_data)
        return getattr(IndexValue, type(index).__name__)(_name=index.name, **params)

    def _serialize_range_index(index):
        params = _parse_property(index, False)
        return IndexValue.RangeIndex(_slice=slice(index._start, index._stop, index._step),
                                     _name=index.name, **params)

    def _serialize_multi_index(index):
        kw = _parse_property(index, store_data)
        kw['_sortorder'] = index.sortorder
        return IndexValue.MultiIndex(_names=index.names, **kw)

    if isinstance(index_value, pd.RangeIndex):
        return IndexValue(_index_value=_serialize_range_index(index_value))
    elif isinstance(index_value, pd.MultiIndex):
        return IndexValue(_index_value=_serialize_multi_index(index_value))
    else:
        return IndexValue(_index_value=_serialize_index(index_value))


def rechunk_monotonic_index_min_max(*chunk_indexes_min_max):
    all_min, all_min_close, all_max, all_max_close = zip(*chunk_indexes_min_max[0])

    for chunk_index_min_max in chunk_indexes_min_max[1:]:
        for index_min, index_max in chunk_index_min_max:
            min_effect_idx = np.searchsorted(all_min, index_min)
            max_effect_idx = np.searchsorted(all_max, index_max, side='right')

            # for min, check another one, according to its max
            if min_effect_idx > 0:
                if all_max_close[min_effect_idx - 1] and \
                        all_max[min_effect_idx - 1] >= index_min:
                    min_effect_idx -= 1
                elif not all_max_close[min_effect_idx - 1] and \
                        all_max[min_effect_idx - 1] > index_min:
                    min_effect_idx -= 1

            # for max, check another one, according to its min
            if max_effect_idx < len(all_max) - 1:
                if all_min_close[max_effect_idx + 1] and \
                        all_min[max_effect_idx + 1] <= index_max:
                    max_effect_idx += 1
                elif not all_max_close[max_effect_idx + 1] and \
                        all_min[max_effect_idx + 1] < index_max:
                    max_effect_idx += 1

            if min_effect_idx == max_effect_idx:
                all_min.insert(min_effect_idx, index_min)
                all_max.insert(min_effect_idx, index_max)
                all_min_close.insert(min_effect_idx, True)
                all_max_close.insert(max_effect_idx, True)
