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

import itertools
import operator
import functools
from contextlib import contextmanager
from numbers import Integral

import numpy as np
import pandas as pd
from pandas.core.dtypes.cast import find_common_type
try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pass

from ..core import Entity, ExecutableTuple
from ..lib.mmh3 import hash as mmh_hash
from ..tensor.utils import dictify_chunk_size, normalize_chunk_sizes
from ..utils import tokenize, sbytes


def hash_index(index, size):
    def func(x, size):
        return mmh_hash(sbytes(x)) % size

    f = functools.partial(func, size=size)
    idx_to_grouped = index.groupby(index.map(f))
    return [idx_to_grouped.get(i, list()) for i in range(size)]


def hash_dataframe_on(df, on, size, level=None):
    if on is None:
        idx = df.index
        if level is not None:
            idx = idx.to_frame(False)[level]
        hashed_label = pd.util.hash_pandas_object(idx, categorize=False)
    elif callable(on):
        # todo optimization can be added, if ``on`` is a numpy ufunc or sth can be vectorized
        hashed_label = pd.util.hash_pandas_object(df.index.map(on), categorize=False)
    else:
        if isinstance(on, list):
            to_concat = []
            for v in on:
                if isinstance(v, pd.Series):
                    to_concat.append(v)
                else:
                    to_concat.append(df[v])
            data = pd.concat(to_concat, axis=1)
        else:
            data = df[on]
        hashed_label = pd.util.hash_pandas_object(data, index=False, categorize=False)
    idx_to_grouped = df.index.groupby(hashed_label % size)
    return [idx_to_grouped.get(i, pd.Index([])).unique() for i in range(size)]


def hash_dtypes(dtypes, size):
    hashed_indexes = hash_index(dtypes.index, size)
    return [dtypes[index] for index in hashed_indexes]


def sort_dataframe_inplace(df, *axis):
    for ax in axis:
        df.sort_index(axis=ax, inplace=True)
    return df


def _get_range_index_start(pd_range_index):
    try:
        return pd_range_index.start
    except AttributeError:  # pragma: no cover
        return pd_range_index._start


def _get_range_index_stop(pd_range_index):
    try:
        return pd_range_index.stop
    except AttributeError:  # pragma: no cover
        return pd_range_index._stop


def _get_range_index_step(pd_range_index):
    try:
        return pd_range_index.step
    except AttributeError:  # pragma: no cover
        return pd_range_index._step


def is_pd_range_empty(pd_range_index):
    start, stop, step = _get_range_index_start(pd_range_index), \
                        _get_range_index_stop(pd_range_index), \
                        _get_range_index_step(pd_range_index)
    return (start >= stop and step >= 0) or (start <= stop and step < 0)


def decide_dataframe_chunk_sizes(shape, chunk_size, memory_usage):
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
    from ..config import options

    chunk_size = dictify_chunk_size(shape, chunk_size)
    average_memory_usage = memory_usage / shape[0]

    nleft = len(shape) - len(chunk_size)
    if nleft < 0:
        raise ValueError("chunks have more than two dimensions")
    if nleft == 0:
        return normalize_chunk_sizes(shape, tuple(chunk_size[j] for j in range(len(shape))))

    max_chunk_size = options.chunk_store_limit

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
        col_chunk_store = [average_memory_usage[acc[i]: acc[i + 1]].sum()
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
            col_chunk_store.append(average_memory_usage.iloc[start: start + cs].sum())
            col_left_size -= cs
        if row_left_size > 0:
            max_col_chunk_store = max(col_chunk_store)
            cs = min(row_left_size, int(max_chunk_size / max_col_chunk_store))
            row_chunk_size.append(cs)
            row_left_size -= cs

        if col_left_size == 0 and row_left_size == 0:
            break

    return tuple(row_chunk_size), tuple(col_chunk_size)


def decide_series_chunk_size(shape, chunk_size, memory_usage):
    from ..config import options

    chunk_size = dictify_chunk_size(shape, chunk_size)
    average_memory_usage = memory_usage / shape[0]

    if len(chunk_size) == len(shape):
        return normalize_chunk_sizes(shape, chunk_size[0])

    max_chunk_size = options.chunk_store_limit
    series_chunk_size = max_chunk_size / average_memory_usage
    return normalize_chunk_sizes(shape, int(series_chunk_size))


def parse_index(index_value, *args, store_data=False, key=None):
    from .core import IndexValue

    def _extract_property(index, tp, ret_data):
        kw = {
            '_min_val': _get_index_min(index),
            '_max_val': _get_index_max(index),
            '_min_val_close': True,
            '_max_val_close': True,
            '_key': key or _tokenize_index(index, *args),
        }
        if ret_data:
            kw['_data'] = index.values
        for field in tp._FIELDS:
            if field in kw or field == '_data':
                continue
            val = getattr(index, field.lstrip('_'), None)
            if val is not None:
                kw[field] = val
        return kw

    def _tokenize_index(index, *token_objects):
        if not index.empty:
            return tokenize(index)
        else:
            return tokenize(index, *token_objects)

    def _get_index_min(index):
        try:
            return index.min()
        except TypeError:
            return None

    def _get_index_max(index):
        try:
            return index.max()
        except TypeError:
            return None

    def _serialize_index(index):
        tp = getattr(IndexValue, type(index).__name__)
        properties = _extract_property(index, tp, store_data)
        return tp(**properties)

    def _serialize_range_index(index):
        if is_pd_range_empty(index):
            properties = {
                '_is_monotonic_increasing': True,
                '_is_monotonic_decreasing': False,
                '_is_unique': True,
                '_min_val': _get_index_min(index),
                '_max_val': _get_index_max(index),
                '_min_val_close': True,
                '_max_val_close': False,
                '_key': key or _tokenize_index(index, *args),
                '_name': index.name,
                '_dtype': index.dtype,
            }
        else:
            properties = _extract_property(index, IndexValue.RangeIndex, False)
        return IndexValue.RangeIndex(_slice=slice(_get_range_index_start(index),
                                                  _get_range_index_stop(index),
                                                  _get_range_index_step(index)),
                                     **properties)

    def _serialize_multi_index(index):
        kw = _extract_property(index, IndexValue.MultiIndex, store_data)
        kw['_sortorder'] = index.sortorder
        return IndexValue.MultiIndex(**kw)

    if index_value is None:
        return IndexValue(_index_value=IndexValue.Index(
            _is_monotonic_increasing=False,
            _is_monotonic_decreasing=False,
            _is_unique=False,
            _min_val=None,
            _max_val=None,
            _min_val_close=True,
            _max_val_close=True,
            _key=key or tokenize(*args),
        ))
    if isinstance(index_value, pd.RangeIndex):
        return IndexValue(_index_value=_serialize_range_index(index_value))
    elif isinstance(index_value, pd.MultiIndex):
        return IndexValue(_index_value=_serialize_multi_index(index_value))
    else:
        return IndexValue(_index_value=_serialize_index(index_value))


def gen_unknown_index_value(index_value, *args):
    pd_index = index_value.to_pandas()
    if isinstance(pd_index, pd.RangeIndex):
        return parse_index(pd.RangeIndex(-1), *args)
    elif not isinstance(pd_index, pd.MultiIndex):
        return parse_index(pd.Index([], dtype=pd_index.dtype), *args)
    else:
        i = pd.MultiIndex.from_arrays([c[:0] for c in pd_index.levels],
                                      names=pd_index.names)
        return parse_index(i, *args)


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
    ...                 (20, True, 22, True)]
    >>> right_min_max = [(2, True, 6, True), (7, True, 9, True), (10, True, 14, True),
    ...                  (18, True, 19, True)]
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

    if left_increase is False:
        left_idx_to_min_max = list(reversed(left_idx_to_min_max))
    if right_increase is False:
        right_idx_to_min_max = list(reversed(right_idx_to_min_max))

    return left_idx_to_min_max, right_idx_to_min_max


def build_split_idx_to_origin_idx(splits, increase=True):
    # splits' len is equal to the original chunk size on a specified axis,
    # splits is sth like [[(0, True, 2, True), (2, False, 3, True)]]
    # which means there is one input chunk, and will be split into 2 out chunks
    # in this function, we want to build a new dict from the out chunk index to
    # the original chunk index and the inner position, like {0: (0, 0), 1: (0, 1)}
    if increase is False:
        splits = list(reversed(splits))
    out_idx = itertools.count(0)
    res = dict()
    for origin_idx, _ in enumerate(splits):
        for pos in range(len(splits[origin_idx])):
            if increase is False:
                o_idx = len(splits) - origin_idx - 1
            else:
                o_idx = origin_idx
            res[next(out_idx)] = o_idx, pos
    return res


def _generate_value(dtype, fill_value):
    # special handle for datetime64 and timedelta64
    dispatch = {
        np.datetime64: pd.Timestamp,
        np.timedelta64: pd.Timedelta,
    }
    # otherwise, just use dtype.type itself to convert
    convert = dispatch.get(dtype.type, dtype.type)
    return convert(fill_value)


def build_empty_df(dtypes, index=None):
    columns = dtypes.index
    # duplicate column may exist,
    # so use RangeIndex first
    df = pd.DataFrame(columns=pd.RangeIndex(len(columns)), index=index)
    for i, d in enumerate(dtypes):
        df[i] = pd.Series(dtype=d, index=index)
    df.columns = columns
    return df


def build_df(df_obj, fill_value=1, size=1):
    empty_df = build_empty_df(df_obj.dtypes, index=df_obj.index_value.to_pandas()[:0])
    dtypes = empty_df.dtypes
    record = [_generate_value(dtype, fill_value) for dtype in empty_df.dtypes]
    if isinstance(empty_df.index, pd.MultiIndex):
        index = tuple(_generate_value(level.dtype, fill_value) for level in empty_df.index.levels)
        empty_df.loc[index, ] = record
    else:
        index = _generate_value(empty_df.index.dtype, fill_value)
        empty_df.loc[index] = record

    empty_df = pd.concat([empty_df] * size)
    # make sure dtypes correct for MultiIndex
    for i, dtype in enumerate(dtypes.tolist()):
        s = empty_df.iloc[:, i]
        if s.dtype != dtype:
            empty_df.iloc[:, i] = s.astype(dtype)
    return empty_df


def build_empty_series(dtype, index=None, name=None):
    return pd.Series(dtype=dtype, index=index, name=name)


def build_series(series_obj, fill_value=1, size=1):
    empty_series = build_empty_series(series_obj.dtype, index=series_obj.index_value.to_pandas()[:0])
    record = _generate_value(series_obj.dtype, fill_value)
    if isinstance(empty_series.index, pd.MultiIndex):
        index = tuple(_generate_value(level.dtype, fill_value) for level in empty_series.index.levels)
        empty_series.loc[index, ] = record
    else:
        if isinstance(empty_series.index.dtype, pd.CategoricalDtype):
            index = None
        else:
            index = _generate_value(empty_series.index.dtype, fill_value)
        empty_series.loc[index] = record

    empty_series = pd.concat([empty_series] * size)
    # make sure dtype correct for MultiIndex
    empty_series = empty_series.astype(series_obj.dtype, copy=False)
    return empty_series


def concat_index_value(index_values, store_data=False):
    result = pd.Index([])
    if not isinstance(index_values, (list, tuple)):
        index_values = [index_values]
    for index_value in index_values:
        if isinstance(index_value, pd.Index):
            result = result.append(index_value)
        else:
            result = result.append(index_value.to_pandas())
    return parse_index(result, store_data=store_data)


def build_concatenated_rows_frame(df):
    from .operands import ObjectType
    from .merge.concat import DataFrameConcat

    # When the df isn't splitted along the column axis, return the df directly.
    if df.chunk_shape[1] == 1:
        return df

    columns = concat_index_value([df.cix[0, idx].columns_value for idx in range(df.chunk_shape[1])],
                                 store_data=True)
    columns_size = columns.to_pandas().size

    out_chunks = []
    for idx in range(df.chunk_shape[0]):
        out_chunk = DataFrameConcat(axis=1, object_type=ObjectType.dataframe).new_chunk(
            [df.cix[idx, k] for k in range(df.chunk_shape[1])], index=(idx, 0),
            shape=(df.cix[idx, 0].shape[0], columns_size), dtypes=df.dtypes,
            index_value=df.cix[idx, 0].index_value, columns_value=columns)
        out_chunks.append(out_chunk)

    return DataFrameConcat(axis=1, object_type=ObjectType.dataframe).new_dataframe(
        [df], chunks=out_chunks, nsplits=((chunk.shape[0] for chunk in out_chunks), (df.shape[1],)),
        shape=df.shape, dtypes=df.dtypes,
        index_value=df.index_value, columns_value=df.columns_value)


def _filter_range_index(pd_range_index, min_val, min_val_close, max_val, max_val_close):
    if is_pd_range_empty(pd_range_index):
        return pd_range_index

    raw_min, raw_max, step = pd_range_index.min(), pd_range_index.max(), _get_range_index_step(pd_range_index)

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


def infer_index_value(left_index_value, right_index_value):
    from .core import IndexValue

    if isinstance(left_index_value.value, IndexValue.RangeIndex) and \
            isinstance(right_index_value.value, IndexValue.RangeIndex):
        if left_index_value.value.slice == right_index_value.value.slice:
            return left_index_value
        return parse_index(pd.Int64Index([]), left_index_value, right_index_value)

    # when left index and right index is identical, and both of them are elements unique,
    # we can infer that the out index should be identical also
    if left_index_value.is_unique and right_index_value.is_unique and \
            left_index_value.key == right_index_value.key:
        return left_index_value

    left_index = left_index_value.to_pandas()
    right_index = right_index_value.to_pandas()
    out_index = pd.Index([], dtype=find_common_type([left_index.dtype, right_index.dtype]))
    return parse_index(out_index, left_index_value, right_index_value)


def filter_index_value(index_value, min_max, store_data=False):
    from .core import IndexValue

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


def indexing_index_value(index_value, indexes, store_data=False):
    pd_index = index_value.to_pandas()
    if not index_value.has_value():
        new_index_value = parse_index(pd_index, indexes, store_data=store_data)
        new_index_value._index_value._min_val = index_value.min_val
        new_index_value._index_value._min_val_close = index_value.min_val_close
        new_index_value._index_value._max_val = index_value.max_val
        new_index_value._index_value._max_val_close = index_value.max_val_close
        return new_index_value
    else:
        if isinstance(indexes, Integral):
            return parse_index(pd_index[[indexes]], store_data=store_data)
        elif isinstance(indexes, Entity):
            if isinstance(pd_index, pd.RangeIndex):
                return parse_index(
                    pd.RangeIndex(-1), indexes, index_value, store_data=False)
            else:
                return parse_index(
                    type(pd_index)([]), indexes, index_value, store_data=False)
        if isinstance(indexes, tuple):
            return parse_index(pd_index[list(indexes)], store_data=store_data)
        else:
            return parse_index(pd_index[indexes], store_data=store_data)


def merge_index_value(to_merge_index_values, store_data=False):
    """
    Merge index value according to their chunk index.
    :param to_merge_index_values: Dict object. {index: index_value}
    :return: Merged index_value
    """
    index_value = None
    min_val, min_val_close, max_val, max_val_close = None, None, None, None
    for _, chunk_index_value in sorted(to_merge_index_values.items()):
        if index_value is None:
            index_value = chunk_index_value.to_pandas()
            min_val, min_val_close, max_val, max_val_close = \
                chunk_index_value.min_val, \
                chunk_index_value.min_val_close, \
                chunk_index_value.max_val, \
                chunk_index_value.max_val_close
        else:
            index_value = index_value.append(chunk_index_value.to_pandas())
            if chunk_index_value.min_val is not None:
                if min_val is None or min_val > chunk_index_value.min_val:
                    min_val = chunk_index_value.min_val
                    min_val_close = chunk_index_value.min_val_close
            if chunk_index_value.max_val is not None:
                if max_val is None or max_val < chunk_index_value.max_val:
                    max_val = chunk_index_value.max_val
                    max_val_close = chunk_index_value.max_val_close

    new_index_value = parse_index(index_value, store_data=store_data)
    if not new_index_value.has_value():
        new_index_value._index_value._min_val = min_val
        new_index_value._index_value._min_val_close = min_val_close
        new_index_value._index_value._max_val = max_val
        new_index_value._index_value._max_val_close = max_val_close
    return new_index_value


def infer_dtypes(left_dtypes, right_dtypes, operator):
    left = build_empty_df(left_dtypes)
    right = build_empty_df(right_dtypes)
    return operator(left, right).dtypes


def infer_dtype(left_dtype, right_dtype, operator):
    left = build_empty_series(left_dtype)
    right = build_empty_series(right_dtype)
    return operator(left, right).dtype


def filter_dtypes(dtypes, column_min_max):
    left_filter = operator.ge if column_min_max[1] else operator.gt
    left = left_filter(dtypes.index, column_min_max[0])
    right_filter = operator.le if column_min_max[3] else operator.lt
    right = right_filter(dtypes.index, column_min_max[2])
    return dtypes[left & right]


def in_range_index(i, pd_range_index):
    """
    Check whether the input `i` is within `pd_range_index` which is a pd.RangeIndex.
    """
    start, stop, step = _get_range_index_start(pd_range_index), \
        _get_range_index_stop(pd_range_index), \
        _get_range_index_step(pd_range_index)
    if step > 0 and start <= i < stop and (i - start) % step == 0:
        return True
    if step < 0 and start >= i > stop and (start - i) % step == 0:
        return True
    return False


def wrap_notimplemented_exception(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotImplementedError:
            return NotImplemented

    return wrapper


def validate_axis(axis, tileable=None):
    if axis == 'index':
        axis = 0
    elif axis == 'columns':
        axis = 1

    illegal = False
    try:
        axis = operator.index(axis)
        if axis < 0 or (tileable is not None and axis >= tileable.ndim):
            illegal = True
    except TypeError:
        illegal = True

    if illegal:
        raise ValueError('No axis named {} for '
                         'object type {}'.format(axis, type(tileable)))
    return axis


def standardize_range_index(chunks, axis=0):
    from .base.standardize_range_index import ChunkStandardizeRangeIndex

    row_chunks = dict((k, next(v)) for k, v in itertools.groupby(chunks, key=lambda x: x.index[axis]))
    row_chunks = [row_chunks[i] for i in range(len(row_chunks))]

    out_chunks = []
    for c in chunks:
        inputs = row_chunks[:c.index[axis]] + [c]
        op = ChunkStandardizeRangeIndex(
            prepare_inputs=[False] * (len(inputs) - 1) + [True], axis=axis, object_type=c.op.object_type)
        out_chunks.append(op.new_chunk(inputs, **c.params.copy()))

    return out_chunks


def fetch_corner_data(df_or_series, session=None) -> pd.DataFrame:
    """
    Fetch corner DataFrame or Series for repr usage.

    :param df_or_series: DataFrame or Series
    :return: corner DataFrame
    """
    from .indexing.iloc import iloc

    max_rows = pd.get_option('display.max_rows')
    try:
        min_rows = pd.get_option('display.min_rows')
        min_rows = min(min_rows, max_rows)
    except KeyError:  # pragma: no cover
        # display.min_rows is introduced in pandas 0.25
        min_rows = max_rows

    index_size = None
    if df_or_series.shape[0] > max_rows and \
            df_or_series.shape[0] > min_rows // 2 * 2 + 2:
        # for pandas, greater than max_rows
        # will display min_rows
        # thus we fetch min_rows + 2 lines
        index_size = min_rows // 2 + 1

    if index_size is None:
        return df_or_series.fetch(session=session)
    else:
        head = iloc(df_or_series)[:index_size]
        tail = iloc(df_or_series)[-index_size:]
        head_data, tail_data = \
            ExecutableTuple([head, tail]).fetch(session=session)
        return pd.concat([head_data, tail_data], axis='index')


class ReprSeries(pd.Series):
    def __init__(self, corner_data, real_shape):
        super().__init__(corner_data)
        self._real_shape = real_shape

    def __len__(self):
        # As we only fetch corner data to repr,
        # the length would be wrong and we have no way to control,
        # thus we just overwrite the length to show the real one
        return self._real_shape[0]


@contextmanager
def create_sa_connection(con, **kwargs):
    import sqlalchemy as sa
    from sqlalchemy.engine import Connection, Engine

    # process con
    engine = None
    if isinstance(con, Connection):
        # connection create by user
        close = False
        dispose = False
    elif isinstance(con, Engine):
        con = con.connect()
        close = True
        dispose = False
    else:
        engine = sa.create_engine(con, **kwargs)
        con = engine.connect()
        close = True
        dispose = True

    try:
        yield con
    finally:
        if close:
            con.close()
        if dispose:
            engine.dispose()


def arrow_table_to_pandas_dataframe(arrow_table, use_arrow_dtype=True, **kw):
    if not use_arrow_dtype:
        # if not use arrow string, just return
        return arrow_table.to_pandas(**kw)

    from .arrays import ArrowStringArray

    table = arrow_table
    schema = arrow_table.schema

    string_field_names = list()
    string_arrays = list()
    string_indexes = list()
    other_field_names = list()
    other_arrays = list()
    for i, arrow_type in enumerate(schema.types):
        if arrow_type == pa.string():
            string_field_names.append(schema.names[i])
            string_indexes.append(i)
            string_arrays.append(table.columns[i])
        else:
            other_field_names.append(schema.names[i])
            other_arrays.append(table.columns[i])

    df = pa.Table.from_arrays(
        other_arrays, names=other_field_names).to_pandas(**kw)
    for string_index, string_name, string_array in \
            zip(string_indexes, string_field_names, string_arrays):
        df.insert(string_index, string_name,
                  pd.Series(ArrowStringArray(string_array)))

    return df
