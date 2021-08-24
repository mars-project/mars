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

import functools
import itertools
import operator
from contextlib import contextmanager
from numbers import Integral

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.extensions import ExtensionDtype
from pandas.core.dtypes.cast import find_common_type

from ..core import Entity, ExecutableTuple
from ..lib.mmh3 import hash as mmh_hash
from ..tensor.utils import dictify_chunk_size, normalize_chunk_sizes
from ..utils import tokenize, sbytes, lazy_import, ModulePlaceholder

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = ModulePlaceholder('pyarrow')

cudf = lazy_import('cudf', globals=globals(), rename='cudf')


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
        if cudf and isinstance(idx, cudf.Index):  # pragma: no cover
            idx = idx.to_pandas()
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
    idx_to_grouped = pd.RangeIndex(0, len(hashed_label)).groupby(hashed_label % size)
    return [idx_to_grouped.get(i, pd.Index([])) for i in range(size)]


def hash_dtypes(dtypes, size):
    hashed_indexes = hash_index(dtypes.index, size)
    return [dtypes[index] for index in hashed_indexes]


def sort_dataframe_inplace(df, *axis):
    for ax in axis:
        df.sort_index(axis=ax, inplace=True)
    return df


@functools.lru_cache(1)
def _get_range_index_type():
    if cudf is not None:
        return pd.RangeIndex, cudf.RangeIndex
    else:
        return pd.RangeIndex


@functools.lru_cache(1)
def _get_multi_index_type():
    if cudf is not None:
        return pd.MultiIndex, cudf.MultiIndex
    else:
        return pd.MultiIndex


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
        pass
    try:  # pragma: no cover
        return pd_range_index._step
    except AttributeError:  # pragma: no cover
        return 1  # cudf does not support step arg


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
        row_left_size = -1
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
        col_left_size = -1

    while True:
        nbytes_occupied = np.prod([max(it) for it in (row_chunk_size, col_chunk_store) if it])
        dim_size = np.maximum(int(np.power(max_chunk_size / nbytes_occupied, 1 / float(nleft))), 1)

        if col_left_size == 0:
            col_chunk_size.append(0)

        if row_left_size == 0:
            row_chunk_size.append(0)

        # check col first
        if col_left_size > 0:
            cs = min(col_left_size, dim_size)
            col_chunk_size.append(cs)
            start = int(np.sum(col_chunk_size[:-1]))
            col_chunk_store.append(average_memory_usage.iloc[start: start + cs].sum())
            col_left_size -= cs
        if row_left_size > 0:
            if col_chunk_store:
                max_col_chunk_store = max(col_chunk_store)
                cs = min(row_left_size, int(max_chunk_size / max_col_chunk_store))
            else:
                cs = row_left_size
            row_chunk_size.append(cs)
            row_left_size -= cs

        if col_left_size <= 0 and row_left_size <= 0:
            break

    return tuple(row_chunk_size), tuple(col_chunk_size)


def decide_series_chunk_size(shape, chunk_size, memory_usage):
    from ..config import options

    chunk_size = dictify_chunk_size(shape, chunk_size)
    average_memory_usage = memory_usage / shape[0] if shape[0] != 0 else memory_usage

    if len(chunk_size) == len(shape):
        return normalize_chunk_sizes(shape, chunk_size[0])

    if all(s == 0 for s in shape):
        # skip when shape is 0
        return tuple((s,) for s in shape)

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
        except (ValueError, AttributeError):
            if isinstance(index, pd.IntervalIndex):
                return None
            raise
        except TypeError:
            return None

    def _get_index_max(index):
        try:
            return index.max()
        except (ValueError, AttributeError):
            if isinstance(index, pd.IntervalIndex):
                return None
            raise
        except TypeError:
            return None

    def _serialize_index(index):
        tp = getattr(IndexValue, type(index).__name__)
        properties = _extract_property(index, tp, store_data)
        properties['_name'] = index.name
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
        kw['_dtypes'] = [lev.dtype for lev in index.levels]
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
    if hasattr(index_value, 'to_pandas'):  # pragma: no cover
        # convert cudf.Index to pandas
        index_value = index_value.to_pandas()

    if isinstance(index_value, _get_range_index_type()):
        return IndexValue(_index_value=_serialize_range_index(index_value))
    elif isinstance(index_value, _get_multi_index_type()):
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
        pd.CategoricalDtype.type: lambda x: pd.CategoricalDtype([x]),
        # for object, we do not know the actual dtype,
        # just convert to str for common usage
        np.object_: lambda x: str(fill_value),
    }
    # otherwise, just use dtype.type itself to convert
    convert = dispatch.get(dtype.type, dtype.type)
    return convert(fill_value)


def build_empty_df(dtypes, index=None):
    columns = dtypes.index
    length = len(index) if index is not None else 0
    record = [[_generate_value(dtype, 1) for dtype in dtypes]] * max(1, length)

    # duplicate column may exist,
    # so use RangeIndex first
    df = pd.DataFrame(record, columns=range(len(dtypes)), index=index)
    for i, dtype in enumerate(dtypes):
        s = df.iloc[:, i]
        if not pd.api.types.is_dtype_equal(s.dtype, dtype):
            df.iloc[:, i] = s.astype(dtype)

    df.columns = columns
    return df[:length] if len(df) > length else df


def build_df(df_obj, fill_value=1, size=1, ensure_string=False):
    dfs = []
    if not isinstance(size, (list, tuple)):
        sizes = [size]
    else:
        sizes = size

    if not isinstance(fill_value, (list, tuple)):
        fill_values = [fill_value]
    else:
        fill_values = fill_value

    for size, fill_value in zip(sizes, fill_values):
        dtypes = df_obj.dtypes
        record = [[_generate_value(dtype, fill_value) for dtype in dtypes]] * size
        df = pd.DataFrame(record)
        df.columns = dtypes.index

        if len(record) != 0:  # columns is empty in some cases
            target_index = df_obj.index_value.to_pandas()
            if isinstance(target_index, pd.MultiIndex):
                index_val = tuple(_generate_value(level.dtype, fill_value)
                                  for level in target_index.levels)
                df.index = pd.MultiIndex.from_tuples([index_val] * size, names=target_index.names)
            else:
                index_val = _generate_value(target_index.dtype, fill_value)
                df.index = pd.Index([index_val] * size, name=target_index.name)

        # make sure dtypes correct
        for i, dtype in enumerate(dtypes):
            s = df.iloc[:, i]
            if not pd.api.types.is_dtype_equal(s.dtype, dtype):
                df.iloc[:, i] = s.astype(dtype)
        dfs.append(df)
    if len(dfs) == 1:
        ret_df = dfs[0]
    else:
        ret_df = pd.concat(dfs)

    if ensure_string:
        obj_dtypes = df_obj.dtypes[df_obj.dtypes == np.dtype('O')]
        ret_df[obj_dtypes.index] = ret_df[obj_dtypes.index].radd('O')
    return ret_df


def build_empty_series(dtype, index=None, name=None):
    length = len(index) if index is not None else 0
    return pd.Series([_generate_value(dtype, 1) for _ in range(length)],
                     dtype=dtype, index=index, name=name)


def build_series(series_obj, fill_value=1, size=1, name=None, ensure_string=False):
    seriess = []
    if not isinstance(size, (list, tuple)):
        sizes = [size]
    else:
        sizes = size

    if not isinstance(fill_value, (list, tuple)):
        fill_values = [fill_value]
    else:
        fill_values = fill_value

    for size, fill_value in zip(sizes, fill_values):
        empty_series = build_empty_series(series_obj.dtype, name=name,
                                          index=series_obj.index_value.to_pandas()[:0])
        record = _generate_value(series_obj.dtype, fill_value)
        if isinstance(empty_series.index, pd.MultiIndex):
            index = tuple(_generate_value(level.dtype, fill_value) for level in empty_series.index.levels)
            empty_series = empty_series.reindex(
                index=pd.MultiIndex.from_tuples([index], names=empty_series.index.names))
            empty_series.iloc[0] = record
        else:
            if isinstance(empty_series.index.dtype, pd.CategoricalDtype):
                index = None
            else:
                index = _generate_value(empty_series.index.dtype, fill_value)
            empty_series.loc[index] = record

        empty_series = pd.concat([empty_series] * size)
        # make sure dtype correct for MultiIndex
        empty_series = empty_series.astype(series_obj.dtype, copy=False)
        seriess.append(empty_series)

    if len(seriess) == 1:
        ret_series = seriess[0]
    else:
        ret_series = pd.concat(seriess)

    if ensure_string and series_obj.dtype == np.dtype('O'):
        ret_series = ret_series.radd('O')
    return ret_series


def concat_index_value(index_values, store_data=False):
    if not isinstance(index_values, (list, tuple)):
        index_values = [index_values]
    result = index_values[0]
    if not isinstance(result, pd.Index):
        result = result.to_pandas()
    for index_value in index_values[1:]:
        if isinstance(index_value, pd.Index):
            result = result.append(index_value)
        else:
            result = result.append(index_value.to_pandas())
    return parse_index(result, store_data=store_data)


def build_concatenated_rows_frame(df):
    from ..core import OutputType
    from .merge.concat import DataFrameConcat

    # When the df isn't split along the column axis, return the df directly.
    if df.chunk_shape[1] == 1:
        return df

    columns = concat_index_value([df.cix[0, idx].columns_value for idx in range(df.chunk_shape[1])],
                                 store_data=True)
    columns_size = columns.to_pandas().size

    out_chunks = []
    for idx in range(df.chunk_shape[0]):
        out_chunk = DataFrameConcat(axis=1, output_types=[OutputType.dataframe]).new_chunk(
            [df.cix[idx, k] for k in range(df.chunk_shape[1])], index=(idx, 0),
            shape=(df.cix[idx, 0].shape[0], columns_size), dtypes=df.dtypes,
            index_value=df.cix[idx, 0].index_value, columns_value=columns)
        out_chunks.append(out_chunk)

    return DataFrameConcat(axis=1, output_types=[OutputType.dataframe]).new_dataframe(
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


def merge_index_value(to_merge_index_values: dict, store_data: bool = False):
    """
    Merge index value according to their chunk index.

    Parameters
    ----------
    to_merge_index_values : dict
        index to index_value
    store_data : bool
        store data in index_value

    Returns
    -------
    merged_index_value
    """

    pd_index = None
    min_val, min_val_close, max_val, max_val_close = None, None, None, None
    for _, chunk_index_value in sorted(to_merge_index_values.items()):
        if pd_index is None:
            pd_index = chunk_index_value.to_pandas()
            min_val, min_val_close, max_val, max_val_close = \
                chunk_index_value.min_val, \
                chunk_index_value.min_val_close, \
                chunk_index_value.max_val, \
                chunk_index_value.max_val_close
        else:
            cur_pd_index = chunk_index_value.to_pandas()
            if store_data or (
                    isinstance(pd_index, pd.RangeIndex) and
                    isinstance(cur_pd_index, pd.RangeIndex) and
                    cur_pd_index.step == pd_index.step and
                    cur_pd_index.start == pd_index.stop
            ):
                # range index that is continuous
                pd_index = pd_index.append(cur_pd_index)
            else:
                pd_index = pd.Index([], dtype=pd_index.dtype)
            if chunk_index_value.min_val is not None:
                try:
                    if min_val is None or min_val > chunk_index_value.min_val:
                        min_val = chunk_index_value.min_val
                        min_val_close = chunk_index_value.min_val_close
                except TypeError:
                    # min_value has different types that cannot compare
                    # just stop compare
                    continue
            if chunk_index_value.max_val is not None:
                if max_val is None or max_val < chunk_index_value.max_val:
                    max_val = chunk_index_value.max_val
                    max_val_close = chunk_index_value.max_val_close

    index_value = parse_index(pd_index, store_data=store_data)
    if not index_value.has_value():
        index_value._index_value._min_val = min_val
        index_value._index_value._min_val_close = min_val_close
        index_value._index_value._max_val = max_val
        index_value._index_value._max_val_close = max_val_close
    return index_value


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
        raise ValueError(f'No axis named {axis} for object type {type(tileable)}')
    return axis


def validate_axis_style_args(data, args, kwargs, arg_name, method_name):  # pragma: no cover
    """Argument handler for mixed index, columns / axis functions

    In an attempt to handle both `.method(index, columns)`, and
    `.method(arg, axis=.)`, we have to do some bad things to argument
    parsing. This translates all arguments to `{index=., columns=.}` style.

    Parameters
    ----------
    data : DataFrame
    args : tuple
        All positional arguments from the user
    kwargs : dict
        All keyword arguments from the user
    arg_name, method_name : str
        Used for better error messages

    Returns
    -------
    kwargs : dict
        A dictionary of keyword arguments. Doesn't modify ``kwargs``
        inplace, so update them with the return value here.
    """
    out = {}
    # Goal: fill 'out' with index/columns-style arguments
    # like out = {'index': foo, 'columns': bar}

    # Start by validating for consistency
    axes_names = ['index'] if data.ndim == 1 else ['index', 'columns']
    if "axis" in kwargs and any(x in kwargs for x in axes_names):
        msg = "Cannot specify both 'axis' and any of 'index' or 'columns'."
        raise TypeError(msg)

    # First fill with explicit values provided by the user...
    if arg_name in kwargs:
        if args:
            msg = f"{method_name} got multiple values for argument '{arg_name}'"
            raise TypeError(msg)

        axis = axes_names[validate_axis(kwargs.get("axis", 0), data)]
        out[axis] = kwargs[arg_name]

    # More user-provided arguments, now from kwargs
    for k, v in kwargs.items():
        try:
            ax = axes_names[validate_axis(k, data)]
        except ValueError:
            pass
        else:
            out[ax] = v

    # All user-provided kwargs have been handled now.
    # Now we supplement with positional arguments, emitting warnings
    # when there's ambiguity and raising when there's conflicts

    if len(args) == 0:
        pass  # It's up to the function to decide if this is valid
    elif len(args) == 1:
        axis = axes_names[validate_axis(kwargs.get("axis", 0), data)]
        out[axis] = args[0]
    elif len(args) == 2:
        if "axis" in kwargs:
            # Unambiguously wrong
            msg = "Cannot specify both 'axis' and any of 'index' or 'columns'"
            raise TypeError(msg)

        msg = (
            "Interpreting call\n\t'.{method_name}(a, b)' as "
            "\n\t'.{method_name}(index=a, columns=b)'.\nUse named "
            "arguments to remove any ambiguity."
        )
        raise TypeError(msg.format(method_name=method_name))
    else:
        msg = f"Cannot specify all of '{arg_name}', 'index', 'columns'."
        raise TypeError(msg)
    return out


def validate_output_types(**kwargs):
    from ..core import OutputType

    output_type = kwargs.pop('object_type', None) or kwargs.pop('output_type', None)
    output_types = kwargs.pop('output_types', None) \
        or ([output_type] if output_type is not None else None)
    return [getattr(OutputType, v.lower()) if isinstance(v, str) else v for v in output_types] \
        if output_types else None


def standardize_range_index(chunks, axis=0):
    from .base.standardize_range_index import ChunkStandardizeRangeIndex

    row_chunks = dict((k, next(v)) for k, v in itertools.groupby(chunks, key=lambda x: x.index[axis]))
    row_chunks = [row_chunks[i] for i in range(len(row_chunks))]

    out_chunks = []
    for c in chunks:
        inputs = row_chunks[:c.index[axis]] + [c]
        op = ChunkStandardizeRangeIndex(
            pure_depends=[True] * (len(inputs) - 1) + [False], axis=axis, output_types=c.op.output_types)
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
        return df_or_series._fetch(session=session)
    else:
        head = iloc(df_or_series)[:index_size]
        tail = iloc(df_or_series)[-index_size:]
        head_data, tail_data = \
            ExecutableTuple([head, tail]).fetch(session=session)
        xdf = cudf if head.op.is_gpu() else pd
        return xdf.concat([head_data, tail_data], axis='index')


class ReprSeries(pd.Series):
    def __init__(self, corner_data, real_shape):
        super().__init__(corner_data)
        self._real_shape = real_shape

    def __len__(self):
        # As we only fetch corner data to repr,
        # the length would be wrong and we have no way to control,
        # thus we just overwrite the length to show the real one
        return self._real_shape[0]


def filter_dtypes_by_index(dtypes, index):
    try:
        new_dtypes = dtypes.loc[index].dropna()
    except KeyError:
        dtypes_idx = dtypes.index.to_frame().merge(index.to_frame()) \
            .set_index(list(range(dtypes.index.nlevels))).index
        new_dtypes = dtypes.loc[dtypes_idx]
        new_dtypes.index.names = dtypes.index.names
    return new_dtypes


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

    from .arrays import ArrowStringArray, ArrowListArray

    table: pa.Table = arrow_table
    schema: pa.Schema = arrow_table.schema

    arrow_field_names = list()
    arrow_arrays = list()
    arrow_indexes = list()
    other_field_names = list()
    other_arrays = list()
    for i, arrow_type in enumerate(schema.types):
        if arrow_type == pa.string() or isinstance(arrow_type, pa.ListType):
            arrow_field_names.append(schema.names[i])
            arrow_indexes.append(i)
            arrow_arrays.append(table.columns[i])
        else:
            other_field_names.append(schema.names[i])
            other_arrays.append(table.columns[i])

    df: pd.DataFrame = pa.Table.from_arrays(
        other_arrays, names=other_field_names).to_pandas(**kw)
    for arrow_index, arrow_name, arrow_array in \
            zip(arrow_indexes, arrow_field_names, arrow_arrays):
        if arrow_array.type == pa.string():
            series = pd.Series(ArrowStringArray(arrow_array))
        else:
            assert isinstance(arrow_array.type, pa.ListType)
            series = pd.Series(ArrowListArray(arrow_array))
        df.insert(arrow_index, arrow_name, series)

    return df


def contain_arrow_dtype(dtypes):
    from .arrays import ArrowStringDtype

    return any(isinstance(dtype, ArrowStringDtype) for dtype in dtypes)


def to_arrow_dtypes(dtypes, test_df=None):
    from .arrays import ArrowStringDtype

    new_dtypes = dtypes.copy()
    for i in range(len(dtypes)):
        dtype = dtypes.iloc[i]
        if is_string_dtype(dtype):
            if test_df is not None:
                series = test_df.iloc[:, i]
                # check value
                non_na_series = series[series.notna()]
                if len(non_na_series) > 0:
                    first_value = non_na_series.iloc[0]
                    if isinstance(first_value, str):
                        new_dtypes.iloc[i] = ArrowStringDtype()
                else:  # pragma: no cover
                    # empty, set arrow string dtype
                    new_dtypes.iloc[i] = ArrowStringDtype()
            else:
                # empty, set arrow string dtype
                new_dtypes.iloc[i] = ArrowStringDtype()
    return new_dtypes


def make_dtype(dtype):
    if isinstance(dtype, (np.dtype, ExtensionDtype)):
        return dtype
    return np.dtype(dtype) if dtype is not None else None


def make_dtypes(dtypes):
    if dtypes is None:
        return None
    if not isinstance(dtypes, pd.Series):
        if isinstance(dtypes, dict):
            dtypes = pd.Series(dtypes.values(), index=dtypes.keys())
        else:
            dtypes = pd.Series(dtypes)
    return dtypes.apply(make_dtype)


def is_dataframe(x):
    if cudf is not None:  # pragma: no cover
        if isinstance(x, cudf.DataFrame):
            return True
    return isinstance(x, pd.DataFrame)


def is_series(x):
    if cudf is not None:  # pragma: no cover
        if isinstance(x, cudf.Series):
            return True
    return isinstance(x, pd.Series)


def is_index(x):
    if cudf is not None:  # pragma: no cover
        if isinstance(x, cudf.Index):
            return True
    return isinstance(x, pd.Index)


def get_xdf(x):
    if cudf is not None:  # pragma: no cover
        if isinstance(x, (cudf.DataFrame, cudf.Series, cudf.Index)):
            return cudf
    return pd


def is_cudf(x):
    if cudf is not None:  # pragma: no cover
        if isinstance(x, (cudf.DataFrame, cudf.Series, cudf.Index)):
            return True
    return False
