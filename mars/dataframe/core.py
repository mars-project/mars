#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import weakref
from collections.abc import Iterable
from io import StringIO
from typing import Union, Dict, Any

import numpy as np
import pandas as pd

from ..core import ChunkData, Chunk, Tileable, HasShapeTileableData, \
    HasShapeTileable, OutputType, register_output_types, \
    _ExecuteAndFetchMixin, ENTITY_TYPE, is_build_mode
from ..core.entity.utils import refresh_tileable_shape
from ..lib.groupby_wrapper import GroupByWrapper
from ..serialization.serializables import Serializable, FieldTypes, DataTypeField, AnyField, \
    SeriesField, BoolField, Int32Field, StringField, ListField, SliceField, \
    TupleField, OneOfField, ReferenceField, NDArrayField, IntervalArrayField
from ..utils import on_serialize_shape, on_deserialize_shape, on_serialize_numpy_type, \
    ceildiv, tokenize
from .utils import fetch_corner_data, ReprSeries, parse_index, merge_index_value


class IndexValue(Serializable):
    """
    Meta class for index, held by IndexData, SeriesData and DataFrameData
    """
    __slots__ = ()

    class IndexBase(Serializable):
        _key = StringField('key')  # to identify if the index is the same
        _is_monotonic_increasing = BoolField('is_monotonic_increasing')
        _is_monotonic_decreasing = BoolField('is_monotonic_decreasing')
        _is_unique = BoolField('is_unique')
        _should_be_monotonic = BoolField('should_be_monotonic')
        _max_val = AnyField('max_val', on_serialize=on_serialize_numpy_type)
        _max_val_close = BoolField('max_val_close')
        _min_val = AnyField('min_val', on_serialize=on_serialize_numpy_type)
        _min_val_close = BoolField('min_val_close')

        @property
        def is_monotonic_increasing(self):
            return self._is_monotonic_increasing

        @property
        def is_monotonic_decreasing(self):
            return self._is_monotonic_decreasing

        @property
        def is_unique(self):
            return self._is_unique

        @property
        def should_be_monotonic(self):
            return self._should_be_monotonic

        @should_be_monotonic.setter
        def should_be_monotonic(self, val):
            self._should_be_monotonic = val

        @property
        def min_val(self):
            return self._min_val

        @property
        def min_val_close(self):
            return self._min_val_close

        @property
        def max_val(self):
            return self._max_val

        @property
        def max_val_close(self):
            return self._max_val_close

        @property
        def key(self):
            return self._key

        @property
        def inferred_type(self):
            return None

        def to_pandas(self):
            kw = {field.tag: getattr(self, attr, None)
                  for attr, field in self._FIELDS.items()
                  if attr not in super(type(self), self)._FIELDS}
            kw = {k: v for k, v in kw.items() if v is not None}
            if kw.get('data') is None:
                kw['data'] = []
            return getattr(pd, type(self).__name__)(**kw)

    class Index(IndexBase):
        _name = AnyField('name')
        _data = NDArrayField('data')
        _dtype = DataTypeField('dtype')

    class RangeIndex(IndexBase):
        _name = AnyField('name')
        _slice = SliceField('slice')
        _dtype = DataTypeField('dtype')

        @property
        def slice(self):
            return self._slice

        @property
        def dtype(self):
            return getattr(self, '_dtype', np.dtype(np.intc))

        def to_pandas(self):
            slc = self._slice
            return pd.RangeIndex(slc.start, slc.stop, slc.step,
                                 name=getattr(self, '_name', None))

    class CategoricalIndex(IndexBase):
        _name = AnyField('name')
        _data = NDArrayField('data')
        _categories = AnyField('categories')
        _ordered = BoolField('ordered')

        @property
        def inferred_type(self):
            return 'categorical'

    class IntervalIndex(IndexBase):
        _name = AnyField('name')
        _data = IntervalArrayField('data')
        _closed = StringField('closed')

        @property
        def inferred_type(self):
            return 'interval'

    class DatetimeIndex(IndexBase):
        _name = AnyField('name')
        _data = NDArrayField('data')
        _freq = AnyField('freq')
        _start = AnyField('start')
        _periods = AnyField('periods')
        _end = AnyField('end')
        _closed = AnyField('closed')
        _tz = AnyField('tz')
        _ambiguous = AnyField('ambiguous')
        _dayfirst = BoolField('dayfirst')
        _yearfirst = BoolField('yearfirst')

        @property
        def inferred_type(self):
            return 'datetime64'

        @property
        def freq(self):
            return getattr(self, '_freq', None)

    class TimedeltaIndex(IndexBase):
        _name = AnyField('name')
        _data = NDArrayField('data')
        _unit = AnyField('unit')
        _freq = AnyField('freq')
        _start = AnyField('start')
        _periods = AnyField('periods')
        _end = AnyField('end')
        _closed = AnyField('closed')

        @property
        def inferred_type(self):
            return 'timedelta64'

    class PeriodIndex(IndexBase):
        _name = AnyField('name')
        _data = NDArrayField('data')
        _freq = AnyField('freq')
        _start = AnyField('start')
        _periods = AnyField('periods')
        _end = AnyField('end')
        _year = AnyField('year')
        _month = AnyField('month')
        _quater = AnyField('quater')
        _day = AnyField('day')
        _hour = AnyField('hour')
        _minute = AnyField('minute')
        _second = AnyField('second')
        _tz = AnyField('tz')
        _dtype = DataTypeField('dtype')

        @property
        def inferred_type(self):
            return 'period'

    class Int64Index(IndexBase):
        _name = AnyField('name')
        _data = NDArrayField('data')
        _dtype = DataTypeField('dtype')

        @property
        def inferred_type(self):
            return 'integer'

    class UInt64Index(IndexBase):
        _name = AnyField('name')
        _data = NDArrayField('data')
        _dtype = DataTypeField('dtype')

        @property
        def inferred_type(self):
            return 'integer'

    class Float64Index(IndexBase):
        _name = AnyField('name')
        _data = NDArrayField('data')
        _dtype = DataTypeField('dtype')

        @property
        def inferred_type(self):
            return 'floating'

    class MultiIndex(IndexBase):
        _names = ListField('names', on_serialize=list)
        _dtypes = ListField('dtypes', on_serialize=list)
        _data = NDArrayField('data')
        _sortorder = Int32Field('sortorder')

        @property
        def inferred_type(self):
            return 'mixed'

        @property
        def names(self) -> list:
            return self._names

        def to_pandas(self):
            data = getattr(self, '_data', None)
            sortorder = getattr(self, '_sortorder', None)
            if data is None:
                return pd.MultiIndex.from_arrays([np.array([], dtype=dtype) for dtype in self._dtypes],
                                                 sortorder=sortorder, names=self._names)
            return pd.MultiIndex.from_tuples([tuple(d) for d in data], sortorder=sortorder,
                                             names=self._names)

    _index_value = OneOfField('index_value', index=Index,
                              range_index=RangeIndex, categorical_index=CategoricalIndex,
                              interval_index=IntervalIndex, datetime_index=DatetimeIndex,
                              timedelta_index=TimedeltaIndex, period_index=PeriodIndex,
                              int64_index=Int64Index, uint64_index=UInt64Index,
                              float64_index=Float64Index, multi_index=MultiIndex)

    def __mars_tokenize__(self):
        # return object for tokenize
        # todo fix this when index support is fixed
        if hasattr(self, '_key'):
            return self._key
        try:
            v = self._index_value
        except AttributeError:
            return None
        return [type(v).__name__] + [getattr(v, f, None) for f in v.__slots__]

    @property
    def value(self):
        return self._index_value

    @property
    def key(self):
        return self._index_value.key

    @property
    def is_monotonic_increasing(self):
        return self._index_value.is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        return self._index_value.is_monotonic_decreasing

    @property
    def is_monotonic_increasing_or_decreasing(self):
        return self.is_monotonic_increasing or self.is_monotonic_decreasing

    @property
    def should_be_monotonic(self):
        return self._index_value.should_be_monotonic

    @property
    def is_unique(self):
        return self._index_value.is_unique

    @property
    def min_val(self):
        return self._index_value.min_val

    @property
    def min_val_close(self):
        return self._index_value.min_val_close

    @property
    def max_val(self):
        return self._index_value.max_val

    @property
    def max_val_close(self):
        return self._index_value.max_val_close

    @property
    def min_max(self):
        return self._index_value.min_val, self._index_value.min_val_close, \
               self._index_value.max_val, self._index_value.max_val_close

    @property
    def name(self):
        return getattr(self._index_value, '_name', None)

    @property
    def inferred_type(self):
        return self._index_value.inferred_type

    def has_value(self):
        if isinstance(self._index_value, self.RangeIndex):
            if np.isnan(self._index_value.max_val):
                return False
            else:
                return True
        elif getattr(self._index_value, '_data', None) is not None:
            return True
        return False

    def to_pandas(self):
        return self._index_value.to_pandas()


class DtypesValue(Serializable):
    """
    Meta class for dtypes.
    """
    __slots__ = ()

    _key = StringField('key')
    _value = SeriesField('value')

    def __init__(self, key=None, value=None, **kw):
        super().__init__(_key=key, _value=value, **kw)
        if self._key is None:
            self._key = tokenize(self._value)

    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value


def refresh_index_value(tileable: ENTITY_TYPE):
    index_to_index_values = dict()
    for chunk in tileable.chunks:
        if chunk.ndim == 1:
            index_to_index_values[chunk.index] = chunk.index_value
        elif chunk.index[1] == 0:
            index_to_index_values[chunk.index] = chunk.index_value
    index_value = merge_index_value(
        index_to_index_values, store_data=False)
    index_value._index_value.should_be_monotonic = getattr(
        tileable.index_value, 'should_be_monotonic', None)
    tileable._index_value = index_value


def refresh_dtypes(tileable: ENTITY_TYPE):
    all_dtypes = [
        c.dtypes_value.value for c in tileable.chunks
        if c.index[0] == 0]
    dtypes = pd.concat(all_dtypes)
    tileable._dtypes = dtypes
    columns_values = parse_index(
        dtypes.index, store_data=True)
    columns_values._index_value.should_be_monotonic = getattr(
        tileable.columns_value, 'should_be_monotonic', None)
    tileable._columns_value = columns_values
    tileable._dtypes_value = DtypesValue(
        key=tokenize(dtypes), value=dtypes)


class IndexChunkData(ChunkData):
    __slots__ = ()
    type_name = 'Index'

    # required fields
    _shape = TupleField('shape', FieldTypes.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)
    # optional field
    _dtype = DataTypeField('dtype')
    _name = AnyField('name')
    _index_value = ReferenceField('index_value', IndexValue)

    def __init__(self, op=None, shape=None, index=None, dtype=None, name=None,
                 index_value=None, **kw):
        super().__init__(_op=op, _shape=shape, _index=index, _dtype=dtype, _name=name,
                         _index_value=index_value, **kw)

    @property
    def params(self) -> Dict[str, Any]:
        # params return the properties which useful to rebuild a new chunk
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'index': self.index,
            'index_value': self.index_value,
            'name': self.name
        }

    @params.setter
    def params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        params.pop('index', None)  # index not needed to update
        new_shape = params.pop('shape', None)
        if new_shape is not None:
            self._shape = new_shape
        dtype = params.pop('dtype', None)
        if dtype is not None:
            self._dtype = dtype
        index_value = params.pop('index_value', None)
        if index_value is not None:
            self._index_value = index_value
        name = params.pop('name', None)
        if name is not None:
            self._name = name
        if params:  # pragma: no cover
            raise TypeError(f'Unknown params: {list(params)}')

    @classmethod
    def get_params_from_data(cls, data: pd.Index) -> Dict[str, Any]:
        return {
            'shape': data.shape,
            'dtype': data.dtype,
            'index_value': parse_index(data, store_data=False),
            'name': data.name
        }

    @property
    def shape(self):
        return getattr(self, '_shape', None)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self):
        return self._name

    @property
    def index_value(self):
        return self._index_value


class IndexChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (IndexChunkData,)
    type_name = 'Index'


def _on_deserialize_index_value(index_value):
    if index_value is None:
        return
    try:
        getattr(index_value, 'value')
        return index_value
    except AttributeError:
        return


class _ToPandasMixin(_ExecuteAndFetchMixin):
    __slots__ = ()

    def to_pandas(self, session=None, **kw):
        return self._execute_and_fetch(session=session, **kw)


class _BatchedFetcher:
    __slots__ = ()

    def _iter(self, batch_size=1000, session=None, **kw):
        from .indexing.iloc import iloc

        size = self.shape[0]
        n_batch = ceildiv(size, batch_size)

        if n_batch > 1:
            for i in range(n_batch):
                batch_data = iloc(self)[batch_size * i: batch_size * (i + 1)]
                yield batch_data._fetch(session=session, **kw)
        else:
            yield self._fetch(session=session, **kw)

    def iterbatch(self, batch_size=1000, session=None, **kw):
        # trigger execution
        self.execute(session=session, **kw)
        return self._iter(batch_size=batch_size, session=session)

    def fetch(self, session=None, **kw):
        from .indexing.iloc import DataFrameIlocGetItem, SeriesIlocGetItem

        batch_size = kw.pop('batch_size', 1000)
        if isinstance(self.op, (DataFrameIlocGetItem, SeriesIlocGetItem)):
            # see GH#1871
            # already iloc, do not trigger batch fetch
            return self._fetch(session=session, **kw)
        else:
            batches = list(self._iter(batch_size=batch_size,
                                      session=session, **kw))
            return pd.concat(batches) if len(batches) > 1 else batches[0]


class IndexData(HasShapeTileableData, _ToPandasMixin):
    __slots__ = ()
    type_name = 'Index'

    # optional field
    _dtype = DataTypeField('dtype')
    _name = AnyField('name')
    _names = AnyField('names')
    _index_value = ReferenceField('index_value', IndexValue, on_deserialize=_on_deserialize_index_value)
    _chunks = ListField('chunks', FieldTypes.reference(IndexChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [IndexChunk(it) for it in x] if x is not None else x)

    def __init__(self, op=None, shape=None, nsplits=None, dtype=None,
                 name=None, names=None, index_value=None, chunks=None, **kw):
        super().__init__(_op=op, _shape=shape, _nsplits=nsplits, _dtype=dtype, _name=name,
                         _names=names, _index_value=index_value, _chunks=chunks, **kw)

    @property
    def params(self) -> Dict[str, Any]:
        # params return the properties which useful to rebuild a new tileable object
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'name': self.name,
            'index_value': self.index_value,
        }

    @params.setter
    def params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        new_shape = params.pop('shape', None)
        if new_shape is not None:
            self._shape = new_shape
        dtype = params.pop('dtype', None)
        if dtype is not None:
            self._dtype = dtype
        index_value = params.pop('index_value', None)
        if index_value is not None:
            self._index_value = index_value
        name = params.pop('name', None)
        if name is not None:
            self._name = name
        if params:  # pragma: no cover
            raise TypeError(f'Unknown params: {list(params)}')

    def refresh_params(self):
        # refresh params when chunks updated
        refresh_tileable_shape(self)
        refresh_index_value(self)
        if self._dtype is None:
            self._dtype = self.chunks[0].dtype
        if self._name is None:
            self._name = self.chunks[0].name

    def _to_str(self, representation=False):
        if is_build_mode() or len(self._executed_sessions) == 0:
            # in build mode, or not executed, just return representation
            if representation:
                return f'Index <op={type(self._op).__name__}, key={self.key}'
            else:
                return f'Index(op={type(self._op).__name__})'
        else:
            data = self.fetch(session=self._executed_sessions[-1])
            return repr(data) if repr(data) else str(data)

    def __str__(self):
        return self._to_str(representation=False)

    def __repr__(self):
        return self._to_str(representation=True)

    def _to_mars_tensor(self, dtype=None, order='K', extract_multi_index=False):
        tensor = self.to_tensor(extract_multi_index=extract_multi_index)
        dtype = dtype if dtype is not None else tensor.dtype
        return tensor.astype(dtype=dtype, order=order, copy=False)

    def __mars_tensor__(self, dtype=None, order='K'):
        return self._to_mars_tensor(dtype=dtype, order=order)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None) or self.op.dtype

    @property
    def name(self):
        return self._name

    @property
    def names(self):
        return getattr(self, '_names', None) or [self.name]

    @property
    def index_value(self) -> IndexValue:
        return self._index_value

    @property
    def inferred_type(self):
        return self._index_value.inferred_type

    def to_tensor(self, dtype=None, extract_multi_index=False):
        from ..tensor.datasource.from_dataframe import from_index
        return from_index(self, dtype=dtype, extract_multi_index=extract_multi_index)


class Index(HasShapeTileable, _ToPandasMixin):
    __slots__ = '_df_or_series', '_parent_key', '_axis'
    _allow_data_type_ = (IndexData,)
    type_name = 'Index'

    def __new__(cls, data: Union[pd.Index, IndexData] = None, **_):
        if data is not None and not isinstance(data, pd.Index):
            # create corresponding Index class
            # according to type of index_value
            clz = globals()[type(data.index_value.value).__name__]
        else:
            clz = cls
        return object.__new__(clz)

    def __len__(self):
        return len(self._data)

    def __mars_tensor__(self, dtype=None, order='K'):
        return self._data.__mars_tensor__(dtype=dtype, order=order)

    def _get_df_or_series(self):
        obj = getattr(self, '_df_or_series', None)
        if obj is not None:
            return obj()
        return None

    def _set_df_or_series(self, df_or_series, axis):
        self._df_or_series = weakref.ref(df_or_series)
        self._parent_key = df_or_series.key
        self._axis = axis

    @property
    def name(self):
        return self._data.name

    @name.setter
    def name(self, value):
        df_or_series = self._get_df_or_series()
        if df_or_series is not None and df_or_series.key == self._parent_key:
            df_or_series.rename_axis(value, axis=self._axis, inplace=True)
            self.data = df_or_series.axes[self._axis].data
        else:
            self.rename(value, inplace=True)

    @property
    def names(self):
        return self._data.names

    @names.setter
    def names(self, value):
        df_or_series = self._get_df_or_series()
        if df_or_series is not None:
            df_or_series.rename_axis(value, axis=self._axis, inplace=True)
            self.data = df_or_series.axes[self._axis].data
        else:
            self.rename(value, inplace=True)

    @property
    def values(self):
        return self.to_tensor()

    def to_frame(self, index: bool = True, name=None):
        """
        Create a DataFrame with a column containing the Index.

        Parameters
        ----------
        index : bool, default True
            Set the index of the returned DataFrame as the original Index.

        name : object, default None
            The passed name should substitute for the index name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame containing the original Index data.

        See Also
        --------
        Index.to_series : Convert an Index to a Series.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> import mars.dataframe as md
        >>> idx = md.Index(['Ant', 'Bear', 'Cow'], name='animal')
        >>> idx.to_frame().execute()
               animal
        animal
        Ant       Ant
        Bear     Bear
        Cow       Cow

        By default, the original Index is reused. To enforce a new Index:

        >>> idx.to_frame(index=False).execute()
          animal
        0    Ant
        1   Bear
        2    Cow

        To override the name of the resulting column, specify `name`:

        >>> idx.to_frame(index=False, name='zoo').execute()
            zoo
        0   Ant
        1  Bear
        2   Cow
        """
        from . import dataframe_from_tensor

        if isinstance(self.index_value.value, IndexValue.MultiIndex):
            old_names = self.index_value.value.names

            if name is not None and not isinstance(name, Iterable) or isinstance(name, str):
                raise TypeError("'name' must be a list / sequence of column names.")

            name = list(name if name is not None else old_names)
            if len(name) != len(old_names):
                raise ValueError("'name' should have same length as number of levels on index.")

            columns = [old or new or idx for idx, (old, new) in enumerate(zip(old_names, name))]
        else:
            columns = [name or self.name or 0]
        index_ = self if index else None
        return dataframe_from_tensor(self._data._to_mars_tensor(self, extract_multi_index=True),
                                     index=index_, columns=columns)

    def to_series(self, index=None, name=None):
        """
        Create a Series with both index and values equal to the index keys.

        Useful with map for returning an indexer based on an index.

        Parameters
        ----------
        index : Index, optional
            Index of resulting Series. If None, defaults to original index.
        name : str, optional
            Dame of resulting Series. If None, defaults to name of original
            index.

        Returns
        -------
        Series
            The dtype will be based on the type of the Index values.
        """
        from . import series_from_index

        return series_from_index(self, index=index, name=name)


class RangeIndex(Index):
    __slots__ = ()


class CategoricalIndex(Index):
    __slots__ = ()


class IntervalIndex(Index):
    __slots__ = ()


class DatetimeIndex(Index):
    __slots__ = ()


class TimedeltaIndex(Index):
    __slots__ = ()


class PeriodIndex(Index):
    __slots__ = ()


class Int64Index(Index):
    __slots__ = ()


class UInt64Index(Index):
    __slots__ = ()


class Float64Index(Index):
    __slots__ = ()


class MultiIndex(Index):
    __slots__ = ()


class BaseSeriesChunkData(ChunkData):
    __slots__ = ()

    # required fields
    _shape = TupleField('shape', FieldTypes.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)
    # optional field
    _dtype = DataTypeField('dtype')
    _name = AnyField('name')
    _index_value = ReferenceField('index_value', IndexValue, on_deserialize=_on_deserialize_index_value)

    def __init__(self, op=None, shape=None, index=None, dtype=None, name=None,
                 index_value=None, **kw):
        super().__init__(_op=op, _shape=shape, _index=index, _dtype=dtype, _name=name,
                         _index_value=index_value, **kw)

    def _get_params(self) -> Dict[str, Any]:
        # params return the properties which useful to rebuild a new chunk
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'index': self.index,
            'index_value': self.index_value,
            'name': self.name
        }

    def _set_params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        params.pop('index', None)  # index not needed to update
        new_shape = params.pop('shape', None)
        if new_shape is not None:
            self._shape = new_shape
        dtype = params.pop('dtype', None)
        if dtype is not None:
            self._dtype = dtype
        index_value = params.pop('index_value', None)
        if index_value is not None:
            self._index_value = index_value
        name = params.pop('name', None)
        if name is not None:
            self._name = name
        if params:  # pragma: no cover
            raise TypeError(f'Unknown params: {list(params)}')

    params = property(_get_params, _set_params)

    @classmethod
    def get_params_from_data(cls, data: pd.Series) -> Dict[str, Any]:
        return {
            'shape': data.shape,
            'dtype': data.dtype,
            'index_value': parse_index(data.index, store_data=False),
            'name': data.name
        }

    @property
    def shape(self):
        return getattr(self, '_shape', None)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self):
        return self._name

    @property
    def index_value(self):
        return self._index_value


class SeriesChunkData(BaseSeriesChunkData):
    type_name = 'Series'


class SeriesChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (SeriesChunkData,)
    type_name = 'Series'


class BaseSeriesData(HasShapeTileableData, _ToPandasMixin):
    __slots__ = '_cache', '_accessors'

    # optional field
    _dtype = DataTypeField('dtype')
    _name = AnyField('name')
    _index_value = ReferenceField('index_value', IndexValue, on_deserialize=_on_deserialize_index_value)
    _chunks = ListField('chunks', FieldTypes.reference(SeriesChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [SeriesChunk(it) for it in x] if x is not None else x)

    def __init__(self, op=None, shape=None, nsplits=None, dtype=None,
                 name=None, index_value=None, chunks=None, **kw):
        super().__init__(_op=op, _shape=shape, _nsplits=nsplits, _dtype=dtype, _name=name,
                         _index_value=index_value, _chunks=chunks, **kw)
        self._accessors = dict()

    def _get_params(self) -> Dict[str, Any]:
        # params return the properties which useful to rebuild a new tileable object
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'name': self.name,
            'index_value': self.index_value,
        }

    def _set_params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        new_shape = params.pop('shape', None)
        if new_shape is not None:
            self._shape = new_shape
        dtype = params.pop('dtype', None)
        if dtype is not None:
            self._dtype = dtype
        index_value = params.pop('index_value', None)
        if index_value is not None:
            self._index_value = index_value
        name = params.pop('name', None)
        if name is not None:
            self._name = name
        if params:  # pragma: no cover
            raise TypeError(f'Unknown params: {list(params)}')

    params = property(_get_params, _set_params)

    def refresh_params(self):
        # refresh params when chunks updated
        refresh_tileable_shape(self)
        refresh_index_value(self)
        if self._dtype is None:
            self._dtype = self.chunks[0].dtype
        if self._name is None:
            self._name = self.chunks[0].name

    def _to_str(self, representation=False):
        if is_build_mode() or len(self._executed_sessions) == 0:
            # in build mode, or not executed, just return representation
            if representation:
                return f'{self.type_name} <op={type(self._op).__name__}, key={self.key}>'
            else:
                return f'{self.type_name}(op={type(self._op).__name__})'
        else:
            corner_data = fetch_corner_data(
                self, session=self._executed_sessions[-1])

            buf = StringIO()
            max_rows = pd.get_option('display.max_rows')
            corner_max_rows = max_rows if self.shape[0] <= max_rows else \
                corner_data.shape[0] - 1  # make sure max_rows < corner_data

            with pd.option_context('display.max_rows', corner_max_rows):
                if self.shape[0] <= max_rows:
                    corner_series = corner_data
                else:
                    corner_series = ReprSeries(corner_data, self.shape)
                buf.write(repr(corner_series) if representation else str(corner_series))

            return buf.getvalue()

    def __str__(self):
        return self._to_str(representation=False)

    def __repr__(self):
        return self._to_str(representation=False)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None) or self.op.dtype

    @property
    def name(self):
        return self._name

    @property
    def index_value(self):
        return self._index_value

    @property
    def index(self):
        from .datasource.index import from_tileable

        return from_tileable(self)

    @property
    def axes(self):
        return [self.index]

    @property
    def empty(self):
        shape = getattr(self, '_shape')
        if np.any(np.isnan(shape)):
            raise ValueError('Tileable object must be executed first')
        return shape == (0,)

    def to_tensor(self, dtype=None):
        from ..tensor.datasource.from_dataframe import from_series
        return from_series(self, dtype=dtype)

    @staticmethod
    def from_tensor(in_tensor, index=None, name=None):
        from .datasource.from_tensor import series_from_tensor
        return series_from_tensor(in_tensor, index=index, name=name)


class SeriesData(_BatchedFetcher, BaseSeriesData):
    type_name = 'Series'

    def __mars_tensor__(self, dtype=None, order='K'):
        tensor = self.to_tensor()
        dtype = dtype if dtype is not None else tensor.dtype
        return tensor.astype(dtype=dtype, order=order, copy=False)

    def iteritems(self, batch_size=10000, session=None):
        for batch_data in self.iterbatch(batch_size=batch_size, session=session):
            yield from getattr(batch_data, 'iteritems')()

    items = iteritems

    def to_dict(self, into=dict, batch_size=10000, session=None):
        fetch_kwargs = dict(batch_size=batch_size)
        return self.to_pandas(session=session, fetch_kwargs=fetch_kwargs).to_dict(into=into)


class Series(HasShapeTileable, _ToPandasMixin):
    __slots__ = '_cache',
    _allow_data_type_ = (SeriesData,)
    type_name = 'Series'

    def to_tensor(self, dtype=None):
        return self._data.to_tensor(dtype=dtype)

    def from_tensor(self, in_tensor, index=None, name=None):
        return self._data.from_tensor(in_tensor, index=index, name=name)

    @property
    def ndim(self):
        """
        Return an int representing the number of axes / array dimensions.

        Return 1 if Series. Otherwise return 2 if DataFrame.

        See Also
        --------
        ndarray.ndim : Number of array dimensions.

        Examples
        --------
        >>> import mars.dataframe as md
        >>> s = md.Series({'a': 1, 'b': 2, 'c': 3})
        >>> s.ndim.execute()
        1

        >>> df = md.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.ndim.execute()
        2
        """
        return super().ndim

    @property
    def index(self):
        """
        The index (axis labels) of the Series.
        """
        idx = self._data.index
        idx._set_df_or_series(self, 0)
        return idx

    @index.setter
    def index(self, new_index):
        self.set_axis(new_index, axis=0, inplace=True)

    @property
    def name(self):
        return self._data.name

    @name.setter
    def name(self, val):
        from .indexing.rename import DataFrameRename
        op = DataFrameRename(new_name=val, output_types=[OutputType.series])
        new_series = op(self)
        self.data = new_series.data

    @property
    def dtype(self):
        """
        Return the dtype object of the underlying data.
        """
        return self._data.dtype

    def copy(self, deep=True):  # pylint: disable=arguments-differ
        """
        Make a copy of this object's indices and data.

        When ``deep=True`` (default), a new object will be created with a
        copy of the calling object's data and indices. Modifications to
        the data or indices of the copy will not be reflected in the
        original object (see notes below).

        When ``deep=False``, a new object will be created without copying
        the calling object's data or index (only references to the data
        and index are copied). Any changes to the data of the original
        will be reflected in the shallow copy (and vice versa).

        Parameters
        ----------
        deep : bool, default True
            Make a deep copy, including a copy of the data and the indices.
            With ``deep=False`` neither the indices nor the data are copied.

        Returns
        -------
        copy : Series or DataFrame
            Object type matches caller.
        """
        if deep:
            return super().copy()
        else:
            return super()._view()

    def __len__(self):
        return len(self._data)

    def __mars_tensor__(self, dtype=None, order='K'):
        return self._data.__mars_tensor__(dtype=dtype, order=order)

    def keys(self):
        """
        Return alias for index.

        Returns
        -------
        Index
            Index of the Series.
        """
        return self.index

    @property
    def values(self):
        return self.to_tensor()

    def iteritems(self, batch_size=10000, session=None):
        """
        Lazily iterate over (index, value) tuples.

        This method returns an iterable tuple (index, value). This is
        convenient if you want to create a lazy iterator.

        Returns
        -------
        iterable
            Iterable of tuples containing the (index, value) pairs from a
            Series.

        See Also
        --------
        DataFrame.items : Iterate over (column name, Series) pairs.
        DataFrame.iterrows : Iterate over DataFrame rows as (index, Series) pairs.

        Examples
        --------
        >>> import mars.dataframe as md
        >>> s = md.Series(['A', 'B', 'C'])
        >>> for index, value in s.items():
        ...     print(f"Index : {index}, Value : {value}")
        Index : 0, Value : A
        Index : 1, Value : B
        Index : 2, Value : C
        """
        return self._data.iteritems(batch_size=batch_size, session=session)

    items = iteritems

    def to_dict(self, into=dict, batch_size=10000, session=None):
        """
        Convert Series to {label -> value} dict or dict-like object.

        Parameters
        ----------
        into : class, default dict
            The collections.abc.Mapping subclass to use as the return
            object. Can be the actual class or an empty
            instance of the mapping type you want.  If you want a
            collections.defaultdict, you must pass it initialized.

        Returns
        -------
        collections.abc.Mapping
            Key-value representation of Series.

        Examples
        --------
        >>> import mars.dataframe as md
        >>> s = md.Series([1, 2, 3, 4])
        >>> s.to_dict()
        {0: 1, 1: 2, 2: 3, 3: 4}
        >>> from collections import OrderedDict, defaultdict
        >>> s.to_dict(OrderedDict)
        OrderedDict([(0, 1), (1, 2), (2, 3), (3, 4)])
        >>> dd = defaultdict(list)
        >>> s.to_dict(dd)
        defaultdict(<class 'list'>, {0: 1, 1: 2, 2: 3, 3: 4})
        """
        return self._data.to_dict(into=into, batch_size=batch_size, session=session)

    def to_frame(self, name=None):
        """
        Convert Series to DataFrame.

        Parameters
        ----------
        name : object, default None
            The passed name should substitute for the series name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame representation of Series.

        Examples
        --------
        >>> import mars.dataframe as md
        >>> s = md.Series(["a", "b", "c"], name="vals")
        >>> s.to_frame().execute()
          vals
        0    a
        1    b
        2    c
        """
        from . import dataframe_from_tensor

        name = name or self.name or 0
        return dataframe_from_tensor(self, columns=[name])


class BaseDataFrameChunkData(ChunkData):
    __slots__ = '_dtypes_value',

    # required fields
    _shape = TupleField('shape', FieldTypes.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)
    # optional fields
    _dtypes = SeriesField('dtypes')
    _index_value = ReferenceField('index_value', IndexValue, on_deserialize=_on_deserialize_index_value)
    _columns_value = ReferenceField('columns_value', IndexValue)

    def __init__(self, op=None, shape=None, index=None, dtypes=None,
                 index_value=None, columns_value=None, **kw):
        super().__init__(_op=op, _shape=shape, _index=index, _dtypes=dtypes,
                         _index_value=index_value, _columns_value=columns_value, **kw)
        self._dtypes_value = None

    def __len__(self):
        return self.shape[0]

    def _get_params(self) -> Dict[str, Any]:
        # params return the properties which useful to rebuild a new chunk
        return {
            'shape': self.shape,
            'dtypes': self.dtypes,
            'dtypes_value': self.dtypes_value,
            'index': self.index,
            'index_value': self.index_value,
            'columns_value': self.columns_value,
        }

    def _set_params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        params.pop('index', None)  # index not needed to update
        new_shape = params.pop('shape', None)
        if new_shape is not None:
            self._shape = new_shape
        index_value = params.pop('index_value', None)
        if index_value is not None:
            self._index_value = index_value
        dtypes = params.pop('dtypes', None)
        if dtypes is not None:
            self._dtypes = dtypes
        columns_value = params.pop('columns_value', None)
        if columns_value is not None:
            self._columns_value = columns_value
        dtypes_value = params.pop('dtypes_value', None)
        if dtypes_value is not None:
            if dtypes is None:
                self._dtypes = dtypes_value.value
            if columns_value is None:
                self._columns_value = parse_index(
                    self._dtypes.index, store_data=True)
            self._dtypes_value = dtypes_value
        if params:  # pragma: no cover
            raise TypeError(f'Unknown params: {list(params)}')

    params = property(_get_params, _set_params)

    @classmethod
    def get_params_from_data(cls, data: pd.DataFrame) -> Dict[str, Any]:
        parse_index(data.index, store_data=False)
        return {
            'shape': data.shape,
            'index_value': parse_index(data.index, store_data=False),
            'dtypes_value': DtypesValue(key=tokenize(data.dtypes),
                                        value=data.dtypes)
        }

    @property
    def shape(self):
        return getattr(self, '_shape', None)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtypes(self):
        dt = getattr(self, '_dtypes', None)
        if dt is not None:
            return dt
        return getattr(self.op, 'dtypes', None)

    @property
    def dtypes_value(self):
        if self._dtypes_value is not None:
            return self._dtypes_value
        # TODO(qinxuye): when creating Dataframe,
        #  dtypes_value instead of dtypes later must be passed into
        dtypes = self.dtypes
        if dtypes is not None:
            self._dtypes_value = DtypesValue(
                key=tokenize(dtypes), value=dtypes)
            return self._dtypes_value

    @property
    def index_value(self):
        return self._index_value

    @property
    def columns_value(self):
        return self._columns_value


class DataFrameChunkData(BaseDataFrameChunkData):
    type_name = 'DataFrame'


class DataFrameChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (DataFrameChunkData,)
    type_name = 'DataFrame'

    def __len__(self):
        return len(self._data)


class BaseDataFrameData(HasShapeTileableData, _ToPandasMixin):
    __slots__ = '_accessors', '_dtypes_value'

    # optional fields
    _dtypes = SeriesField('dtypes')
    _index_value = ReferenceField('index_value', IndexValue, on_deserialize=_on_deserialize_index_value)
    _columns_value = ReferenceField('columns_value', IndexValue)
    _chunks = ListField('chunks', FieldTypes.reference(DataFrameChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [DataFrameChunk(it) for it in x] if x is not None else x)

    def __init__(self, op=None, shape=None, nsplits=None, dtypes=None,
                 index_value=None, columns_value=None, chunks=None, **kw):
        super().__init__(_op=op, _shape=shape, _nsplits=nsplits, _dtypes=dtypes,
                         _index_value=index_value, _columns_value=columns_value,
                         _chunks=chunks, **kw)
        self._accessors = dict()
        self._dtypes_value = None

    def _get_params(self) -> Dict[str, Any]:
        # params return the properties which useful to rebuild a new tileable object
        return {
            'shape': self.shape,
            'dtypes': self.dtypes,
            'index_value': self.index_value,
            'columns_value': self.columns_value,
            'dtypes_value': self.dtypes_value
        }

    def _set_params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        new_shape = params.pop('shape', None)
        if new_shape is not None:
            self._shape = new_shape
        index_value = params.pop('index_value', None)
        if index_value is not None:
            self._index_value = index_value
        dtypes = params.pop('dtypes', None)
        if dtypes is not None:
            self._dtypes = dtypes
        columns_value = params.pop('columns_value', None)
        if columns_value is not None:
            self._columns_value = columns_value
        dtypes_value = params.pop('dtypes_value', None)
        if dtypes_value is not None:
            if dtypes is None:
                self._dtypes = dtypes_value.value
            if columns_value is None:
                self._columns_value = parse_index(
                    self._dtypes.index, store_data=True)
            self._dtypes_value = dtypes_value
        if params:  # pragma: no cover
            raise TypeError(f'Unknown params: {list(params)}')

    params = property(_get_params, _set_params)

    def refresh_params(self):
        # refresh params when chunks updated
        refresh_tileable_shape(self)
        refresh_index_value(self)
        refresh_dtypes(self)

    @property
    def dtypes(self):
        dt = getattr(self, '_dtypes', None)
        if dt is not None:
            return dt
        return getattr(self.op, 'dtypes', None)

    @property
    def dtypes_value(self):
        if self._dtypes_value is not None:
            return self._dtypes_value
        # TODO(qinxuye): when creating Dataframe,
        #  dtypes_value instead of dtypes later must be passed into
        dtypes = self.dtypes
        if dtypes is not None:
            self._dtypes_value = DtypesValue(
                key=tokenize(dtypes), value=dtypes)
            return self._dtypes_value

    @property
    def index_value(self):
        return self._index_value

    @property
    def columns_value(self):
        return self._columns_value

    @property
    def empty(self):
        shape = getattr(self, '_shape')
        if np.any(np.isnan(shape)):
            raise ValueError('Tileable object must be executed first')
        return (0 in shape)

    def to_tensor(self, dtype=None):
        from ..tensor.datasource.from_dataframe import from_dataframe
        return from_dataframe(self, dtype=dtype)

    @staticmethod
    def from_tensor(in_tensor, index=None, columns=None):
        from .datasource.from_tensor import dataframe_from_tensor
        return dataframe_from_tensor(in_tensor, index=index, columns=columns)

    @staticmethod
    def from_records(records, **kw):
        from .datasource.from_records import from_records
        return from_records(records, **kw)

    @property
    def index(self):
        from .datasource.index import from_tileable

        return from_tileable(self)

    @property
    def columns(self):
        from .datasource.index import from_pandas as from_pandas_index

        return from_pandas_index(self.dtypes.index, store_data=True)

    @property
    def axes(self):
        return [self.index, self.columns]


class DataFrameData(_BatchedFetcher, BaseDataFrameData):
    type_name = 'DataFrame'

    def _to_str(self, representation=False):
        if is_build_mode() or len(self._executed_sessions) == 0:
            # in build mode, or not executed, just return representation
            if representation:
                return f'{self.type_name} <op={type(self._op).__name__}, key={self.key}>'
            else:
                return f'{self.type_name}(op={type(self._op).__name__})'
        else:
            corner_data = fetch_corner_data(
                self, session=self._executed_sessions[-1])

            buf = StringIO()
            max_rows = pd.get_option('display.max_rows')

            if self.shape[0] <= max_rows:
                buf.write(repr(corner_data) if representation else str(corner_data))
            else:
                # remember we cannot directly call repr(df),
                # because the [... rows x ... columns] may show wrong rows
                with pd.option_context('display.show_dimensions', False,
                                       'display.max_rows', corner_data.shape[0] - 1):
                    if representation:
                        s = repr(corner_data)
                    else:
                        s = str(corner_data)
                    buf.write(s)
                if pd.get_option('display.show_dimensions'):
                    n_rows, n_cols = self.shape
                    buf.write(f"\n\n[{n_rows} rows x {n_cols} columns]")

            return buf.getvalue()

    def __str__(self):
        return self._to_str(representation=False)

    def __repr__(self):
        return self._to_str(representation=True)

    def __mars_tensor__(self, dtype=None, order='K'):
        return self.to_tensor().astype(dtype=dtype, order=order, copy=False)

    def _repr_html_(self):
        if len(self._executed_sessions) == 0:
            # not executed before, fall back to normal repr
            raise NotImplementedError

        corner_data = fetch_corner_data(
            self, session=self._executed_sessions[-1])

        buf = StringIO()
        max_rows = pd.get_option('display.max_rows')
        if self.shape[0] <= max_rows:
            buf.write(corner_data._repr_html_())
        else:
            with pd.option_context('display.show_dimensions', False,
                                   'display.max_rows', corner_data.shape[0] - 1):
                buf.write(corner_data._repr_html_().rstrip().rstrip('</div>'))
            if pd.get_option('display.show_dimensions'):
                n_rows, n_cols = self.shape
                buf.write(f"<p>{n_rows} rows × {n_cols} columns</p>\n")
            buf.write('</div>')

        return buf.getvalue()

    def items(self):
        for col_name in self.dtypes.index:
            yield col_name, self[col_name]

    iteritems = items

    def iterrows(self, batch_size=1000, session=None):
        for batch_data in self.iterbatch(batch_size=batch_size, session=session):
            yield from getattr(batch_data, 'iterrows')()

    def itertuples(self, index=True, name='Pandas', batch_size=1000, session=None):
        for batch_data in self.iterbatch(batch_size=batch_size, session=session):
            yield from getattr(batch_data, 'itertuples')(index=index, name=name)


class DataFrame(HasShapeTileable, _ToPandasMixin):
    __slots__ = '_cache',
    _allow_data_type_ = (DataFrameData,)
    type_name = 'DataFrame'

    def __len__(self):
        return len(self._data)

    def to_tensor(self):
        return self._data.to_tensor()

    def from_tensor(self, in_tensor, index=None, columns=None):
        return self._data.from_tensor(in_tensor, index=index, columns=columns)

    def from_records(self, records, **kw):
        return self._data.from_records(records, **kw)

    def __mars_tensor__(self, dtype=None, order='K'):
        return self._data.__mars_tensor__(dtype=dtype, order=order)

    def __getattr__(self, key):
        try:
            return getattr(self._data, key)
        except AttributeError:
            if key in self.dtypes:
                return self[key]
            else:
                raise

    def __dir__(self):
        result = list(super().__dir__())
        return sorted(result + [k for k in self.dtypes.index if isinstance(k, str) and k.isidentifier()])

    @property
    def ndim(self):
        """
        Return an int representing the number of axes / array dimensions.

        Return 1 if Series. Otherwise return 2 if DataFrame.

        See Also
        --------
        ndarray.ndim : Number of array dimensions.

        Examples
        --------
        >>> import mars.dataframe as md
        >>> s = md.Series({'a': 1, 'b': 2, 'c': 3})
        >>> s.ndim.execute()
        1

        >>> df = md.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.ndim.execute()
        2
        """
        return super().ndim

    @property
    def index(self):
        idx = self._data.index
        idx._set_df_or_series(self, 0)
        return idx

    @index.setter
    def index(self, new_index):
        self.set_axis(new_index, axis=0, inplace=True)

    @property
    def columns(self):
        col = self._data.columns
        col._set_df_or_series(self, 1)
        return col

    @columns.setter
    def columns(self, new_columns):
        self.set_axis(new_columns, axis=1, inplace=True)

    def keys(self):
        """
        Get the 'info axis' (see Indexing for more).

        This is index for Series, columns for DataFrame.

        Returns
        -------
        Index
            Info axis.
        """
        return self.columns

    @property
    def values(self):
        return self.to_tensor()

    @property
    def dtypes(self):
        """
        Return the dtypes in the DataFrame.

        This returns a Series with the data type of each column.
        The result's index is the original DataFrame's columns. Columns
        with mixed types are stored with the ``object`` dtype. See
        :ref:`the User Guide <basics.dtypes>` for more.

        Returns
        -------
        pandas.Series
            The data type of each column.

        Examples
        --------
        >>> import mars.dataframe as md
        >>> df = md.DataFrame({'float': [1.0],
        ...                    'int': [1],
        ...                    'datetime': [md.Timestamp('20180310')],
        ...                    'string': ['foo']})
        >>> df.dtypes
        float              float64
        int                  int64
        datetime    datetime64[ns]
        string              object
        dtype: object
        """
        return self._data.dtypes

    def iterrows(self, batch_size=1000, session=None):
        """
        Iterate over DataFrame rows as (index, Series) pairs.

        Yields
        ------
        index : label or tuple of label
            The index of the row. A tuple for a `MultiIndex`.
        data : Series
            The data of the row as a Series.

        it : generator
            A generator that iterates over the rows of the frame.

        See Also
        --------
        DataFrame.itertuples : Iterate over DataFrame rows as namedtuples of the values.
        DataFrame.items : Iterate over (column name, Series) pairs.

        Notes
        -----

        1. Because ``iterrows`` returns a Series for each row,
           it does **not** preserve dtypes across the rows (dtypes are
           preserved across columns for DataFrames). For example,

           >>> import mars.dataframe as md
           >>> df = md.DataFrame([[1, 1.5]], columns=['int', 'float'])
           >>> row = next(df.iterrows())[1]
           >>> row
           int      1.0
           float    1.5
           Name: 0, dtype: float64
           >>> print(row['int'].dtype)
           float64
           >>> print(df['int'].dtype)
           int64

           To preserve dtypes while iterating over the rows, it is better
           to use :meth:`itertuples` which returns namedtuples of the values
           and which is generally faster than ``iterrows``.

        2. You should **never modify** something you are iterating over.
           This is not guaranteed to work in all cases. Depending on the
           data types, the iterator returns a copy and not a view, and writing
           to it will have no effect.
        """
        return self._data.iterrows(batch_size=batch_size, session=session)

    def itertuples(self, index=True, name='Pandas', batch_size=1000, session=None):
        """
        Iterate over DataFrame rows as namedtuples.

        Parameters
        ----------
        index : bool, default True
            If True, return the index as the first element of the tuple.
        name : str or None, default "Pandas"
            The name of the returned namedtuples or None to return regular
            tuples.

        Returns
        -------
        iterator
            An object to iterate over namedtuples for each row in the
            DataFrame with the first field possibly being the index and
            following fields being the column values.

        See Also
        --------
        DataFrame.iterrows : Iterate over DataFrame rows as (index, Series)
            pairs.
        DataFrame.items : Iterate over (column name, Series) pairs.

        Notes
        -----
        The column names will be renamed to positional names if they are
        invalid Python identifiers, repeated, or start with an underscore.
        On python versions < 3.7 regular tuples are returned for DataFrames
        with a large number of columns (>254).

        Examples
        --------
        >>> import mars.dataframe as md
        >>> df = md.DataFrame({'num_legs': [4, 2], 'num_wings': [0, 2]},
        ...                   index=['dog', 'hawk'])
        >>> df.execute()
              num_legs  num_wings
        dog          4          0
        hawk         2          2
        >>> for row in df.itertuples():
        ...     print(row)
        ...
        Pandas(Index='dog', num_legs=4, num_wings=0)
        Pandas(Index='hawk', num_legs=2, num_wings=2)

        By setting the `index` parameter to False we can remove the index
        as the first element of the tuple:

        >>> for row in df.itertuples(index=False):
        ...     print(row)
        ...
        Pandas(num_legs=4, num_wings=0)
        Pandas(num_legs=2, num_wings=2)

        With the `name` parameter set we set a custom name for the yielded
        namedtuples:

        >>> for row in df.itertuples(name='Animal'):
        ...     print(row)
        ...
        Animal(Index='dog', num_legs=4, num_wings=0)
        Animal(Index='hawk', num_legs=2, num_wings=2)
        """
        return self._data.itertuples(batch_size=batch_size, session=session,
                                     index=index, name=name)

    def assign(self, **kwargs):
        """
        Assign new columns to a DataFrame.
        Returns a new object with all original columns in addition to new ones.
        Existing columns that are re-assigned will be overwritten.

        Parameters
        ----------
        **kwargs : dict of {str: callable or Series}
            The column names are keywords. If the values are
            callable, they are computed on the DataFrame and
            assigned to the new columns. The callable must not
            change input DataFrame (though pandas doesn't check it).
            If the values are not callable, (e.g. a Series, scalar, or array),
            they are simply assigned.

        Returns
        -------
        DataFrame
            A new DataFrame with the new columns in addition to
            all the existing columns.

        Notes
        -----
        Assigning multiple columns within the same ``assign`` is possible.
        Later items in 'kwargs' may refer to newly created or modified
        columns in 'df'; items are computed and assigned into 'df' in order.

        Examples
        --------
        >>> import mars.dataframe as md
        >>> df = md.DataFrame({'temp_c': [17.0, 25.0]},
        ...                   index=['Portland', 'Berkeley'])
        >>> df.execute()
                  temp_c
        Portland    17.0
        Berkeley    25.0

        Where the value is a callable, evaluated on `df`:

        >>> df.assign(temp_f=lambda x: x.temp_c * 9 / 5 + 32).execute()
                  temp_c  temp_f
        Portland    17.0    62.6
        Berkeley    25.0    77.0

        Alternatively, the same behavior can be achieved by directly
        referencing an existing Series or sequence:

        >>> df.assign(temp_f=df['temp_c'] * 9 / 5 + 32).execute()
                  temp_c  temp_f
        Portland    17.0    62.6
        Berkeley    25.0    77.0

        You can create multiple columns within the same assign where one
        of the columns depends on another one defined within the same assign:

        >>> df.assign(temp_f=lambda x: x['temp_c'] * 9 / 5 + 32,
        ...           temp_k=lambda x: (x['temp_f'] +  459.67) * 5 / 9).execute()
                  temp_c  temp_f  temp_k
        Portland    17.0    62.6  290.15
        Berkeley    25.0    77.0  298.15
        """

        def apply_if_callable(maybe_callable, obj, **kwargs):
            if callable(maybe_callable):
                return maybe_callable(obj, **kwargs)

            return maybe_callable

        data = self.copy()

        for k, v in kwargs.items():
            data[k] = apply_if_callable(v, data)
        return data


class DataFrameGroupByChunkData(BaseDataFrameChunkData):
    type_name = 'DataFrameGroupBy'

    _key_dtypes = SeriesField('key_dtypes')
    _selection = AnyField('selection')

    @property
    def key_dtypes(self):
        return self._key_dtypes

    @property
    def selection(self):
        return self._selection

    def _get_params(self) -> Dict[str, Any]:
        p = super()._get_params()
        p.update(dict(key_dtypes=self.key_dtypes, selection=self.selection))
        return p

    def _set_params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        key_dtypes = params.pop('key_dtypes', None)
        if key_dtypes is not None:
            self._key_dtypes = key_dtypes
        selection = params.pop('selection', None)
        if selection is not None:
            self._selection = selection
        super()._set_params(params)

    params = property(_get_params, _set_params)

    @classmethod
    def get_params_from_data(cls, data: GroupByWrapper) -> Dict[str, Any]:
        params = super().get_params_from_data(data.obj)
        if data.selection:
            dtypes = params['dtypes_value'].value[data.selection]
            params['dtypes_value'] = DtypesValue(value=dtypes)
            params['shape'] = data.shape
        return params

    def __init__(self, key_dtypes=None, selection=None, **kw):
        super().__init__(_key_dtypes=key_dtypes, _selection=selection, **kw)


class DataFrameGroupByChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (DataFrameGroupByChunkData,)
    type_name = 'DataFrameGroupBy'

    def __len__(self):
        return len(self._data)


class SeriesGroupByChunkData(BaseSeriesChunkData):
    type_name = 'SeriesGroupBy'

    _key_dtypes = AnyField('key_dtypes')

    @property
    def key_dtypes(self):
        return self._key_dtypes

    def _get_params(self) -> Dict[str, Any]:
        p = super()._get_params()
        p['key_dtypes'] = self.key_dtypes
        return p

    def _set_params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        key_dtypes = params.pop('key_dtypes', None)
        if key_dtypes is not None:
            self._key_dtypes = key_dtypes
        super()._set_params(new_params)

    params = property(_get_params, _set_params)

    @classmethod
    def get_params_from_data(cls, data: GroupByWrapper):
        series_name = data.selection or data.obj.name
        if hasattr(data.obj, 'dtype'):
            dtype = data.obj.dtype
        else:
            dtype = data.obj.dtypes[series_name]

        return {
            'shape': (data.obj.shape[0],),
            'dtype': dtype,
            'index_value': parse_index(data.obj.index, store_data=False),
            'name': series_name,
        }

    def __init__(self, key_dtypes=None, **kw):
        super().__init__(_key_dtypes=key_dtypes, **kw)


class SeriesGroupByChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (SeriesGroupByChunkData,)
    type_name = 'SeriesGroupBy'

    def __len__(self):
        return len(self._data)


class DataFrameGroupByData(BaseDataFrameData):
    type_name = 'DataFrameGroupBy'

    _key_dtypes = SeriesField('key_dtypes')
    _selection = AnyField('selection')
    _chunks = ListField('chunks', FieldTypes.reference(DataFrameGroupByChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [DataFrameGroupByChunk(it) for it in x] if x is not None else x)

    @property
    def key_dtypes(self):
        return self._key_dtypes

    @property
    def selection(self):
        return self._selection

    def _get_params(self) -> Dict[str, Any]:
        p = super()._get_params()
        p.update(dict(key_dtypes=self.key_dtypes, selection=self.selection))
        return p

    def _set_params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        key_dtypes = params.pop('key_dtypes', None)
        if key_dtypes is not None:
            self._key_dtypes = key_dtypes
        selection = params.pop('selection', None)
        if selection is not None:
            self._selection = selection
        super()._set_params(params)

    params = property(_get_params, _set_params)

    def __init__(self, key_dtypes=None, selection=None, **kw):
        super().__init__(_key_dtypes=key_dtypes, _selection=selection, **kw)

    def _equal(self, o):
        # FIXME We need to implemented a true `==` operator for DataFrameGroupby
        if is_build_mode():
            return self is o
        else:
            return self == o


class SeriesGroupByData(BaseSeriesData):
    type_name = 'SeriesGroupBy'

    _key_dtypes = AnyField('key_dtypes')
    _chunks = ListField('chunks', FieldTypes.reference(SeriesGroupByChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [SeriesGroupByChunk(it) for it in x] if x is not None else x)

    @property
    def key_dtypes(self):
        return self._key_dtypes

    def _get_params(self) -> Dict[str, Any]:
        p = super()._get_params()
        p['key_dtypes'] = self.key_dtypes
        return p

    def _set_params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        key_dtypes = params.pop('key_dtypes', None)
        if key_dtypes is not None:
            self._key_dtypes = key_dtypes
        super()._set_params(params)

    params = property(_get_params, _set_params)

    def __init__(self, key_dtypes=None, **kw):
        super().__init__(_key_dtypes=key_dtypes, **kw)

    def _equal(self, o):
        # FIXME We need to implemented a true `==` operator for DataFrameGroupby
        if is_build_mode():
            return self is o
        else:
            return self == o


class GroupBy(Tileable, _ToPandasMixin):
    __slots__ = ()


class DataFrameGroupBy(GroupBy):
    __slots__ = ()
    _allow_data_type_ = (DataFrameGroupByData,)
    type_name = 'DataFrameGroupBy'

    def __eq__(self, other):
        return self._equal(other)

    def __hash__(self):
        # NB: we have customized __eq__ explicitly, thus we need define __hash__ explicitly as well.
        return super().__hash__()

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            if item in self.dtypes:
                return self[item]
            else:
                raise

    def __dir__(self):
        result = list(super().__dir__())
        return sorted(result + [k for k in self.dtypes.index if isinstance(k, str) and k.isidentifier()])


class SeriesGroupBy(GroupBy):
    __slots__ = ()
    _allow_data_type_ = (SeriesGroupByData,)
    type_name = 'SeriesGroupBy'

    def __eq__(self, other):
        return self._equal(other)

    def __hash__(self):
        # NB: we have customized __eq__ explicitly, thus we need define __hash__ explicitly as well.
        return super().__hash__()


class CategoricalChunkData(ChunkData):
    __slots__ = ()
    type_name = 'Categorical'

    # required fields
    _shape = TupleField('shape', FieldTypes.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)
    # optional field
    _dtype = DataTypeField('dtype')
    _categories_value = ReferenceField('categories_value', IndexValue,
                                       on_deserialize=_on_deserialize_index_value)

    def __init__(self, op=None, shape=None, index=None, dtype=None,
                 categories_value=None, **kw):
        super().__init__(_op=op, _shape=shape, _index=index, _dtype=dtype,
                         _categories_value=categories_value, **kw)

    @property
    def params(self) -> Dict[str, Any]:
        # params return the properties which useful to rebuild a new chunk
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'index': self.index,
            'categories_value': self.categories_value,
        }

    @params.setter
    def params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        params.pop('index', None)  # index not needed to update
        new_shape = params.pop('shape', None)
        if new_shape is not None:
            self._shape = new_shape
        dtype = params.pop('dtype', None)
        if dtype is not None:
            self._dtype = dtype
        categories_value = params.pop('categories_value', None)
        if categories_value is not None:
            self._categories_value = categories_value
        if params:  # pragma: no cover
            raise TypeError(f'Unknown params: {list(params)}')

    @classmethod
    def get_params_from_data(cls, data: pd.Categorical) -> Dict[str, Any]:
        return {
            'shape': data.shape,
            'dtype': data.dtype,
            'categories_value': parse_index(data.categories, store_data=True)
        }

    @property
    def shape(self):
        return getattr(self, '_shape', None)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def categories_value(self):
        return self._categories_value


class CategoricalChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (CategoricalChunkData,)
    type_name = 'Categorical'


class CategoricalData(HasShapeTileableData, _ToPandasMixin):
    __slots__ = '_cache',
    type_name = 'Categorical'

    # optional field
    _dtype = DataTypeField('dtype')
    _categories_value = ReferenceField('categories_value', IndexValue, on_deserialize=_on_deserialize_index_value)
    _chunks = ListField('chunks', FieldTypes.reference(CategoricalChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [CategoricalChunk(it) for it in x] if x is not None else x)

    def __init__(self, op=None, shape=None, nsplits=None, dtype=None,
                 categories_value=None, chunks=None, **kw):
        super().__init__(_op=op, _shape=shape, _nsplits=nsplits, _dtype=dtype,
                         _categories_value=categories_value, _chunks=chunks, **kw)

    @property
    def params(self) -> Dict[str, Any]:
        # params return the properties which useful to rebuild a new tileable object
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'categories_value': self.categories_value,
        }

    @params.setter
    def params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        new_shape = params.pop('shape', None)
        if new_shape is not None:
            self._shape = new_shape
        dtype = params.pop('dtype', None)
        if dtype is not None:
            self._dtype = dtype
        categories_value = params.pop('categories_value', None)
        if categories_value is not None:
            self._categories_value = categories_value
        if params:  # pragma: no cover
            raise TypeError(f'Unknown params: {list(params)}')

    def refresh_params(self):
        # refresh params when chunks updated
        refresh_tileable_shape(self)
        if self._dtype is None:
            self._dtype = self.chunks[0].dtype
        if self._categories_value is None:
            categories = []
            for chunk in self.chunks:
                categories.extend(chunk.categories_value.to_pandas())
            self._categories_value = parse_index(
                pd.Categorical(categories).categories, store_data=True)

    def _to_str(self, representation=False):
        if is_build_mode() or len(self._executed_sessions) == 0:
            # in build mode, or not executed, just return representation
            if representation:
                return f'{self.type_name} <op={type(self.op).__name__}, key={self.key}>'
            else:
                return f'{self.type_name}(op={type(self.op).__name__})'
        else:
            data = self.fetch(session=self._executed_sessions[-1])
            return repr(data) if repr(data) else str(data)

    def __str__(self):
        return self._to_str(representation=False)

    def __repr__(self):
        return self._to_str(representation=True)

    def _equal(self, o):
        # FIXME We need to implemented a true `==` operator for DataFrameGroupby
        if is_build_mode():
            return self is o
        else:  # pragma: no cover
            return self == o

    @property
    def dtype(self):
        return getattr(self, '_dtype', None) or self.op.dtype

    @property
    def categories_value(self):
        return self._categories_value

    def __eq__(self, other):
        return self._equal(other)

    def __hash__(self):
        # NB: we have customized __eq__ explicitly, thus we need define __hash__ explicitly as well.
        return super().__hash__()


class Categorical(HasShapeTileable, _ToPandasMixin):
    __slots__ = ()
    _allow_data_type_ = (CategoricalData,)
    type_name = 'Categorical'

    def __len__(self):
        return len(self._data)

    def __eq__(self, other):
        return self._equal(other)

    def __hash__(self):
        # NB: we have customized __eq__ explicitly, thus we need define __hash__ explicitly as well.
        return super().__hash__()


INDEX_TYPE = (Index, IndexData)
INDEX_CHUNK_TYPE = (IndexChunk, IndexChunkData)
SERIES_TYPE = (Series, SeriesData)
SERIES_CHUNK_TYPE = (SeriesChunk, SeriesChunkData)
DATAFRAME_TYPE = (DataFrame, DataFrameData)
DATAFRAME_CHUNK_TYPE = (DataFrameChunk, DataFrameChunkData)
DATAFRAME_GROUPBY_TYPE = (DataFrameGroupBy, DataFrameGroupByData)
DATAFRAME_GROUPBY_CHUNK_TYPE = (DataFrameGroupByChunk, DataFrameGroupByChunkData)
SERIES_GROUPBY_TYPE = (SeriesGroupBy, SeriesGroupByData)
SERIES_GROUPBY_CHUNK_TYPE = (SeriesGroupByChunk, SeriesGroupByChunkData)
GROUPBY_TYPE = (GroupBy,) + DATAFRAME_GROUPBY_TYPE + SERIES_GROUPBY_TYPE
GROUPBY_CHUNK_TYPE = DATAFRAME_GROUPBY_CHUNK_TYPE + SERIES_GROUPBY_CHUNK_TYPE
CATEGORICAL_TYPE = (Categorical, CategoricalData)
CATEGORICAL_CHUNK_TYPE = (CategoricalChunk, CategoricalChunkData)
TILEABLE_TYPE = INDEX_TYPE + SERIES_TYPE + DATAFRAME_TYPE + GROUPBY_TYPE + CATEGORICAL_TYPE
CHUNK_TYPE = INDEX_CHUNK_TYPE + SERIES_CHUNK_TYPE + DATAFRAME_CHUNK_TYPE + \
             GROUPBY_CHUNK_TYPE + CATEGORICAL_CHUNK_TYPE

register_output_types(OutputType.dataframe, DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)
register_output_types(OutputType.series, SERIES_TYPE, SERIES_CHUNK_TYPE)
register_output_types(OutputType.index, INDEX_TYPE, INDEX_CHUNK_TYPE)
register_output_types(OutputType.categorical, CATEGORICAL_TYPE, CATEGORICAL_CHUNK_TYPE)
register_output_types(OutputType.dataframe_groupby, DATAFRAME_GROUPBY_TYPE, DATAFRAME_GROUPBY_CHUNK_TYPE)
register_output_types(OutputType.series_groupby, SERIES_GROUPBY_TYPE, SERIES_GROUPBY_CHUNK_TYPE)
