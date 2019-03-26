#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from ..core import ChunkData, Chunk, Entity, TilesableData
from ..serialize import Serializable, ValueType, ProviderType, DataTypeField, AnyField, SeriesField, \
    BoolField, Int64Field, Int32Field, StringField, ListField, SliceField, OneOfField, ReferenceField


class IndexValue(Serializable):
    __slots__ = ()

    class IndexBase(Serializable):
        _key = StringField('key')  # to identify if the index is the same
        _is_monotonic_increasing = BoolField('is_monotonic_increasing')
        _is_monotonic_decreasing = BoolField('is_monotonic_decreasing')
        _is_unique = BoolField('is_unique')
        _should_be_monotonic = BoolField('should_be_monotonic')
        _max_val = AnyField('max_val')
        _max_val_close = BoolField('max_val_close')
        _min_val = AnyField('min_val')
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

    class Index(IndexBase):
        _name = AnyField('name')
        _data = ListField('data')
        _dtype = DataTypeField('dtype')

    class RangeIndex(IndexBase):
        _name = AnyField('name')
        _slice = SliceField('slice')

    class CategoricalIndex(IndexBase):
        _name = AnyField('name')
        _categories = ListField('categories')
        _ordered = BoolField('ordered')

    class IntervalIndex(IndexBase):
        _name = AnyField('name')
        _data = ListField('data')
        _closed = BoolField('closed')

    class DatetimeIndex(IndexBase):
        _name = AnyField('name')
        _data = ListField('data')
        _freq = AnyField('freq')
        _start = AnyField('start')
        _periods = Int64Field('periods')
        _end = AnyField('end')
        _closed = AnyField('closed')
        _tz = AnyField('tz')
        _dayfirst = BoolField('dayfirst')
        _yearfirst = BoolField('yearfirst')

    class TimedeltaIndex(IndexBase):
        _name = AnyField('name')
        _data = ListField('data')
        _unit = AnyField('unit')
        _freq = AnyField('freq')
        _start = AnyField('start')
        _periods = Int64Field('periods')
        _end = AnyField('end')
        _closed = AnyField('closed')

    class PeriodIndex(IndexBase):
        _name = AnyField('name')
        _data = ListField('data')
        _freq = AnyField('freq')
        _start = AnyField('start')
        _periods = Int64Field('periods')
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

    class Int64Index(IndexBase):
        _name = AnyField('name')
        _data = ListField('data')
        _dtype = DataTypeField('dtype')

    class UInt64Index(IndexBase):
        _name = AnyField('name')
        _data = ListField('data')
        _dtype = DataTypeField('dtype')

    class Float64Index(IndexBase):
        _name = AnyField('name')
        _data = ListField('data')
        _dtype = DataTypeField('dtype')

    class MultiIndex(IndexBase):
        _names = ListField('name')
        _data = ListField('data')
        _sortorder = Int32Field('sortorder')

    _index_value = OneOfField('index_value', index=Index,
                              range_index=RangeIndex, categorical_index=CategoricalIndex,
                              interval_index=IntervalIndex, datetime_index=DatetimeIndex,
                              timedelta_index=TimedeltaIndex, period_index=PeriodIndex,
                              int64_index=Int64Field, uint64_index=UInt64Index,
                              float64_index=Float64Index, multi_index=MultiIndex)

    def __mars_tokenize__(self):
        # return object for tokenize
        v = self._index_value
        return [type(v).__name__] + [getattr(v, f, None) for f in v.__slots__]

    @property
    def value(self):
        return self._index_value

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
    def is_unique(self):
        return self._index_value.is_unique

    @property
    def min_val(self):
        return self._index_value.min_val

    @property
    def max_val(self):
        return self._index_value.max_val


class IndexChunkData(ChunkData):
    __slots__ = ()

    # optional field
    _dtype = DataTypeField('dtype')
    _index_value = ReferenceField('index_value', IndexValue)

    @property
    def dtype(self):
        return self._dtype

    @property
    def index_value(self):
        return self._index_value


class IndexChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (IndexChunkData,)


class IndexData(TilesableData):
    __slots__ = ()

    # optional field
    _dtype = DataTypeField('dtype')
    _index_value = ReferenceField('index_value', IndexValue)
    _chunks = ListField('chunks', ValueType.reference(IndexChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [IndexChunk(it) for it in x] if x is not None else x)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from ..serialize.protos.dataframe_pb2 import IndexDef
            return IndexDef
        return super(IndexData, cls).cls(provider)

    def __repr__(self):
        return 'Index <op={0}, key={1}>'.format(self.op.__class__.__name__, self.key)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None) or self.op.dtype

    @property
    def index_value(self):
        return self._index_value


class Index(Entity):
    _allow_data_type_ = (IndexData,)


class SeriesChunkData(ChunkData):
    __slots__ = ()

    # optional field
    _dtype = DataTypeField('dtype')
    _index_value = ReferenceField('index_value', IndexValue)

    @property
    def dtype(self):
        return self._dtype

    @property
    def index_value(self):
        return self._index_value


class SeriesChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (SeriesChunkData,)


class SeriesData(TilesableData):
    __slots__ = ()

    # optional field
    _dtype = DataTypeField('dtype')
    _name = AnyField('name')
    _index_value = ReferenceField('index_value', IndexValue)
    _chunks = ListField('chunks', ValueType.reference(SeriesChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [SeriesChunk(it) for it in x] if x is not None else x)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None) or self.op.dtype

    @property
    def name(self):
        return self._name

    @property
    def index_value(self):
        return self._index_value


class Series(Entity):
    __slots__ = ()
    _allow_data_type_ = (SeriesData,)


class DataFrameChunkData(ChunkData):
    __slots__ = ()

    # optional field
    _dtypes = SeriesField('dtypes')
    _index_value = ReferenceField('index_value', IndexValue)
    _columns_value = ReferenceField('columns_value', IndexValue)

    @property
    def dtypes(self):
        return getattr(self, '_dtypes', None) or getattr(self.op, 'dtypes', None)

    @property
    def index_value(self):
        return self._index_value

    @property
    def columns(self):
        return self._columns_value


class DataFrameChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (DataFrameChunkData,)


class DataFrameData(TilesableData):
    __slots__ = ()

    # optional field
    _dtypes = SeriesField('dtypes')
    _index_value = ReferenceField('index_value', IndexValue)
    _columns_value = ReferenceField('columns_value', IndexValue)
    _chunks = ListField('chunks', ValueType.reference(DataFrameChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [DataFrameChunk(it) for it in x] if x is not None else x)

    @property
    def dtypes(self):
        return getattr(self, '_dtypes', None) or getattr(self.op, 'dtypes', None)

    @property
    def index_value(self):
        return self._index_value

    @property
    def columns(self):
        return self._columns_value


class DataFrame(Entity):
    __slots__ = ()
    _allow_data_type_ = (DataFrameData,)


INDEX_TYPE = (Index, IndexData)
SERIES_TYPE = (Series, SeriesData)
DATAFRAME_TYPE = (DataFrame, DataFrameData)
CHUNK_TYPE = (DataFrameChunk, DataFrameChunkData)
