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
from ..serialize import Serializable, ProviderType, DataTypeField, AnyField, SeriesField, \
    BoolField, Int64Field, Int32Field, ListField, SliceField, OneOfField, ReferenceField


class IndexValue(Serializable):
    __slots__ = ()

    class Index(Serializable):
        _name = AnyField('name')
        _data = ListField('data')
        _dtype = DataTypeField('dtype')

    class RangeIndex(Serializable):
        _name = AnyField('name')
        _slice = SliceField('slice')

    class CategoricalIndex(Serializable):
        _name = AnyField('name')
        _categories = ListField('categories')
        _ordered = BoolField('ordered')

    class IntervalIndex(Serializable):
        _name = AnyField('name')
        _data = ListField('data')
        _closed = BoolField('closed')

    class DatetimeIndex(Serializable):
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

    class TimedeltaIndex(Serializable):
        _name = AnyField('name')
        _data = ListField('data')
        _unit = AnyField('unit')
        _freq = AnyField('freq')
        _start = AnyField('start')
        _periods = Int64Field('periods')
        _end = AnyField('end')
        _closed = AnyField('closed')

    class PeriodIndex(Serializable):
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

    class Int64Index(Serializable):
        _name = AnyField('name')
        _data = ListField('data')
        _dtype = DataTypeField('dtype')

    class UInt64Index(Serializable):
        _name = AnyField('name')
        _data = ListField('data')
        _dtype = DataTypeField('dtype')

    class Float64Index(Serializable):
        _name = AnyField('name')
        _data = ListField('data')
        _dtype = DataTypeField('dtype')

    class MultiIndex(Serializable):
        _names = ListField('name')
        _levels = ListField('levels')
        _labels = ListField('labels')
        _sortorder = Int32Field('sortorder')

    _index_value = OneOfField('index_value', index=Index,
                              range_index=RangeIndex, categorical_index=CategoricalIndex,
                              interval_index=IntervalIndex, datetime_index=DatetimeIndex,
                              timedelta_index=TimedeltaIndex, period_index=PeriodIndex,
                              int64_index=Int64Field, uint64_index=UInt64Index,
                              float64_index=Float64Index, multi_index=MultiIndex)


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

    @property
    def dtypes(self):
        return getattr(self, '_dtypes', None) or getattr(self.op, 'dtypes', None)

    @property
    def index_value(self):
        return self._index_value

    @property
    def columns(self):
        return self._columns


class DataFrameChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (DataFrameChunkData,)


class DataFrameData(TilesableData):
    __slots__ = ()

    # optional field
    _dtypes = SeriesField('dtypes')
    _index_value = ReferenceField('index_value', IndexValue)

    @property
    def dtypes(self):
        return getattr(self, '_dtypes', None) or getattr(self.op, 'dtypes', None)

    @property
    def index_value(self):
        return self._index_value

    @property
    def columns(self):
        return self._columns


class DataFrame(Entity):
    __slots__ = ()
    _allow_data_type_ = (DataFrameData,)


DATAFRAME_TYPE = (DataFrame, DataFrameData)
CHUNK_TYPE = (DataFrameChunk, DataFrameChunkData)
