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

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pass

from ..utils import on_serialize_shape, on_deserialize_shape, on_serialize_numpy_type
from ..core import ChunkData, Chunk, TileableEntity, TileableData, is_eager_mode
from ..serialize import Serializable, ValueType, ProviderType, DataTypeField, AnyField, \
    SeriesField, BoolField, Int64Field, Int32Field, StringField, ListField, SliceField, \
    TupleField, OneOfField, ReferenceField, NDArrayField


class IndexValue(Serializable):
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

        def to_pandas(self):
            kw = {field.tag_name(None): getattr(self, attr, None)
                  for attr, field in self._FIELDS.items()
                  if attr not in super(type(self), self)._FIELDS}
            if kw['data'] is None:
                kw['data'] = []
            return getattr(pd, type(self).__name__)(**kw)

    class Index(IndexBase):
        _name = AnyField('name')
        _data = NDArrayField('data')
        _dtype = DataTypeField('dtype')

    class RangeIndex(IndexBase):
        _name = AnyField('name')
        _slice = SliceField('slice')

        @property
        def slice(self):
            return self._slice

        def to_pandas(self):
            slc = self._slice
            return pd.RangeIndex(slc.start, slc.stop, slc.step)

    class CategoricalIndex(IndexBase):
        _name = AnyField('name')
        _data = NDArrayField('data')
        _categories = ListField('categories')
        _ordered = BoolField('ordered')

    class IntervalIndex(IndexBase):
        _name = AnyField('name')
        _data = NDArrayField('data')
        _closed = BoolField('closed')

    class DatetimeIndex(IndexBase):
        _name = AnyField('name')
        _data = NDArrayField('data')
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
        _data = NDArrayField('data')
        _unit = AnyField('unit')
        _freq = AnyField('freq')
        _start = AnyField('start')
        _periods = Int64Field('periods')
        _end = AnyField('end')
        _closed = AnyField('closed')

    class PeriodIndex(IndexBase):
        _name = AnyField('name')
        _data = NDArrayField('data')
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
        _data = NDArrayField('data')
        _dtype = DataTypeField('dtype')

    class UInt64Index(IndexBase):
        _name = AnyField('name')
        _data = NDArrayField('data')
        _dtype = DataTypeField('dtype')

    class Float64Index(IndexBase):
        _name = AnyField('name')
        _data = NDArrayField('data')
        _dtype = DataTypeField('dtype')

    class MultiIndex(IndexBase):
        _names = ListField('name')
        _data = NDArrayField('data')
        _sortorder = Int32Field('sortorder')

        def to_pandas(self):
            data = getattr(self, '_data', None)
            if data is None:
                return pd.MultiIndex.from_arrays([[], []], sortorder=self._sortorder,
                                                 names=self._names)
            return pd.MultiIndex.from_tuples(np.asarray(data), sortorder=self._sortorder,
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

    def has_value(self):
        if isinstance(self._index_value, self.RangeIndex):
            return True
        elif getattr(self._index_value, '_data', None) is not None:
            return True
        return False

    def to_pandas(self):
        return self._index_value.to_pandas()

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from ..serialize.protos.indexvalue_pb2 import IndexValue as IndexValueDef
            return IndexValueDef
        return super(IndexValue, cls).cls(provider)


class IndexChunkData(ChunkData):
    __slots__ = ()

    # required fields
    _shape = TupleField('shape', ValueType.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)
    # optional field
    _dtype = DataTypeField('dtype')
    _name = AnyField('name')
    _index_value = ReferenceField('index_value', IndexValue)

    def __init__(self, op=None, shape=None, index=None, dtype=None, name=None,
                 index_value=None, **kw):
        super(IndexChunkData, self).__init__(_op=op, _shape=shape, _index=index,
                                             _dtype=dtype, _name=name,
                                             _index_value=index_value, **kw)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from ..serialize.protos.dataframe_pb2 import IndexChunkDef
            return IndexChunkDef
        return super(IndexChunkData, cls).cls(provider)

    @property
    def params(self):
        # params return the properties which useful to rebuild a new chunk
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'index': self.index,
            'index_value': self.index_value,
            'name': self.name
        }

    @property
    def shape(self):
        return getattr(self, '_shape', None)

    @property
    def dtype(self):
        return self._dtype

    @property
    def index_value(self):
        return self._index_value


class IndexChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (IndexChunkData,)


class IndexData(TileableData):
    __slots__ = ()

    # optional field
    _dtype = DataTypeField('dtype')
    _name = AnyField('name')
    _index_value = ReferenceField('index_value', IndexValue)
    _chunks = ListField('chunks', ValueType.reference(IndexChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [IndexChunk(it) for it in x] if x is not None else x)

    def __init__(self, op=None, shape=None, nsplits=None, dtype=None,
                 name=None, index_value=None, chunks=None, **kw):
        super(IndexData, self).__init__(_op=op, _shape=shape, _nsplits=nsplits,
                                        _dtype=dtype, _name=name, _index_value=index_value,
                                        _chunks=chunks, **kw)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from ..serialize.protos.dataframe_pb2 import IndexDef
            return IndexDef
        return super(IndexData, cls).cls(provider)

    @property
    def params(self):
        # params return the properties which useful to rebuild a new tileable object
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'name': self.name,
            'index_value': self.index_value,
        }

    def __str__(self):
        if is_eager_mode():
            return 'Index(op={0}, data=\n{1})'.format(self.op.__class__.__name__,
                                                      str(self.fetch()))
        else:
            return 'Index(op={0})'.format(self.op.__class__.__name__)

    def __repr__(self):
        if is_eager_mode():
            return 'Index <op={0}, key={1}, data=\n{2}>'.format(self.op.__class__.__name__,
                                                                self.key,
                                                                repr(self.fetch()))
        else:
            return 'Index <op={0}, key={1}>'.format(self.op.__class__.__name__,
                                                    self.key)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None) or self.op.dtype

    @property
    def index_value(self):
        return self._index_value


class Index(TileableEntity):
    __slots__ = ()
    _allow_data_type_ = (IndexData,)


class SeriesChunkData(ChunkData):
    __slots__ = ()

    # required fields
    _shape = TupleField('shape', ValueType.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)
    # optional field
    _dtype = DataTypeField('dtype')
    _name = AnyField('name')
    _index_value = ReferenceField('index_value', IndexValue)

    def __init__(self, op=None, shape=None, index=None, dtype=None, name=None,
                 index_value=None, **kw):
        super(SeriesChunkData, self).__init__(_op=op, _shape=shape, _index=index,
                                              _dtype=dtype, _name=name,
                                              _index_value=index_value, **kw)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from ..serialize.protos.dataframe_pb2 import SeriesChunkDef
            return SeriesChunkDef
        return super(SeriesChunkData, cls).cls(provider)

    @property
    def params(self):
        # params return the properties which useful to rebuild a new chunk
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'index': self.index,
            'index_value': self.index_value,
            'name': self.name
        }

    @property
    def shape(self):
        return getattr(self, '_shape', None)

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self):
        return self._name

    @property
    def index_value(self):
        return self._index_value


class SeriesChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (SeriesChunkData,)


class SeriesData(TileableData):
    __slots__ = ()

    # optional field
    _dtype = DataTypeField('dtype')
    _name = AnyField('name')
    _index_value = ReferenceField('index_value', IndexValue)
    _chunks = ListField('chunks', ValueType.reference(SeriesChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [SeriesChunk(it) for it in x] if x is not None else x)

    def __init__(self, op=None, shape=None, nsplits=None, dtype=None,
                 name=None, index_value=None, chunks=None, **kw):
        super(SeriesData, self).__init__(_op=op, _shape=shape, _nsplits=nsplits,
                                         _dtype=dtype, _name=name, _index_value=index_value,
                                         _chunks=chunks, **kw)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from ..serialize.protos.dataframe_pb2 import SeriesDef
            return SeriesDef
        return super(SeriesData, cls).cls(provider)

    @property
    def params(self):
        # params return the properties which useful to rebuild a new tileable object
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'name': self.name,
            'index_value': self.index_value,
        }

    def __str__(self):
        if is_eager_mode():
            return 'Series(op={0}, data=\n{1})'.format(self.op.__class__.__name__,
                                                       str(self.fetch()))
        else:
            return 'Series(op={0})'.format(self.op.__class__.__name__)

    def __repr__(self):
        if is_eager_mode():
            return 'Series <op={0}, key={1}, data=\n{2}>'.format(self.op.__class__.__name__,
                                                                 self.key,
                                                                 repr(self.fetch()))
        else:
            return 'Series <op={0}, key={1}>'.format(self.op.__class__.__name__,
                                                     self.key)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None) or self.op.dtype

    @property
    def name(self):
        return self._name

    @property
    def index_value(self):
        return self._index_value


class Series(TileableEntity):
    __slots__ = ()
    _allow_data_type_ = (SeriesData,)


class DataFrameChunkData(ChunkData):
    __slots__ = ()

    # required fields
    _shape = TupleField('shape', ValueType.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)
    # optional fields
    _dtypes = SeriesField('dtypes')
    _index_value = ReferenceField('index_value', IndexValue)
    _columns_value = ReferenceField('columns_value', IndexValue)

    def __init__(self, op=None, shape=None, index=None, dtypes=None,
                 index_value=None, columns_value=None, **kw):
        super(DataFrameChunkData, self).__init__(_op=op, _shape=shape, _index=index,
                                                 _dtypes=dtypes, _index_value=index_value,
                                                 _columns_value=columns_value, **kw)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from ..serialize.protos.dataframe_pb2 import DataFrameChunkDef
            return DataFrameChunkDef
        return super(DataFrameChunkData, cls).cls(provider)

    @property
    def params(self):
        # params return the properties which useful to rebuild a new chunk
        return {
            'shape': self.shape,
            'dtypes': self.dtypes,
            'index': self.index,
            'index_value': self.index_value,
            'columns_value': self.columns,
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
    def index_value(self):
        return self._index_value

    @property
    def columns(self):
        return self._columns_value


class DataFrameChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (DataFrameChunkData,)


class DataFrameData(TileableData):
    __slots__ = ()

    # optional fields
    _dtypes = SeriesField('dtypes')
    _index_value = ReferenceField('index_value', IndexValue)
    _columns_value = ReferenceField('columns_value', IndexValue)
    _chunks = ListField('chunks', ValueType.reference(DataFrameChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [DataFrameChunk(it) for it in x] if x is not None else x)

    def __init__(self, op=None, shape=None, nsplits=None, dtypes=None,
                 index_value=None, columns_value=None, chunks=None, **kw):
        super(DataFrameData, self).__init__(_op=op, _shape=shape, _nsplits=nsplits,
                                            _dtypes=dtypes, _index_value=index_value,
                                            _columns_value=columns_value,
                                            _chunks=chunks, **kw)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from ..serialize.protos.dataframe_pb2 import DataFrameDef
            return DataFrameDef
        return super(DataFrameData, cls).cls(provider)

    def __str__(self):
        if is_eager_mode():
            return 'DataFrame(op={0}, data=\n{1})'.format(self.op.__class__.__name__,
                                                          str(self.fetch()))
        else:
            return 'DataFrame(op={0})'.format(self.op.__class__.__name__)

    def __repr__(self):
        if is_eager_mode():
            return 'DataFrame <op={0}, key={1}, data=\n{2}>'.format(self.op.__class__.__name__,
                                                                    self.key,
                                                                    repr(self.fetch()))
        else:
            return 'DataFrame <op={0}, key={1}>'.format(self.op.__class__.__name__,
                                                        self.key)

    @property
    def params(self):
        # params return the properties which useful to rebuild a new tileable object
        return {
            'shape': self.shape,
            'dtypes': self.dtypes,
            'index_value': self.index_value,
            'columns_value': self.columns
        }

    @property
    def dtypes(self):
        dt = getattr(self, '_dtypes', None)
        if dt is not None:
            return dt
        return getattr(self.op, 'dtypes', None)

    @property
    def index_value(self):
        return self._index_value

    @property
    def columns(self):
        return self._columns_value

    def _equal(self, o):
        from ..core import build_mode
        # FIXME We need to implemented a true `==` operator for DataFrame, current we just need
        # to do `self is o` under build mode to make the `iloc.__setitem__` happy.
        if build_mode().is_build_mode:
            return self is o
        else:
            return self == o

    def to_tensor(self):
        from ..tensor.datasource.from_dataframe import from_dataframe
        return from_dataframe(self)

    @staticmethod
    def from_tensor(in_tensor):
        from .datasource.from_tensor import from_tensor
        return from_tensor(in_tensor)

    @staticmethod
    def from_records(records, **kw):
        from .datasource.from_records import from_records
        return from_records(records, **kw)


class DataFrame(TileableEntity):
    __slots__ = ()
    _allow_data_type_ = (DataFrameData,)

    def __eq__(self, other):
        return self._equal(other)

    def __hash__(self):
        # NB: we have customized __eq__ explicitly, thus we need define __hash__ explicitly as well.
        return super(DataFrame, self).__hash__()

    def to_tensor(self):
        return self._data.to_tensor()

    def from_tensor(self, in_tensor):
        return self._data.from_tensor(in_tensor)

    def from_records(self, records, **kw):
        return self._data.from_records(records, **kw)

    def __mars_tensor__(self, dtype=None, order='K'):
        return self._data.to_tensor().astype(dtype=dtype, order=order, copy=False)

    def __getattr__(self, key):
        try:
            return getattr(self._data, key)
        except AttributeError:
            if key in self.dtypes:
                return self[key]
            else:
                raise


INDEX_TYPE = (Index, IndexData)
INDEX_CHUNK_TYPE = (IndexChunk, IndexChunkData)
SERIES_TYPE = (Series, SeriesData)
SERIES_CHUNK_TYPE = (SeriesChunk, SeriesChunkData)
DATAFRAME_TYPE = (DataFrame, DataFrameData)
DATAFRAME_CHUNK_TYPE = (DataFrameChunk, DataFrameChunkData)
TILEABLE_TYPE = INDEX_TYPE + SERIES_TYPE + DATAFRAME_TYPE
CHUNK_TYPE = INDEX_CHUNK_TYPE + SERIES_CHUNK_TYPE + DATAFRAME_CHUNK_TYPE
