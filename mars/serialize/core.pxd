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


cpdef enum PrimitiveType:
    bool = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    uint8 = 6
    uint16 = 7
    uint32 = 8
    uint64 = 9
    float16 = 10
    float32 = 11
    float64 = 12
    bytes = 13
    unicode = 14
    complex64 = 24
    complex128 = 25


cpdef enum ExtendType:
    slice = 15
    arr = 16
    dtype = 17
    key = 18
    datetime64 = 19
    timedelta64 = 20


cdef class Identity:
    cdef public object type
    cdef public str name


cdef class List:
    cdef public object type
    cdef public str name


cdef class Tuple:
    cdef public object type
    cdef public str name


cdef class Dict:
    cdef public object key_type
    cdef public object value_type
    cdef public str name


cdef class Reference:
    cdef public object model
    cdef public str name


cdef class OneOf:
    cdef public tuple references


cdef class ValueType:
    pass


cdef class SelfReferenceOverwritten(Exception):
    pass


cdef class Field:
    cdef object tag
    cdef object default_val
    cdef str _tag_name
    cdef object _type
    cdef object _model_cls

    cdef public bint weak_ref
    cdef public str attr
    cdef public object on_serialize
    cdef public object on_deserialize

    cpdef str tag_name(self, Provider provider)
    cpdef serialize(self, Provider provider, model_instance, obj)
    cpdef deserialize(self, Provider provider, model_instance, obj, list callbacks, dict key_to_instance)


cdef class AnyField(Field):
    pass


cdef class IdentityField(Field):
    pass


cdef class BoolField(Field):
    pass


cdef class Int8Field(Field):
    pass


cdef class Int16Field(Field):
    pass


cdef class Int32Field(Field):
    pass


cdef class Int64Field(Field):
    pass


cdef class UInt8Field(Field):
    pass


cdef class UInt16Field(Field):
    pass


cdef class UInt32Field(Field):
    pass


cdef class UInt64Field(Field):
    pass


cdef class Float16Field(Field):
    pass


cdef class Float32Field(Field):
    pass


cdef class Float64Field(Field):
    pass


cdef class Complex64Field(Field):
    pass


cdef class Complex128Field(Field):
    pass


cdef class StringField(Field):
    pass


cdef class BytesField(Field):
    pass


cdef class UnicodeField(Field):
    pass


cdef class KeyField(Field):
    pass


cdef class NDArrayField(Field):
    pass


cdef class Datetime64Field(Field):
    pass


cdef class Timedelta64Field(Field):
    pass


cdef class DataTypeField(Field):
    pass


cdef class ListField(Field):
    cdef public object _nest_ref


cdef class TupleField(Field):
    pass


cdef class DictField(Field):
    pass


cdef class ReferenceField(Field):
    cdef public object _model


cdef class OneOfField(Field):
    cdef public list fields


cdef class KeyPlaceholder:
    cdef public str key
    cdef public str id


cdef class AttrWrapper:
    cdef object _obj

    cpdef asdict(self)


cpdef enum ProviderType:
    protobuf = 1
    json = 2


cdef class Provider:
    cdef public object type

    cpdef serialize_field(self, Field field, model_instance, obj)
    cpdef serialize_model(self, model_instance, obj=?)
    cpdef serialize_attribute_as_dict(self, model_instance, obj=?)
    cpdef deserialize_model(self, model_cls, obj, list callbacks, dict key_to_instance)
