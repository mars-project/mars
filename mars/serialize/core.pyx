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

import importlib
import inspect
import copy
from collections import Iterable

from ..compat import six, OrderedDict


from cpython.version cimport PY_MAJOR_VERSION

cdef bint PY2 = PY_MAJOR_VERSION == 2
cdef bint PY3 = PY_MAJOR_VERSION == 3


cdef class Identity:
    def __init__(self, tp=None):
        if tp is None:
            tp = PrimitiveType.unicode if six.PY3 else PrimitiveType.bytes
        self.type = tp
        self.name = 'id'

    def __call__(self, tp):
        return Identity(tp)


cdef class List:
    def __init__(self, tp=None):
        self.type = tp
        self.name = 'list'

    def __call__(self, tp):
        assert self.type is None
        return List(tp)


cdef class Tuple:
    def __init__(self, *tps):
        if len(tps) == 1 and not isinstance(tps[0], Iterable):
            self.type = tps[0]
        else:
            self.type = tps if tps else None
        self.name = 'tuple'

    def __call__(self, *tps):
        assert self.type is None
        return Tuple(*tps)


cdef class Dict:
    def __init__(self, key_type=None, value_type=None):
        self.key_type = key_type
        self.value_type = value_type
        self.name = 'dict'

    def __call__(self, key_type, value_type):
        assert self.key_type is None and self.value_type is None
        return Dict(key_type, value_type)


cdef class Reference:
    def __init__(self, model):
        self.model = model
        self.name = 'reference'


cdef class OneOf:
    def __init__(self, *references):
        self.references = references


cdef class ValueType:
    bool = PrimitiveType.bool
    int8 = PrimitiveType.int8
    int16 = PrimitiveType.int16
    int32 = PrimitiveType.int32
    int64 = PrimitiveType.int64
    uint8 = PrimitiveType.uint8
    uint16 = PrimitiveType.uint16
    uint32 = PrimitiveType.uint32
    uint64 = PrimitiveType.uint64
    float16 = PrimitiveType.float16
    float32 = PrimitiveType.float32
    float64 = PrimitiveType.float64
    bytes = PrimitiveType.bytes
    unicode = PrimitiveType.unicode
    string = PrimitiveType.unicode if PY3 else PrimitiveType.bytes
    complex64 = PrimitiveType.complex64
    complex128 = PrimitiveType.complex128
    slice = ExtendType.slice
    arr = ExtendType.arr
    dtype = ExtendType.dtype
    key = ExtendType.key
    datetime64 = ExtendType.datetime64
    timedelta64 = ExtendType.timedelta64
    index = ExtendType.index
    series = ExtendType.series
    dataframe = ExtendType.dataframe

    identity = Identity()

    list = List()
    tuple = Tuple()
    dict = Dict()

    reference = Reference
    oneof = OneOf


cdef class Field:
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        self._model_cls = None
        self.attr = None

        self.tag = tag
        self._tag_name = None
        if not callable(self.tag):
            self._tag_name = self.tag
        self.default_val = default
        self.weak_ref = weak_ref
        self.on_serialize = on_serialize
        self.on_deserialize = on_deserialize

    @property
    def model(self):
        return self._model_cls

    @model.setter
    def model(self, model_cls):
        self._model_cls = model_cls

    cpdef str tag_name(self, Provider provider):
        if self._tag_name is None:
            return self.tag(provider)
        return self._tag_name

    cpdef serialize(self, Provider provider, model_instance, obj):
        return provider.serialize_field(self, model_instance, obj)

    cpdef deserialize(self, Provider provider, model_instance, obj, list callbacks, dict key_to_instance):
        return provider.deserialize_field(self, model_instance, obj, callbacks, key_to_instance)

    @property
    def type(self):
        return self._type

    @property
    def default(self):
        return self.default_val


cdef class AnyField(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(AnyField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = None


cdef class IdentityField(Field):
    def __init__(self, tag, tp=None, default=None, bint weak_ref=False,
                 on_serialize=None, on_deserialize=None):
        super(IdentityField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        if tp is not None:
            self._type = ValueType.identity(tp)
        else:
            self._type = ValueType.identity


cdef class BoolField(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(BoolField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.bool


cdef class Int8Field(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(Int8Field, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.int8


cdef class Int16Field(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(Int16Field, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.int16


cdef class Int32Field(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(Int32Field, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.int32


cdef class Int64Field(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(Int64Field, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.int64


cdef class UInt8Field(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(UInt8Field, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.uint8


cdef class UInt16Field(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(UInt16Field, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.uint16


cdef class UInt32Field(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(UInt32Field, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.uint32


cdef class UInt64Field(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(UInt64Field, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.uint64


cdef class Float16Field(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(Float16Field, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.float64


cdef class Float32Field(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(Float32Field, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.float32


cdef class Float64Field(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(Float64Field, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.float64


cdef class Complex64Field(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(Complex64Field, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.complex64


cdef class Complex128Field(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(Complex128Field, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.complex128


cdef class StringField(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(StringField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.bytes if PY2 else ValueType.unicode


cdef class BytesField(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(BytesField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.bytes


cdef class UnicodeField(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(UnicodeField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.unicode


cdef class KeyField(Field):
    # this field is to store the HasKey object
    # we only store the key when pickling the serializable object

    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(KeyField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.key


cdef class NDArrayField(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(NDArrayField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.arr


cdef class Datetime64Field(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(Datetime64Field, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.datetime64


cdef class Timedelta64Field(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(Timedelta64Field, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.timedelta64


cdef class DataTypeField(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(DataTypeField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.dtype


cdef class IndexField(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(IndexField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.index


cdef class SeriesField(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(SeriesField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.series


cdef class DataFrameField(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(DataFrameField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.dataframe


cdef class SliceField(Field):
    def __init__(self, tag, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(SliceField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.slice


cdef inline _handle_nest_reference(field, ref):
    if not isinstance(ref, Reference):
        return ref

    reference_field = ReferenceField(None, ref.model)
    reference_field.model = field.model
    return reference_field.type


cdef class ListField(Field):
    def __init__(self, tag, tp=None, default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(ListField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        if tp is not None and type(tp) == Reference:
            self._type = None
            self._nest_ref = tp
        else:
            self._type = ValueType.list(tp)

    @property
    def model(self):
        return self._model_cls

    @model.setter
    def model(self, new_model_cls):
        if getattr(self, '_nest_ref', None) is not None and \
                self._nest_ref.model == 'self' and self._model_cls is not None and \
                new_model_cls is not None:
            raise SelfReferenceOverwritten('self reference is overwritten')
        self._model_cls = new_model_cls

    @property
    def type(self):
        if self._type is None:
            self._type = ValueType.list(
                _handle_nest_reference(self, self._nest_ref))
        return self._type


cdef class TupleField(Field):
    def __init__(self, tag, *tps, **kwargs):
        default = kwargs.pop('default', None)
        weak_ref = kwargs.pop('weak_ref', False)
        on_serialize = kwargs.pop('on_serialize', None)
        on_deserialize = kwargs.pop('on_deserialize', None)
        super(TupleField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        if len(tps) == 1 and isinstance(tps[0], Iterable):
            tps = tps[0]
        self._type = ValueType.tuple(*tps)


cdef class DictField(Field):
    def __init__(self, tag, key_type=None, value_type=None,
                 default=None, bint weak_ref=False, on_serialize=None, on_deserialize=None):
        super(DictField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        self._type = ValueType.dict(key_type, value_type)


cdef class ReferenceField(Field):
    def __init__(self, tag, model, default=None, bint weak_ref=False,
                 on_serialize=None, on_deserialize=None):
        super(ReferenceField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)
        if not isinstance(model, str):
            self._type = ValueType.reference(model)
        else:
            self._type = None
            self._model = model

    @property
    def model(self):
        return self._model_cls

    @model.setter
    def model(self, new_model_cls):
        if getattr(self, '_model', None) == 'self' and \
                self._model_cls is not None and new_model_cls is not None:
            raise SelfReferenceOverwritten('self reference is overwritten')
        self._model_cls = new_model_cls

    @property
    def type(self):
        if not self._type:
            if self._model == 'self':
                self._type = ValueType.reference(self.model)
            elif '.' not in self._model:
                model = getattr(inspect.getmodule(self.model), self._model)
                self._type = ValueType.reference(model)
            else:
                module, name = self._model.rsplit('.', 1)
                model = getattr(importlib.import_module(module), name)
                self._type = ValueType.reference(model)

        return self._type


cdef class OneOfField(Field):
    def __init__(self, tag, **kw):
        default = kw.pop('default', None)
        weak_ref = kw.pop('weak_ref', False)
        on_serialize = kw.pop('on_serialize', None)
        on_deserialize = kw.pop('on_deserialize', None)
        super(OneOfField, self).__init__(
            tag, default=default, weak_ref=weak_ref,
            on_serialize=on_serialize, on_deserialize=on_deserialize)

        self.fields = [ReferenceField(tag, model)
                       for tag, model in kw.items()]

    @property
    def type(self):
        if self._type is None:
            self._type = ValueType.oneof(*[f.type for f in self.fields])
        return self._type

    @property
    def attrs(self):
        return [f.attr for f in self.fields]


cdef inline set_model(dict fields, cls):
    cdef str slot
    cdef bint modified

    for slot, field in fields.items():
        if not isinstance(field, OneOfField):
            try:
                field.model = cls
            except SelfReferenceOverwritten:
                field = copy.copy(field)
                # reset old model after copy
                field.model = None
                field.model = cls
                cls._FIELDS[slot] = field
        else:
            one_field_fields = []
            modified = False
            for f in field.fields:
                try:
                    f.model = cls
                except SelfReferenceOverwritten:
                    f = copy.copy(f)
                    # reset old model after copy
                    f.model = None
                    f.model = cls
                    modified = True
                f.attr = field.attr
                one_field_fields.append(f)
            if modified:
                field.fields = one_field_fields


class SerializableMetaclass(type):
    def __new__(mcs, str name, tuple bases, dict kv):
        cdef list slots
        cdef set sslots
        cdef object props
        cdef dict fields
        cdef str k
        cdef object v

        slots = list(kv.pop('__slots__', ()))
        sslots = set(slots)

        props = dict()
        for base in bases:
            if hasattr(base, '_FIELDS'):
                props.update(base._FIELDS)
        props.update(kv)
        props = OrderedDict(sorted(props.items(),
                                   key=lambda x: x[0] in kv, reverse=True))

        fields = {}
        for k, v in props.items():
            if not isinstance(v, Field):
                continue

            if k not in sslots:
                slots.append(k)

            if isinstance(v, IdentityField):
                if '_ID_FIELD' not in kv:
                    kv['_ID_FIELD'] = (v,)
                else:
                    kv['_ID_FIELD'] += (v,)

            fields[k] = v
            v.attr = k

            if k in kv:
                del kv[k]

        kv['_FIELDS'] = fields
        kv['__slots__'] = tuple(slots)

        cls = type.__new__(mcs, name, bases, kv)
        set_model(fields, cls)
        return cls


class Serializable(six.with_metaclass(SerializableMetaclass)):
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        for slot, arg in zip(self.__slots__, args):
            object.__setattr__(self, slot, arg)

        for key, val in kwargs.items():
            object.__setattr__(self, key, val)

    @classmethod
    def cls(cls, Provider provider):
        if provider.type == ProviderType.json:
            return dict

        raise TypeError('Unknown provider type `{0}` for class `{1}`'.format(
            ProviderType(provider.type).name, cls.__name__))

    def serialize(self, Provider provider, obj=None):
        return provider.serialize_model(self, obj=obj)

    @classmethod
    def deserialize(cls, Provider provider, obj, list callbacks=None, dict key_to_instance=None):
        call_cb = callbacks is None
        callbacks = [] if callbacks is None else callbacks
        key_to_instance = {} if key_to_instance is None else key_to_instance
        obj = provider.deserialize_model(cls, obj, callbacks, key_to_instance)
        if call_cb:
            [cb(key_to_instance) for cb in callbacks]
        return obj

    def to_pb(self, obj=None):
        from .pbserializer import ProtobufSerializeProvider

        provider = ProtobufSerializeProvider()
        return self.serialize(provider, obj=obj)

    @classmethod
    def from_pb(cls, obj):
        from .pbserializer import ProtobufSerializeProvider

        provider = ProtobufSerializeProvider()
        return cls.deserialize(provider, obj)

    def to_json(self, obj=None):
        from .jsonserializer import JsonSerializeProvider

        provider = JsonSerializeProvider()
        return self.serialize(provider, obj=obj)

    @classmethod
    def from_json(cls, obj):
        from .jsonserializer import JsonSerializeProvider

        provider = JsonSerializeProvider()
        return cls.deserialize(provider, obj)


cdef class AttrWrapper:
    def __init__(self, obj):
        self._obj = obj

    cpdef asdict(self):
        return dict(self._obj)

    def __getattr__(self, item):
        return self._obj[item]

    def __setattr__(self, key, value):
        self._obj[key] = value


class AttributeAsDict(Serializable):
    attr_tag = None

    @classmethod
    def cls(cls, Provider provider):
        if provider.type == ProviderType.protobuf and cls.attr_tag is None:
            from .protos.value_pb2 import Value
            return Value
        return super(AttributeAsDict, cls).cls(provider)

    def serialize(self, Provider provider, obj=None):
        return provider.serialize_attribute_as_dict(self, obj=obj)

    @classmethod
    def deserialize(cls, Provider provider, obj, list callbacks=None, dict key_to_instance=None):
        call_cb = callbacks is None
        callbacks = [] if callbacks is None else callbacks
        key_to_instance = {} if key_to_instance is None else key_to_instance
        obj = provider.deserialize_attribute_as_dict(
            cls, obj, callbacks, key_to_instance)
        if call_cb:
            [cb(key_to_instance) for cb in callbacks]
        return obj


class HasKey(object):
    __slots__ = '_key', '_id'


cdef class KeyPlaceholder:
    def __init__(self, key, id):
        self.key = key
        self.id = id


cpdef serializes(Provider provider, objects):
    return [obj.serialize(provider) if obj is not None else None for obj in objects]


cpdef list deserializes(Provider provider, list models, list objects):
    cdef list callbacks
    cdef dict key_to_instances
    cdef list objs

    callbacks = []
    key_to_instance = {}
    objs = []
    for model, pb_obj in zip(models, objects):
        if model is type(None):
            objs.append(None)
            continue
        objs.append(model.deserialize(provider, pb_obj, callbacks, key_to_instance))
    [cb(key_to_instance) for cb in callbacks]
    return objs


cdef class Provider:
    cpdef serialize_field(self, Field field, model_instance, obj):
        raise NotImplementedError

    cpdef serialize_model(self, model_instance, obj=None):
        if obj is None:
            obj = model_instance.cls(self)()

        for name, field in model_instance._FIELDS.items():
            field.serialize(self, model_instance, obj)

        return obj

    cpdef serialize_attribute_as_dict(self, model_instance, obj=None):
        return self.serialize_model(model_instance, obj=obj)

    def deserialize_field(self, Field field, model_instance, obj, list callbacks, dict key_to_instance):
        raise NotImplementedError

    cpdef deserialize_model(self, model_cls, obj, list callbacks, dict key_to_instance):
        cdef object kw
        cdef IdentityField id_field
        cdef object model_instance
        cdef str name
        cdef Field field

        kw = AttrWrapper(dict())
        if hasattr(model_cls, '_ID_FIELD'):
            for id_field in model_cls._ID_FIELD:
                id_field.deserialize(self, kw, obj, callbacks, key_to_instance)
        model_instance = model_cls(**kw.asdict())

        for name, field in model_instance._FIELDS.items():
            field.deserialize(self, model_instance, obj, callbacks, key_to_instance)

        if hasattr(model_instance, 'key') and hasattr(model_instance, 'id'):
            key_to_instance[model_instance.key, model_instance.id] = model_instance
        return model_instance

    def deserialize_attribute_as_dict(self, model_cls, obj, list callbacks, dict key_to_instance):
        return self.deserialize_model(model_cls, obj, callbacks, key_to_instance)
