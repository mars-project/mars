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
import os
import operator
import threading
import weakref
from typing import Dict, List, Type, Tuple

import cloudpickle

from ...lib.fury import ListSerializer, NullableSerializer, PrimitiveSerializer, Int32ListSerializer, \
    Int64ListSerializer, StringListSerializer, Float64ListSerializer, Int32TupleSerializer, Int64TupleSerializer, \
    Float64TupleSerializer, StringTupleSerializer, StringKVDictSerializer, DictSerializer, Buffer
from ...utils import no_default
from ..core import Serializer, Placeholder, buffered
from .field import Field, ListField, TupleField, DictField
from .field_type import (
    PrimitiveFieldType,
    ListType,
    TupleType,
    DictType,
    DtypeType,
    DatetimeType,
    TimedeltaType,
    TZInfoType,
    PrimitiveType,
)


_primitive_field_types = (
    PrimitiveFieldType,
    DtypeType,
    DatetimeType,
    TimedeltaType,
    TZInfoType,
)

_is_ci = (os.environ.get("CI") or "0").lower() in ("1", "true")


def _is_field_primitive_compound(field: Field):
    if field.on_serialize is not None or field.on_deserialize is not None:
        return False

    def check_type(field_type):
        if isinstance(field_type, _primitive_field_types):
            return True
        if isinstance(field_type, (ListType, TupleType)):
            if all(
                check_type(element_type) or element_type is Ellipsis
                for element_type in field_type._field_types
            ):
                return True
        if isinstance(field_type, DictType):
            if all(
                isinstance(element_type, _primitive_field_types)
                or element_type is Ellipsis
                for element_type in (field_type.key_type, field_type.value_type)
            ):
                return True
        return False

    return check_type(field.field_type)


def get_basic_serializer(field_type, nullable=True):
    from mars.lib.fury import Buffer

    if not isinstance(field_type, PrimitiveFieldType):
        return None
    if nullable:
        if field_type.type == PrimitiveType.bool:
            return Buffer.write_nullable_bool, Buffer.read_nullable_bool
        elif field_type.type in {PrimitiveType.int8, PrimitiveType.int16, PrimitiveType.int32,
                                 PrimitiveType.uint8, PrimitiveType.uint16}:
            return Buffer.write_nullable_varint32, Buffer.read_nullable_varint32
        elif field_type.type == {PrimitiveType.int64, PrimitiveType.uint32}:
            return Buffer.write_nullable_varint64, Buffer.read_nullable_varint64
        elif field_type.type == {PrimitiveType.float16, PrimitiveType.float32}:
            return Buffer.write_nullable_float32, Buffer.read_nullable_float32
        elif field_type.type == PrimitiveType.float64:
            return Buffer.write_nullable_float64, Buffer.read_nullable_float64
        elif field_type.type == PrimitiveType.string:
            return Buffer.write_nullable_string, Buffer.read_nullable_string
        elif field_type.type == PrimitiveType.bytes:
            return Buffer.write_nullable_bytes, Buffer.read_nullable_bytes
        else:
            return None
    if field_type.type == PrimitiveType.bool:
        return Buffer.write_bool, Buffer.read_bool
    elif field_type.type == PrimitiveType.int8:
        return Buffer.write_int8, Buffer.read_int8
    elif field_type.type == PrimitiveType.int16:
        return Buffer.write_int16, Buffer.read_int16
    elif field_type.type in {PrimitiveType.uint8, PrimitiveType.uint16, PrimitiveType.int32}:
        return Buffer.write_varint32, Buffer.read_varint32
    elif field_type.type in {PrimitiveType.int64, PrimitiveType.uint32}:
        return Buffer.write_varint64, Buffer.read_varint64
    elif field_type.type in {PrimitiveType.float16, PrimitiveType.float32}:
        return Buffer.write_float32, Buffer.read_float32
    elif field_type.type == PrimitiveType.float64:
        return Buffer.write_float64, Buffer.read_float64
    elif field_type.type == PrimitiveType.string:
        return Buffer.write_string, Buffer.read_string
    elif field_type.type == PrimitiveType.bytes:
        return Buffer.write_bytes, Buffer.read_bytes
    else:
        return None


def get_fury_serializer(field: Field):
    if field.on_serialize is not None or field.on_deserialize is not None:
        return None

    field_type = field.field_type
    if isinstance(field, ListField):
        assert isinstance(field_type, ListType)
        element_type = field_type.element_types()[0]
        if element_type == PrimitiveType.int32:
            return Int32ListSerializer(field.nullable, field.elements_nullable)
        elif element_type == PrimitiveType.int64:
            return Int64ListSerializer(field.nullable, field.elements_nullable)
        elif element_type == PrimitiveType.float64:
            return Float64ListSerializer(field.nullable, field.elements_nullable)
        elif element_type == PrimitiveType.string:
            return StringListSerializer(field.nullable, field.elements_nullable)
        element_serializer = get_basic_serializer(element_type, field.elements_nullable)
        if element_serializer is None:
            return None
        if field.elements_nullable:
            elem_serializer = NullableSerializer(*element_serializer)
        else:
            elem_serializer = PrimitiveSerializer(*element_serializer)
        return ListSerializer(field.nullable, field.elements_nullable, elem_serializer)

    if isinstance(field, TupleField):
        assert isinstance(field_type, TupleType)
        if not field_type.is_homogeneous():
            return None  # TODO support heterogeneous types
        element_type = field_type.element_types()[0]
        if element_type == PrimitiveType.int32:
            return Int32TupleSerializer(field.nullable, field.elements_nullable)
        elif element_type == PrimitiveType.int64:
            return Int64TupleSerializer(field.nullable, field.elements_nullable)
        elif element_type == PrimitiveType.float64:
            return Float64TupleSerializer(field.nullable, field.elements_nullable)
        elif element_type == PrimitiveType.string:
            return StringTupleSerializer(field.nullable, field.elements_nullable)
        return None  # TODO support other types

    if isinstance(field, DictField):
        assert isinstance(field_type, DictType)
        if field_type.key_type == PrimitiveType.string and field_type.value_type == PrimitiveType.string:
            return StringKVDictSerializer(field.nullable, field.key_nullable, field.value_nullable)
        key_serializer = get_basic_serializer(field_type.key_type, field.key_nullable)
        value_serializer = get_basic_serializer(field_type.key_type, field.key_nullable)
        if key_serializer is not None and value_serializer is not None:
            return DictSerializer(field.nullable, field.key_nullable,
                                  field.value_nullable, key_serializer, value_serializer)
    if isinstance(field_type, PrimitiveType):
        return get_basic_serializer(field_type, field.nullable)
    return None


fury_buffer = threading.local()


def get_fury_write_buffer() -> Buffer:
    buffer = getattr(fury_buffer, "write_buffer", None)
    if buffer is None:
        buffer = Buffer.allocate(32)
        fury_buffer.write_buffer = buffer
    return buffer


def get_fury_read_buffer() -> Buffer:
    buffer = getattr(fury_buffer, "read_buffer", None)
    if buffer is None:
        buffer = Buffer.allocate(32)
        fury_buffer.read_buffer = buffer
    return buffer


_disable_fury_serialization = (os.environ.get("DISABLE_FURY") or "0").lower() in ("1", "true")


class SerializableMeta(type):
    def __new__(mcs, name: str, bases: Tuple[Type], properties: Dict):
        # All the fields including base fields.
        all_fields = dict()

        for base in bases:
            if hasattr(base, "_FIELDS"):
                all_fields.update(base._FIELDS)

        properties_without_fields = {}
        properties_field_slot_names = []
        for k, v in properties.items():
            if not isinstance(v, Field):
                properties_without_fields[k] = v
                continue

            field = all_fields.get(k)
            if field is None:
                properties_field_slot_names.append(k)
            else:
                v.name = field.name
                v.get = field.get
                v.set = field.set
                v.__delete__ = field.__delete__
            all_fields[k] = v

        # Make field order deterministic to serialize it as list instead of dict.
        all_fields = dict(sorted(all_fields.items(), key=operator.itemgetter(0)))
        pickle_fields = []
        non_pickle_fields = []
        fury_serializable_fields = []
        fields_fury_serializers = []
        fields_fury_deserializers = []
        non_fury_serializable_fields = []
        for v in all_fields.values():
            if _is_field_primitive_compound(v):
                pickle_fields.append(v)
                fury_serializer = get_fury_serializer(v)
                if fury_serializer is None:
                    non_fury_serializable_fields.append(v)
                else:
                    fury_serializable_fields.append(v)
                    if isinstance(fury_serializer, tuple):
                        fields_fury_serializers.append(fury_serializer[0])
                        fields_fury_deserializers.append(fury_serializer[1])
                    else:
                        fields_fury_serializers.append(fury_serializer.write)
                        fields_fury_deserializers.append(fury_serializer.read)
            else:
                non_pickle_fields.append(v)

        slots = set(properties.pop("__slots__", set()))
        slots.update(properties_field_slot_names)

        properties = properties_without_fields
        properties["_FIELDS"] = all_fields
        properties["FURY_SERIALIZABLE_FIELDS"] = fury_serializable_fields
        properties["NON_FURY_SERIALIZABLE_FIELDS"] = non_fury_serializable_fields
        properties["FIELDS_FURY_SERIALIZERS"] = fields_fury_serializers
        properties["FIELDS_FURY_DESERIALIZERS"] = fields_fury_deserializers
        properties["_PRIMITIVE_FIELDS"] = fury_serializable_fields + non_fury_serializable_fields
        properties["_NON_PRIMITIVE_FIELDS"] = non_pickle_fields
        properties["__slots__"] = tuple(slots)

        clz = type.__new__(mcs, name, bases, properties)
        # Bind slot member_descriptor with field.
        for name in properties_field_slot_names:
            member_descriptor = getattr(clz, name)
            field = all_fields[name]
            field.name = member_descriptor.__name__
            field.get = member_descriptor.__get__
            field.set = member_descriptor.__set__
            field.__delete__ = member_descriptor.__delete__
            setattr(clz, name, field)

        return clz


class Serializable(metaclass=SerializableMeta):
    __slots__ = ("__weakref__",)

    _cache_primitive_serial = False
    _enable_fury_serialization = True

    _FIELDS: Dict[str, Field]
    _PRIMITIVE_FIELDS: List[Field]
    FURY_SERIALIZABLE_FIELDS: List[Field]
    NON_FURY_SERIALIZABLE_FIELDS: List[Field]
    FIELDS_FURY_SERIALIZERS: List
    FIELDS_FURY_DESERIALIZERS: List
    _NON_PRIMITIVE_FIELDS: List[Field]

    def __init__(self, *args, **kwargs):
        fields = self._FIELDS
        if args:  # pragma: no cover
            values = dict(zip(fields, args))
            values.update(kwargs)
        else:
            values = kwargs
        for k, v in values.items():
            fields[k].set(self, v)

    def __repr__(self):
        values = ", ".join(
            [
                "{}={!r}".format(slot, getattr(self, slot, None))
                for slot in self.__slots__
            ]
        )
        return "{}({})".format(self.__class__.__name__, values)

    def copy(self) -> "Serializable":
        copied = type(self)()
        copied_fields = copied._FIELDS
        for k, field in self._FIELDS.items():
            try:
                # Slightly faster than getattr.
                value = field.get(self, k)
                copied_fields[k].set(copied, value)
            except AttributeError:
                continue
        return copied


_primitive_serial_cache = weakref.WeakKeyDictionary()


class SerializableSerializer(Serializer):
    """
    Leverage DictSerializer to perform serde.
    """

    @classmethod
    def _get_field_values(cls, obj: Serializable, fields):
        values = []
        for field in fields:
            try:
                value = field.get(obj)
                if field.on_serialize is not None:
                    value = field.on_serialize(value)
            except AttributeError:
                # Most field values are not None, serialize by list is more efficient than dict.
                value = no_default
            values.append(value)
        return values

    @buffered
    def serial(self, obj: Serializable, context: Dict):
        if obj._cache_primitive_serial and obj in _primitive_serial_cache:
            primitive_vals = _primitive_serial_cache[obj]
        else:
            primitive_vals = self._get_field_values(obj, obj._PRIMITIVE_FIELDS)
            if obj._enable_fury_serialization:
                buffer = get_fury_write_buffer()
                buffer.writer_index = 0
                for idx, fury_serializer in enumerate(obj.FIELDS_FURY_SERIALIZERS):
                    fury_serializer(buffer, primitive_vals[idx])
                fury_data = buffer.to_bytes(length=buffer.writer_index)
                primitive_vals = primitive_vals[len(obj.FIELDS_FURY_SERIALIZERS):]
                primitive_vals.append(fury_data)
            if obj._cache_primitive_serial:
                primitive_vals = cloudpickle.dumps(primitive_vals)
                _primitive_serial_cache[obj] = primitive_vals
        compound_vals = self._get_field_values(obj, obj._NON_PRIMITIVE_FIELDS)
        return (type(obj), primitive_vals), [compound_vals], False

    @staticmethod
    def _set_field_value(obj: Serializable, field: Field, value):
        if value is no_default:
            return
        if type(value) is Placeholder:
            if field.on_deserialize is not None:
                value.callbacks.append(
                    lambda v: field.set(obj, field.on_deserialize(v))
                )
            else:
                value.callbacks.append(lambda v: field.set(obj, v))
        else:
            if field.on_deserialize is not None:
                field.set(obj, field.on_deserialize(value))
            else:
                field.set(obj, value)

    def deserial(self, serialized: Tuple, context: Dict, subs: List) -> Serializable:
        obj_class, primitives = serialized

        if type(primitives) is not list:
            primitives = cloudpickle.loads(primitives)
        obj = obj_class()
        if obj._enable_fury_serialization:
            fury_data = primitives[-1]
            buffer = get_fury_read_buffer()
            buffer.point_to_bytes(fury_data)
            vals = []
            for fury_deserializer in obj.FIELDS_FURY_DESERIALIZERS:
                vals.append(fury_deserializer(buffer))
            primitives = vals.extend(primitives)
        if primitives:
            for field, value in zip(obj_class._PRIMITIVE_FIELDS, primitives):
                self._set_field_value(obj, field, value)

        if obj_class._NON_PRIMITIVE_FIELDS:
            for field, value in zip(obj_class._NON_PRIMITIVE_FIELDS, subs[0]):
                self._set_field_value(obj, field, value)

        return obj


SerializableSerializer.register(Serializable)
