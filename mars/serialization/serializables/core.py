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

import os
import weakref
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, List, Type, Tuple

import cloudpickle

from ...core.mode import is_kernel_mode, is_build_mode
from ...utils import no_default
from ..core import Serializer, Placeholder, buffered
from .field import Field, OneOfField
from .field_type import (
    PrimitiveFieldType,
    ListType,
    TupleType,
    DictType,
    DtypeType,
    DatetimeType,
    TimedeltaType,
    TZInfoType,
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


class SerializableMeta(type):
    def __new__(mcs, name: str, bases: Tuple[Type], properties: Dict):
        new_properties = dict()
        for base in bases:
            if hasattr(base, "_FIELDS"):
                new_properties.update(base._FIELDS)
        new_properties.update(properties)
        # make field order deterministic to serialize it as list instead of dict
        properties = OrderedDict()
        for k, v in sorted(new_properties.items(), key=lambda item: item[0]):
            properties[k] = v

        # make field order deterministic to serialize it as list instead of dict
        property_to_fields = OrderedDict()
        pickle_fields = []
        non_pickle_fields = []
        # filter out all fields
        for k, v in properties.items():
            if not isinstance(v, Field):
                continue

            property_to_fields[k] = v
            v._attr_name = k
            if _is_field_primitive_compound(v):
                pickle_fields.append(v)
            else:
                non_pickle_fields.append(v)

        properties["_FIELDS"] = property_to_fields
        properties["_PRIMITIVE_FIELDS"] = pickle_fields
        properties["_NON_PRIMITIVE_FIELDS"] = non_pickle_fields
        slots = set(properties.pop("__slots__", set()))
        if property_to_fields:
            slots.add("_FIELD_VALUES")
        properties["__slots__"] = tuple(slots)

        clz = type.__new__(mcs, name, bases, properties)
        return clz


class Serializable(metaclass=SerializableMeta):
    __slots__ = ("__weakref__",)

    _cache_primitive_serial = False

    _FIELDS: Dict[str, Field]
    _FIELD_VALUES: Dict[str, Any]
    _PRIMITIVE_FIELDS: List[str]
    _NON_PRIMITIVE_FIELDS: List[str]

    def __init__(self, *args, **kwargs):
        if args:  # pragma: no cover
            values = dict(zip(self._FIELDS, args))
            values.update(kwargs)
        else:
            values = kwargs
        if not _is_ci or is_kernel_mode() or is_build_mode():
            self._FIELD_VALUES = values
        else:
            self._FIELD_VALUES = dict()
            for k, v in values.items():
                setattr(self, k, v)

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
        copied._FIELD_VALUES = self._FIELD_VALUES.copy()
        return copied


_primitive_serial_cache = weakref.WeakKeyDictionary()


class SerializableSerializer(Serializer):
    """
    Leverage DictSerializer to perform serde.
    """

    @classmethod
    def _get_field_values(cls, obj: Serializable, fields):
        attr_to_values = obj._FIELD_VALUES
        values = []
        for field in fields:
            attr_name = field.attr_name
            try:
                value = attr_to_values[attr_name]
                if field.on_serialize:
                    value = field.on_serialize(value)
            except KeyError:
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
            if obj._cache_primitive_serial:
                primitive_vals = cloudpickle.dumps(primitive_vals)
                _primitive_serial_cache[obj] = primitive_vals

        compound_vals = self._get_field_values(obj, obj._NON_PRIMITIVE_FIELDS)
        return (type(obj), primitive_vals), [compound_vals], False

    @classmethod
    def _set_field_value(cls, attr_to_values: dict, field: Field, value):
        if value is no_default:
            return
        attr_to_values[field.attr_name] = value
        if type(field) is not OneOfField:
            if value is not None:
                if field.on_deserialize:

                    def cb(v, field_):
                        attr_to_values[field_.attr_name] = field_.on_deserialize(v)

                else:

                    def cb(v, field_):
                        attr_to_values[field_.attr_name] = v

                if type(value) is Placeholder:
                    value.callbacks.append(partial(cb, field_=field))
                else:
                    cb(value, field)

    def deserial(self, serialized: Tuple, context: Dict, subs: List) -> Serializable:
        obj_class, primitives = serialized
        attr_to_values = dict()

        if type(primitives) is not list:
            primitives = cloudpickle.loads(primitives)

        if primitives:
            for field, value in zip(obj_class._PRIMITIVE_FIELDS, primitives):
                self._set_field_value(attr_to_values, field, value)

        if obj_class._NON_PRIMITIVE_FIELDS:
            for field, value in zip(obj_class._NON_PRIMITIVE_FIELDS, subs[0]):
                self._set_field_value(attr_to_values, field, value)

        obj = obj_class()
        obj._FIELD_VALUES = attr_to_values
        return obj


SerializableSerializer.register(Serializable)
