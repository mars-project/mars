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

from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Generator, List, Type, Tuple

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


_basic_field_type = (
    PrimitiveFieldType,
    DtypeType,
    DatetimeType,
    TimedeltaType,
    TZInfoType,
)


def serialize_by_pickle(field: Field):
    if field.on_serialize is not None or field.on_deserialize is not None:
        return False

    def check_type(field_type):
        if isinstance(field_type, _basic_field_type):
            return True
        if isinstance(field_type, (ListType, TupleType)):
            if all(
                check_type(element_type) or element_type is Ellipsis
                for element_type in field_type._field_types
            ):
                return True
        if isinstance(field_type, DictType):
            if all(
                isinstance(element_type, _basic_field_type) or element_type is Ellipsis
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
            if serialize_by_pickle(v):
                pickle_fields.append(v)
            else:
                non_pickle_fields.append(v)

        properties["_FIELDS"] = property_to_fields
        properties["_PICKLE_FIELDS"] = pickle_fields
        properties["_NON_PICKLE_FIELDS"] = non_pickle_fields
        slots = set(properties.pop("__slots__", set()))
        if property_to_fields:
            slots.add("_FIELD_VALUES")
        properties["__slots__"] = tuple(slots)

        clz = type.__new__(mcs, name, bases, properties)
        return clz


class Serializable(metaclass=SerializableMeta):
    __slots__ = ()

    _FIELDS: Dict[str, Field]
    _FIELD_VALUES: Dict[str, Any]

    def __init__(self, *args, **kwargs):
        if args:  # pragma: no cover
            values = dict(zip(self.__slots__, args))
            values.update(kwargs)
            self._FIELD_VALUES = values
        else:
            self._FIELD_VALUES = kwargs

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


class _SkipStub:
    pass


class SerializableSerializer(Serializer):
    """
    Leverage DictSerializer to perform serde.
    """

    serializer_name = "serializable"

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
                value = _SkipStub  # Most field values are not None, serialize by list is more efficient than dict.
            values.append(value)
        return values

    @buffered
    def serialize(self, obj: Serializable, context: Dict):
        pickles = self._get_field_values(obj, obj._PICKLE_FIELDS)
        composed_values = self._get_field_values(obj, obj._NON_PICKLE_FIELDS)

        value_headers = [None] * len(composed_values)
        value_sizes = [0] * len(composed_values)
        value_buffers = []
        for idx, val in enumerate(composed_values):
            value_headers[idx], val_buf = yield val
            value_sizes[idx] = len(val_buf)
            value_buffers.extend(val_buf)

        header = {"class": type(obj)}
        if pickles:
            header["pickles"] = pickles
        if composed_values:
            header["value_headers"] = value_headers
            header["value_sizes"] = value_sizes
        return header, value_buffers

    def _set_field_value(self, attr_to_values: dict, field: Field, value):
        if value is _SkipStub:
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

    def deserialize(
        self, header: Dict, buffers: List, context: Dict
    ) -> Generator[Any, Any, Serializable]:
        obj_class: Type[Serializable] = header.pop("class")
        attr_to_values = dict()
        pickles = header.get("pickles")
        if pickles:
            for value, field in zip(pickles, obj_class._PICKLE_FIELDS):
                self._set_field_value(attr_to_values, field, value)
        if obj_class._NON_PICKLE_FIELDS:
            pos = 0
            value_headers = header.get("value_headers")
            value_sizes = header.get("value_sizes")
            for field, value_header, value_size in zip(
                obj_class._NON_PICKLE_FIELDS, value_headers, value_sizes
            ):
                value = (
                    yield value_header,
                    buffers[pos : pos + value_size],
                )  # noqa: E999
                pos += value_size
                self._set_field_value(attr_to_values, field, value)
        obj = obj_class()
        obj._FIELD_VALUES = attr_to_values
        return obj


SerializableSerializer.register(Serializable)
