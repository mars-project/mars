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
    DtypeType,
    DatetimeType,
    TimedeltaType,
    FunctionType,
    TZInfoType,
)


_basic_field_type = (
    PrimitiveFieldType,
    DtypeType,
    DatetimeType,
    TimedeltaType,
    FunctionType,
    TZInfoType,
)


def serialize_by_pickle(field_type: Field):
    if field_type.on_serialize is not None or field_type.on_deserialize is not None:
        return False
    if isinstance(field_type.field_type, _basic_field_type):
        return True
    if isinstance(field_type, (ListType, TupleType)):
        if all(
            isinstance(element_type, _basic_field_type) or element_type is Ellipsis
            for element_type in field_type._field_types
        ):
            return True
    return False


class SerializableMeta(type):
    def __new__(mcs, name: str, bases: Tuple[Type], properties: Dict):
        # make field order deterministic to serialize it as list instead of dict
        new_properties = OrderedDict()
        for base in bases:
            if hasattr(base, "_FIELDS"):
                new_properties.update(base._FIELDS)
        new_properties.update(properties)
        properties = new_properties

        # make field order deterministic to serialize it as list instead of dict
        property_to_fields = OrderedDict()
        basic_fields = OrderedDict()
        composed_fields = OrderedDict()
        # filter out all fields
        for k, v in properties.items():
            if not isinstance(v, Field):
                continue

            property_to_fields[k] = v
            v._attr_name = k
            if serialize_by_pickle(v):
                basic_fields[k] = v
            else:
                composed_fields[k] = v

        properties["_FIELDS"] = property_to_fields
        properties["_BASIC_FIELDS"] = basic_fields
        properties["_COMPOSED_FIELDS"] = composed_fields
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
        for index, field in enumerate(fields.values()):
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
        basic_values = self._get_field_values(obj, obj._BASIC_FIELDS)
        composed_values = self._get_field_values(obj, obj._COMPOSED_FIELDS)

        value_headers = [None] * len(composed_values)
        value_sizes = [0] * len(composed_values)
        value_buffers = []
        for idx, val in enumerate(composed_values):
            value_headers[idx], val_buf = yield val
            value_sizes[idx] = len(val_buf)
            value_buffers.extend(val_buf)

        header = {
            "basic_values": basic_values,
            "value_headers": value_headers,
            "value_sizes": value_sizes,
            "class": type(obj),
        }
        return header, value_buffers

    def _set_field_value(self, attr_to_values: dict, field: Field, value):
        if value is _SkipStub:
            return
        if not isinstance(field, OneOfField):
            if value is not None:
                if field.on_deserialize:

                    def cb(v, field_):
                        attr_to_values[field_.attr_name] = field_.on_deserialize(v)

                else:

                    def cb(v, field_):
                        attr_to_values[field_.attr_name] = v

                if isinstance(value, Placeholder):
                    value.callbacks.append(partial(cb, field_=field))
                else:
                    cb(value, field)

    def deserialize(
        self, header: Dict, buffers: List, context: Dict
    ) -> Generator[Any, Any, Serializable]:
        obj_class: Type[Serializable] = header.pop("class")
        basic_values = header["basic_values"]
        attr_to_values = dict()
        for value, field in zip(basic_values, obj_class._BASIC_FIELDS.values()):
            self._set_field_value(attr_to_values, field, value)
        pos = 0
        for field, value_header, value_size in zip(
            obj_class._COMPOSED_FIELDS.values(),
            header["value_headers"],
            header["value_sizes"],
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
