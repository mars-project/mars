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

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Generator, List, Type, Tuple
from weakref import WeakKeyDictionary

from ..core import Serializer, Placeholder, buffered
from ...utils import dataslots
from .field import Field, OneOfField, AbstractFieldType
from .field_type import (
    PrimitiveFieldType,
    ListType,
    TupleType,
    DtypeType,
    DatetimeType,
    TimedeltaType,
    FunctionType,
    NamedtupleType,
    TZInfoType,
)


class SerializableMeta(type):
    def __new__(mcs, name: str, bases: Tuple[Type], properties: Dict):
        new_properties = dict()
        for base in bases:
            if hasattr(base, "_FIELDS"):
                new_properties.update(base._FIELDS)
        new_properties.update(properties)
        properties = new_properties

        property_to_fields = dict()
        # filter out all fields
        for k, v in properties.items():
            if not isinstance(v, Field):
                continue

            property_to_fields[k] = v
            properties[k] = v
            v._attr_name = k

        properties["_FIELDS"] = property_to_fields
        slots = set(properties.pop("__slots__", set()))
        if property_to_fields:
            slots.add("_FIELD_VALUES")
        properties["__slots__"] = tuple(slots)

        clz = type.__new__(mcs, name, bases, properties)
        return clz


class Serializable(metaclass=SerializableMeta):
    __slots__ = ("__weakref__",)
    _cache_serialize = True

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


_serialize_with_pickle_field_types = (
    PrimitiveFieldType,
    DtypeType,
    DatetimeType,
    TimedeltaType,
    FunctionType,
    NamedtupleType,
    TZInfoType,
)


def _serialize_with_pickle(field_type: AbstractFieldType) -> bool:
    if isinstance(field_type, _serialize_with_pickle_field_types):
        return True
    if isinstance(field_type, (ListType, TupleType)):
        if all(
            _serialize_with_pickle(element_type) or element_type is Ellipsis
            for element_type in field_type.element_types
        ):
            return True
    return False


@dataslots
@dataclass
class _PickleObj:
    keys: list = field(default_factory=list)
    values: list = field(default_factory=list)


class SerializableSerializer(Serializer):
    """
    Leverage DictSerializer to perform serde.
    """

    serializer_name = "serializable"
    _cached = WeakKeyDictionary()

    @classmethod
    def _get_tag_to_field_values(cls, obj: Serializable):
        fields = obj._FIELDS
        attr_to_values = obj._FIELD_VALUES
        tag_to_field_values = dict()

        for field in fields.values():
            tag = field.tag
            attr_name = field.attr_name
            try:
                value = attr_to_values[attr_name]
            except KeyError:
                continue
            if field.on_serialize:
                value = field.on_serialize(value)
            tag_to_field_values[tag] = (field, value)

        return tag_to_field_values

    @buffered
    def serialize(self, obj: Serializable, context: Dict):
        tag_to_values = self._get_tag_to_field_values(obj)

        value_headers = []
        value_keys = []
        value_sizes = []
        value_buffers = []
        pickle_obj = cached_pickle_obj = None
        need_cache = type(obj)._cache_serialize
        try:
            cached_pickle_obj = SerializableSerializer._cached[obj]
            is_cached = True
        except TypeError:
            pickle_obj = _PickleObj()
            is_cached = False
            need_cache = False
        except KeyError:
            pickle_obj = _PickleObj()
            is_cached = False
        for key, (field, val) in tag_to_values.items():
            if _serialize_with_pickle(field.field_type):
                if not is_cached:
                    pickle_obj.keys.append(key)
                    pickle_obj.values.append(val)
            else:
                value_keys.append(key)
                value_header, val_buf = yield val
                value_headers.append(value_header)
                value_sizes.append(len(val_buf))
                value_buffers.extend(val_buf)
        if not is_cached:
            pickled_header, pickled = yield pickle_obj
            if need_cache:
                SerializableSerializer._cached[obj] = (pickled_header, pickled)
        else:
            pickled_header, pickled = cached_pickle_obj
        value_headers.append(pickled_header)
        value_sizes.append(len(pickled))
        value_buffers.extend(pickled)

        header = {
            "value_keys": value_keys,
            "value_headers": value_headers,
            "value_sizes": value_sizes,
            "class": type(obj),
        }
        return header, value_buffers

    def deserialize(
        self, header: Dict, buffers: List, context: Dict
    ) -> Generator[Any, Any, Serializable]:
        obj_class: Type[Serializable] = header.pop("class")

        tag_to_values = dict()
        pos = 0
        for value_key, value_header, value_size in zip(
            header["value_keys"], header["value_headers"], header["value_sizes"]
        ):
            tag_to_values[value_key] = (
                yield value_header,
                buffers[pos : pos + value_size],
            )  # noqa: E999
            pos += value_size
        pickled_size = header["value_sizes"][-1]
        pickle_obj = (
            yield header["value_headers"][-1],
            buffers[pos : pos + pickled_size],
        )
        for k, v in zip(pickle_obj.keys, pickle_obj.values):
            tag_to_values[k] = v

        attr_to_values = dict()
        for field in obj_class._FIELDS.values():
            try:
                value = attr_to_values[field.attr_name] = tag_to_values[field.tag]
            except KeyError:
                continue
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

        obj = obj_class()
        obj._FIELD_VALUES = attr_to_values
        return obj


SerializableSerializer.register(Serializable)
