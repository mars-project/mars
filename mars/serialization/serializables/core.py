# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

import cloudpickle
from typing import Dict, Tuple, List, Type, Any

from ..core import DictSerializer
from .field import _STORE_VALUE_PROPERTY, Field, FunctionField, OneOfField


class SerializableMeta(type):
    def __new__(mcs,
                name: str,
                bases: Tuple[Type],
                properties: Dict):
        properties = properties.copy()
        for base in bases:
            if hasattr(base, '_FIELDS'):
                properties.update(base._FIELDS)

        property_to_fields = dict()
        # filter out all fields
        for k, v in properties.items():
            if not isinstance(v, Field):
                continue

            property_to_fields[k] = v
            properties[k] = v
            v._attr_name = k

        properties['_FIELDS'] = property_to_fields
        slots = set(properties.pop('__slots__', set()))
        if property_to_fields:
            slots.add(_STORE_VALUE_PROPERTY)
        properties['__slots__'] = tuple(slots)

        clz = type.__new__(mcs, name, bases, properties)
        return clz


class Serializable(metaclass=SerializableMeta):
    __slots__ = ()

    _FIELDS: Dict[str, Field]
    _STORE_VALUE_PROPERTY: Dict[str, Any]

    def __init__(self, *args, **kwargs):
        setattr(self, _STORE_VALUE_PROPERTY, dict())

        for slot, arg in zip(self.__slots__, args):  # pragma: no cover
            object.__setattr__(self, slot, arg)

        for key, val in kwargs.items():
            object.__setattr__(self, key, val)


class SerializableSerializer(DictSerializer):
    """
    Leverage DictSerializer to perform serde.
    """
    serializer_name = 'serializable'

    def serialize(self, obj: Serializable, context: Dict):
        fields = obj._FIELDS
        tag_to_values = getattr(obj, _STORE_VALUE_PROPERTY).copy()

        for field in fields.values():
            tag = field.tag
            try:
                value = tag_to_values[tag]
            except KeyError:
                continue
            if field.on_serialize:
                value = tag_to_values[tag] = field.on_serialize(value)
            if isinstance(field, FunctionField):
                tag_to_values[tag] = cloudpickle.dumps(value)

        header, buffers = super().serialize(tag_to_values, context)
        header['class'] = type(obj)
        return header, buffers

    def deserialize(self, header: Dict, buffers: List, context: Dict) -> Serializable:
        obj_class: Type[Serializable] = header.pop('class')
        tag_to_values = super().deserialize(header, buffers, context)

        property_to_values = dict()
        for property_name, field in obj_class._FIELDS.items():
            if isinstance(field, OneOfField):
                value = None
                for ref_field in field.reference_fields:
                    try:
                        value = tag_to_values.pop(ref_field.tag)
                    except KeyError:
                        continue
                if value is None:
                    continue
            else:
                try:
                    value = tag_to_values[field.tag]
                except KeyError:
                    continue
            if value is not None and field.on_deserialize:
                value = field.on_deserialize(value)
            if isinstance(field, FunctionField):
                value = cloudpickle.loads(value)
            property_to_values[property_name] = value

        obj = obj_class()
        for prop, value in property_to_values.items():
            setattr(obj, prop, value)
        return obj


SerializableSerializer.register(Serializable)
