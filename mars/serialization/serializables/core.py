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

from functools import partial
from typing import Any, Dict, Generator, List, Type, Tuple

from ..core import Serializer, Placeholder, buffered
from .field import Field, OneOfField


class SerializableMeta(type):
    def __new__(mcs,
                name: str,
                bases: Tuple[Type],
                properties: Dict):
        new_properties = dict()
        for base in bases:
            if hasattr(base, '_FIELDS'):
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

        properties['_FIELDS'] = property_to_fields
        slots = set(properties.pop('__slots__', set()))
        if property_to_fields:
            slots.add('_FIELD_VALUES')
        properties['__slots__'] = tuple(slots)

        clz = type.__new__(mcs, name, bases, properties)
        return clz


class Serializable(metaclass=SerializableMeta):
    __slots__ = ()

    _FIELDS: Dict[str, Field]
    _FIELD_VALUES: Dict[str, Any]

    def __init__(self, *args, **kwargs):
        setattr(self, '_FIELD_VALUES', dict())

        for slot, arg in zip(self.__slots__, args):  # pragma: no cover
            object.__setattr__(self, slot, arg)

        for key, val in kwargs.items():
            object.__setattr__(self, key, val)

    def __repr__(self):
        values = ', '.join(['{}={!r}'.format(slot, getattr(self, slot, None)) for slot in self.__slots__])
        return '{}({})'.format(self.__class__.__name__, values)


class SerializableSerializer(Serializer):
    """
    Leverage DictSerializer to perform serde.
    """
    serializer_name = 'serializable'

    @classmethod
    def _get_tag_to_values(cls, obj: Serializable):
        fields = obj._FIELDS
        tag_to_values = obj._FIELD_VALUES.copy()

        for field in fields.values():
            tag = field.tag
            try:
                value = tag_to_values[tag]
            except KeyError:
                continue
            if field.on_serialize:
                tag_to_values[tag] = field.on_serialize(value)

        return tag_to_values

    @buffered
    def serialize(self, obj: Serializable, context: Dict):
        tag_to_values = self._get_tag_to_values(obj)

        keys = [None] * len(tag_to_values)
        value_headers = [None] * len(tag_to_values)
        value_sizes = [0] * len(tag_to_values)
        value_buffers = []
        for idx, (key, val) in enumerate(tag_to_values.items()):
            keys[idx] = key
            value_headers[idx], val_buf = yield val
            value_sizes[idx] = len(val_buf)
            value_buffers.extend(val_buf)

        header = {
            'keys': keys,
            'value_headers': value_headers,
            'value_sizes': value_sizes,
            'class': type(obj),
        }
        return header, value_buffers

    def deserialize(self, header: Dict, buffers: List,
                    context: Dict) -> Generator[Any, Any, Serializable]:
        obj_class: Type[Serializable] = header.pop('class')

        tag_to_values = dict()
        pos = 0
        for key, value_header, value_size in \
                zip(header['keys'], header['value_headers'], header['value_sizes']):
            tag_to_values[key] = yield value_header, buffers[pos:pos + value_size]  # noqa: E999
            pos += value_size

        for field in obj_class._FIELDS.values():
            if not isinstance(field, OneOfField):
                try:
                    value = tag_to_values[field.tag]
                except KeyError:
                    continue
                if value is not None:
                    if field.on_deserialize:
                        def cb(v, field_):
                            tag_to_values[field_.tag] = field_.on_deserialize(v)
                    else:
                        def cb(v, field_):
                            tag_to_values[field_.tag] = v

                    if isinstance(value, Placeholder):
                        value.callbacks.append(partial(cb, field_=field))
                    else:
                        cb(value, field)

        obj = obj_class()
        obj._FIELD_VALUES = tag_to_values
        return obj


SerializableSerializer.register(Serializable)
