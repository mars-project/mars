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

from functools import wraps
from typing import Dict

from ..serialization.serializables import Serializable, StringField
from ..serialization.serializables.core import SerializableSerializer
from ..utils import tokenize


class Base(Serializable):
    _no_copy_attrs_ = {'_id'}
    _init_update_key_ = True

    _key = StringField('key')
    _id = StringField('id')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._init_update_key_ and (not hasattr(self, '_key') or not self._key):
            self._update_key()
        if not hasattr(self, '_id') or not self._id:
            self._id = str(id(self))

    @property
    def _keys_(self):
        cls = type(self)
        member = '__keys_' + cls.__name__
        try:
            return getattr(cls, member)
        except AttributeError:
            slots = sorted(self._FIELDS)
            setattr(cls, member, slots)
            return slots

    @property
    def _values_(self):
        return [getattr(self, k, None) for k in self._keys_
                if k not in self._no_copy_attrs_]

    def __mars_tokenize__(self):
        if hasattr(self, '_key'):
            return self._key
        else:
            return (type(self), *self._values_)

    def _obj_set(self, k, v):
        object.__setattr__(self, k, v)

    def _update_key(self):
        self._obj_set('_key', tokenize(type(self).__name__, *self._values_))
        return self

    def reset_key(self):
        self._obj_set('_key', None)
        return self

    def __copy__(self):
        return self.copy()

    def copy(self):
        return self.copy_to(type(self)(_key=self.key))

    def copy_to(self, target):
        for attr in self._FIELDS:
            if (attr.startswith('__') and attr.endswith('__')) or attr in self._no_copy_attrs_:
                # we don't copy id to identify that the copied one is new
                continue
            try:
                attr_val = getattr(self, attr)
            except AttributeError:
                continue
            setattr(target, attr, attr_val)

        return target

    def copy_from(self, obj):
        obj.copy_to(self)

    @property
    def key(self):
        return self._key

    @property
    def id(self):
        return self._id


def buffered(func):
    @wraps(func)
    def wrapped(self, obj: Base, context: Dict):
        obj_id = (obj.key, obj.id)
        if obj_id in context:
            return {
                       'id': id(context[obj_id]),
                       'serializer': 'ref',
                       'buf_num': 0,
                   }, []
        else:
            context[obj_id] = obj
            return func(self, obj, context)
    return wrapped


class BaseSerializer(SerializableSerializer):
    @buffered
    def serialize(self, obj: Serializable, context: Dict):
        return (yield from super().serialize(obj, context))


BaseSerializer.register(Base)


class MarsError(Exception):
    pass
