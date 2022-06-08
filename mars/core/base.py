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
from typing import Dict, Tuple, Type

from ..serialization.core import Placeholder, fast_id
from ..serialization.serializables import Serializable, StringField
from ..serialization.serializables.core import SerializableSerializer
from ..utils import tokenize


class Base(Serializable):
    _no_copy_attrs_ = {"_id"}
    _init_update_key_ = True

    _key = StringField("key", default=None)
    _id = StringField("id")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._init_update_key_ and (not hasattr(self, "_key") or not self._key):
            self._update_key()
        if not hasattr(self, "_id") or not self._id:
            self._id = str(id(self))

    @property
    def _keys_(self):
        cls = type(self)
        member = "__keys_" + cls.__name__
        try:
            return getattr(cls, member)
        except AttributeError:
            slots = sorted(self._FIELDS)
            setattr(cls, member, slots)
            return slots

    @property
    def _copy_tags_(self):
        cls = type(self)
        member = f"__copy_tags_{cls.__name__}"
        try:
            return getattr(cls, member)
        except AttributeError:
            slots = sorted(
                f.name for k, f in self._FIELDS.items() if k not in self._no_copy_attrs_
            )
            setattr(cls, member, slots)
            return slots

    @property
    def _values_(self):
        values = []
        fields = self._FIELDS
        for k in self._copy_tags_:
            try:
                values.append(fields[k].get(self))
            except AttributeError:
                values.append(None)
        return values

    def __mars_tokenize__(self):
        try:
            return self._key
        except AttributeError:  # pragma: no cover
            self._update_key()
            return self._key

    def _obj_set(self, k, v):
        object.__setattr__(self, k, v)

    def _update_key(self):
        self._obj_set("_key", tokenize(type(self).__name__, *self._values_))
        return self

    def reset_key(self):
        self._obj_set("_key", None)
        return self

    def __copy__(self):
        return self.copy()

    def copy(self):
        return self.copy_to(type(self)(_key=self.key))

    def copy_to(self, target: "Base"):
        target_fields = target._FIELDS
        no_copy_attrs = self._no_copy_attrs_
        for k, field in self._FIELDS.items():
            if k in no_copy_attrs:
                continue
            try:
                # Slightly faster than getattr.
                value = field.__get__(self, k)
                target_fields[k].set(target, value)
            except AttributeError:
                continue

        return target

    def copy_from(self, obj):
        obj.copy_to(self)

    @property
    def key(self):
        return self._key

    @property
    def id(self):
        return self._id

    def to_kv(self, exclude_fields: Tuple[str], accept_value_types: Tuple[Type]):
        fields = self._FIELDS
        kv = {}
        no_value = object()
        for name, field in fields.items():
            if name not in exclude_fields:
                value = getattr(self, name, no_value)
                if value is not no_value and isinstance(value, accept_value_types):
                    kv[field.tag] = value
        return kv


def buffered_base(func):
    @wraps(func)
    def wrapped(self, obj: Base, context: Dict):
        obj_id = (obj.key, obj.id)
        if obj_id in context:
            return Placeholder(fast_id(context[obj_id]))
        else:
            context[obj_id] = obj
            return func(self, obj, context)

    return wrapped


class BaseSerializer(SerializableSerializer):
    @buffered_base
    def serial(self, obj: Base, context: Dict):
        return super().serial(obj, context)


BaseSerializer.register(Base)


class MarsError(Exception):
    pass


class ExecutionError(MarsError):
    def __init__(self, nested_error: BaseException):
        super().__init__(nested_error)
        self.nested_error = nested_error
