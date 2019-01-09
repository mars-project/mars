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

import numpy as np

from .compat import six, izip
from .utils import tokenize, AttributeDict, on_serialize_shape, on_deserialize_shape
from .serialize import ValueType, ProviderType, Serializable, AttributeAsDict, \
    ListField, TupleField, DictField, DataTypeField, KeyField, BoolField, StringField


class Base(object):
    __slots__ = ()
    _no_copy_attrs_ = set()

    def __init__(self, *args, **kwargs):
        for slot, arg in izip(self.__slots__, args):
            object.__setattr__(self, slot, arg)

        for key, val in six.iteritems(kwargs):
            object.__setattr__(self, key, val)

    @property
    def _values_(self):
        return [getattr(self, k, None) for k in self.__slots__
                if k not in self._no_copy_attrs_]


class BaseWithKey(Base):
    __slots__ = '_key', '_id'
    _no_copy_attrs_ = {'_id'}
    _init_update_key_ = True

    def __init__(self, *args, **kwargs):
        super(BaseWithKey, self).__init__(*args, **kwargs)

        if self._init_update_key_ and (not hasattr(self, '_key') or not self._key):
            self.update_key()
        if not hasattr(self, '_id') or not self._id:
            self._id = str(id(self))

    def _obj_set(self, k, v):
        object.__setattr__(self, k, v)

    def update_key(self):
        self._obj_set('_key', tokenize(type(self), *self._values_))
        return self

    def reset_key(self):
        self._obj_set('_key', None)
        return self

    def update_id(self, new_id=None):
        new_id = new_id if new_id is not None else str(id(self))
        self._obj_set('_id', new_id)

    def __copy__(self):
        return self.copy()

    def copy(self):
        return self.copy_to(type(self)(_key=self.key))

    def copy_to(self, target):
        for attr in self.__slots__:
            if (attr.startswith('__') and attr.endswith('__')) or attr in self._no_copy_attrs_:
                # we don't copy id to identify that the copied one is new
                continue
            if hasattr(self, attr):
                setattr(target, attr, getattr(self, attr))

        return target

    def copy_from(self, obj):
        obj.copy_to(self)

    @property
    def key(self):
        return self._key

    @property
    def id(self):
        return self._id


class Entity(object):
    __slots__ = '_data',
    _allow_data_type_ = ()

    def __init__(self, data):
        self._check_data(data)
        self._data = data

    def _check_data(self, data):
        if data is not None and not isinstance(data, self._allow_data_type_):
            raise TypeError('Expect {0}, got {1}'.format(self._allow_data_type_, type(data)))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._check_data(new_data)
        self._data = new_data

    def __copy__(self):
        return self.copy()

    def copy(self):
        self.copy_to(type(self)(None))

    def copy_to(self, target):
        target.data = self._data

    def copy_from(self, obj):
        self.data = obj.data

    def __getattr__(self, attr):
        return getattr(self._data, attr)

    def __setattr__(self, key, value):
        try:
            super(Entity, self).__setattr__(key, value)
        except AttributeError:
            return setattr(self._data, key, value)


class SerializableWithKey(BaseWithKey, Serializable):
    _key = StringField('key')
    _id = StringField('id')


class AttributeAsDictKey(BaseWithKey, AttributeAsDict):
    _key = StringField('key')
    _id = StringField('id')


class ChunkData(SerializableWithKey):
    __slots__ = '__weakref__',

    # required fields
    _op = KeyField('op')  # store key of operand here
    _shape = TupleField('shape', ValueType.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)
    # optional fields
    _dtype = DataTypeField('dtype')
    _index = TupleField('index', ValueType.uint32)
    _cached = BoolField('cached')
    _composed = ListField('composed', ValueType.reference('self'))
    _params = DictField('params', key_type=ValueType.string, on_deserialize=AttributeDict)

    def __init__(self, *args, **kwargs):
        extras = AttributeDict((k, kwargs.pop(k)) for k in set(kwargs) - set(self.__slots__))
        kwargs['_params'] = kwargs.pop('_params', extras)
        super(ChunkData, self).__init__(*args, **kwargs)

    def __repr__(self):
        return 'Chunk <op={0}, key={1}>'.format(self.op.__class__.__name__, self.key)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from .serialize.protos.chunk_pb2 import ChunkDef
            return ChunkDef
        return super(ChunkData, cls).cls(provider)

    @property
    def shape(self):
        return getattr(self, '_shape', None)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def index(self):
        return getattr(self, '_index', None)

    @property
    def op(self):
        try:
            return self._op
        except AttributeError:
            return None

    @property
    def cached(self):
        return getattr(self, '_cached', None)

    @property
    def inputs(self):
        return self.op.inputs

    @inputs.setter
    def inputs(self, new_inputs):
        self.op.inputs = new_inputs

    @property
    def dtype(self):
        return getattr(self, '_dtype', None) or self.op.dtype

    @property
    def nbytes(self):
        return np.prod(self.shape) * self.dtype.itemsize

    @property
    def composed(self):
        return getattr(self, '_composed', None)

    @property
    def device(self):
        return self.op.device

    def is_sparse(self):
        return self.op.is_sparse()

    issparse = is_sparse

    def update_key(self):
        object.__setattr__(self, '_key', tokenize(
            type(self), *(getattr(self, k, None) for k in self.__slots__ if k != '_index')))


class Chunk(Entity):
    __slots__ = ()
    _allow_data_type_ = (ChunkData,)
