#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import builtins
import itertools
from operator import attrgetter
from typing import List
from weakref import WeakKeyDictionary, WeakSet, ref

import numpy as np

from .utils import tokenize, AttributeDict, on_serialize_shape, \
    on_deserialize_shape, on_serialize_nsplits, enter_build_mode, build_mode
from .serialize import HasKey, HasData, ValueType, ProviderType, Serializable, AttributeAsDict, \
    TupleField, ListField, DictField, KeyField, BoolField, StringField, ReferenceField
from .tiles import Tileable, handler


class Base(HasKey):
    __slots__ = ()
    _no_copy_attrs_ = {'_id'}
    _init_update_key_ = True

    def __init__(self, *args, **kwargs):
        for slot, arg in zip(self.__slots__, args):
            object.__setattr__(self, slot, arg)

        for key, val in kwargs.items():
            object.__setattr__(self, key, val)

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
            slots = sorted(self.__slots__)
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


class Entity(HasData):
    __slots__ = ()
    _allow_data_type_ = ()

    def __init__(self, data):
        self._check_data(data)
        self._data = data

    def __dir__(self):
        obj_dir = object.__dir__(self)
        if self._data is not None:
            obj_dir = sorted(set(dir(self._data) + obj_dir))
        return obj_dir

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__repr__()

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
        return self.copy_to(type(self)(None))

    def copy_to(self, target):
        target.data = self._data
        return target

    def copy_from(self, obj):
        self.data = obj.data

    def tiles(self):
        new_entity = self.copy()
        new_entity.data = handler.tiles(self.data)
        return new_entity

    def _inplace_tile(self):
        return handler.inplace_tile(self)

    def __getattr__(self, attr):
        return getattr(self._data, attr)

    def __setattr__(self, key, value):
        try:
            object.__setattr__(self, key, value)
        except AttributeError:
            return setattr(self._data, key, value)


class SerializableWithKey(Base, Serializable):
    _key = StringField('key')
    _id = StringField('id')


class AttributeAsDictKey(Base, AttributeAsDict):
    _key = StringField('key')
    _id = StringField('id')


class EntityData(SerializableWithKey):
    __slots__ = '__weakref__', '_siblings'

    # required fields
    _op = KeyField('op')  # store key of operand here
    # optional fields
    _extra_params = DictField('extra_params', key_type=ValueType.string, on_deserialize=AttributeDict)

    def __init__(self, *args, **kwargs):
        extras = AttributeDict((k, kwargs.pop(k)) for k in set(kwargs) - set(self.__slots__))
        kwargs['_extra_params'] = kwargs.pop('_extra_params', extras)
        super().__init__(*args, **kwargs)

    @property
    def op(self):
        return getattr(self, '_op', None)

    @property
    def inputs(self):
        return self.op.inputs or []

    @inputs.setter
    def inputs(self, new_inputs):
        self.op.inputs = new_inputs

    def is_sparse(self):
        return self.op.is_sparse()

    issparse = is_sparse

    @property
    def extra_params(self):
        return self._extra_params


ENTITY_TYPE = (EntityData, Entity)


class ChunkData(EntityData):
    __slots__ = ()

    # optional fields
    _index = TupleField('index', ValueType.uint32)
    _cached = BoolField('cached')

    def __repr__(self):
        if self.op.stage is None:
            return 'Chunk <op={0}, key={1}>'.format(type(self.op).__name__, self.key)
        else:
            return 'Chunk <op={0}, stage={1}, key={2}>'.format(
                type(self.op).__name__, self.op.stage.name, self.key)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from .serialize.protos.chunk_pb2 import ChunkDef
            return ChunkDef
        return super().cls(provider)

    @property
    def index(self):
        return getattr(self, '_index', None)

    @property
    def cached(self):
        return getattr(self, '_cached', None)

    @property
    def composed(self):
        return getattr(self, '_composed', None)

    @property
    def device(self):
        return self.op.device

    def _update_key(self):
        object.__setattr__(self, '_key', tokenize(
            type(self).__name__, *(getattr(self, k, None) for k in self._keys_ if k != '_index')))


class Chunk(Entity):
    __slots__ = ()
    _allow_data_type_ = (ChunkData,)


CHUNK_TYPE = (ChunkData, Chunk)


class ObjectChunkData(ChunkData):
    # chunk whose data could be any serializable
    __slots__ = ()

    def __init__(self, op=None, index=None, **kw):
        super().__init__(_op=op, _index=index, **kw)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from .serialize.protos.object_pb2 import ObjectChunkDef
            return ObjectChunkDef
        return super().cls(provider)

    @property
    def params(self):
        # params return the properties which useful to rebuild a new chunk
        return {
            'index': self.index,
        }


class ObjectChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (ObjectChunkData,)


def _on_serialize_composed(composed):
    return [c.data if isinstance(c, Entity) else c for c in composed]


class FuseChunkData(ChunkData):
    __slots__ = '_inited',

    class ChunkRef(Serializable):
        _chunk = ReferenceField('chunk', None)

        @property
        def chunk(self):
            return self._chunk

    _composed = ListField('composed', ValueType.reference(None),
                          on_serialize=_on_serialize_composed)

    def __init__(self, *args, **kwargs):
        self._inited = False
        super().__init__(*args, **kwargs)
        self._extra_params = {}
        self._inited = True

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from .serialize.protos.fusechunk_pb2 import FuseChunkDef
            return FuseChunkDef
        return super().cls(provider)

    @property
    def params(self):
        # params return the properties which useful to rebuild a new chunk
        return self.composed[-1].params

    def __getattr__(self, attr):
        if not self._inited:
            return object.__getattribute__(self, attr)
        if attr in self._extra_params:
            return self._extra_params[attr]
        return getattr(self.composed[-1], attr)

    @property
    def nbytes(self):
        return np.prod(self.shape) * self.dtype.itemsize


class FuseChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (FuseChunkData,)


FUSE_CHUNK_TYPE = (FuseChunkData, FuseChunk)


class _ExecutableMixin:
    __slots__ = ()

    def execute(self, session=None, **kw):
        from .session import Session

        if 'fetch' in kw and kw['fetch']:
            raise ValueError('Does not support fetch=True for `.execute()`,'
                             'please use `.fetch()` instead')
        else:
            kw['fetch'] = False

        if session is None:
            session = Session.default_or_local()

        # no more fetch, thus just fire run
        session.run(self, **kw)
        # return Tileable or ExecutableTuple itself
        return self

    def fetch(self, session=None, **kw):
        from .session import Session

        if session is None:
            session = Session.default_or_local()
        return session.fetch(self, **kw)

    def _attach_session(self, session):
        _cleaner.register(self, session)
        self._executed_sessions.append(session)


class _ExecuteAndFetchMixin:
    __slots__ = ()

    def _execute_and_fetch(self, session=None, **kw):
        if session is None and len(self._executed_sessions) > 0:
            session = self._executed_sessions[-1]
        try:
            # fetch first, to reduce the potential cost of submitting a graph
            return self.fetch(session=session)
        except ValueError:
            # not execute before
            return self.execute(session=session, **kw).fetch(session=session)


class _ToObjectMixin(_ExecuteAndFetchMixin):
    __slots__ = ()

    def to_object(self, session=None, **kw):
        return self._execute_and_fetch(session=session, **kw)


class TileableData(EntityData, Tileable, _ExecutableMixin):
    __slots__ = '_cix', '_entities', '_executed_sessions'
    _no_copy_attrs_ = SerializableWithKey._no_copy_attrs_ | {'_cix'}

    # optional fields
    # `nsplits` means the sizes of chunks for each dimension
    _nsplits = TupleField('nsplits', ValueType.tuple(ValueType.uint64),
                          on_serialize=on_serialize_nsplits)

    def __init__(self, *args, **kwargs):
        if kwargs.get('_nsplits', None) is not None:
            kwargs['_nsplits'] = tuple(tuple(s) for s in kwargs['_nsplits'])

        super().__init__(*args, **kwargs)

        if hasattr(self, '_chunks') and self._chunks:
            self._chunks = sorted(self._chunks, key=attrgetter('index'))

        self._entities = WeakSet()
        self._executed_sessions = []

    @property
    def chunk_shape(self):
        if hasattr(self, '_nsplits') and self._nsplits is not None:
            return tuple(map(len, self._nsplits))

    @property
    def chunks(self) -> List["Chunk"]:
        return getattr(self, '_chunks', None)

    @property
    def nsplits(self):
        return getattr(self, '_nsplits', None)

    @nsplits.setter
    def nsplits(self, new_nsplits):
        self._nsplits = new_nsplits

    @property
    def params(self) -> dict:
        # params return the properties which useful to rebuild a new tileable object
        return dict()

    @property
    def cix(self):
        if self.ndim == 0:
            return ChunksIndexer(self)

        try:
            if getattr(self, '_cix', None) is None:
                self._cix = ChunksIndexer(self)
            return self._cix
        except (TypeError, ValueError):
            return ChunksIndexer(self)

    @property
    def entities(self):
        return self._entities

    def is_coarse(self):
        return not hasattr(self, '_chunks') or self._chunks is None or len(self._chunks) == 0

    @enter_build_mode
    def attach(self, entity):
        self._entities.add(entity)

    @enter_build_mode
    def detach(self, entity):
        self._entities.discard(entity)


class TileableEntity(Entity):
    __slots__ = '__weakref__',

    def __init__(self, data):
        super().__init__(data)
        if self._data is not None:
            self._data.attach(self)
            if self._data.op.create_view:
                entity_view_handler.add_observer(self._data.inputs[0], self)

    def __copy__(self):
        return self._view()

    def _view(self):
        return super().copy()

    def copy(self):
        new_op = self.op.copy().reset_key()
        if new_op.create_view:
            # if the operand is a view, make it a copy
            new_op._create_view = False
        new_outs = new_op.new_tileables(self.op.inputs, kws=[t.params for t in self.op.outputs],
                                        output_limit=len(self.op.outputs),
                                        **self._data.extra_params)
        pos = -1
        for i, out in enumerate(self.op.outputs):
            if self._data is out:
                pos = i
                break
        assert pos >= 0
        return new_outs[pos]

    @Entity.data.setter
    def data(self, new_data):
        self._check_data(new_data)
        if self._data is None:
            self._data = new_data
            self._data.attach(self)
        else:
            entity_view_handler.data_changed(self._data, new_data)


TILEABLE_TYPE = (TileableEntity, TileableData)


class HasShapeTileableData(TileableData):
    # required fields
    _shape = TupleField('shape', ValueType.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        try:
            return self.shape[0]
        except IndexError:
            if build_mode().is_build_mode:
                return 0
            raise TypeError('len() of unsized object')

    @property
    def shape(self):
        if hasattr(self, '_shape') and self._shape is not None:
            return self._shape
        if hasattr(self, '_nsplits') and self._nsplits is not None:
            self._shape = tuple(builtins.sum(nsplit) for nsplit in self._nsplits)
            return self._shape

    def _update_shape(self, new_shape):
        self._shape = new_shape

    @property
    def size(self):
        return np.prod(self.shape).item()

    @property
    def params(self):
        # params return the properties which useful to rebuild a new tileable object
        return {
            'shape': self.shape
        }

    def _equals(self, o):
        return self is o


class HasShapeTileableEnity(TileableEntity):
    __slots__ = ()

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

    def execute(self, session=None, **kw):
        self._data.execute(session, **kw)
        return self


class ObjectData(TileableData, _ToObjectMixin):
    __slots__ = ()

    # optional fields
    _chunks = ListField('chunks', ValueType.reference(ObjectChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [ObjectChunk(it) for it in x] if x is not None else x)

    def __init__(self, op=None, nsplits=None, chunks=None, **kw):
        super().__init__(_op=op, _nsplits=nsplits, _chunks=chunks, **kw)

    def __repr__(self):
        return 'Object <op={0}, key={1}>'.format(type(self.op).__name__, self.key)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from .serialize.protos.object_pb2 import ObjectDef
            return ObjectDef
        return super().cls(provider)

    @property
    def params(self):
        # params return the properties which useful to rebuild a new tileable object
        return {
        }


class Object(Entity, _ToObjectMixin):
    __slots__ = ()
    _allow_data_type_ = (ObjectData,)


OBJECT_TYPE = (ObjectData, Object)
OBJECT_CHUNK_TYPE = (ObjectChunkData, ObjectChunk)


class ChunksIndexer(object):
    __slots__ = '_tileable',

    def __init__(self, tileable):
        self._tileable = tileable

    def __getitem__(self, item):
        """
        The indices for `cix` can be [x, y] or [x, :]. For the former the result will be
        a single chunk, and for the later the result will be a list of chunks (flattened).

        The length of indices must be the same with `chunk_shape` of tileable.
        """
        if isinstance(item, tuple):
            if len(item) == 0 and self._tileable.is_scalar():
                return self._tileable.chunks[0]
            if len(item) != self._tileable.ndim:
                raise ValueError('Cannot get chunk by %s, expect length %d' % (
                    item, self._tileable.ndim))
            slices, singleton = [], True
            for it, dim in zip(item, self._tileable.chunk_shape):
                if isinstance(it, slice):
                    slices.append(range(dim)[it])
                    singleton = False
                elif np.issubdtype(type(it), np.integer):
                    slices.append([it if it >= 0 else dim + it])
                else:
                    raise TypeError('Cannot get chunk by %s, invalid value has type %s' % (
                        it, type(it)))

            indexes = tuple(zip(*itertools.product(*slices)))

            flat_index = np.ravel_multi_index(indexes, self._tileable.chunk_shape)
            if singleton:
                return self._tileable._chunks[flat_index[0]]
            else:
                return [self._tileable._chunks[idx] for idx in flat_index]

        raise ValueError('Cannot get {0} chunk by {1}'.format(type(self._tileable).__name__, item))


class ExecutableTuple(tuple, _ExecutableMixin, _ToObjectMixin):
    def __init__(self, *_):
        super().__init__()
        self._executed_sessions = []

    def execute(self, session=None, **kw):
        if len(self) == 0:
            return self
        return super().execute(session=session, **kw)

    def fetch(self, session=None, **kw):
        if len(self) == 0:
            return tuple()
        return super().fetch(session=session, **kw)

    def _attach_session(self, session):
        super()._attach_session(session)
        for t in self:
            if hasattr(t, '_attach_session'):
                t._attach_session(session)


class _TileableSession(object):
    def __init__(self, tensor, session):
        key = tensor.key, tensor.id

        def cb(_, sess=ref(session)):
            s = sess()
            if s:
                s.decref(key)
        self._tensor = ref(tensor, cb)


class _TileableDataCleaner(object):
    def __init__(self):
        self._tileable_to_sessions = WeakKeyDictionary()

    @enter_build_mode
    def register(self, tensor, session):
        if tensor in self._tileable_to_sessions:
            self._tileable_to_sessions[tensor].append(_TileableSession(tensor, session))
        else:
            self._tileable_to_sessions[tensor] = [_TileableSession(tensor, session)]


# we don't use __del__ to avoid potential Circular reference
_cleaner = _TileableDataCleaner()


class EntityDataModificationHandler(object):
    def __init__(self):
        self._data_to_entities = WeakKeyDictionary()

    def _add_observer(self, data, entity):
        # only tileable data should be considered
        assert isinstance(data, TileableData)
        assert isinstance(entity, TileableEntity)

        if data not in self._data_to_entities:
            self._data_to_entities[data] = WeakSet()

        self._data_to_entities[data].add(entity)

    @enter_build_mode
    def add_observer(self, data, entity):
        self._add_observer(data, entity)

    def _update_observe_data(self, observer, data, new_data):
        self._data_to_entities.get(data, set()).discard(observer)
        self._add_observer(new_data, observer)

    @staticmethod
    def _set_data(entity, data):
        entity._data.detach(entity)
        entity._data = data
        data.attach(entity)

    @staticmethod
    def _get_data(obj):
        return obj.data if isinstance(obj, Entity) else obj

    @enter_build_mode
    def data_changed(self, old_data, new_data):
        notified = set()
        processed_data = set()
        old_to_new = {old_data: new_data}
        q = [old_data]
        while len(q) > 0:
            data = q.pop()

            # handle entities
            for entity in data.entities:
                self._set_data(entity, old_to_new[data])
                notified.add(entity)

            observers = {ob for ob in self._data_to_entities.pop(data, set())
                         if ob not in notified}
            for ob in observers:
                new_data = self._get_data(ob.op.on_input_modify(old_to_new[data]))
                old_data = ob.data
                self._update_observe_data(ob, ob.data, new_data)
                old_to_new[old_data] = new_data
                if old_data not in processed_data:
                    q.append(old_data)
                    processed_data.add(old_data)
                notified.add(ob)

            if data.op.create_view:
                old_input_data = data.inputs[0]
                new_input_data = self._get_data(data.op.on_output_modify(old_to_new[data]))
                old_to_new[old_input_data] = new_input_data
                if old_input_data not in processed_data:
                    q.append(old_input_data)
                    processed_data.add(old_input_data)


entity_view_handler = EntityDataModificationHandler()
