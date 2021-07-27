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

import builtins
import inspect
import itertools
from operator import attrgetter
from typing import List, Callable, Generator
from weakref import WeakSet, WeakKeyDictionary

import numpy as np

from ...serialization.serializables import FieldTypes, TupleField
from ...typing import OperandType, TileableType, ChunkType
from ...utils import on_serialize_shape, on_deserialize_shape, on_serialize_nsplits
from ..base import Base
from ..mode import enter_mode, is_build_mode
from .chunks import Chunk
from .core import EntityData, Entity
from .executable import _ExecutableMixin


class NotSupportTile(Exception):
    pass


class OperandTilesHandler:
    _handlers = dict()

    @classmethod
    def _get_op_cls(cls, op: OperandType):
        if isinstance(op, type):
            return op
        return type(op)

    @classmethod
    def register(cls,
                 op: OperandType,
                 tile_handler: Callable[[OperandType], TileableType]):
        cls._handlers[cls._get_op_cls(op)] = tile_handler

    @classmethod
    def unregister(cls,
                   op: OperandType):
        del cls._handlers[cls._get_op_cls(op)]

    @classmethod
    def get_handler(cls, op: OperandType) -> Callable[[OperandType], List[TileableType]]:
        op_cls = cls._get_op_cls(op)
        return cls._handlers.get(op_cls, op_cls.tile)

    @classmethod
    def tile(cls, tileables: List[TileableType]) \
            -> Generator[List[ChunkType], List[ChunkType], List[TileableType]]:
        op = tileables[0].op
        # pre tile
        op.pre_tile(op)
        tiled_result = None
        try:
            tile_handler = cls.get_handler(op)
            if inspect.isgeneratorfunction(tile_handler):
                # op.tile can be a generator function,
                # each time an operand yield some chunks,
                # they will be put into ChunkGraph and executed first.
                # After execution, resume from the yield place.
                tiled_result = yield from tile_handler(op)
            else:
                # without iterative tiling
                tiled_result = tile_handler(op)
        finally:
            op.post_tile(op, tiled_result)

        if not isinstance(tiled_result, list):
            tiled_result = [tiled_result]
        tiled_results = [t.data if hasattr(t, 'data') else t
                         for t in tiled_result]
        assert len(tileables) == len(tiled_results)
        if any(inspect.isgenerator(r) for r in tiled_results):  # pragma: no cover
            raise TypeError(f'tiled result cannot be generator '
                            f'when tiling {op}')
        cls._assign_to(tiled_results, tileables)
        return tileables

    @classmethod
    def _assign_to(cls,
                   tile_after_tensor_datas: List["TileableData"],
                   tile_before_tensor_datas: List["TileableData"]):
        assert len(tile_after_tensor_datas) == len(tile_before_tensor_datas)

        for tile_after_tensor_data, tile_before_tensor_data in \
                zip(tile_after_tensor_datas, tile_before_tensor_datas):
            if tile_before_tensor_data is None:
                # garbage collected
                continue
            tile_after_tensor_data.copy_to(tile_before_tensor_data)
            tile_before_tensor_data.op.outputs = tile_before_tensor_datas

    @enter_mode(kernel=True)
    def dispatch(self, op: OperandType):
        op_cls = self._get_op_cls(op)
        tiled = None
        cause = None

        if op_cls in self._handlers:
            tiled = self._handlers[op_cls](op)
        else:
            try:
                tiled = op_cls.tile(op)
            except NotImplementedError as ex:
                cause = ex
                for super_cls in op_cls.__mro__:
                    if super_cls in self._handlers:
                        h = self._handlers[op_cls] = self._handlers[super_cls]
                        tiled = h(op)
                        break

        if tiled is not None:
            return tiled if isinstance(tiled, list) else [tiled]
        else:
            raise NotImplementedError(
                f'{type(op)} does not support tile') from cause


handler = OperandTilesHandler()
register = OperandTilesHandler.register
unregister = OperandTilesHandler.unregister


class _ChunksIndexer:
    __slots__ = '_tileable',

    def __init__(self, tileable):
        self._tileable = tileable

    def __getitem__(self, item):
        """
        The indices for `cix` can be [x, y] or [x, :].
        For the former the result will be a single chunk,
        and for the later the result will be a list of chunks (flattened).

        The length of indices must be the same with `chunk_shape` of tileable.
        """
        if isinstance(item, tuple):
            if len(item) == 0 and self._tileable.is_scalar():
                return self._tileable.chunks[0]
            if len(item) != self._tileable.ndim:
                raise ValueError(f'Cannot get chunk by {item}, '
                                 f'expect length {self._tileable.ndim}')
            slices, singleton = [], True
            for it, dim in zip(item, self._tileable.chunk_shape):
                if isinstance(it, slice):
                    slices.append(range(dim)[it])
                    singleton = False
                elif np.issubdtype(type(it), np.integer):
                    slices.append([it if it >= 0 else dim + it])
                else:
                    raise TypeError(f'Cannot get chunk by {it}, '
                                    f'invalid value has type {type(it)}')

            indexes = tuple(zip(*itertools.product(*slices)))

            flat_index = np.ravel_multi_index(indexes, self._tileable.chunk_shape)
            if singleton:
                return self._tileable._chunks[flat_index[0]]
            else:
                return [self._tileable._chunks[idx] for idx in flat_index]

        raise ValueError(f'Cannot get {type(self._tileable).__name__} chunk by {item}')


class EntityDataModificationHandler:
    def __init__(self):
        self._data_to_entities = WeakKeyDictionary()

    def _add_observer(self, data, entity):
        # only tileable data should be considered
        assert isinstance(data, TileableData)
        assert isinstance(entity, Tileable)

        if data not in self._data_to_entities:
            self._data_to_entities[data] = WeakSet()

        self._data_to_entities[data].add(entity)

    @enter_mode(build=True)
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

    @enter_mode(build=True)
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


class TileableData(EntityData, _ExecutableMixin):
    __slots__ = '_cix', '_entities', '_executed_sessions'
    _no_copy_attrs_ = Base._no_copy_attrs_ | {'_cix'}

    # optional fields
    # `nsplits` means the sizes of chunks for each dimension
    _nsplits = TupleField('nsplits', FieldTypes.tuple(FieldTypes.uint64),
                          on_serialize=on_serialize_nsplits)

    def __init__(self: TileableType, *args, **kwargs):
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
    def chunks(self) -> List[Chunk]:
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
            return _ChunksIndexer(self)

        try:
            if getattr(self, '_cix', None) is None:
                self._cix = _ChunksIndexer(self)
            return self._cix
        except (TypeError, ValueError):
            return _ChunksIndexer(self)

    @property
    def entities(self):
        return self._entities

    def is_coarse(self):
        if not hasattr(self, '_chunks'):
            return True
        if not self._chunks:
            return True
        return False

    @enter_mode(build=True)
    def attach(self, entity):
        self._entities.add(entity)

    @enter_mode(build=True)
    def detach(self, entity):
        self._entities.discard(entity)


class Tileable(Entity):
    __slots__ = '__weakref__',

    def __init__(self, data: TileableType=None, **kw):
        super().__init__(data=data, **kw)
        if self._data is not None:
            self._data.attach(self)
            if self._data.op.create_view:
                entity_view_handler.add_observer(
                    self._data.inputs[0], self)

    def __copy__(self):
        return self._view()

    def _view(self):
        return super().copy()

    def copy(self: TileableType) -> TileableType:
        new_op = self.op.copy()
        if new_op.create_view:
            # if the operand is a view, make it a copy
            new_op.create_view = False
        params = []
        for o in self.op.outputs:
            param = o.params
            param['_key'] = o.key
            param.update(o.extra_params)
            params.append(param)
        new_outs = new_op.new_tileables(self.op.inputs, kws=params,
                                        output_limit=len(params))
        pos = -1
        for i, out in enumerate(self.op.outputs):
            # create a ref to copied one
            new_out = new_outs[i]
            if not hasattr(new_out.data, '_siblings'):
                new_out.data._siblings = []
            new_out.data._siblings.append(out)

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


TILEABLE_TYPE = (Tileable, TileableData)


class HasShapeTileableData(TileableData):
    # required fields
    _shape = TupleField('shape', FieldTypes.int64,
                        on_serialize=on_serialize_shape,
                        on_deserialize=on_deserialize_shape)

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        try:
            return self.shape[0]
        except IndexError:
            if is_build_mode():
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


class HasShapeTileable(Tileable):
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
        result = self.data.execute(session=session, **kw)
        if isinstance(result, TILEABLE_TYPE):
            return self
        else:
            return result
