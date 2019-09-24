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

import threading
import itertools
from operator import attrgetter
from weakref import WeakKeyDictionary, WeakSet, ref

import numpy as np

from .compat import six, izip, builtins
from .utils import tokenize, AttributeDict, on_serialize_shape, \
    on_deserialize_shape, on_serialize_nsplits, is_eager_mode, kernel_mode, calc_data_size
from .serialize import HasKey, ValueType, ProviderType, Serializable, AttributeAsDict, \
    TupleField, ListField, DictField, KeyField, BoolField, StringField, OneOfField
from .tiles import Tileable, handler
from .graph import DAG


class Base(HasKey):
    __slots__ = ()
    _no_copy_attrs_ = {'_id'}
    _init_update_key_ = True

    def __init__(self, *args, **kwargs):
        for slot, arg in izip(self.__slots__, args):
            object.__setattr__(self, slot, arg)

        for key, val in six.iteritems(kwargs):
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

    def _obj_set(self, k, v):
        object.__setattr__(self, k, v)

    def _update_key(self):
        self._obj_set('_key', tokenize(type(self), *self._values_))
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


class Entity(object):
    __slots__ = '_data',
    _allow_data_type_ = ()

    def __init__(self, data):
        self._check_data(data)
        self._data = data

    def __dir__(self):
        from .lib.lib_utils import dir2
        obj_dir = dir2(self)
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

    def __getattr__(self, attr):
        return getattr(self._data, attr)

    def __setattr__(self, key, value):
        try:
            super(Entity, self).__setattr__(key, value)
        except AttributeError:
            return setattr(self._data, key, value)


_threading_local = threading.local()


class BuildMode(object):
    def __init__(self):
        self.is_build_mode = False
        self._old_mode = None
        self._enter_times = 0

    def __enter__(self):
        if self._enter_times == 0:
            # check to prevent nested enter and exit
            self._old_mode = self.is_build_mode
            self.is_build_mode = True
        self._enter_times += 1

    def __exit__(self, *_):
        self._enter_times -= 1
        if self._enter_times == 0:
            self.is_build_mode = self._old_mode
            self._old_mode = None


def build_mode():
    ret = getattr(_threading_local, 'build_mode', None)
    if ret is None:
        ret = BuildMode()
        _threading_local.build_mode = ret

    return ret


def enter_build_mode(func):
    """
    Decorator version of build_mode.

    :param func: function
    :return: the result of function
    """
    def inner(*args, **kwargs):
        with build_mode():
            return func(*args, **kwargs)
    return inner


class SerializableWithKey(Base, Serializable):
    _key = StringField('key')
    _id = StringField('id')


class AttributeAsDictKey(Base, AttributeAsDict):
    _key = StringField('key')
    _id = StringField('id')


class ChunkData(SerializableWithKey):
    __slots__ = '__weakref__', '_siblings'

    # required fields
    _op = KeyField('op')  # store key of operand here
    # optional fields
    _index = TupleField('index', ValueType.uint32)
    _cached = BoolField('cached')
    _extra_params = DictField('extra_params', key_type=ValueType.string, on_deserialize=AttributeDict)

    def __init__(self, *args, **kwargs):
        extras = AttributeDict((k, kwargs.pop(k)) for k in set(kwargs) - set(self.__slots__))
        kwargs['_extra_params'] = kwargs.pop('_extra_params', extras)
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
    def composed(self):
        return getattr(self, '_composed', None)

    @property
    def device(self):
        return self.op.device

    def is_sparse(self):
        return self.op.is_sparse()

    issparse = is_sparse

    def _update_key(self):
        object.__setattr__(self, '_key', tokenize(
            type(self), *(getattr(self, k, None) for k in self._keys_ if k != '_index')))


class Chunk(Entity):
    __slots__ = ()
    _allow_data_type_ = (ChunkData,)


def _on_serialize_composed(composed):
    return [FuseChunkData.ChunkRef(c.data if isinstance(c, Entity) else c) for c in composed]


def _on_deserialize_composed(refs):
    return [r.chunk for r in refs]


class FuseChunkData(ChunkData):
    __slots__ = '_inited',

    class ChunkRef(Serializable):
        _chunk = OneOfField('chunk', tensor_chunk='mars.tensor.core.TensorChunkData',
                            tensor='mars.tensor.core.TensorData',
                            dataframe_chunk='mars.dataframe.core.DataFrameChunkData',
                            dataframe='mars.dataframe.core.DataFrameData',
                            index_chunk='mars.dataframe.core.IndexChunkData',
                            index='mars.dataframe.core.IndexData',
                            series_chunk='mars.dataframe.core.SeriesChunkData',
                            series='mars.dataframe.core.SeriesData')

        @property
        def chunk(self):
            return self._chunk

    _composed = ListField('composed', ValueType.reference(ChunkRef),
                          on_serialize=_on_serialize_composed,
                          on_deserialize=_on_deserialize_composed)

    def __init__(self, *args, **kwargs):
        self._inited = False
        super(FuseChunkData, self).__init__(*args, **kwargs)
        self._extra_params = {}
        self._inited = True

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from .serialize.protos.fusechunk_pb2 import FuseChunkDef
            return FuseChunkDef
        return super(FuseChunkData, cls).cls(provider)

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


class TileableData(SerializableWithKey, Tileable):
    __slots__ = '__weakref__', '_siblings', '_cix', '_entities'
    _no_copy_attrs_ = SerializableWithKey._no_copy_attrs_ | {'_cix'}

    # required fields
    _op = KeyField('op')
    _shape = TupleField('shape', ValueType.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)
    # optional fields
    # `nsplits` means the sizes of chunks for each dimension
    _nsplits = TupleField('nsplits', ValueType.tuple(ValueType.uint64),
                          on_serialize=on_serialize_nsplits)
    _extra_params = DictField('extra_params', key_type=ValueType.string, on_deserialize=AttributeDict)

    def __init__(self, *args, **kwargs):
        extras = AttributeDict((k, kwargs.pop(k)) for k in set(kwargs) - set(self.__slots__))
        kwargs['_extra_params'] = kwargs.pop('_extra_params', extras)
        if kwargs.get('_nsplits', None) is not None:
            kwargs['_nsplits'] = tuple(tuple(s) for s in kwargs['_nsplits'])

        super(TileableData, self).__init__(*args, **kwargs)

        if hasattr(self, '_chunks') and self._chunks:
            self._chunks = sorted(self._chunks, key=attrgetter('index'))

        self._entities = WeakSet()

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
    def chunk_shape(self):
        if hasattr(self, '_nsplits') and self._nsplits is not None:
            return tuple(map(len, self._nsplits))

    @property
    def chunks(self):
        return getattr(self, '_chunks', None)

    @property
    def op(self):
        return getattr(self, '_op', None)

    @property
    def nsplits(self):
        return getattr(self, '_nsplits', None)

    @nsplits.setter
    def nsplits(self, new_nsplits):
        self._nsplits = new_nsplits

    @property
    def size(self):
        return np.prod(self.shape).item()

    @property
    def inputs(self):
        return self.op.inputs or []

    @inputs.setter
    def inputs(self, new_inputs):
        self.op.inputs = new_inputs

    @property
    def params(self):
        # params return the properties which useful to rebuild a new tileable object
        return {
            'shape': self.shape
        }

    @property
    def extra_params(self):
        return self._extra_params

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

    def to_coarse(self):
        if self.is_coarse():
            return self
        new_entity = self.copy()
        new_entity._obj_set('_id', self._id)
        new_entity._chunks = None
        if self.inputs is None or len(self.inputs) == 0:
            new_entity.extra_params.update({'raw_chunk_size': self.nsplits})
        return new_entity

    def is_sparse(self):
        return self.op.is_sparse()

    issparse = is_sparse

    @enter_build_mode
    def attach(self, entity):
        self._entities.add(entity)

    @enter_build_mode
    def detach(self, entity):
        self._entities.discard(entity)

    def tiles(self):
        return handler.tiles(self)

    def single_tiles(self):
        return handler.single_tiles(self)

    @kernel_mode
    def build_graph(self, graph=None, cls=DAG, tiled=False, compose=True, executed_keys=None):
        from .utils import build_fetch

        executed_keys = set(executed_keys or [])
        if tiled and self.is_coarse():
            self.tiles()

        graph = graph if graph is not None else cls()
        keys = None

        if tiled:
            nodes = list(c.data for c in self.chunks)
            keys = list(c.key for c in self.chunks)
        else:
            nodes = list(self.op.outputs)

        node_to_fetch = dict()

        def _generate_fetch_node(n):
            if n in node_to_fetch:
                return node_to_fetch[n]
            fn = build_fetch(n, coarse=True).data
            node_to_fetch[n] = fn
            return fn

        visited = set()
        while len(nodes) > 0:
            node = nodes.pop()

            # replace executed tensor/chunk by tensor/chunk with fetch op
            if node.key in executed_keys:
                node = _generate_fetch_node(node)

            visited.add(node)
            if not graph.contains(node):
                graph.add_node(node)
            children = node.inputs or []
            for c in children:
                if c.key in executed_keys:
                    visited.add(c)
                    c = _generate_fetch_node(c)
                if not graph.contains(c):
                    graph.add_node(c)
                if not graph.has_successor(c, node):
                    graph.add_edge(c, node)
            nodes.extend([c for c in itertools.chain(*[inp.op.outputs for inp in node.inputs or []])
                          if c not in visited])
        if tiled and compose:
            graph.compose(keys=keys)

        if not tiled and any(not n.is_coarse() for n in graph):
            return self._to_coarse_graph(graph)

        return graph

    @staticmethod
    def _to_coarse_graph(graph):
        new_graph = type(graph)()
        visited = dict()
        for n in graph:
            if n not in visited:
                new_node = n.to_coarse()
                visited[n] = new_node
                new_graph.add_node(new_node)
            for succ in graph.successors(n):
                if succ not in visited:
                    new_node = succ.to_coarse()
                    visited[succ] = new_node
                    new_graph.add_node(new_node)
                new_graph.add_edge(visited[n], visited[succ])
        return new_graph

    def visualize(self, graph_attrs=None, node_attrs=None, **kw):
        from graphviz import Source

        g = self.build_graph(**kw)
        dot = g.to_dot(graph_attrs=graph_attrs, node_attrs=node_attrs,
                       result_chunk_keys={c.key for c in self.chunks})

        return Source(dot)

    def execute(self, session=None, **kw):
        from .session import Session

        if session is None:
            session = Session.default_or_local()
        return session.run(self, **kw)

    def fetch(self, session=None, **kw):
        from .session import Session

        if session is None:
            session = Session.default_or_local()
        return session.fetch(self, **kw)

    def _set_execute_session(self, session):
        _cleaner.register(self, session)

    _execute_session = property(fset=_set_execute_session)


class TileableEntity(Entity):
    __slots__ = '__weakref__',

    def __init__(self, data):
        super(TileableEntity, self).__init__(data)
        if self._data is not None:
            self._data.attach(self)
            if self._data.op.create_view:
                entity_view_handler.add_observer(self._data.inputs[0], self)

    def __copy__(self):
        return self.view()

    def view(self):
        return super(TileableEntity, self).copy()

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
        if self._data is None:
            self._data = new_data
            self._data.attach(self)
        else:
            entity_view_handler.data_changed(self._data, new_data)


class ChunksIndexer(object):
    __slots__ = '_tileable',

    def __init__(self, tileable):
        self._tileable = tileable

    def __getitem__(self, item):
        '''
        The indices for `cix` can be [x, y] or [x, :]. For the former the result will be
        a single chunk, and for the later the result will be a list of chunks (flattened).

        The length of indices must be the same with `chunk_shape` of tileable.
        '''
        if isinstance(item, tuple):
            if len(item) == 0 and self._tileable.is_scalar():
                return self._tileable.chunks[0]
            if len(item) != self._tileable.ndim:
                raise ValueError('Cannot get tensor chunk by %s, expect length %d' % (
                    item, self._tileable.ndim))
            slices, singleton = [], True
            for it, dim in zip(item, self._tileable.chunk_shape):
                if isinstance(it, slice):
                    slices.append(range(dim)[it])
                    singleton = False
                elif np.issubdtype(type(it), np.integer):
                    slices.append([it if it >= 0 else dim + it])
                else:
                    raise TypeError('Cannot get tensor chunk by %s, invalid value has type %s' % (
                        it, type(it)))

            indexes = tuple(zip(*itertools.product(*slices)))

            flat_index = np.ravel_multi_index(indexes, self._tileable.chunk_shape)
            if singleton:
                return self._tileable._chunks[flat_index[0]]
            else:
                return [self._tileable._chunks[idx] for idx in flat_index]

        raise ValueError('Cannot get tensor chunk by {0}'.format(item))


class TileableOperandMixin(object):
    __slots__ = ()

    def check_inputs(self, inputs):
        pass

    @classmethod
    def _check_if_gpu(cls, inputs):
        if inputs is not None and \
                len([inp for inp in inputs
                     if inp is not None and getattr(inp, 'op', None) is not None]) > 0:
            if all(inp.op.gpu is True for inp in inputs):
                return True
            elif all(inp.op.gpu is False for inp in inputs):
                return False

    def _create_chunk(self, output_idx, index, **kw):
        raise NotImplementedError

    def _new_chunks(self, inputs, kws=None, **kw):
        output_limit = kw.pop('output_limit', None)
        if output_limit is None:
            output_limit = getattr(self, 'output_limit')

        self.check_inputs(inputs)
        getattr(self, '_set_inputs')(inputs)
        if getattr(self, '_gpu', None) is None:
            self._gpu = self._check_if_gpu(self._inputs)
        if getattr(self, '_key', None) is None:
            getattr(self, '_update_key')()

        chunks = []
        for j in range(output_limit):
            create_chunk_kw = kw.copy()
            if kws:
                create_chunk_kw.update(kws[j])
            index = create_chunk_kw.pop('index', None)
            chunk = self._create_chunk(j, index, **create_chunk_kw)
            chunks.append(chunk)

        setattr(self, 'outputs', chunks)
        if len(chunks) > 1:
            # for each output chunk, hold the reference to the other outputs
            # so that either no one or everyone are gc collected
            for j, t in enumerate(chunks):
                t.data._siblings = [c.data for c in chunks[:j] + chunks[j + 1:]]
        return chunks

    def new_chunks(self, inputs, kws=None, **kwargs):
        """
        Create chunks.
        A chunk is a node in a fine grained graph, all the chunk objects are created by
        calling this function, it happens mostly in tiles.
        The generated chunks will be set as this operand's outputs and each chunk will
        hold this operand as it's op.
        :param inputs: input chunks
        :param kws: kwargs for each output
        :param kwargs: common kwargs for all outputs

        .. note::
            It's a final method, do not override.
            Override the method `_new_chunks` if needed.
        """
        return self._new_chunks(inputs, kws=kws, **kwargs)

    def new_chunk(self, inputs, kws=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new chunk with more than 1 outputs')

        return self.new_chunks(inputs, kws=kws, **kw)[0]

    def _create_tileable(self, output_idx, **kw):
        raise NotImplementedError

    def _new_tileables(self, inputs, kws=None, **kw):
        output_limit = kw.pop('output_limit', None)
        if output_limit is None:
            output_limit = getattr(self, 'output_limit')

        self.check_inputs(inputs)
        getattr(self, '_set_inputs')(inputs)
        if getattr(self, '_gpu', None) is None:
            self._gpu = self._check_if_gpu(self._inputs)
        if getattr(self, '_key', None) is None:
            getattr(self, '_update_key')()  # update key when inputs are set

        tileables = []
        for j in range(output_limit):
            create_tensor_kw = kw.copy()
            if kws:
                create_tensor_kw.update(kws[j])
            tileable = self._create_tileable(j, **create_tensor_kw)
            tileables.append(tileable)

        setattr(self, 'outputs', tileables)
        if len(tileables) > 1:
            # for each output tileable, hold the reference to the other outputs
            # so that either no one or everyone are gc collected
            for j, t in enumerate(tileables):
                t.data._siblings = [tileable.data for tileable in tileables[:j] + tileables[j + 1:]]
        return tileables

    def new_tileables(self, inputs, kws=None, **kw):
        """
        Create tileable objects(Tensors or DataFrames).
        This is a base function for create tileable objects like tensors or dataframes,
        it will be called inside the `new_tensors` and `new_dataframes`.
        If eager mode is on, it will trigger the execution after tileable objects are created.
        :param inputs: input tileables
        :param kws: kwargs for each output
        :param kw: common kwargs for all outputs

        .. note::
            It's a final method, do not override.
            Override the method `_new_tileables` if needed.
        """

        tileables = self._new_tileables(inputs, kws=kws, **kw)
        if is_eager_mode():
            ExecutableTuple(tileables).execute(fetch=False)
        return tileables

    def new_tileable(self, inputs, kws=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new chunk with more than 1 outputs')

        return self.new_tileables(inputs, kws=kws, **kw)[0]

    @classmethod
    def execute(cls, ctx, op):
        raise NotImplementedError

    @classmethod
    def estimate_size(cls, ctx, op):
        exec_size = 0
        outputs = op.outputs
        if all(not c.is_sparse() and hasattr(c, 'nbytes') and not np.isnan(c.nbytes) for c in outputs):
            for out in outputs:
                ctx[out.key] = (out.nbytes, out.nbytes)

        for inp in op.inputs or ():
            try:
                exec_size += ctx[inp.key][0]
            except KeyError:
                if not op.sparse:
                    inp_size = calc_data_size(inp)
                    if not np.isnan(inp_size):
                        exec_size += inp_size
        exec_size = int(exec_size)

        total_out_size = 0
        chunk_sizes = dict()
        for out in outputs:
            try:
                chunk_size = calc_data_size(out) if not out.is_sparse() else exec_size
                if np.isnan(chunk_size):
                    raise TypeError
                chunk_sizes[out.key] = chunk_size
                total_out_size += chunk_size
            except (AttributeError, TypeError, ValueError):
                pass
        exec_size = max(exec_size, total_out_size)
        for out in outputs:
            if out.key in ctx:
                continue
            if out.key in chunk_sizes:
                store_size = chunk_sizes[out.key]
            else:
                store_size = max(exec_size // len(outputs),
                                 total_out_size // max(len(chunk_sizes), 1))
            try:
                if out.is_sparse():
                    max_sparse_size = out.nbytes + np.dtype(np.int64).itemsize * np.prod(out.shape) * out.ndim
                else:
                    max_sparse_size = np.nan
            except TypeError:  # pragma: no cover
                max_sparse_size = np.nan
            if not np.isnan(max_sparse_size):
                store_size = min(store_size, max_sparse_size)
            ctx[out.key] = (store_size, exec_size // len(outputs))


class ExecutableTuple(tuple):
    def execute(self, session=None, **kw):
        from .session import Session

        if session is None:
            session = Session.default_or_local()
        return session.run(*self, **kw)


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
