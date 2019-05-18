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


import itertools
from operator import attrgetter, mul
from weakref import WeakKeyDictionary, ref
from collections import Iterable
import threading

import numpy as np

from ..core import Entity
from ..compat import builtins, reduce
from ..graph import DAG
from ..tiles import Tilesable, handler
from ..serialize import SerializableWithKey, ValueType, ProviderType, \
    ListField, TupleField, DictField, DataTypeField, KeyField, BoolField
from ..utils import AttributeDict, on_serialize_shape, on_deserialize_shape, tokenize
from .expressions.utils import get_chunk_slices


class ChunkData(SerializableWithKey):
    __slots__ = '__weakref__', '_siblings'

    # required fields
    _shape = TupleField('shape', ValueType.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)
    _op = KeyField('op')  # store key of operand here
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
            from ..serialize.protos.chunk_pb2 import ChunkDef
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

    def _update_key(self):
        object.__setattr__(self, '_key', tokenize(
            type(self), *(getattr(self, k, None) for k in self._keys_ if k != '_index')))


class Chunk(Entity):
    __slots__ = ()
    _allow_data_type_ = (ChunkData,)


class TensorData(SerializableWithKey, Tilesable):
    __slots__ = '__weakref__', '_siblings', '_cix'
    _no_copy_attrs_ = SerializableWithKey._no_copy_attrs_ | {'_cix'}

    # required fields
    _shape = TupleField('shape', ValueType.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)
    _dtype = DataTypeField('dtype')
    _op = KeyField('op')
    # optional fields
    # `nsplits` means the sizes of chunks for each dimension
    _nsplits = TupleField('nsplits', ValueType.tuple(ValueType.uint64))
    _chunks = ListField('chunks', ValueType.reference(ChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [Chunk(it) for it in x] if x is not None else x)
    _params = DictField('params', key_type=ValueType.string, on_deserialize=AttributeDict)

    def __init__(self, *args, **kwargs):
        extras = AttributeDict((k, kwargs.pop(k)) for k in set(kwargs) - set(self.__slots__))
        kwargs['_params'] = kwargs.pop('_params', extras)
        if '_nsplits' in kwargs:
            kwargs['_nsplits'] = tuple(tuple(s) for s in kwargs['_nsplits'])

        super(TensorData, self).__init__(*args, **kwargs)

        if hasattr(self, '_chunks') and self._chunks:
            self._chunks = sorted(self._chunks, key=attrgetter('index'))

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from ..serialize.protos.tensor_pb2 import TensorDef
            return TensorDef
        return super(TensorData, cls).cls(provider)

    def __repr__(self):
        return 'Tensor <op={0}, key={1}>'.format(self.op.__class__.__name__, self.key)

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
    def nbytes(self):
        return np.prod(self.shape) * self.dtype.itemsize

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
    def dtype(self):
        return getattr(self, '_dtype', None) or self.op.dtype

    @property
    def params(self):
        return self._params

    @property
    def real(self):
        from .expressions.arithmetic import real
        return real(self)

    @property
    def imag(self):
        from .expressions.arithmetic import imag
        return imag(self)

    def get_chunk_slices(self, idx):
        return get_chunk_slices(self.nsplits, idx)

    def is_coarse(self):
        return not hasattr(self, '_chunks') or self._chunks is None or len(self._chunks) == 0

    def to_coarse(self):
        if self.is_coarse():
            return self
        new_entity = self.copy()
        new_entity._obj_set('_id', self._id)
        new_entity._chunks = None
        if self.inputs is None or len(self.inputs) == 0:
            new_entity.params.update({'raw_chunk_size': self.nsplits})
        return new_entity

    def is_scalar(self):
        return self.ndim == 0

    isscalar = is_scalar

    def is_sparse(self):
        return self.op.is_sparse()

    issparse = is_sparse

    def tosparse(self):
        if self.issparse():
            return self

        from .expressions.datasource import fromdense
        return fromdense(self)

    def todense(self):
        if not self.issparse():
            return self

        from .expressions.datasource import fromsparse
        return fromsparse(self)

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

    def tiles(self):
        return handler.tiles(self)

    def single_tiles(self):
        return handler.single_tiles(self)

    def build_graph(self, graph=None, cls=DAG, tiled=False, compose=True):
        if tiled and self.is_coarse():
            self.tiles()

        graph = graph if graph is not None else cls()
        keys = None

        if tiled:
            nodes = list(c.data for c in self.chunks)
            keys = list(c.key for c in self.chunks)
        else:
            nodes = list(self.op.outputs)
        visited = set()
        while len(nodes) > 0:
            chunk = nodes.pop()
            visited.add(chunk)
            if not graph.contains(chunk):
                graph.add_node(chunk)
            children = chunk.inputs or []
            for c in children:
                if not graph.contains(c):
                    graph.add_node(c)
                if not graph.has_successor(c, chunk):
                    graph.add_edge(c, chunk)
            nodes.extend([c for c in itertools.chain(*[inp.op.outputs for inp in chunk.inputs or []])
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

    def transpose(self, *axes):
        """
        Returns a view of the tensor with axes transposed.

        For a 1-D tensor, this has no effect. (To change between column and
        row vectors, first cast the 1-D tensor into a matrix object.)
        For a 2-D tensor, this is the usual matrix transpose.
        For an n-D tensor, if axes are given, their order indicates how the
        axes are permuted (see Examples). If axes are not provided and
        ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
        ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

        Parameters
        ----------
        axes : None, tuple of ints, or `n` ints

         * None or no argument: reverses the order of the axes.

         * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
           `i`-th axis becomes `a.transpose()`'s `j`-th axis.

         * `n` ints: same as an n-tuple of the same ints (this form is
           intended simply as a "convenience" alternative to the tuple form)

        Returns
        -------
        out : Tensor
            View of `a`, with axes suitably permuted.

        See Also
        --------
        Tensor.T : Tensor property returning the tensor transposed.

        Examples
        --------
        >>> import mars.tensor as mt

        >>> a = mt.array([[1, 2], [3, 4]])
        >>> a.execute()
        array([[1, 2],
               [3, 4]])
        >>> a.transpose().execute()
        array([[1, 3],
               [2, 4]])
        >>> a.transpose((1, 0))
        array([[1, 3],
               [2, 4]])
        >>> a.transpose(1, 0).execute()
        array([[1, 3],
               [2, 4]])
        """
        from .expressions.base import transpose

        if len(axes) == 1 and isinstance(axes[0], Iterable):
            axes = axes[0]

        return transpose(self, axes)

    @property
    def T(self):
        """
        Same as self.transpose(), except that self is returned if
        self.ndim < 2.

        Examples
        --------
        >>> import mars.tensor as mt

        >>> x = mt.array([[1.,2.],[3.,4.]])
        >>> x.execute()
        array([[ 1.,  2.],
               [ 3.,  4.]])
        >>> x.T.execute()
        array([[ 1.,  3.],
               [ 2.,  4.]])
        >>> x = mt.array([1.,2.,3.,4.])
        >>> x.execute()
        array([ 1.,  2.,  3.,  4.])
        >>> x.T.execute()
        array([ 1.,  2.,  3.,  4.])
        """
        return self.transpose()

    def reshape(self, shape, *shapes):
        """
        Returns a tensor containing the same data with a new shape.

        Refer to `mt.reshape` for full documentation.

        See Also
        --------
        mt.reshape : equivalent function

        Notes
        -----
        Unlike the free function `mt.reshape`, this method on `Tensor` allows
        the elements of the shape parameter to be passed in as separate arguments.
        For example, ``a.reshape(10, 11)`` is equivalent to
        ``a.reshape((10, 11))``.
        """
        from .expressions.reshape import reshape

        if isinstance(shape, Iterable):
            shape = tuple(shape)
        else:
            shape = (shape,)
        shape += shapes

        return reshape(self, shape)

    def ravel(self):
        """
        Return a flattened tensor.

        Refer to `mt.ravel` for full documentation.

        See Also
        --------
        mt.ravel : equivalent function
        """
        from .expressions.base import ravel

        return ravel(self)

    flatten = ravel

    def totiledb(self, uri, ctx=None, key=None, timestamp=None):
        from .expressions.datastore import totiledb

        return totiledb(uri, self, ctx=ctx, key=key, timestamp=timestamp)

    def _equals(self, o):
        return self is o

    def execute(self, session=None, **kw):
        from ..session import Session

        if session is None:
            session = Session.default_or_local()
        return session.run(self, **kw)

    def _set_execute_session(self, session):
        _cleaner.register(self, session)

    _execute_session = property(fset=_set_execute_session)

    def visualize(self, graph_attrs=None, node_attrs=None, **kw):
        from graphviz import Source

        g = self.build_graph(**kw)
        dot = g.to_dot(graph_attrs=graph_attrs, node_attrs=node_attrs)

        return Source(dot)


class ExecutableTuple(tuple):
    def execute(self, session=None, **kw):
        from ..session import Session

        if session is None:
            session = Session.default_or_local()
        return session.run(*self, **kw)


class ChunksIndexer(object):
    __slots__ = '_tensor',

    def __init__(self, tensor):
        self._tensor = tensor

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) == 0 and self._tensor.is_scalar():
                return self._tensor.chunks[0]
            elif all(np.issubdtype(type(it), np.integer) for it in item):
                if len(item) != self._tensor.ndim:
                    raise ValueError('Cannot get tensor chunk by %s, expect length %d' % (
                        item, self._tensor.ndim))

                s = self._tensor.chunk_shape
                item = tuple(i if i >= 0 else i + s for i, s in zip(item, s))
                idx = sum(idx * reduce(mul, s[i+1:], 1) for i, idx
                          in zip(itertools.count(0), item))
                return self._tensor._chunks[idx]

        raise ValueError('Cannot get tensor chunk by {0}'.format(item))


class Tensor(Entity):
    __slots__ = ()
    _allow_data_type_ = (TensorData,)

    def __dir__(self):
        from ..lib.lib_utils import dir2
        obj_dir = dir2(self)
        if self._data is not None:
            obj_dir = sorted(set(dir(self._data) + obj_dir))
        return obj_dir

    def __len__(self):
        return len(self._data)

    def copy(self):
        return Tensor(self._data)

    def tiles(self):
        return handler.tiles(self)

    def single_tiles(self):
        return handler.single_tiles(self)

    @property
    def shape(self):
        return self.data.shape

    @shape.setter
    def shape(self, new_shape):
        self._data = self._data.reshape(new_shape).data

    def _update_shape(self, new_shape):
        self._data._update_shape(new_shape)

    @property
    def real(self):
        return self.data.real

    @real.setter
    def real(self, new_real):
        from .expressions.arithmetic.setreal import set_real

        self._data = set_real(self._data, new_real).data

    @property
    def imag(self):
        return self.data.imag

    @imag.setter
    def imag(self, new_imag):
        from .expressions.arithmetic.setimag import set_imag

        self._data = set_imag(self._data, new_imag).data

    def __array__(self, dtype=None):
        return np.asarray(self.execute(), dtype=dtype)

    def __array_function__(self, func, types, args, kwargs):
        from .. import tensor as module

        for submodule in func.__module__.split('.')[1:]:
            try:
                module = getattr(module, submodule)
            except AttributeError:
                return NotImplemented
        if not hasattr(module, func.__name__):
            return NotImplemented
        mars_func = getattr(module, func.__name__)
        if mars_func is func:
            # avoid Numpy func
            return NotImplemented
        return mars_func(*args, **kwargs)

    def execute(self, session=None, **kw):
        return self._data.execute(session, **kw)


class SparseTensor(Tensor):
    __slots__ = ()


TENSOR_TYPE = (Tensor, TensorData)
CHUNK_TYPE = (Chunk, ChunkData)

_threading_local = threading.local()


class _TensorSession(object):
    def __init__(self, tensor, session):
        key = tensor.key, tensor.id

        def cb(_, sess=ref(session)):
            s = sess()
            if s:
                s.decref(key)
        self._tensor = ref(tensor, cb)


class _TensorCleaner(object):
    def __init__(self):
        self._tensor_to_sessions = WeakKeyDictionary()

    def register(self, tensor, session):
        with build_mode():
            self._tensor_to_sessions[tensor] = _TensorSession(tensor, session)


# we don't use __del__ to decref because a tensor holds an op,
# and op's outputs contains the tensor, so a circular references exists
_cleaner = _TensorCleaner()


class BuildMode(object):
    def __init__(self):
        self.is_build_mode = False
        self._old_mode = None

    def __enter__(self):
        if self._old_mode is None:
            # check to prevent nested enter and exit
            self._old_mode = self.is_build_mode
            self.is_build_mode = True

    def __exit__(self, *_):
        if self._old_mode is not None:
            self.is_build_mode = self._old_mode
            self._old_mode = None


def build_mode():
    ret = getattr(_threading_local, 'build_mode', None)
    if ret is None:
        ret = BuildMode()
        _threading_local.build_mode = ret

    return ret
