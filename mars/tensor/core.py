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


from collections import Iterable, defaultdict
from datetime import datetime

import numpy as np

from ..core import Entity, ChunkData, Chunk, TileableData, is_eager_mode, build_mode
from ..tiles import handler
from ..serialize import ProviderType, ValueType, DataTypeField, ListField, TupleField, BoolField, StringField
from ..utils import log_unhandled, on_serialize_shape, on_deserialize_shape
from .expressions.utils import get_chunk_slices

import logging
logger = logging.getLogger(__name__)


class TensorChunkData(ChunkData):
    __slots__ = ()

    # required fields
    _shape = TupleField('shape', ValueType.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)
    # optional fields
    _dtype = DataTypeField('dtype')

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from ..serialize.protos.tensor_pb2 import TensorChunkDef
            return TensorChunkDef
        return super(TensorChunkData, cls).cls(provider)

    @property
    def params(self):
        # params return the properties which useful to rebuild a new chunk
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'index': self.index,
        }

    def __len__(self):
        try:
            return self.shape[0]
        except IndexError:
            if build_mode().is_build_mode:
                return 0
            raise TypeError('len() of unsized object')

    @property
    def shape(self):
        return getattr(self, '_shape', None)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape).item()

    @property
    def dtype(self):
        return getattr(self, '_dtype', None) or self.op.dtype

    @property
    def nbytes(self):
        return np.prod(self.shape) * self.dtype.itemsize


class TensorChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (TensorChunkData,)

    def __len__(self):
        return len(self._data)


class TensorData(TileableData):
    __slots__ = ()

    # required fields
    _dtype = DataTypeField('dtype')
    _chunks = ListField('chunks', ValueType.reference(TensorChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [TensorChunk(it) for it in x] if x is not None else x)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from ..serialize.protos.tensor_pb2 import TensorDef
            return TensorDef
        return super(TensorData, cls).cls(provider)

    def __str__(self):
        if is_eager_mode():
            return 'Tensor(op={0}, shape={1}, data=\n{2})'.format(self.op.__class__.__name__,
                                                                  self.shape, str(self.fetch()))
        else:
            return 'Tensor(op={0}, shape={1})'.format(self.op.__class__.__name__, self.shape)

    def __repr__(self):
        if is_eager_mode():
            return 'Tensor <op={0}, shape={1}, key={2}, data=\n{3}>'.format(self.op.__class__.__name__,
                                                                            self.shape, self.key,
                                                                            repr(self.fetch()))
        else:
            return 'Tensor <op={0}, shape={1}, key={2}>'.format(self.op.__class__.__name__,
                                                                self.shape, self.key)

    @property
    def params(self):
        # params return the properties which useful to rebuild a new tileable object
        return {
            'shape': self.shape,
            'dtype': self.dtype
        }

    @property
    def real(self):
        from .expressions.arithmetic import real
        return real(self)

    @property
    def imag(self):
        from .expressions.arithmetic import imag
        return imag(self)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None) or self.op.dtype

    @property
    def nbytes(self):
        return np.prod(self.shape) * self.dtype.itemsize

    def get_chunk_slices(self, idx):
        return get_chunk_slices(self.nsplits, idx)

    def is_scalar(self):
        return self.ndim == 0

    isscalar = is_scalar

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

    def transpose(self, *axes):
        from .expressions.base import transpose

        if len(axes) == 1 and isinstance(axes[0], Iterable):
            axes = axes[0]

        return transpose(self, axes)

    @property
    def T(self):
        return self.transpose()

    def reshape(self, shape, *shapes):
        from .expressions.reshape import reshape

        if isinstance(shape, Iterable):
            shape = tuple(shape)
        else:
            shape = (shape,)
        shape += shapes

        return reshape(self, shape)

    def ravel(self):
        from .expressions.base import ravel

        return ravel(self)

    flatten = ravel

    def _equals(self, o):
        return self is o

    def totiledb(self, uri, ctx=None, key=None, timestamp=None):
        from .expressions.datastore import totiledb

        return totiledb(uri, self, ctx=ctx, key=key, timestamp=timestamp)


class Tensor(Entity):
    __slots__ = ()
    _allow_data_type_ = (TensorData,)

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
        if is_eager_mode():
            return np.asarray(self.fetch(), dtype=dtype)
        else:
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
        return self._data.transpose(*axes)

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
        return self._data.T

    def ravel(self):
        """
        Return a flattened tensor.

        Refer to `mt.ravel` for full documentation.

        See Also
        --------
        mt.ravel : equivalent function
        """
        return self._data.ravel()

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
        return self._data.reshape(shape, *shapes)

    def totiledb(self, uri, ctx=None, key=None, timestamp=None):
        return self._data.totiledb(uri, ctx=ctx, key=key, timestamp=timestamp)

    def execute(self, session=None, **kw):
        return self._data.execute(session, **kw)


class MutableTensorData(TensorData):
    __slots__ = ()

    # required fields
    _name = StringField('name')
    _compression = BoolField("compression")

    @classmethod
    def cls(cls, provider):
        return super(MutableTensorData, cls).cls(provider)

    def __str__(self):
        return 'MutableTensor(op={0}, name={1}, shape={2})'.format(self.op.__class__.__name__,
                                                                   self.name,
                                                                   self.shape)

    def __repr__(self):
        return 'MutableTensor <op={0}, name={1}, shape={2}, key={3}>'.format(self.op.__class__.__name__,
                                                                             self.name,
                                                                             self.shape,
                                                                             self.key)

    @property
    def params(self):
        # params return the properties which useful to rebuild a new tileable object
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'name': self.name,
            'compression': self.compression,
        }

    @property
    def name(self):
        return getattr(self, '_name', None)

    @property
    def compression(self):
        return getattr(self, '_compression', None)


class MutableTensor(Entity):
    __slots__ = ("_chunk_buffers", "_record_type", "_buffer_size")
    _allow_data_type_ = (MutableTensorData,)

    def __init__(self, *args, **kwargs):
        super(MutableTensor, self).__init__(*args, **kwargs)
        self._chunk_buffers = defaultdict(lambda: [])
        self._record_type = np.dtype([("index", np.uint32), ("ts", np.dtype('datetime64[ns]')), ("value", self.dtype)])
        self._buffer_size = np.prod(self.chunks[0].shape)

    def __len__(self):
        return len(self._data)

    @property
    def name(self):
        return self._data.name

    def __setitem__(self, index, value):
        from ..session import Session
        session = Session.default_or_local()
        return session.write_mutable_tensor(self, index, value)

    def seal(self):
        from ..session import Session
        session = Session.default_or_local()
        return session.seal(self)

    @log_unhandled
    def _do_write(self, tensor_index, value):
        ''' Notes [buffer management of mutable tensor]:
        Write operations on a mutable tensor are buffered at client. Every chunk has a
        corresponding buffer in the form of

            {chunk_key: [(index, ts, value)]}

        Every time we write to a chunk, we will append the new operation records to
        the list

        At the end of write, if the buffer size exceeds `buffer_size`, the buffer will be send
        to the corresponding worker.

        The insights for above design are:

        1. `append` on (small) list is fast
        2. We try to flush the (affected) buffer to worker at the end of every write, the buffer
           size is guaranteed to less than 2 * chunk_size.
        '''
        from .expressions.indexing.core import process_index, calc_shape
        from .expressions.indexing.setitem import TensorIndex
        from .expressions.utils import setitem_as_records

        tensor_index = process_index(self, tensor_index)
        output_shape = calc_shape(self.shape, tensor_index)

        index_tensor_op = TensorIndex(dtype=self.dtype, sparse=False, indexes=tensor_index)
        index_tensor = index_tensor_op.new_tensor([self], tuple(output_shape)).single_tiles()
        output_chunks = index_tensor.chunks

        if np.isscalar(value):
            value = self.dtype.type(value)
        else:
            value = np.broadcast_to(value, output_shape).astype(self.dtype)

        nsplits_acc = [np.cumsum((0,) + tuple(c.shape[i] for c in output_chunks
                                              if all(idx == 0 for j, idx in enumerate(c.index) if j != i)))
                       for i in range(len(output_chunks[0].shape))]

        now = np.datetime64(datetime.now())
        affected_chunk_keys = []

        for output_chunk in output_chunks:
            logger.debug("write: chunk idx = {}, output chunk idx = {}, chunk_shape = {}, chunk_index = {}".format(
                output_chunk.op.input.index, output_chunk.index, output_chunk.shape, output_chunk.op.indexes))

            records = self._chunk_buffers[output_chunk.op.input.key]
            records += setitem_as_records(nsplits_acc, output_chunk, value, now)
            affected_chunk_keys.append(output_chunk.op.input.key)

        # Try to flush affected chunks
        return self._do_flush(self._buffer_size, affected_chunk_keys)

    @log_unhandled
    def _do_flush(self, buffer_size_limit=1, affected_chunk_keys=None):
        chunk_records_to_send = dict()
        affected_chunk_keys = affected_chunk_keys or self._chunk_buffers.keys()
        for chunk_key in affected_chunk_keys:
            records = self._chunk_buffers[chunk_key]
            if len(records) >= buffer_size_limit:
                chunk_records_to_send[chunk_key] = np.array(records, dtype=self._record_type)
                self._chunk_buffers[chunk_key] = []
        return chunk_records_to_send


class SparseTensor(Tensor):
    __slots__ = ()


TENSOR_TYPE = (Tensor, TensorData)
CHUNK_TYPE = (TensorChunk, TensorChunkData)
