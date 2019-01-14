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


from weakref import WeakKeyDictionary, ref
from collections import Iterable

from ..core import Entity, ChunkData, Chunk, TilesableData, build_mode
from ..tiles import handler
from ..serialize import ProviderType, DataTypeField
from .expressions.utils import get_chunk_slices


class TensorData(TilesableData):
    __slots__ = ()

    # required fields
    _dtype = DataTypeField('dtype')

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from ..serialize.protos.tensor_pb2 import TensorDef
            return TensorDef
        return super(TensorData, cls).cls(provider)

    def __repr__(self):
        return 'Tensor <op={0}, key={1}>'.format(self.op.__class__.__name__, self.key)

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


class ExecutableTuple(tuple):
    def execute(self, session=None, **kw):
        from ..session import Session

        if session is None:
            session = Session.default_or_local()
        return session.run(*self, **kw)


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
        return np.asarray(self.execute(), dtype=dtype)


class SparseTensor(Tensor):
    __slots__ = ()


TENSOR_TYPE = (Tensor, TensorData)
CHUNK_TYPE = (Chunk, ChunkData)


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

