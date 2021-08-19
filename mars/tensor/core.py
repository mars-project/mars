#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import logging
from collections.abc import Iterable
from enum import Enum
from operator import attrgetter
from typing import Any, Dict

import numpy as np

from ..core import HasShapeTileable, ChunkData, Chunk, HasShapeTileableData, \
    OutputType, register_output_types, _ExecuteAndFetchMixin, is_build_mode
from ..core.entity.utils import refresh_tileable_shape
from ..serialization.serializables import Serializable, FieldTypes, \
    DataTypeField, ListField, TupleField, StringField, AnyField, ReferenceField
from ..utils import on_serialize_shape, on_deserialize_shape
from .utils import get_chunk_slices, fetch_corner_data

logger = logging.getLogger(__name__)


class TensorOrder(Enum):
    # C order
    C_ORDER = 'C'
    # Fortran order
    F_ORDER = 'F'


class TensorChunkData(ChunkData):
    __slots__ = ()
    type_name = 'Tensor'

    # required fields
    _shape = TupleField('shape', FieldTypes.int64,
                        on_serialize=on_serialize_shape,
                        on_deserialize=on_deserialize_shape)
    _order = ReferenceField('order', TensorOrder)
    # optional fields
    _dtype = DataTypeField('dtype')

    def __init__(self, op=None, index=None, shape=None, dtype=None, order=None, **kw):
        if isinstance(order, str):
            order = getattr(TensorOrder, order)
        super().__init__(_op=op, _index=index, _shape=shape, _dtype=dtype, _order=order, **kw)
        if self.order is None and self.op is not None:
            if len(self.inputs) == 0:
                self._order = TensorOrder.C_ORDER
            elif all(hasattr(inp, 'order') and inp.order == TensorOrder.F_ORDER
                     for inp in self.inputs):
                self._order = TensorOrder.F_ORDER
            else:
                self._order = TensorOrder.C_ORDER

    @property
    def params(self) -> Dict[str, Any]:
        # params return the properties which useful to rebuild a new chunk
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'order': self.order,
            'index': self.index,
        }

    @params.setter
    def params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        params.pop('index', None)  # index not needed to update
        new_shape = params.pop('shape', None)
        if new_shape is not None:
            self._shape = new_shape
        dtype = params.pop('dtype', None)
        if dtype is not None:
            self._dtype = dtype
        order = params.pop('order', None)
        if order is not None:
            self._order = order
        if params:  # pragma: no cover
            raise TypeError(f'Unknown params: {list(params)}')

    @classmethod
    def get_params_from_data(cls, data: np.ndarray) -> Dict[str, Any]:
        from .array_utils import is_cupy

        if not is_cupy(data):
            data = np.asarray(data)
        order = TensorOrder.C_ORDER \
            if data.flags['C_CONTIGUOUS'] else TensorOrder.F_ORDER
        return {
            'shape': data.shape,
            'dtype': data.dtype,
            'order': order}

    def __len__(self):
        try:
            return self.shape[0]
        except IndexError:
            if is_build_mode():
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
    def order(self):
        return getattr(self, '_order', None)

    @property
    def nbytes(self):
        return np.prod(self.shape) * self.dtype.itemsize


class TensorChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (TensorChunkData,)
    type_name = 'Tensor'

    def __len__(self):
        return len(self._data)


class TensorData(HasShapeTileableData, _ExecuteAndFetchMixin):
    __slots__ = ()
    type_name = 'Tensor'

    # required fields
    _order = StringField('order', on_serialize=attrgetter('value'), on_deserialize=TensorOrder)
    # optional fields
    _dtype = DataTypeField('dtype')
    _chunks = ListField('chunks', FieldTypes.reference(TensorChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [TensorChunk(it) for it in x] if x is not None else x)

    def __init__(self, op=None, shape=None, dtype=None, order=None, nsplits=None, chunks=None, **kw):
        if isinstance(order, str):
            order = getattr(TensorOrder, order)
        super().__init__(_op=op, _shape=shape, _dtype=dtype, _order=order, _nsplits=nsplits,
                         _chunks=chunks, **kw)
        if self.order is None and self.op is not None:
            if len(self.inputs) == 0:
                self._order = TensorOrder.C_ORDER
            elif all(hasattr(inp, 'order') and inp.order == TensorOrder.F_ORDER
                     for inp in self.inputs):
                self._order = TensorOrder.F_ORDER
            else:
                self._order = TensorOrder.C_ORDER

    def _to_str(self, representation=False):
        if is_build_mode() or len(self._executed_sessions) == 0:
            # in build mode, or not executed, just return representation
            if representation:
                return f'Tensor <op={type(self._op).__name__}, shape={self._shape}, key={self._key}>'
            else:
                return f'Tensor(op={type(self._op).__name__}, shape={self._shape})'
        else:
            print_options = np.get_printoptions()
            threshold = print_options['threshold']

            corner_data = fetch_corner_data(self, session=self._executed_sessions[-1])
            # if less than default threshold, just set it as default,
            # if not, set to corner_data.size - 1 make sure ... exists in repr
            threshold = threshold if self.size <= threshold else corner_data.size - 1
            with np.printoptions(threshold=threshold):
                corner_str = repr(corner_data) if representation else str(corner_data)
            return corner_str

    def __str__(self):
        return self._to_str(representation=False)

    def __repr__(self):
        return self._to_str(representation=True)

    @property
    def params(self):
        # params return the properties which useful to rebuild a new tileable object
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'order': self.order
        }

    @params.setter
    def params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        shape = params.pop('shape', None)
        if shape is not None:
            self._shape = shape
        dtype = params.pop('dtype', None)
        if dtype is not None:
            self._dtype = dtype
        order = params.pop('order', None)
        if order is not None:
            self._order = order
        if params:  # pragma: no cover
            raise TypeError(f'Unknown params: {list(params)}')

    def refresh_params(self):
        refresh_tileable_shape(self)
        if self._dtype is None:
            self._dtype = self.chunks[0].dtype

    @property
    def flags(self):
        c_order = True if self.ndim <= 1 else self.order == TensorOrder.C_ORDER
        f_order = True if self.ndim <= 1 else self.order == TensorOrder.F_ORDER
        return {
            'C_CONTIGUOUS': c_order,
            'F_CONTIGUOUS': f_order
        }

    @property
    def real(self):
        from .arithmetic import real
        return real(self)

    @property
    def imag(self):
        from .arithmetic import imag
        return imag(self)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None) or self.op.dtype

    @property
    def order(self):
        return getattr(self, '_order', None)

    @property
    def nbytes(self):
        return np.prod(self.shape) * self.dtype.itemsize

    def get_chunk_slices(self, idx):
        return get_chunk_slices(self.nsplits, idx)

    def is_scalar(self):
        return self.ndim == 0

    isscalar = is_scalar

    def tosparse(self, missing=None):
        if self.issparse():
            return self

        from .datasource import fromdense
        return fromdense(self, missing=missing)

    def todense(self, fill_value=None):
        if not self.issparse():
            return self

        from .datasource import fromsparse
        return fromsparse(self, fill_value=fill_value)

    def transpose(self, *axes):
        from .base import transpose

        if len(axes) == 1 and isinstance(axes[0], Iterable):
            axes = axes[0]

        return transpose(self, axes)

    @property
    def T(self):
        return self.transpose()

    def reshape(self, shape, *shapes, **kw):
        from .reshape import reshape

        order = kw.pop('order', 'C')
        if kw:
            raise TypeError(
                f"'{next(iter(kw))}' is an invalid keyword argument for this function")

        if isinstance(shape, Iterable):
            shape = tuple(shape)
        else:
            shape = (shape,)
        shape += shapes

        return reshape(self, shape, order=order)

    def totiledb(self, uri, ctx=None, key=None, timestamp=None):
        from .datastore import totiledb

        return totiledb(uri, self, ctx=ctx, key=key, timestamp=timestamp)

    @staticmethod
    def from_dataframe(in_df):
        from .datasource import from_dataframe
        return from_dataframe(in_df)

    def to_dataframe(self, *args, **kwargs):
        from ..dataframe.datasource.from_tensor import dataframe_from_tensor
        return dataframe_from_tensor(self, *args, **kwargs)

    @property
    def flat(self):
        return flatiter(self)

    def to_numpy(self, session=None, **kw):
        return self._execute_and_fetch(session=session, **kw)


class Tensor(HasShapeTileable):
    __slots__ = ()
    _allow_data_type_ = (TensorData,)
    type_name = 'Tensor'

    def __len__(self):
        return len(self._data)

    @property
    def shape(self):
        return self._data.shape

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
        from .arithmetic.setreal import set_real

        self._data = set_real(self._data, new_real).data

    @property
    def imag(self):
        return self.data.imag

    @imag.setter
    def imag(self, new_imag):
        from .arithmetic.setimag import set_imag

        self._data = set_imag(self._data, new_imag).data

    def __array__(self, dtype=None):
        return np.asarray(self.to_numpy(), dtype=dtype)

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

    def view(self):
        return self._view()

    @property
    def ndim(self):
        """
        Number of array dimensions.

        Examples
        --------
        >>> import mars.tensor as mt
        >>> x = mt.array([1, 2, 3])
        >>> x.ndim
        1
        >>> y = mt.zeros((2, 3, 4))
        >>> y.ndim
        3
        """
        return super().ndim

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

    def totiledb(self, uri, ctx=None, key=None, timestamp=None):
        return self._data.totiledb(uri, ctx=ctx, key=key, timestamp=timestamp)

    def copy(self, order='C'):
        return super().copy().astype(self.dtype, order=order, copy=False)

    def sort(self, axis=-1, kind=None, parallel_kind=None, psrs_kinds=None, order=None):
        """
        Sort a tensor, in-place.

        Parameters
        ----------
        axis : int, optional
            Axis along which to sort. Default is -1, which means sort along the
            last axis.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            Sorting algorithm. Default is 'quicksort'.
        parallel_kind: {'PSRS'}, optional
            Parallel sorting algorithm, for the details, refer to:
            http://csweb.cs.wfu.edu/bigiron/LittleFE-PSRS/build/html/PSRSalgorithm.html
        psrs_kinds: list with 3 elements, optional
            Sorting algorithms during PSRS algorithm.
        order : str or list of str, optional
            When `a` is a tensor with fields defined, this argument specifies
            which fields to compare first, second, etc.  A single field can
            be specified as a string, and not all fields need be specified,
            but unspecified fields will still be used, in the order in which
            they come up in the dtype, to break ties.

        See Also
        --------
        numpy.sort : Return a sorted copy of a tensor.
        argsort : Indirect sort.
        lexsort : Indirect stable sort on multiple keys.
        searchsorted : Find elements in sorted tensor.
        partition: Partial sort.

        Notes
        -----
        See ``sort`` for notes on the different sorting algorithms.

        Examples
        --------
        >>> import mars.tensor as mt
        >>> a = mt.array([[1,4], [3,1]])
        >>> a.sort(axis=1)
        >>> a.execute()
        array([[1, 4],
               [1, 3]])
        >>> a.sort(axis=0)
        >>> a.execute()
        array([[1, 3],
               [1, 4]])

        Use the `order` keyword to specify a field to use when sorting a
        structured tensor:

        >>> a = mt.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])
        >>> a.sort(order='y')
        >>> a.execute()
        array([('c', 1), ('a', 2)],
              dtype=[('x', '|S1'), ('y', '<i4')])
        """
        from .base import sort

        self._data = sort(self, axis=axis, kind=kind, parallel_kind=parallel_kind,
                          psrs_kinds=psrs_kinds, order=order).data

    def partition(self, kth, axis=-1, kind='introselect', order=None, **kw):
        """
        Rearranges the elements in the tensor in such a way that the value of the
        element in kth position is in the position it would be in a sorted tensor.
        All elements smaller than the kth element are moved before this element and
        all equal or greater are moved behind it. The ordering of the elements in
        the two partitions is undefined.

        Parameters
        ----------
        kth : int or sequence of ints
            Element index to partition by. The kth element value will be in its
            final sorted position and all smaller elements will be moved before it
            and all equal or greater elements behind it.
            The order of all elements in the partitions is undefined.
            If provided with a sequence of kth it will partition all elements
            indexed by kth of them into their sorted position at once.
        axis : int, optional
            Axis along which to sort. Default is -1, which means sort along the
            last axis.
        kind : {'introselect'}, optional
            Selection algorithm. Default is 'introselect'.
        order : str or list of str, optional
            When `a` is a tensor with fields defined, this argument specifies
            which fields to compare first, second, etc. A single field can
            be specified as a string, and not all fields need to be specified,
            but unspecified fields will still be used, in the order in which
            they come up in the dtype, to break ties.

        See Also
        --------
        mt.partition : Return a partitioned copy of an tensor.
        argpartition : Indirect partition.
        sort : Full sort.

        Notes
        -----
        See ``mt.partition`` for notes on the different algorithms.

        Examples
        --------
        >>> import mars.tensor as mt
        >>> a = mt.array([3, 4, 2, 1])
        >>> a.partition(3)
        >>> a.execute()
        array([2, 1, 3, 4])

        >>> a.partition((1, 3))
        >>> a.execute()
        array([1, 2, 3, 4])
        """
        from .base import partition

        self._data = partition(self, kth, axis=axis,
                               kind=kind, order=order, **kw).data

    @property
    def flat(self):
        """
        Flat iterator object to iterate over arrays.

        A `flatiter` iterator is returned by ``x.flat`` for any tensor `x`.
        It allows iterating over the tensor as if it were a 1-D array,
        either in a for-loop or by calling its `next` method.

        Iteration is done in row-major, C-style order (the last
        index varying the fastest). The iterator can also be indexed using
        basic slicing or advanced indexing.

        See Also
        --------
        Tensor.flat : Return a flat iterator over a tensor.
        Tensor.flatten : Returns a flattened copy of a tensor.

        Examples
        --------
        >>> import mars.tensor as mt

        >>> x = mt.arange(6).reshape(2, 3)
        >>> fl = x.flat

        >>> fl[2:4].execute()
        array([2, 3])
        """
        return self._data.flat

    def from_dataframe(self, in_df):
        return self._data.from_dataframe(in_df)

    def to_dataframe(self, *args, **kwargs):
        return self._data.to_dataframe(*args, **kwargs)

    def to_numpy(self, session=None, **kw):
        return self._data.to_numpy(session, **kw)


SparseTensor = Tensor


class flatiter(object):
    def __init__(self, tensor):
        # flatten creates a copy
        self._flatten_tensor = tensor.flatten()
        # ravel creates a view
        self._ravel_tensor = tensor.ravel()

    def __getitem__(self, item):
        # a.flat[item] create a copy
        return self._flatten_tensor[item]

    def __setitem__(self, key, value):
        # a.flat[item] = value will apply changes to original tensor
        self._ravel_tensor[key] = value


class Indexes(Serializable):
    indexes = AnyField('indexes')


TENSOR_TYPE = (Tensor, TensorData)
TENSOR_CHUNK_TYPE = (TensorChunk, TensorChunkData)

register_output_types(OutputType.tensor, TENSOR_TYPE, TENSOR_CHUNK_TYPE)
register_output_types(OutputType.scalar, TENSOR_TYPE, TENSOR_CHUNK_TYPE)
