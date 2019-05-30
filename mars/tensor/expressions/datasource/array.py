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

from .... import opcodes as OperandDef
from ....lib.sparse.core import issparse, get_array_module, cp, cps, sps
from ....utils import on_serialize_shape, on_deserialize_shape
from ....serialize import ValueType, NDArrayField, TupleField
from ...core import TENSOR_TYPE, Tensor
from ..utils import get_chunk_slices
from .core import TensorNoInput
from .scalar import scalar


class ArrayDataSource(TensorNoInput):
    """
    Represents data from numpy or cupy array
    """

    _op_type_ = OperandDef.TENSOR_DATA_SOURCE

    _data = NDArrayField('data')

    def __init__(self, data=None, dtype=None, gpu=None, **kw):
        if dtype is not None:
            dtype = np.dtype(dtype)
        elif data is not None:
            dtype = np.dtype(data.dtype)
        super(ArrayDataSource, self).__init__(_data=data, _dtype=dtype, _gpu=gpu, **kw)

    @property
    def data(self):
        return self._data

    def to_chunk_op(self, *args):
        _, idx, chunk_size = args
        chunk_op = self.copy().reset_key()
        chunk_op._data = self.data[get_chunk_slices(chunk_size, idx)]

        return chunk_op


class CSRMatrixDataSource(TensorNoInput):
    """
    Represents data from sparse array include scipy sparse or cupy sparse matrix.
    """

    _op_type_ = OperandDef.SPARSE_MATRIX_DATA_SOURCE

    _indices = NDArrayField('indices')
    _indptr = NDArrayField('indptr')
    _data = NDArrayField('data')
    _shape = TupleField('shape', ValueType.int64,
                        on_serialize=on_serialize_shape, on_deserialize=on_deserialize_shape)

    def __init__(self, indices=None, indptr=None, data=None, shape=None,
                 dtype=None, gpu=None, **kw):
        super(CSRMatrixDataSource, self).__init__(_indices=indices, _indptr=indptr,
                                                  _data=data, _shape=shape, _dtype=dtype,
                                                  _gpu=gpu, _sparse=True, **kw)

    def to_chunk_op(self, *args):
        _, idx, chunk_size = args

        xps = cps if self._gpu else sps
        if len(self._shape) == 1:
            shape = (1, self._shape[0])
        else:
            shape = self._shape
        data = xps.csr_matrix(
            (self._data, self._indices, self._indptr), shape)
        chunk_data = data[get_chunk_slices(chunk_size, idx)]

        chunk_op = self.copy().reset_key()
        chunk_op._data = chunk_data.data
        chunk_op._indices = chunk_data.indices
        chunk_op._indptr = chunk_data.indptr
        chunk_shape = chunk_data.shape[1:] \
            if len(self._shape) == 1 else chunk_data.shape
        chunk_op._shape = chunk_shape

        return chunk_op

    @property
    def indices(self):
        return self._indices

    @property
    def indptr(self):
        return self._indptr

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape


def _from_spmatrix(spmatrix, dtype=None, chunk_size=None, gpu=None):
    if gpu is None and cp is not None and get_array_module(spmatrix) is cp:
        gpu = True
    if dtype and spmatrix.dtype != dtype:
        spmatrix = spmatrix.astype(dtype)
    spmatrix = spmatrix.tocsr()
    op = CSRMatrixDataSource(indices=spmatrix.indices, indptr=spmatrix.indptr,
                             data=spmatrix.data, shape=spmatrix.shape,
                             dtype=spmatrix.dtype, gpu=gpu)
    return op(spmatrix.shape, chunk_size=chunk_size)


def tensor(data, dtype=None, chunk_size=None, gpu=None, sparse=False):
    if isinstance(data, TENSOR_TYPE):
        if dtype is not None and data.dtype != dtype:
            return data.astype(dtype)
        return data
    elif isinstance(data, (tuple, list)) and all(isinstance(d, TENSOR_TYPE) for d in data):
        from ..merge import stack

        data = stack(data)
        if dtype is not None:
            data = data.astype(dtype)
        return data
    elif np.isscalar(data):
        return scalar(data, dtype=dtype)
    elif issparse(data):
        return _from_spmatrix(data, dtype=dtype, chunk_size=chunk_size, gpu=gpu)
    else:
        m = get_array_module(data)
        data = m.asarray(data, dtype=dtype)
        if gpu is None and cp is not None and m is cp:
            gpu = True

    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            return scalar(data.item(), dtype=dtype)
        op = ArrayDataSource(data, dtype=dtype, gpu=gpu)
        t = op(data.shape, chunk_size=chunk_size)
        if sparse and not t.issparse():
            return t.tosparse()
        return t
    else:
        raise ValueError('Cannot create tensor by given data: {0}'.format(data))


def array(x, dtype=None, copy=True, ndmin=None, chunk_size=None):
    """
    Create a tensor.

    Parameters
    ----------
    object : array_like
        An array, any object exposing the array interface, an object whose
        __array__ method returns an array, or any (nested) sequence.
    dtype : data-type, optional
        The desired data-type for the array.  If not given, then the type will
        be determined as the minimum type required to hold the objects in the
        sequence.  This argument can only be used to 'upcast' the array.  For
        downcasting, use the .astype(t) method.
    copy : bool, optional
        If true (default), then the object is copied.  Otherwise, a copy will
        only be made if __array__ returns a copy, if obj is a nested sequence,
        or if a copy is needed to satisfy any of the other requirements
        (`dtype`, `order`, etc.).
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting
        array should have.  Ones will be pre-pended to the shape as
        needed to meet this requirement.
    chunk_size: int, tuple, optional
        Specifies chunk size for each dimension.

    Returns
    -------
    out : Tensor
        An tensor object satisfying the specified requirements.

    See Also
    --------
    empty, empty_like, zeros, zeros_like, ones, ones_like, full, full_like

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.array([1, 2, 3]).execute()
    array([1, 2, 3])

    Upcasting:

    >>> mt.array([1, 2, 3.0]).execute()
    array([ 1.,  2.,  3.])

    More than one dimension:

    >>> mt.array([[1, 2], [3, 4]]).execute()
    array([[1, 2],
           [3, 4]])

    Minimum dimensions 2:

    >>> mt.array([1, 2, 3], ndmin=2).execute()
    array([[1, 2, 3]])

    Type provided:

    >>> mt.array([1, 2, 3], dtype=complex).execute()
    array([ 1.+0.j,  2.+0.j,  3.+0.j])

    """
    raw_x = x
    x = tensor(x, chunk_size=chunk_size)
    if copy and x is raw_x:
        x = Tensor(x.data)
    while ndmin is not None and x.ndim < ndmin:
        x = x[np.newaxis, :]
    if dtype is not None and x.dtype != dtype:
        x = x.astype(dtype)
    return x


def asarray(x, dtype=None):
    """Convert the input to an array.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to a tensor.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and tensors.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.

    Returns
    -------
    out : Tensor
        Tensor interpretation of `a`.  No copy is performed if the input
        is already an ndarray with matching dtype and order.  If `a` is a
        subclass of ndarray, a base class ndarray is returned.

    Examples
    --------
    Convert a list into an array:

    >>> import mars.tensor as mt

    >>> a = [1, 2]
    >>> mt.asarray(a).execute()
    array([1, 2])

    Existing arrays are not copied:

    >>> a = mt.array([1, 2])
    >>> mt.asarray(a) is a
    True

    If `dtype` is set, array is copied only if dtype does not match:

    >>> a = mt.array([1, 2], dtype=mt.float32)
    >>> mt.asarray(a, dtype=mt.float32) is a
    True
    >>> mt.asarray(a, dtype=mt.float64) is a
    False
    """
    return array(x, dtype=dtype, copy=False)
