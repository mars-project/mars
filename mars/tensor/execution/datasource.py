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

import logging

from .array import array_module, device
from ...lib import sparse
from ...lib.sparse.core import get_sparse_module, get_array_module, cps, sps, naked
from ...lib.sparse import SparseNDArray
from ..expressions import datasource
from .utils import get_tiledb_ctx

logger = logging.getLogger(__name__)


def _create_array(op):
    xp = array_module(op.gpu)

    def inner(func, *args, **kwargs):
        with device(op.device):
            return getattr(xp, func)(*args, **kwargs)

    return inner


def _tensor_ones(ctx, chunk):
    ctx[chunk.key] = _create_array(chunk.op)('ones', chunk.shape, dtype=chunk.op.dtype)


def _tensor_zeros(ctx, chunk):
    if chunk.issparse():
        ctx[chunk.key] = sparse.zeros(chunk.shape, dtype=chunk.op.dtype, gpu=chunk.op.gpu)
    else:
        ctx[chunk.key] = _create_array(chunk.op)('zeros', chunk.shape, dtype=chunk.op.dtype)


def _tensor_empty(ctx, chunk):
    ctx[chunk.key] = _create_array(chunk.op)('empty', chunk.shape, dtype=chunk.op.dtype)


def _tensor_full(ctx, chunk):
    ctx[chunk.key] = _create_array(chunk.op)('full', chunk.shape,
                                             chunk.op.fill_value, dtype=chunk.op.dtype)


def _tensor_ones_like(ctx, chunk):
    if chunk.issparse():
        in_data = naked(ctx[chunk.inputs[0].key])
        xps = get_sparse_module(in_data)
        xp = get_array_module(in_data)
        ctx[chunk.key] = SparseNDArray(xps.csr_matrix(
            (xp.ones_like(in_data.data, dtype=chunk.op.dtype),
             in_data.indices, in_data.indptr), shape=in_data.shape
        ))
    else:
        ctx[chunk.key] = _create_array(chunk.op)(
            'ones_like', ctx[chunk.inputs[0].key], dtype=chunk.op.dtype)


def _tensor_zeros_like(ctx, chunk):
    if chunk.issparse():
        in_data = naked(ctx[chunk.inputs[0].key])
        xps = get_sparse_module(in_data)
        xp = get_array_module(in_data)
        ctx[chunk.key] = SparseNDArray(xps.csr_matrix(
            (xp.zeros_like(in_data.data, dtype=chunk.op.dtype),
             in_data.indices, in_data.indptr), shape=in_data.shape
        ))
    else:
        ctx[chunk.key] = _create_array(chunk.op)(
            'zeros_like', ctx[chunk.inputs[0].key], dtype=chunk.op.dtype)


def _tensor_empty_like(ctx, chunk):
    if chunk.issparse():
        in_data = naked(ctx[chunk.inputs[0].key])
        xps = get_array_module(in_data)
        xp = get_array_module(in_data)
        ctx[chunk.key] = SparseNDArray(xps.csr_matrix(
            (xp.empty_like(in_data.data, dtype=chunk.op.dtype),
             in_data.indices, in_data.indptr), shape=in_data.shape
        ))
    else:
        ctx[chunk.key] = _create_array(chunk.op)(
            'empty_like', ctx[chunk.inputs[0].key], dtype=chunk.op.dtype)


def _tensor_arange(ctx, chunk):
    ctx[chunk.key] = _create_array(chunk.op)(
        'arange', chunk.op.start, chunk.op.stop, chunk.op.step, dtype=chunk.op.dtype)


def _tensor_diag(ctx, chunk):
    if chunk.issparse():
        ctx[chunk.key] = sparse.diag(ctx[chunk.inputs[0].key], k=chunk.op.k, gpu=chunk.op.gpu)
    else:
        ctx[chunk.key] = _create_array(chunk.op)(
            'diag', ctx[chunk.inputs[0].key], k=chunk.op.k)


def _tensor_eye(ctx, chunk):
    if chunk.issparse():
        ctx[chunk.key] = sparse.eye(chunk.shape[0], M=chunk.shape[1], k=chunk.op.k,
                                    dtype=chunk.op.dtype, gpu=chunk.op.gpu)
    else:
        ctx[chunk.key] = _create_array(chunk.op)(
            'eye', chunk.shape[0], M=chunk.shape[1], k=chunk.op.k, dtype=chunk.op.dtype)


def _tensor_linspace(ctx, chunk):
    ctx[chunk.key] = _create_array(chunk.op)(
        'linspace', chunk.op.start, chunk.op.stop, num=chunk.op.num,
        endpoint=chunk.op.endpoint, dtype=chunk.op.dtype)


def _tensor_tri(ctx, chunk):
    f = 'triu' if isinstance(chunk.op, datasource.TensorTriu) else 'tril'
    if chunk.op.sparse:
        ctx[chunk.key] = getattr(sparse, f)(ctx[chunk.inputs[0].key], k=chunk.op.k)
    else:
        ctx[chunk.key] = _create_array(chunk.op)(
            f, ctx[chunk.inputs[0].key], chunk.op.k)


def _tensor_array_data_source(ctx, chunk):
    ctx[chunk.key] = array_module(chunk.op.gpu).asarray(chunk.op.data)


def _tensor_csr_matrix_data_source(ctx, chunk):
    xps = cps if chunk.op.gpu else sps
    chunk_shape = (1, chunk.op.shape[0]) if chunk.ndim == 1 else chunk.op.shape
    ctx[chunk.key] = SparseNDArray(xps.csr_matrix(
        (chunk.op.data, chunk.op.indices, chunk.op.indptr), shape=chunk_shape
    ), shape=chunk.op.shape)


def _tensor_sparse_to_dense(ctx, chunk):
    ctx[chunk.key] = ctx[chunk.inputs[0].key].toarray()


def _tensor_dense_to_sparse(ctx, chunk):
    in_data = naked(ctx[chunk.inputs[0].key])
    xps = cps if chunk.op.gpu else sps
    ctx[chunk.key] = SparseNDArray(xps.csr_matrix(in_data), shape=chunk.shape)


def _tensor_tiledb(ctx, chunk):
    import tiledb

    xp = array_module(chunk.op.gpu)

    axis_offsets = [offset + dim_start for offset, dim_start
                    in zip(chunk.op.axis_offsets, chunk.op.tiledb_dim_starts)]
    tiledb_ctx = get_tiledb_ctx(chunk.op.tiledb_config)
    uri = chunk.op.tiledb_uri
    key = chunk.op.tiledb_key
    timestamp = chunk.op.tiledb_timestamp

    slcs = []
    for axis in range(chunk.ndim):
        axis_offset = axis_offsets[axis]
        axis_length = chunk.shape[axis]
        slcs.append(slice(axis_offset, axis_offset + axis_length))

    if not chunk.issparse():
        # read dense array from tiledb
        with tiledb.DenseArray(uri=uri, ctx=tiledb_ctx, key=key, timestamp=timestamp) as tiledb_arr:
            ctx[chunk.key] = tiledb_arr[tuple(slcs)]
    else:
        # read sparse array from tiledb
        with tiledb.SparseArray(uri=uri, ctx=tiledb_ctx, key=key, timestamp=timestamp) as tiledb_arr:
            if tiledb_arr.ndim > 2:
                raise NotImplementedError(
                    'Does not support to read array with more than 2 dimensions')

            data = tiledb_arr[tuple(slcs)]
            coords = data['coords']
            value = data[tiledb_arr.attr(0).name]
            if tiledb_arr.ndim == 2:
                # 2-d
                ij = tuple(coords[tiledb_arr.domain.dim(k).name] - axis_offsets[k]
                           for k in range(tiledb_arr.ndim))
                spmatrix = sps.coo_matrix((value, ij), shape=chunk.shape)
                ctx[chunk.key] = SparseNDArray(spmatrix)
            else:
                # 1-d
                ij = xp.zeros(coords.shape), \
                     coords[tiledb_arr.domain.dim(0).name] - axis_offsets[0]
                spmatrix = sps.coo_matrix((value, ij), shape=(1,) + chunk.shape)
                ctx[chunk.key] = SparseNDArray(spmatrix, shape=chunk.shape)


def _tensor_fetch_chunk(ctx, chunk):
    # nothing need to do
    return


def _scalar(ctx, chunk):
    if chunk.ndim != 0:
        raise ValueError('Missing op for chunk')

    ctx[chunk.key] = _create_array(chunk.op)('asarray', chunk.op.data)


def register_data_source_handler():
    from .core import register

    register(datasource.TensorOnes, _tensor_ones)
    register(datasource.TensorZeros, _tensor_zeros)
    register(datasource.TensorEmpty, _tensor_empty)
    register(datasource.TensorFull, _tensor_full)
    register(datasource.TensorOnesLike, _tensor_ones_like)
    register(datasource.TensorZerosLike, _tensor_zeros_like)
    register(datasource.TensorEmptyLike, _tensor_empty_like)
    register(datasource.TensorArange, _tensor_arange)
    register(datasource.TensorDiag, _tensor_diag)
    register(datasource.TensorEye, _tensor_eye)
    register(datasource.TensorLinspace, _tensor_linspace)
    register(datasource.TensorTriu, _tensor_tri)
    register(datasource.TensorTril, _tensor_tri)
    register(datasource.ArrayDataSource, _tensor_array_data_source)
    register(datasource.CSRMatrixDataSource, _tensor_csr_matrix_data_source)
    register(datasource.SparseToDense, _tensor_sparse_to_dense)
    register(datasource.DenseToSparse, _tensor_dense_to_sparse)
    register(datasource.TensorTileDBDataSource, _tensor_tiledb)
    register(datasource.TensorFetchChunk, _tensor_fetch_chunk)
    register(datasource.Scalar, _scalar)
