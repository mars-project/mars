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
try:
    import tiledb
except (ImportError, OSError):  # pragma: no cover
    tiledb = None

from ...lib.sparse import SparseNDArray
from ...lib.sparse.core import sps
from ..expressions import datastore
from .utils import get_tiledb_ctx


def _store_tiledb(ctx, chunk):
    tiledb_ctx = get_tiledb_ctx(chunk.op.tiledb_config)
    uri = chunk.op.tiledb_uri
    key = chunk.op.tiledb_key
    timestamp = chunk.op.tiledb_timestamp
    axis_offsets = chunk.op.axis_offsets

    if not chunk.issparse():
        # dense
        to_store = np.ascontiguousarray(ctx[chunk.op.input.key])
        slcs = []
        for axis in range(chunk.ndim):
            axis_offset = int(axis_offsets[axis])
            axis_length = int(chunk.op.input.shape[axis])
            slcs.append(slice(axis_offset, axis_offset + axis_length))
        with tiledb.DenseArray(uri=uri, ctx=tiledb_ctx, mode='w',
                               key=key, timestamp=timestamp) as arr:
            arr[tuple(slcs)] = to_store
        ctx[chunk.key] = np.empty((0,) * chunk.ndim, dtype=chunk.dtype)
    else:
        # sparse
        to_store = ctx[chunk.op.input.key].spmatrix.tocoo()
        if to_store.nnz > 0:
            with tiledb.SparseArray(uri=uri, ctx=tiledb_ctx, mode='w',
                                    key=key, timestamp=timestamp) as arr:
                if chunk.ndim == 1:
                    vec = to_store.col if to_store.shape[0] == 1 else to_store.row
                    vec += axis_offsets[0]
                    arr[vec] = to_store.data
                else:
                    i, j = to_store.row + axis_offsets[0], to_store.col + axis_offsets[1]
                    arr[i, j] = to_store.data
        ctx[chunk.key] = SparseNDArray(sps.csr_matrix((0, 0), dtype=chunk.dtype),
                                       shape=chunk.shape)


def register_data_store_handler():
    from .core import register

    register(datastore.TensorTileDBDataStore, _store_tiledb)
