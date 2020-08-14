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

import numpy as np
try:
    import tiledb
except (ImportError, OSError):  # pragma: no cover
    tiledb = None


def get_tiledb_schema_from_tensor(tensor, tiledb_ctx, nsplits, **kw):
    from ..core import TensorOrder

    ctx = tiledb_ctx

    dims = []
    for d in range(tensor.ndim):
        extent = tensor.shape[d]
        domain = (0, extent - 1)
        tile = max(nsplits[d])
        dims.append(tiledb.Dim(name="", domain=domain, tile=tile, dtype=np.int64, ctx=ctx))
    dom = tiledb.Domain(*dims, ctx=ctx)
    att = tiledb.Attr(ctx=ctx, dtype=tensor.dtype)
    cell_order = 'C' if tensor.order == TensorOrder.C_ORDER else 'F'
    return tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,),
                              sparse=tensor.issparse(), cell_order=cell_order, **kw)


def check_tiledb_array_with_tensor(tensor, tiledb_array):
    if tensor.ndim != tiledb_array.ndim:
        # ndim
        raise ValueError('ndim of TileDB Array to store is different to tensor, '
                         f'expect {tensor.ndim}, got {tiledb_array.ndim}')
    if tensor.shape != tiledb_array.shape:
        # shape
        raise ValueError('shape of TileDB Array to store is different to tensor, '
                         f'expect {tensor.shape}, got {tiledb_array.shape}')
    if tensor.dtype != tiledb_array.attr(0).dtype:
        # dtype
        raise ValueError('dtype of TileDB Array to store is different to tensor, '
                         f'expect {tensor.dtype}, got {tiledb_array.domain.dtype}')
