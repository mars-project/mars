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

from ... import opcodes as OperandDef
from ...serialize import ValueType, DictField, TupleField, StringField, Int64Field
from ...lib.sparse.core import sps
from ...lib.sparse import SparseNDArray
from ..core import TensorOrder
from .core import TensorNoInput


class TensorTileDBDataSource(TensorNoInput):
    _op_type_ = OperandDef.TENSOR_FROM_TILEDB

    _tiledb_config = DictField('tiledb_config')
    # URI of array to open
    _tiledb_uri = StringField('tiledb_uri')
    # tiledb dim start
    _tiledb_dim_starts = TupleField('tiledb_dim_starts', ValueType.int64)
    # encryption key to decrypt if provided
    _tiledb_key = StringField('tiledb_key')
    # open array at a given timestamp if provided
    _tiledb_timestamp = Int64Field('tiledb_timestamp')
    _axis_offsets = TupleField('axis_offsets', ValueType.int64)

    def __init__(self, tiledb_config=None, tiledb_uri=None, tiledb_dim_starts=None,
                 tiledb_key=None, tiledb_timstamp=None, dtype=None,
                 gpu=None, sparse=None, **kw):
        super().__init__(
            _tiledb_config=tiledb_config, _tiledb_uri=tiledb_uri,
            _tiledb_dim_starts=tiledb_dim_starts,
            _tiledb_key=tiledb_key, _tiledb_timestamp=tiledb_timstamp,
            _dtype=dtype, _gpu=gpu, _sparse=sparse, **kw)

    @property
    def tiledb_config(self):
        return self._tiledb_config

    @property
    def tiledb_uri(self):
        return self._tiledb_uri

    @property
    def tiledb_dim_starts(self):
        return self._tiledb_dim_starts

    @property
    def tiledb_key(self):
        return self._tiledb_key

    @property
    def tiledb_timestamp(self):
        return self._tiledb_timestamp

    @property
    def axis_offsets(self):
        return self._axis_offsets

    def to_chunk_op(self, *args):
        _, chunk_idx, nsplits = args
        chunk_op = super().to_chunk_op(*args)
        axis_offsets = []
        for axis, idx in enumerate(chunk_idx):
            axis_offsets.append(sum(nsplits[axis][:idx]))
        chunk_op._axis_offsets = tuple(axis_offsets)
        return chunk_op

    @classmethod
    def execute(cls, ctx, op):
        import tiledb
        chunk = op.outputs[0]
        from ..array_utils import array_module
        from ..utils import get_tiledb_ctx

        xp = array_module(op.gpu)

        axis_offsets = [offset + dim_start for offset, dim_start
                        in zip(op.axis_offsets, op.tiledb_dim_starts)]
        tiledb_ctx = get_tiledb_ctx(op.tiledb_config)
        uri = op.tiledb_uri
        key = op.tiledb_key
        timestamp = op.tiledb_timestamp

        slcs = []
        for axis in range(chunk.ndim):
            axis_offset = axis_offsets[axis]
            axis_length = chunk.shape[axis]
            slcs.append(slice(axis_offset, axis_offset + axis_length))

        if not op.sparse:
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


def fromtiledb(uri, ctx=None, key=None, timestamp=None, gpu=False):
    import tiledb

    raw_ctx = ctx
    if raw_ctx is None:
        ctx = tiledb.Ctx()

    # get metadata from tiledb
    try:
        tiledb_arr = tiledb.DenseArray(uri=uri, ctx=ctx, key=key, timestamp=timestamp)
        sparse = False
    except ValueError:
        # if the array is not dense, ValueError will be raised by tiledb
        tiledb_arr = tiledb.SparseArray(uri=uri, ctx=ctx, key=key, timestamp=timestamp)
        sparse = True

    if tiledb_arr.nattr > 1:
        raise NotImplementedError('Does not supported TileDB array schema '
                                  'with more than 1 attr')
    tiledb_dim_starts = tuple(tiledb_arr.domain.dim(j).domain[0].item()
                              for j in range(tiledb_arr.ndim))
    if any(isinstance(s, float) for s in tiledb_dim_starts):
        raise ValueError('Does not support TileDB array schema '
                         'whose dimensions has float domain')

    dtype = tiledb_arr.attr(0).dtype
    tiledb_config = None if raw_ctx is None else ctx.config().dict()
    tensor_order = TensorOrder.C_ORDER \
        if tiledb_arr.schema.cell_order == 'row-major' else TensorOrder.F_ORDER
    op = TensorTileDBDataSource(tiledb_config=tiledb_config, tiledb_uri=uri,
                                tiledb_key=key, tiledb_timstamp=timestamp,
                                tiledb_dim_starts=tiledb_dim_starts,
                                gpu=gpu, sparse=sparse, dtype=dtype)
    chunk_size = tuple(int(tiledb_arr.domain.dim(i).tile)
                       for i in range(tiledb_arr.domain.ndim))
    return op(tiledb_arr.shape, chunk_size=chunk_size, order=tensor_order)
