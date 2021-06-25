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

import numpy as np
try:
    import tiledb
except (ImportError, OSError):  # pragma: no cover
    tildb = None

from ... import opcodes as OperandDef
from ...serialization.serializables import FieldTypes, DictField, \
    TupleField, StringField, Int64Field, KeyField
from ...lib.sparse import SparseNDArray
from ...lib.sparse.core import sps
from ..datasource import tensor as astensor
from ..operands import TensorOperandMixin, TensorOperand
from ..utils import get_tiledb_ctx
from .core import TensorDataStore
from .utils import check_tiledb_array_with_tensor, get_tiledb_schema_from_tensor


class TensorTileDBDataStore(TensorDataStore):
    _op_type_ = OperandDef.TENSOR_STORE_TILEDB

    _input = KeyField('input')
    _tiledb_config = DictField('tiledb_config')
    # URI of array to write
    _tiledb_uri = StringField('tiledb_uri')
    # encryption key to decrypt if provided
    _tiledb_key = StringField('tiledb_key')
    # open array at a given timestamp if provided
    _tiledb_timestamp = Int64Field('tiledb_timestamp')
    _axis_offsets = TupleField('axis_offsets', FieldTypes.int64)

    def __init__(self, tiledb_config=None, tiledb_uri=None, tiledb_key=None,
                 tiledb_timestamp=None, **kw):
        super().__init__(
            _tiledb_config=tiledb_config, _tiledb_uri=tiledb_uri, _tiledb_key=tiledb_key,
            _tiledb_timestamp=tiledb_timestamp, **kw)

    @property
    def tiledb_config(self):
        return self._tiledb_config

    @property
    def tiledb_uri(self):
        return self._tiledb_uri

    @property
    def tiledb_key(self):
        return self._tiledb_key

    @property
    def tiledb_timestamp(self):
        return self._tiledb_timestamp

    @property
    def axis_offsets(self):
        return self._axis_offsets

    @classmethod
    def _get_out_chunk(cls, op, in_chunk):
        chunk_op = op.copy().reset_key()
        nsplits = op.input.nsplits
        axis_offsets = []
        for axis, idx in enumerate(in_chunk.index):
            axis_offsets.append(sum(nsplits[axis][:idx]))
        chunk_op._axis_offsets = tuple(axis_offsets)
        out_chunk_shape = (0,) * in_chunk.ndim
        return chunk_op.new_chunk([in_chunk], shape=out_chunk_shape,
                                  index=in_chunk.index)

    @classmethod
    def _process_out_chunks(cls, op, out_chunks):
        if len(out_chunks) == 1:
            return out_chunks

        consolidate_op = TensorTileDBConsolidate(
            tiledb_config=op.tiledb_config, tiledb_uri=op.tiledb_uri,
            tiledb_key=op.tiledb_key, sparse=op.sparse, dtype=op.dtype
        )
        return consolidate_op.new_chunks(out_chunks, shape=out_chunks[0].shape,
                                         index=(0,) * out_chunks[0].ndim)

    @classmethod
    def tile(cls, op):
        import tiledb

        tensor = super().tile(op)[0]

        ctx = tiledb.Ctx(op.tiledb_config)
        tiledb_array_type = tiledb.SparseArray if tensor.issparse() else tiledb.DenseArray
        try:
            tiledb_array_type(uri=op.tiledb_uri, key=op.tiledb_key,
                              timestamp=op.tiledb_timestamp, ctx=ctx)
        except tiledb.TileDBError:
            # not exist, try to create TileDB Array by given uri
            tiledb_array_schema = get_tiledb_schema_from_tensor(op.input, ctx, op.input.nsplits)
            tiledb_array_type.create(op.tiledb_uri, tiledb_array_schema, key=op.tiledb_key)

        return [tensor]

    @classmethod
    def execute(cls, ctx, op):
        tiledb_ctx = get_tiledb_ctx(op.tiledb_config)
        uri = op.tiledb_uri
        key = op.tiledb_key
        timestamp = op.tiledb_timestamp
        axis_offsets = op.axis_offsets

        chunk = op.outputs[0]
        if not chunk.issparse():
            # dense
            to_store = np.ascontiguousarray(ctx[op.input.key])
            slcs = []
            for axis in range(chunk.ndim):
                axis_offset = int(axis_offsets[axis])
                axis_length = int(op.input.shape[axis])
                slcs.append(slice(axis_offset, axis_offset + axis_length))
            with tiledb.DenseArray(uri=uri, ctx=tiledb_ctx, mode='w',
                                   key=key, timestamp=timestamp) as arr:
                arr[tuple(slcs)] = to_store
            ctx[chunk.key] = np.empty((0,) * chunk.ndim, dtype=chunk.dtype)
        else:
            # sparse
            to_store = ctx[op.input.key].spmatrix.tocoo()
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


class TensorTileDBConsolidate(TensorOperandMixin, TensorOperand):
    _op_type_ = OperandDef.TENSOR_STORE_TILEDB_CONSOLIDATE

    _tiledb_config = DictField('tiledb_config')
    # URI of array to write
    _tiledb_uri = StringField('tiledb_uri')
    # encryption key to decrypt if provided
    _tiledb_key = StringField('tiledb_key')

    def __init__(self, tiledb_config=None, tiledb_uri=None, tiledb_key=None, **kw):
        super().__init__(
            _tiledb_config=tiledb_config, _tiledb_uri=tiledb_uri,
            _tiledb_key=tiledb_key, **kw)

    def calc_shape(self, *inputs_shape):
        return self.outputs[0].shape

    @property
    def tiledb_config(self):
        return self._tiledb_config

    @property
    def tiledb_uri(self):
        return self._tiledb_uri

    @property
    def tiledb_key(self):
        return self._tiledb_key

    @classmethod
    def tile(cls, op):
        raise TypeError(f'{cls.__name__} is a chunk op, cannot be tiled')

    @classmethod
    def execute(cls, ctx, op):
        tiledb_config = tiledb.Config(op.tiledb_config)
        uri = op.tiledb_uri
        key = op.tiledb_key

        tiledb.consolidate(config=tiledb_config, uri=uri, key=key)
        ctx[op.outputs[0].key] = ctx[op.inputs[0].key]


def totiledb(uri, x, ctx=None, key=None, timestamp=None):
    import tiledb

    x = astensor(x)
    raw_ctx = ctx
    if raw_ctx is None:
        ctx = tiledb.Ctx()

    tiledb_array_type = tiledb.SparseArray if x.issparse() else tiledb.DenseArray
    try:
        tiledb_array = tiledb_array_type(uri=uri, key=key, timestamp=timestamp, ctx=ctx)
        # if already created, we will check the shape and dtype
        check_tiledb_array_with_tensor(x, tiledb_array)
    except tiledb.TileDBError:
        # not exist, as we don't know the tile,
        # we will create the tiledb array in the tile of tensor
        pass

    tiledb_config = None if raw_ctx is None else raw_ctx.config().dict()
    op = TensorTileDBDataStore(tiledb_config=tiledb_config, tiledb_uri=uri,
                               tiledb_key=key, tiledb_timestamp=timestamp,
                               dtype=x.dtype, sparse=x.issparse())
    return op(x)
