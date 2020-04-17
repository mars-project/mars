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

from ... import opcodes as OperandDef
from ...serialize import KeyField, AnyField, BoolField, Int32Field
from ...core import Entity, Base
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape, ceildiv, recursive_tile
from ..operands import TensorOperand, TensorOperandMixin
from ..datasource import tensor as astensor
from ..array_utils import as_same_device, device
from ..core import Tensor, TENSOR_TYPE
from ..utils import decide_unify_split


class TensorFillDiagonal(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.FILL_DIAGONAL

    _input = KeyField('input')
    _val = AnyField('val')
    _wrap = BoolField('wrap')
    # used for chunk
    _k = Int32Field('k')

    def __init__(self, val=None, wrap=None, k=None, dtype=None, **kw):
        super().__init__(_val=val, _wrap=wrap, _k=k, _dtype=dtype, **kw)

    @property
    def input(self):
        return self._input

    @property
    def val(self):
        return self._val

    @property
    def wrap(self):
        return self._wrap

    @property
    def k(self):
        return self._k

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if len(self._inputs) == 2:
            self._val = self._inputs[1]

    def __call__(self, a, val=None):
        inputs = [a]
        if val is not None:
            inputs.append(val)
        return self.new_tensor(inputs, shape=a.shape, order=a.order)

    @staticmethod
    def _process_val(val, a, wrap):
        """
        given the `val`, `a`, `wrap` which are the arguments in `fill_diagonal`,
        do some preprocess on `val` includes:

        1. calculate the length to fill on diagonal, 2-d and n-d(n > 2)
           as well as that `wrap` is True and `a` is a tall matrix need to be considered.
        2. if val is a Tensor, rechunk it into one chunk.
        """

        from ..datasource import diag
        from ..base import tile

        is_val_tensor = isinstance(val, TENSOR_TYPE)

        if a.ndim == 2:
            if wrap and TensorFillDiagonal._is_tall(a):
                size = sum(diag(sub).shape[0] for sub
                           in TensorFillDiagonal._split_tall_matrix(a))
            else:
                size = diag(a).shape[0]
        else:
            # every dimension has same shape
            size = a.shape[0]

        repeat_method = tile if is_val_tensor else np.tile
        val_size = val.size
        if val_size < size:
            n = ceildiv(size, val_size)
            val = repeat_method(val, n)[:size]
        elif val_size > size:
            val = val[:size]

        if is_val_tensor and val.ndim > 0:
            val = recursive_tile(val)
            val = val.rechunk({0: val.size})

        return recursive_tile(val) if is_val_tensor else val

    @staticmethod
    def _gen_val(val, diag_idx, cum_sizes):
        """
        Given a tensor-level `val`, calculate the chunk-level `val`.
        Consider both the cases that `val` could be a tensor or ndarray.

        :param val: tensor-level `val`
        :diag_idx: chunk index on the diagonal direction
        :cum_sizes: accumulative chunk sizes on the diagonal direction
        """
        from .slice import TensorSlice

        if val.ndim == 0:
            if isinstance(val, TENSOR_TYPE):
                return val.chunks[0]
            else:
                return val

        if isinstance(val, TENSOR_TYPE):
            start, stop = cum_sizes[diag_idx], cum_sizes[diag_idx + 1]
            slc = slice(start, stop)
            slc_op = TensorSlice(slices=[slc], dtype=val.dtype)
            return slc_op.new_chunk([val.chunks[0]], shape=(stop - start,),
                                    order=val.order, index=(diag_idx,))
        else:
            return val[cum_sizes[diag_idx]: cum_sizes[diag_idx + 1]]

    @classmethod
    def _tile_2d(cls, op, val):
        from ..datasource import diag

        d = diag(op.input)._inplace_tile()
        index_to_diag_chunk = {c.inputs[0].index: c for c in d.chunks}
        cum_sizes = [0] + np.cumsum(d.nsplits[0]).tolist()

        out_chunks = []
        for chunk in op.input.chunks:
            if chunk.index not in index_to_diag_chunk:
                out_chunks.append(chunk)
            else:
                diag_chunk = index_to_diag_chunk[chunk.index]
                diag_idx = diag_chunk.index[0]
                input_chunks = [chunk]
                chunk_val = cls._gen_val(val, diag_idx, cum_sizes)
                if len(op.inputs) == 2:
                    input_chunks.append(chunk_val)
                chunk_op = op.copy().reset_key()
                chunk_op._wrap = False
                chunk_op._k = diag_chunk.op.k
                chunk_op._val = chunk_val
                out_chunk = chunk_op.new_chunk(input_chunks, shape=chunk.shape,
                                               order=chunk.order,
                                               index=chunk.index)
                out_chunks.append(out_chunk)

        out = op.outputs[0]
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=out.shape, order=out.order,
                                  chunks=out_chunks, nsplits=op.input.nsplits)

    @classmethod
    def _tile_nd(cls, op, val):
        # if more than 3d, we will rechunk the tensor into square chunk
        # on the diagonal direction
        in_tensor = op.input
        nsplits = np.array(in_tensor.nsplits)
        if not np.issubdtype(nsplits.dtype, np.integer) or \
                not np.all(np.diff(nsplits, axis=1) == 0):
            # need rechunk
            nsplit = decide_unify_split(*in_tensor.nsplits)
            in_tensor = in_tensor.rechunk(
                tuple(nsplit for _ in range(in_tensor.ndim)))._inplace_tile()
        cum_sizes = [0] + np.cumsum(in_tensor.nsplits[0]).tolist()

        out_chunks = []
        for chunk in in_tensor.chunks:
            if len(set(chunk.index)) == 1:
                # chunk on the diagonal direction
                chunk_op = op.copy().reset_key()
                chunk_op._k = 0
                chunk_inputs = [chunk]
                chunk_val = cls._gen_val(val, chunk.index[0], cum_sizes)
                if len(op.inputs) == 2:
                    chunk_inputs.append(chunk_val)
                chunk_op._val = chunk_val
                out_chunk = chunk_op.new_chunk(chunk_inputs, shape=chunk.shape,
                                               order=chunk.order,
                                               index=chunk.index)
                out_chunks.append(out_chunk)
            else:
                out_chunks.append(chunk)

        out = op.outputs[0]
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=out.shape, order=out.order,
                                  chunks=out_chunks, nsplits=in_tensor.nsplits)

    @classmethod
    def _tile_one_chunk(cls, op, val):
        out = op.outputs[0]
        chunk_op = op.copy().reset_key()
        chunk_inputs = [op.input.chunks[0]]
        if isinstance(val, TENSOR_TYPE):
            chunk_inputs.append(val.chunks[0])
        chunk = chunk_op.new_chunk(chunk_inputs, shape=out.shape,
                                   order=out.order, index=(0,) * out.ndim)
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=out.shape, order=out.order,
                                  chunks=[chunk], nsplits=((s,) for s in out.shape))

    @staticmethod
    def _is_tall(x):
        return x.shape[0] > x.shape[1] + 1

    @staticmethod
    def _split_tall_matrix(a):
        blocksize = a.shape[1] + 1
        n_block = ceildiv(a.shape[0], blocksize)
        return [a[i * blocksize: (i + 1) * blocksize]
                for i in range(n_block)]

    @classmethod
    def tile(cls, op):
        # input tensor must have no unknown chunk shape
        check_chunks_unknown_shape(op.inputs, TilesError)

        in_tensor = op.input
        is_in_tensor_tall = cls._is_tall(in_tensor)

        if op.val.ndim > 0:
            val = cls._process_val(op.val, in_tensor, op.wrap)
        else:
            val = op.val

        if len(in_tensor.chunks) == 1:
            return cls._tile_one_chunk(op, val)

        if op.input.ndim == 2:
            if op.wrap and is_in_tensor_tall:
                from ..merge import concatenate

                sub_tensors = cls._split_tall_matrix(in_tensor)
                for i, sub_tensor in enumerate(sub_tensors):
                    if val.ndim > 0:
                        sub_val = val[i * sub_tensor.shape[1]:
                                      (i + 1) * sub_tensor.shape[1]]
                    else:
                        sub_val = val
                    fill_diagonal(sub_tensor, sub_val, wrap=False)
                out_tensor = concatenate(sub_tensors)
                return [recursive_tile(out_tensor)]
            else:
                return cls._tile_2d(op, val)
        else:
            return cls._tile_nd(op, val)

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)
        a = inputs[0]
        if len(inputs) == 2:
            val = inputs[1]
        else:
            val = op.val

        with device(device_id):
            if not op.k:
                a = a.copy()
                xp.fill_diagonal(a, val, wrap=op.wrap)
            else:
                assert a.ndim == 2
                k = op.k or 0
                n_rows, n_cols = a.shape
                if k > 0:
                    n_cols -= k
                elif k < 0:
                    n_rows += k
                n = min(n_rows, n_cols)

                # generate indices
                rows, cols = np.diag_indices(n)
                if k > 0:
                    cols = cols.copy()
                    cols += k
                elif k < 0:
                    rows = rows.copy()
                    rows -= k

                a = a.copy()
                a[rows, cols] = val

            ctx[op.outputs[0].key] = a


def fill_diagonal(a, val, wrap=False):
    """Fill the main diagonal of the given tensor of any dimensionality.

    For a tensor `a` with ``a.ndim >= 2``, the diagonal is the list of
    locations with indices ``a[i, ..., i]`` all identical. This function
    modifies the input tensor in-place, it does not return a value.

    Parameters
    ----------
    a : Tensor, at least 2-D.
      Tensor whose diagonal is to be filled, it gets modified in-place.

    val : scalar
      Value to be written on the diagonal, its type must be compatible with
      that of the tensor a.

    wrap : bool
      For tall matrices in NumPy version up to 1.6.2, the
      diagonal "wrapped" after N columns. You can have this behavior
      with this option. This affects only tall matrices.

    See also
    --------
    diag_indices, diag_indices_from

    Notes
    -----

    This functionality can be obtained via `diag_indices`, but internally
    this version uses a much faster implementation that never constructs the
    indices and uses simple slicing.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> a = mt.zeros((3, 3), int)
    >>> mt.fill_diagonal(a, 5)
    >>> a.execute()
    array([[5, 0, 0],
           [0, 5, 0],
           [0, 0, 5]])

    The same function can operate on a 4-D tensor:

    >>> a = mt.zeros((3, 3, 3, 3), int)
    >>> mt.fill_diagonal(a, 4)

    We only show a few blocks for clarity:

    >>> a[0, 0].execute()
    array([[4, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])
    >>> a[1, 1].execute()
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 0]])
    >>> a[2, 2].execute()
    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 4]])

    The wrap option affects only tall matrices:

    >>> # tall matrices no wrap
    >>> a = mt.zeros((5, 3), int)
    >>> mt.fill_diagonal(a, 4)
    >>> a.execute()
    array([[4, 0, 0],
           [0, 4, 0],
           [0, 0, 4],
           [0, 0, 0],
           [0, 0, 0]])

    >>> # tall matrices wrap
    >>> a = mt.zeros((5, 3), int)
    >>> mt.fill_diagonal(a, 4, wrap=True)
    >>> a.execute()
    array([[4, 0, 0],
           [0, 4, 0],
           [0, 0, 4],
           [0, 0, 0],
           [4, 0, 0]])

    >>> # wide matrices
    >>> a = mt.zeros((3, 5), int)
    >>> mt.fill_diagonal(a, 4, wrap=True)
    >>> a.execute()
    array([[4, 0, 0, 0, 0],
           [0, 4, 0, 0, 0],
           [0, 0, 4, 0, 0]])

    The anti-diagonal can be filled by reversing the order of elements
    using either `numpy.flipud` or `numpy.fliplr`.

    >>> a = mt.zeros((3, 3), int)
    >>> mt.fill_diagonal(mt.fliplr(a), [1,2,3])  # Horizontal flip
    >>> a.execute()
    array([[0, 0, 1],
           [0, 2, 0],
           [3, 0, 0]])
    >>> mt.fill_diagonal(mt.flipud(a), [1,2,3])  # Vertical flip
    >>> a.execute()
    array([[0, 0, 3],
           [0, 2, 0],
           [1, 0, 0]])

    Note that the order in which the diagonal is filled varies depending
    on the flip function.
    """

    if not isinstance(a, Tensor):
        raise TypeError('`a` should be a tensor, got {}'.format(type(a)))
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    if a.ndim > 2 and len(set(a.shape)) != 1:
        raise ValueError("All dimensions of input must be of equal length")

    # process val
    if isinstance(val, (Base, Entity)):
        val = astensor(val)
        if val.ndim > 1:
            val = val.ravel()
        val_input = val
    else:
        val = np.asarray(val)
        if val.ndim > 1:
            val = val.ravel()
        val_input = None

    op = TensorFillDiagonal(val=val, wrap=wrap, dtype=a.dtype)
    t = op(a, val=val_input)
    a.data = t.data
