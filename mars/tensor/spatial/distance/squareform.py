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

import itertools

import numpy as np

from .... import opcodes as OperandDef
from ....core import recursive_tile
from ....core.operand import OperandStage
from ....serialization.serializables import FieldTypes, KeyField, BoolField, TupleField
from ....config import options
from ....utils import has_unknown_shape, require_module
from ...core import TensorOrder
from ...operands import TensorMapReduceOperand, TensorOperandMixin, TensorShuffleProxy
from ...datasource import ascontiguousarray, array, zeros
from ...arithmetic import equal
from ...utils import decide_chunk_sizes
from ...array_utils import as_same_device, device, cp


class TensorSquareform(TensorMapReduceOperand, TensorOperandMixin):
    _op_type_ = OperandDef.SQUAREFORM

    _input = KeyField('input')
    _checks = BoolField('checks')

    _checks_input = KeyField('checks_input')
    _x_shape = TupleField('x_shape', FieldTypes.int32)
    _reduce_sizes = TupleField('reduce_sizes', FieldTypes.tuple)
    _start_positions = TupleField('start_positions', FieldTypes.int32)

    def __init__(self, checks=None, checks_input=None, x_shape=None,
                 reduce_sizes=None, start_positions=None, **kw):
        super().__init__(_checks=checks, _checks_input=checks_input,
                         _x_shape=x_shape, _reduce_sizes=reduce_sizes,
                         _start_positions=start_positions, **kw)

    @property
    def input(self):
        return self._input

    @property
    def checks(self):
        return self._checks

    @property
    def checks_input(self):
        return self._checks_input

    @property
    def x_shape(self):
        return self._x_shape

    @property
    def reduce_sizes(self):
        return self._reduce_sizes

    @property
    def start_positions(self):
        return self._start_positions

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if self._checks_input is not None:
            self._checks_input = self._inputs[-1]

    def __call__(self, X, force='no', chunk_size=None):
        s = X.shape

        if force.lower() == 'tomatrix':
            if len(s) != 1:
                raise ValueError("Forcing 'tomatrix' but input X is not a "
                                 "distance vector.")
        elif force.lower() == 'tovector':
            if len(s) != 2:
                raise ValueError("Forcing 'tovector' but input X is not a "
                                 "distance matrix.")

        # X = squareform(v)
        if len(s) == 1:
            if s[0] == 0:
                return zeros((1, 1), dtype=X.dtype)

            # Grab the closest value to the square root of the number
            # of elements times 2 to see if the number of elements
            # is indeed a binomial coefficient.
            d = int(np.ceil(np.sqrt(s[0] * 2)))

            # Check that v is of valid dimensions.
            if d * (d - 1) != s[0] * 2:
                raise ValueError('Incompatible vector size. It must be a binomial '
                                 'coefficient n choose 2 for some integer n >= 2.')

            shape = (d, d)
        elif len(s) == 2:
            if s[0] != s[1]:
                raise ValueError('The matrix argument must be square.')

            # One-side of the dimensions is set here.
            d = s[0]

            if d <= 1:
                return array([], dtype=X.dtype)

            shape = ((d * (d - 1)) // 2,)
        else:
            raise ValueError(('The first argument must be one or two dimensional '
                              'tensor. A %d-dimensional tensor is not '
                              'permitted') % len(s))

        return self.new_tensor([X], shape=shape, order=TensorOrder.C_ORDER,
                               raw_chunk_size=chunk_size)

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]
        chunk_size = tensor.extra_params.raw_chunk_size or options.chunk_size
        chunk_size = decide_chunk_sizes(tensor.shape, chunk_size, tensor.dtype.itemsize)
        n_chunk = np.product([len(cs) for cs in chunk_size])

        if len(op.input.chunks) == 1 and n_chunk == 1:
            return cls._tile_one_chunk(op)
        else:
            return (yield from cls._tile_chunks(op, chunk_size))

    @classmethod
    def _tile_one_chunk(cls, op):
        out = op.outputs[0]
        chunk_op = op.copy().reset_key()
        chunk = chunk_op.new_chunk(op.input.chunks, shape=out.shape,
                                   order=out.order,
                                   index=(0,) * out.ndim)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=out.shape,
                                  order=out.order, chunks=[chunk],
                                  nsplits=tuple((s,) for s in out.shape))

    @classmethod
    def _gen_checks_input(cls, op):
        if op.input.ndim != 2 or not op.checks:
            return

        x = op.input
        ret = yield from recursive_tile(equal(x, x.T).all())
        return ret.chunks[0]

    @classmethod
    def _tile_chunks(cls, op, chunk_size):
        if has_unknown_shape(*op.inputs):
            yield
        out = op.outputs[0]

        checks_input = yield from cls._gen_checks_input(op)

        map_chunks = []
        cum_sizes = [[0] + np.cumsum(ns).tolist()
                     for ns in op.input.nsplits]
        to_vec = op.input.ndim == 2
        for in_chunk in op.input.chunks:
            if to_vec and in_chunk.index[0] > in_chunk.index[1]:
                # if apply squareform to 2-d tensor which is symmetric,
                # we don't need to calculate for lower triangle chunks
                continue
            map_chunk_op = TensorSquareform(
                stage=OperandStage.map, checks_input=checks_input, reduce_sizes=chunk_size,
                x_shape=op.input.shape,
                start_positions=tuple(cum_sizes[ax][j]
                                      for ax, j in enumerate(in_chunk.index)),
                dtype=out.dtype, gpu=out.op.gpu)
            chunk_inputs = [in_chunk]
            if checks_input is not None:
                chunk_inputs.append(checks_input)
            map_chunk = map_chunk_op.new_chunk(chunk_inputs, shape=(2, np.nan),
                                               index=in_chunk.index,
                                               order=out.order)
            map_chunks.append(map_chunk)

        proxy_chunk = TensorShuffleProxy(dtype=out.dtype).new_chunk(
            map_chunks, shape=())

        reduce_chunks = []
        out_shape_iter = itertools.product(*chunk_size)
        out_idx_iter = itertools.product(*(range(len(cs)) for cs in chunk_size))
        for out_idx, out_shape in zip(out_idx_iter, out_shape_iter):
            reduce_chunk_op = TensorSquareform(
                stage=OperandStage.reduce,
                dtype=out.dtype)
            reduce_chunk = reduce_chunk_op.new_chunk(
                [proxy_chunk], shape=out_shape, index=out_idx, order=out.order)
            reduce_chunks.append(reduce_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=out.shape, order=out.order,
                                  nsplits=chunk_size, chunks=reduce_chunks)

    @classmethod
    def _to_matrix(cls, ctx, xp, x, op):
        assert x.ndim == 1
        out_chunk_size = op.reduce_sizes
        out_shape = tuple(sum(ns) for ns in out_chunk_size)
        d = out_shape[0]

        # calculate the index for the 1-d chunk
        index = xp.arange(x.shape[0])
        index = xp.add(index, op.start_positions[0], out=index)

        # input length for each row
        row_sizes = xp.arange(d, -1, -1)
        row_sizes[0] = 0
        xp.cumsum(row_sizes[1:], out=row_sizes[1:])
        # calculate row for each element
        rows = xp.searchsorted(row_sizes, index, side='right')
        xp.subtract(rows, 1, out=rows)
        # calculate col for each element
        # offsets
        cols_offsets = xp.arange(1, d + 1)
        cols = xp.empty(x.shape, dtype=np.int32)
        xp.add(xp.subtract(index, row_sizes[rows], out=cols),
               cols_offsets[rows], out=cols)

        cum_sizes = [[0] + np.cumsum(cs).tolist() for cs in out_chunk_size]
        for idx in itertools.product(*(range(len(ns)) for ns in out_chunk_size)):
            i, j = idx
            row_range = cum_sizes[0][i], cum_sizes[0][i + 1]
            col_range = cum_sizes[1][j], cum_sizes[1][j + 1]
            # for upper
            filtered = (rows >= row_range[0]) & (rows < row_range[1]) & \
                       (cols >= col_range[0]) & (cols < col_range[1])
            inds_tup = rows[filtered] - row_range[0], cols[filtered] - col_range[0]
            upper_inds = xp.ravel_multi_index(inds_tup, (out_chunk_size[0][i], out_chunk_size[1][j]))
            upper_values = x[filtered]
            # for lower
            filtered = (rows >= col_range[0]) & (rows < col_range[1]) & \
                       (cols >= row_range[0]) & (cols < row_range[1])
            inds_tup = cols[filtered] - row_range[0], rows[filtered] - col_range[0]
            lower_inds = xp.ravel_multi_index(inds_tup, (out_chunk_size[0][i], out_chunk_size[1][j]))
            lower_values = x[filtered]

            inds = xp.concatenate([upper_inds, lower_inds])
            values = xp.concatenate([upper_values, lower_values])

            ctx[op.outputs[0].key, idx] = inds, values

    @classmethod
    def _to_vector(cls, ctx, xp, x, op):
        out_chunk_size = op.reduce_sizes
        start_poses = op.start_positions

        i_indices, j_indices = xp.mgrid[
            start_poses[0]: start_poses[0] + x.shape[0],
            start_poses[1]: start_poses[1] + x.shape[1]
        ]
        filtered = i_indices < j_indices
        i_indices, j_indices, x = \
            i_indices[filtered], j_indices[filtered], x[filtered]

        d = op.x_shape[0]
        row_sizes = xp.arange(d - 1, -1, -1)
        row_cum_sizes = xp.empty((d + 1,), dtype=int)
        row_cum_sizes[0] = 0
        xp.cumsum(row_sizes, out=row_cum_sizes[1:])
        to_indices = row_cum_sizes[i_indices] + j_indices - (d - row_sizes[i_indices])

        cum_chunk_size = [0] + np.cumsum(out_chunk_size).tolist()
        for i in range(len(out_chunk_size[0])):
            index_range = cum_chunk_size[i], cum_chunk_size[i + 1]
            filtered = (to_indices >= index_range[0]) & (to_indices < index_range[1])
            out_indices = to_indices[filtered] - cum_chunk_size[i]
            ctx[op.outputs[0].key, (i,)] = out_indices, x[filtered]

    @classmethod
    def _execute_map(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)

        if len(inputs) == 2 and not inputs[1]:
            # check fail
            raise ValueError('Distance matrix X must be symmetric.')

        if xp is cp:  # pragma: no cover
            raise NotImplementedError('`squareform` does not support running on GPU yet')

        with device(device_id):
            x = inputs[0]
            if x.ndim == 1:
                cls._to_matrix(ctx, xp, x, op)
            else:
                cls._to_vector(ctx, xp, x, op)

    @classmethod
    def _execute_reduce(cls, ctx, op: "TensorSquareform"):
        raw_inputs = list(op.iter_mapper_data(ctx))
        raw_indices = [inp[0] for inp in raw_inputs]
        raw_dists = [inp[1] for inp in raw_inputs]
        inputs, device_id, xp = as_same_device(
            raw_indices + raw_dists, op.device, ret_extra=True)
        raw_indices = inputs[:len(raw_indices)]
        raw_dists = inputs[len(raw_indices):]
        output = op.outputs[0]

        with device(device_id):
            out_dists = xp.zeros(output.shape, dtype=output.dtype)
            indices = xp.concatenate(raw_indices)
            dists = xp.concatenate(raw_dists)
            out_dists.flat[indices] = dists
            ctx[output.key] = out_dists

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        elif op.stage == OperandStage.reduce:
            cls._execute_reduce(ctx, op)
        else:
            from scipy.spatial.distance import squareform

            (x,), device_id, xp = as_same_device(
                [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)

            if xp is cp:  # pragma: no cover
                raise NotImplementedError('`squareform` does not support running on GPU yet')

            with device(device_id):
                ctx[op.outputs[0].key] = squareform(x, checks=op.checks)


@require_module('scipy.spatial.distance')
def squareform(X, force='no', checks=True, chunk_size=None):
    """
    Convert a vector-form distance vector to a square-form distance
    matrix, and vice-versa.

    Parameters
    ----------
    X : Tensor
        Either a condensed or redundant distance matrix.
    force : str, optional
        As with MATLAB(TM), if force is equal to ``'tovector'`` or
        ``'tomatrix'``, the input will be treated as a distance matrix or
        distance vector respectively.
    checks : bool, optional
        If set to False, no checks will be made for matrix
        symmetry nor zero diagonals. This is useful if it is known that
        ``X - X.T1`` is small and ``diag(X)`` is close to zero.
        These values are ignored any way so they do not disrupt the
        squareform transformation.

    Returns
    -------
    Y : Tensor
        If a condensed distance matrix is passed, a redundant one is
        returned, or if a redundant one is passed, a condensed distance
        matrix is returned.

    Notes
    -----
    1. v = squareform(X)

       Given a square d-by-d symmetric distance matrix X,
       ``v = squareform(X)`` returns a ``d * (d-1) / 2`` (or
       :math:`{n \\choose 2}`) sized vector v.

      :math:`v[{n \\choose 2}-{n-i \\choose 2} + (j-i-1)]` is the distance
      between points i and j. If X is non-square or asymmetric, an error
      is returned.

    2. X = squareform(v)

      Given a ``d*(d-1)/2`` sized v for some integer ``d >= 2`` encoding
      distances as described, ``X = squareform(v)`` returns a d by d distance
      matrix X.  The ``X[i, j]`` and ``X[j, i]`` values are set to
      :math:`v[{n \\choose 2}-{n-i \\choose 2} + (j-i-1)]` and all
      diagonal elements are zero.

    """

    X = ascontiguousarray(X)

    op = TensorSquareform(checks=checks, dtype=X.dtype, gpu=X.op.gpu)
    return op(X, force=force, chunk_size=chunk_size)
