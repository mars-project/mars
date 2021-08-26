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
from typing import Any, List, Tuple, Type

import numpy as np

from ... import opcodes as OperandDef
from ...core import TILEABLE_TYPE
from ...core.operand import OperandStage
from ...serialization.serializables import StringField, \
    AnyField, Int64Field, Int32Field
from ...typing import TileableType, ChunkType
from ...config import options
from ...utils import has_unknown_shape
from ..operands import TensorOperand, TensorOperandMixin
from ..core import TENSOR_TYPE, TensorOrder
from ..datasource.array import tensor as astensor
from ..array_utils import as_same_device, device


class TensorSearchsorted(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.SEARCHSORTED

    v = AnyField('v')
    side = StringField('side')
    combine_size = Int32Field('combine_size')
    # for chunk
    offset = Int64Field('offset')
    size = Int64Field('size')
    n_chunk = Int64Field('n_chunk')

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if isinstance(self.v, TILEABLE_TYPE):
            self.v = self._inputs[1]

    def __call__(self, a, v):
        inputs = [a]
        if isinstance(v, TILEABLE_TYPE):
            inputs.append(v)
            shape = v.shape
        else:
            shape = ()
        return self.new_tensor(inputs, shape=shape, order=TensorOrder.C_ORDER)

    @classmethod
    def _tile_one_chunk(cls, op, a, v, out):
        chunks = []
        if len(op.inputs) == 1:
            v_chunks = [v]
        else:
            v_chunks = v.chunks
        for v_chunk in v_chunks:
            chunk_op = op.copy().reset_key()
            in_chunks = [a.chunks[0]]
            if len(op.inputs) == 2:
                in_chunks.append(v_chunk)
            v_shape = v_chunk.shape if hasattr(v_chunk, 'shape') else ()
            chunk_idx = v_chunk.index if len(op.inputs) == 2 else (0,)
            chunk = chunk_op.new_chunk(in_chunks, shape=v_shape,
                                       index=chunk_idx, order=out.order)
            chunks.append(chunk)
        new_op = op.copy().reset_key()
        nsplits = ((s,) for s in out.shape) if len(op.inputs) == 1 else v.nsplits
        return new_op.new_tensors(op.inputs, out.shape,
                                  chunks=chunks, nsplits=nsplits)

    @classmethod
    def _combine_chunks(cls,
                        to_combine: List[ChunkType],
                        op_type: Type,
                        v: Any,
                        stage: OperandStage,
                        chunk_index: Tuple[int]):
        from ..merge import TensorStack

        dtype = np.dtype(np.intp)
        v_shape = v.shape if hasattr(v, 'shape') else ()
        combine_op = TensorStack(axis=0, dtype=dtype)
        combine_chunk = combine_op.new_chunk(to_combine, shape=v_shape)
        chunk_op = op_type(dtype=dtype, axis=(0,), stage=stage)
        return chunk_op.new_chunk([combine_chunk], shape=v_shape, index=chunk_index,
                                  order=TensorOrder.C_ORDER)

    @classmethod
    def _tile_tree_reduction(cls,
                             op: "TensorSearchsorted",
                             a: TileableType,
                             v: Any,
                             out: TileableType):
        from ..indexing import TensorSlice
        from ..merge import TensorConcatenate
        from ..reduction import TensorMax, TensorMin

        if has_unknown_shape(a):
            yield

        combine_size = op.combine_size or options.combine_size
        n_chunk = len(a.chunks)
        input_len = len(op.inputs)
        v_chunks = [v] if input_len == 1 else v.chunks
        cum_nsplits = [0] + np.cumsum(a.nsplits[0]).tolist()

        input_chunks = []
        offsets = []
        for i in range(n_chunk):
            offset = cum_nsplits[i]
            cur_chunk = a.chunks[i]
            chunk_size = a.shape[0]
            chunks = []
            if i > 0:
                last_chunk = a.chunks[i - 1]
                if last_chunk.shape[0] > 0:
                    slice_chunk_op = TensorSlice(slices=[slice(-1, None)],
                                                 dtype=cur_chunk.dtype)
                    slice_chunk = slice_chunk_op.new_chunk(
                        [last_chunk], shape=(1,), order=out.order)
                    chunks.append(slice_chunk)
                    chunk_size += 1
                    offset -= 1
            chunks.append(cur_chunk)
            if i < n_chunk - 1:
                next_chunk = a.chunks[i + 1]
                if next_chunk.shape[0] > 0:
                    slice_chunk_op = TensorSlice(slices=[slice(1)],
                                                 dtype=cur_chunk.dtype)
                    slice_chunk = slice_chunk_op.new_chunk(
                        [next_chunk], shape=(1,), order=out.order)
                    chunks.append(slice_chunk)
                    chunk_size += 1

            concat_op = TensorConcatenate(dtype=cur_chunk.dtype)
            concat_chunk = concat_op.new_chunk(
                chunks, shape=(chunk_size,), order=out.order,
                index=cur_chunk.index)
            input_chunks.append(concat_chunk)
            offsets.append(offset)

        out_chunks = []
        for v_chunk in v_chunks:
            chunks = []
            v_shape = v_chunk.shape if hasattr(v_chunk, 'shape') else ()
            v_index = v_chunk.index if hasattr(v_chunk, 'index') else (0,)
            for inp_chunk, offset in zip(input_chunks, offsets):
                chunk_op = op.copy().reset_key()
                chunk_op.stage = OperandStage.map
                chunk_op.offset = offset
                chunk_op.n_chunk = n_chunk
                chunk_op.size = a.shape[0]
                chunk_inputs = [inp_chunk]
                if input_len > 1:
                    chunk_inputs.append(v_chunk)
                map_chunk = chunk_op.new_chunk(
                    chunk_inputs, shape=v_shape, index=inp_chunk.index,
                    order=out.order)
                chunks.append(map_chunk)

            op_type = TensorMax if op.side == 'right' else TensorMin
            while len(chunks) > combine_size:
                new_chunks = []
                it = itertools.count(0)
                while True:
                    j = next(it)
                    to_combine = chunks[j * combine_size: (j + 1) * combine_size]
                    if len(to_combine) == 0:
                        break

                    new_chunks.append(
                        cls._combine_chunks(to_combine, op_type, v_chunk,
                                            OperandStage.combine, (j,)))
                chunks = new_chunks

            chunk = cls._combine_chunks(chunks, op_type, v_chunk,
                                        OperandStage.agg, v_index)
            out_chunks.append(chunk)

        new_op = op.copy().reset_key()
        nsplits = ((s,) for s in out.shape) if len(op.inputs) == 1 else v.nsplits
        return new_op.new_tensors(op.inputs, out.shape,
                                  chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def tile(cls, op):
        a = op.inputs[0]
        out = op.outputs[0]
        input_len = len(op.inputs)
        if input_len == 1:
            v = op.v
        else:
            v = op.inputs[1]

        if len(a.chunks) == 1:
            return cls._tile_one_chunk(op, a, v, out)
        return (yield from cls._tile_tree_reduction(op, a, v, out))

    @classmethod
    def _execute_without_stage(cls, xp, a, v, op):
        return xp.searchsorted(a, v, side=op.side)

    @classmethod
    def _execute_map(cls,
                     xp: Any,
                     a: np.ndarray,
                     v: Any,
                     op: "TensorSearchsorted"):
        out = op.outputs[0]
        i = out.index[0]
        side = op.side

        raw_v = v
        v = xp.atleast_1d(v)
        searched = xp.searchsorted(a, v, side=op.side)
        xp.add(searched, op.offset, out=searched)
        a_min, a_max = a[0], a[-1]
        if i == 0:
            # the first chunk
            if a_min == a_max:
                miss = v > a_max
            else:
                miss = v > a_max if side == 'left' else v >= a_max
        elif i == op.n_chunk - 1:
            # the last chunk
            if a_min == a_max:
                miss = v < a_min
            else:
                miss = v <= a_min if side == 'left' else v < a_min
        else:
            if side == 'left' and a_min < a_max:
                miss = (v <= a_min) | (v > a_max)
            elif a_min < a_max:
                miss = (v < a_min) | (v >= a_max)
            else:
                assert a_min == a_max
                miss = v != a_min
        if side == 'right':
            searched[miss] = -1
        else:
            searched[miss] = op.size + 1

        return searched[0] if np.isscalar(raw_v) else searched

    @classmethod
    def execute(cls, ctx, op):
        a = ctx[op.inputs[0].key]
        v = ctx[op.inputs[1].key] if len(op.inputs) == 2 else op.v

        data = []
        if isinstance(a, tuple):
            data.extend(a)
        else:
            data.append(a)
        if len(op.inputs) == 2:
            data.append(v)

        data, device_id, xp = as_same_device(
            data, device=op.device, ret_extra=True)

        if isinstance(a, tuple):
            a = data[:2]
        else:
            a = data[0]
        if len(op.inputs) == 2:
            v = data[-1]

        with device(device_id):
            if op.stage is None:
                ret = cls._execute_without_stage(xp, a, v, op)
            else:
                assert op.stage == OperandStage.map
                ret = cls._execute_map(xp, a, v, op)
            ctx[op.outputs[0].key] = ret


def searchsorted(a, v, side='left', sorter=None, combine_size=None):
    """
    Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted tensor `a` such that, if the
    corresponding elements in `v` were inserted before the indices, the
    order of `a` would be preserved.

    Assuming that `a` is sorted:

    ======  ============================
    `side`  returned index `i` satisfies
    ======  ============================
    left    ``a[i-1] < v <= a[i]``
    right   ``a[i-1] <= v < a[i]``
    ======  ============================

    Parameters
    ----------
    a : 1-D array_like
        Input tensor. If `sorter` is None, then it must be sorted in
        ascending order, otherwise `sorter` must be an array of indices
        that sort it.
    v : array_like
        Values to insert into `a`.
    side : {'left', 'right'}, optional
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index.  If there is no suitable
        index, return either 0 or N (where N is the length of `a`).
    sorter : 1-D array_like, optional
        Optional tensor of integer indices that sort array a into ascending
        order. They are typically the result of argsort.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    indices : tensor of ints
        Array of insertion points with the same shape as `v`.

    See Also
    --------
    sort : Return a sorted copy of a tensor.
    histogram : Produce histogram from 1-D data.

    Notes
    -----
    Binary search is used to find the required insertion points.

    This function is a faster version of the builtin python `bisect.bisect_left`
    (``side='left'``) and `bisect.bisect_right` (``side='right'``) functions,
    which is also vectorized in the `v` argument.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.searchsorted([1,2,3,4,5], 3).execute()
    2
    >>> mt.searchsorted([1,2,3,4,5], 3, side='right').execute()
    3
    >>> mt.searchsorted([1,2,3,4,5], [-10, 10, 2, 3]).execute()
    array([0, 5, 1, 2])

    """

    if not isinstance(a, TENSOR_TYPE) and sorter is not None and \
            not isinstance(sorter, TENSOR_TYPE):
        a = astensor(np.asarray(a)[sorter])
    else:
        a = astensor(a)
        if sorter is not None:
            a = a[sorter]

    if a.ndim != 1:
        raise ValueError('`a` should be 1-d tensor')
    if a.issparse():
        # does not support sparse tensor
        raise ValueError('`a` should be a dense tensor')
    if side not in {'left', 'right'}:
        raise ValueError(f"'{side}' is an invalid value for keyword 'side'")

    if not np.isscalar(v):
        v = astensor(v)

    op = TensorSearchsorted(v=v, side=side, dtype=np.dtype(np.intp),
                            combine_size=combine_size)
    return op(a, v)
