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

import itertools

import numpy as np

from ... import opcodes as OperandDef
from ...operands import OperandStage
from ...serialize import KeyField, StringField, AnyField, Int64Field, Int32Field
from ...config import options
from ...utils import check_chunks_unknown_shape
from ...tiles import TilesError
from ..operands import TensorOperand, TensorOperandMixin
from ..core import TENSOR_TYPE, TensorOrder
from ..datasource.array import tensor as astensor
from ..array_utils import as_same_device, device


class TensorSearchsorted(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.SEARCHSORTED

    _input = KeyField('input')
    _values = AnyField('values')
    _side = StringField('side')
    _combine_size = Int32Field('combine_size')
    # offset is used only for map stage
    _offset = Int64Field('offset')

    def __init__(self, values=None, side=None, dtype=None, gpu=None, combine_size=None,
                 stage=None,  offset=None, **kw):
        super().__init__(_values=values, _side=side, _dtype=dtype, _gpu=gpu,
                         _combine_size=combine_size, _stage=stage, _offset=offset, **kw)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if len(self._inputs) == 2:
            self._values = self._inputs[1]

    @property
    def input(self):
        return self._input

    @property
    def values(self):
        return self._values

    @property
    def side(self):
        return self._side

    @property
    def offset(self):
        return self._offset

    @property
    def combine_size(self):
        return self._combine_size

    @property
    def stage(self):
        return self._stage

    def __call__(self, a, v):
        inputs = [a]
        if isinstance(v, TENSOR_TYPE):
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
    def _combine_chunks(cls, to_combine, op, stage, v, idx):
        from ..merge import TensorStack

        v_shape = v.shape if hasattr(v, 'shape') else ()
        combine_op = TensorStack(axis=0, dtype=op.outputs[0].dtype)
        combine_chunk = combine_op.new_chunk(to_combine, shape=v_shape)
        chunk_op = op.copy().reset_key()
        chunk_op._stage = stage
        in_chunks = [combine_chunk]
        if len(op.inputs) == 2:
            in_chunks.append(v)
        return chunk_op.new_chunk(in_chunks, shape=v_shape, index=idx,
                                  order=op.outputs[0].order)

    @classmethod
    def _tile_tree_reduction(cls, op, a, v, out):
        check_chunks_unknown_shape(op.inputs, TilesError)

        combine_size = op.combine_size or options.combine_size
        input_len = len(op.inputs)
        v_chunks = [v] if input_len == 1 else v.chunks

        out_chunks = []
        for v_chunk in v_chunks:
            offsets = [0] + np.cumsum(a.nsplits[0]).tolist()[:-1]
            v_shape = v_chunk.shape if hasattr(v_chunk, 'shape') else ()
            v_index = v_chunk.index if hasattr(v_chunk, 'index') else (0,)
            chunks = []
            for i, c in enumerate(a.chunks):
                chunk_op = op.copy().reset_key()
                chunk_op._stage = OperandStage.map
                chunk_op._offset = offsets[i]
                in_chunks = [c]
                if input_len == 2:
                    in_chunks.append(v_chunk)
                chunks.append(chunk_op.new_chunk(in_chunks, shape=v_shape,
                                                 index=c.index, order=out.order))

            while len(chunks) > combine_size:
                new_chunks = []
                it = itertools.count(0)
                while True:
                    j = next(it)
                    to_combine = chunks[j * combine_size: (j + 1) * combine_size]
                    if len(to_combine) == 0:
                        break

                    new_chunks.append(
                        cls._combine_chunks(to_combine, op, OperandStage.combine, v_chunk, (j,)))
                chunks = new_chunks

            chunk = cls._combine_chunks(chunks, op, OperandStage.reduce, v_chunk, v_index)
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
            v = op.values
        else:
            v = op.inputs[1]

        if len(a.chunks) == 1:
            return cls._tile_one_chunk(op, a, v, out)
        return cls._tile_tree_reduction(op, a, v, out)

    @classmethod
    def _execute_without_stage(cls, xp, a, v, op):
        return xp.searchsorted(a, v, side=op.side)

    @classmethod
    def _execute_map(cls, xp, a, v, op):
        # in the map phase, calculate the indices and positions
        # for instance, a=[1, 4, 6], v=5, return will be (2, 6)
        indices = xp.atleast_1d(xp.searchsorted(a, v, side=op.side))
        data_indices = indices.copy()
        # if the value is larger than all data
        # for instance, a=[1, 4, 6], v=7
        # return will be (2, 6), not (3, 6), thus needs to subtract 1
        data_indices = xp.subtract(data_indices, 1, out=data_indices,
                                   where=data_indices >= len(a))
        data = a[data_indices]
        if op.offset > 0:
            indices = xp.add(indices, op.offset, out=indices)
        if np.isscalar(v):
            indices, data = indices[0], data[0]

        return indices, data

    @classmethod
    def _execute_combine(cls, xp, a, v, op):
        inp_indices, inp_data = a
        if np.isscalar(v):
            ind = xp.searchsorted(inp_data, v, side=op.side)
            if ind >= len(inp_data):
                ind -= 1
            return inp_indices[ind], inp_data[ind]
        else:
            ret_indices = np.empty(v.shape, dtype=np.intp)
            ret_data = np.empty(v.shape, dtype=inp_data.dtype)
            for idx in itertools.product(*(range(s) for s in v.shape)):
                ind = xp.searchsorted(inp_data[(slice(None),) + idx], v[idx], side=op.side)
                if ind >= len(inp_indices):
                    ind -= 1
                ret_indices[idx] = inp_indices[(ind,) + idx]
                ret_data[idx] = inp_data[(ind,) + idx]
            return ret_indices, ret_data

    @classmethod
    def _execute_reduce(cls, xp, a, v, op):
        inp_indices, inp_data = a
        if np.isscalar(v):
            ind = xp.searchsorted(inp_data, v, side=op.side)
            if ind >= len(inp_indices):
                ind -= 1
            return inp_indices[ind]
        else:
            indices = np.empty(v.shape, dtype=np.intp)
            for idx in itertools.product(*(range(s) for s in v.shape)):
                ind = xp.searchsorted(inp_data[(slice(None),) + idx], v[idx], side=op.side)
                if ind >= len(inp_indices):
                    ind -= 1
                indices[idx] = inp_indices[(ind,) + idx]
            return indices

    @classmethod
    def execute(cls, ctx, op):
        a = ctx[op.inputs[0].key]
        v = ctx[op.inputs[1].key] if len(op.inputs) == 2 else op.values

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
            elif op.stage == OperandStage.map:
                ret = cls._execute_map(xp, a, v, op)
            elif op.stage == OperandStage.combine:
                ret = cls._execute_combine(xp, a, v, op)
            else:
                ret = cls._execute_reduce(xp, a, v, op)
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
        raise ValueError("'{0}' is an invalid value for keyword 'side'".format(side))

    if not np.isscalar(v):
        v = astensor(v)

    op = TensorSearchsorted(values=v, side=side, dtype=np.dtype(np.intp),
                            combine_size=combine_size)
    return op(a, v)
