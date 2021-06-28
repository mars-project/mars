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
import operator
import tempfile
from collections.abc import Iterable

import numpy as np

from ... import opcodes as OperandDef
from ...serialization.serializables import AnyField, BoolField, \
    StringField, TupleField, SliceField
from ..array_utils import device, as_same_device
from ..utils import validate_axis, unify_chunks
from ..datasource import tensor as astensor
from ..operands import TensorOperand, TensorOperandMixin
from ..indexing.slice import TensorSlice


def _get_index(chunk):
    try:
        return chunk.index
    except AttributeError:
        if isinstance(chunk.op, TensorSlice):
            return chunk.inputs[0].index
        raise


def _norm_axis(axis):
    if isinstance(axis, int):
        return axis, True
    if isinstance(axis, Iterable):
        axis = sorted(tuple(axis))
        if len(axis) == 1:
            return axis[0], True
        return axis, False

    assert axis is None
    return None, False


class TensorConcatenate(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.CONCATENATE

    _axis = AnyField('axis')

    # for mmap
    _mmap = BoolField('mmap')
    _file_prefix = StringField('file_prefix')
    _create_mmap_file = BoolField('create_mmap_file')
    _partition_slice = SliceField('partition_slice')
    _total_shape = TupleField('total_shape')

    def __init__(self, axis=None, mmap=None, file_prefix=None, create_mmap_file=None,
                 partition_slice=None, total_shape=None, **kw):
        super().__init__(_axis=axis, _mmap=mmap,
                         _file_prefix=file_prefix,
                         _create_mmap_file=create_mmap_file,
                         _partition_slice=partition_slice,
                         _total_shape=total_shape, **kw)

    @property
    def axis(self):
        return getattr(self, '_axis', None)

    @property
    def mmap(self):
        return self._mmap

    @property
    def file_prefix(self):
        return self._file_prefix

    @property
    def create_mmap_file(self):
        return self._create_mmap_file

    @property
    def partition_slice(self):
        return self._partition_slice

    @property
    def total_shape(self):
        return self._total_shape

    def __call__(self, tensors):
        if len(set(t.ndim for t in tensors)) != 1:
            raise ValueError('all the input tensors must have same number of dimensions')

        axis = self._axis
        shapes = [t.shape[:axis] + t.shape[axis + 1:] for t in tensors]
        if len(set(shapes)) != 1:
            raise ValueError('all the input tensor dimensions '
                             'except for the concatenation axis must match exactly')

        shape = [0 if i == axis else tensors[0].shape[i] for i in range(tensors[0].ndim)]
        shape[axis] = sum(t.shape[axis] for t in tensors)

        if any(np.isnan(s) for i, s in enumerate(shape) if i != axis):
            raise ValueError('cannot concatenate tensor with unknown shape')

        return self.new_tensor(tensors, shape=tuple(shape))

    @classmethod
    def tile(cls, op):
        from ..indexing.slice import TensorSlice

        inputs = op.inputs
        output = op.outputs[0]
        axis = op.axis

        c = itertools.count(inputs[0].ndim)
        tensor_axes = [(t, tuple(i if i != axis else next(c) for i in range(t.ndim)))
                       for t in inputs]
        inputs = yield from unify_chunks(*tensor_axes)

        out_chunk_shape = [0 if i == axis else inputs[0].chunk_shape[i]
                           for i in range(inputs[0].ndim)]
        out_chunk_shape[axis] = sum(t.chunk_shape[axis] for t in inputs)
        out_nsplits = [None if i == axis else inputs[0].nsplits[i]
                       for i in range(inputs[0].ndim)]
        out_nsplits[axis] = tuple(itertools.chain(*[t.nsplits[axis] for t in inputs]))

        out_chunks = []
        axis_cum_chunk_shape = np.cumsum([t.chunk_shape[axis] for t in inputs])
        for out_idx in itertools.product(*[range(s) for s in out_chunk_shape]):
            axis_index = np.searchsorted(axis_cum_chunk_shape, out_idx[axis], side='right')
            t = inputs[axis_index]
            axis_inner_index = out_idx[axis] - \
                (0 if axis_index < 1 else axis_cum_chunk_shape[axis_index - 1])
            idx = out_idx[:axis] + (axis_inner_index,) + out_idx[axis + 1:]
            in_chunk = t.cix[idx]
            if idx == out_idx:
                # if index is the same, just use the input chunk
                out_chunks.append(in_chunk)
            else:
                chunk_op = TensorSlice(slices=[slice(None) for _ in range(in_chunk.ndim)],
                                       dtype=in_chunk.dtype, sparse=in_chunk.op.sparse)
                out_chunk = chunk_op.new_chunk([in_chunk], shape=in_chunk.shape,
                                               index=out_idx, order=output.order)

                out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, output.shape, order=output.order,
                                  nsplits=out_nsplits, chunks=out_chunks)

    @staticmethod
    def _ensure_order(result, order):
        return result.astype(result.dtype, order=order.value, copy=False)

    @classmethod
    def execute(cls, ctx, op):
        if op.mmap:  # pragma: no cover
            cls._execute_with_mmap(ctx, op)
        else:
            cls._execute(ctx, op)

    @classmethod
    def _execute(cls, ctx, op):
        def _base_concatenate(chunk, inputs):
            inputs, device_id, xp = as_same_device(inputs, device=chunk.op.device, ret_extra=True)

            axis, single_axis = _norm_axis(chunk.op.axis)
            if single_axis:
                with device(device_id):
                    res = xp.concatenate(tuple(inputs), axis=axis)
            else:
                axes = axis or list(range(chunk.ndim))
                chunks = [(_get_index(input), data) for input, data in zip(chunk.inputs, inputs)]
                with device(device_id):
                    for i in range(len(axes) - 1):
                        new_chunks = []
                        for idx, cs in itertools.groupby(chunks, key=lambda t: t[0][:-1]):
                            cs = list(map(operator.itemgetter(1), cs))
                            new_chunks.append((idx, xp.concatenate(cs, axis=len(axes) - i - 1)))
                        chunks = new_chunks
                    res = xp.concatenate(list(map(operator.itemgetter(1), chunks)), axis=axes[0])
            return res

        chunk = op.outputs[0]
        inputs = [ctx[input.key] for input in op.inputs]

        if isinstance(inputs[0], tuple):
            ctx[chunk.key] = \
                tuple(cls._ensure_order(_base_concatenate(chunk, [input[i] for input in inputs]), chunk.order)
                      for i in range(len(inputs[0])))
        else:
            ctx[chunk.key] = cls._ensure_order(_base_concatenate(chunk, inputs), chunk.order)

    @classmethod
    def _execute_with_mmap(cls, ctx, op):  # pragma: no cover
        if op.create_mmap_file:
            path = tempfile.mkstemp(prefix=op.file_prefix, suffix='.dat')[1]
            np.memmap(path, dtype=op.dtype, mode='w+', shape=op.total_shape)
            ctx[op.outputs[0].key] = path
        else:
            path = ctx[op.inputs[0].key]
            array = ctx[op.inputs[1].key]
            fp = np.memmap(path, dtype=op.dtype, mode='r+', shape=op.total_shape)
            fp[op.partition_slice] = array
            ctx[op.outputs[0].key] = path


def concatenate(tensors, axis=0):
    """
    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    a1, a2, ... : sequence of array_like
        The tensors must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the tensors will be joined.  Default is 0.

    Returns
    -------
    res : Tensor
        The concatenated tensor.

    See Also
    --------
    array_split : Split a tensor into multiple sub-arrays of equal or
                  near-equal size.
    split : Split tensor into a list of multiple sub-tensors of equal size.
    hsplit : Split tensor into multiple sub-tensors horizontally (column wise)
    vsplit : Split tensor into multiple sub-tensors vertically (row wise)
    dsplit : Split tensor into multiple sub-tensors along the 3rd axis (depth).
    stack : Stack a sequence of tensors along a new axis.
    hstack : Stack tensors in sequence horizontally (column wise)
    vstack : Stack tensors in sequence vertically (row wise)
    dstack : Stack tensors in sequence depth wise (along third dimension)

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([[1, 2], [3, 4]])
    >>> b = mt.array([[5, 6]])
    >>> mt.concatenate((a, b), axis=0).execute()
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> mt.concatenate((a, b.T), axis=1).execute()
    array([[1, 2, 5],
           [3, 4, 6]])

    """
    if axis is None:
        axis = 0
    tensors = [astensor(t) for t in tensors]

    axis = validate_axis(tensors[0].ndim, axis)
    dtype = np.result_type(*(t.dtype for t in tensors))
    sparse = all(t.issparse() for t in tensors)

    op = TensorConcatenate(axis=axis, dtype=dtype, sparse=sparse)
    return op(tensors)
