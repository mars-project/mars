import numpy as np
from itertools import product

from ...serialization.serializables import AnyField, KeyField
from ..datasource import tensor as astensor
from ..operands import TensorHasInput, TensorOperandMixin


def _as_pairs(x, ndim, as_index=False):
    if x is None:
        return ((None, None),) * ndim # pragma: no cover

    x = np.array(x)
    if as_index:
        x = np.round(x).astype(np.intp, copy=False) # pragma: no cover

    if x.ndim < 3:
        if x.size == 1:
            x = x.ravel()
            if as_index and x < 0:
                raise ValueError("index can't contain negative values") # pragma: no cover
            return ((x[0], x[0]),) * ndim

        if x.size == 2 and x.shape != (2, 1):
            x = x.ravel()
            if as_index and (x[0] < 0 or x[1] < 0):
                raise ValueError("index can't contain negative values") # pragma: no cover
            return ((x[0], x[1]),) * ndim

    if as_index and x.min() < 0:
        raise ValueError("index can't contain negative values") # pragma: no cover

    return np.broadcast_to(x, (ndim, 2)).tolist()


class TensorPad(TensorHasInput, TensorOperandMixin):
    _pad_width = AnyField('pad_width')
    _mode = AnyField('mode')
    _pad_kwargs = AnyField('pad_kwargs')
    _output_slice = AnyField('output_slice')
    _input = KeyField('input')
    _axis = AnyField('axis')
    _side = AnyField('side')
    _length = AnyField('length')

    basic_modes = {'constant', 'edge', 'linear_ramp', 'empty'}
    stat_modes = {'maximum', 'minimum', 'mean', 'median'}
    reflect_modes = {'reflect', 'symmetric'}

    def __init__(self, pad_width=None, mode=None, pad_kwargs=None, **kw):
        super().__init__(_pad_width=pad_width, _mode=mode, _pad_kwargs=pad_kwargs, **kw)

    @property
    def pad_width(self):
        return self._pad_width

    @property
    def mode(self):
        return self._mode

    @property
    def pad_kwargs(self):
        return self._pad_kwargs

    @property
    def axis(self):
        return self._axis

    @property
    def side(self):
        return self._side

    @property
    def length(self):
        return self._length

    @classmethod
    def _tile_basic_modes(cls, op: 'TensorPad'):
        inp = op.inputs[0]
        pad_width = np.asarray(op.pad_width)
        chunk_shape = inp.chunk_shape
        nsplits = inp.nsplits
        chunk_shape_arr = np.asarray([[0, shape-1] for shape in chunk_shape])
        out_chunks = []

        for chunk in inp.chunks:
            chunk_index_arr = np.asarray([chunk.index]*2).T
            mask = chunk_shape_arr == chunk_index_arr
            if mask.any():
                chunk_op = op.copy().reset_key()
                chunk_pad_width = np.zeros_like(pad_width)
                chunk_pad_width = np.where(mask, pad_width, chunk_pad_width)
                shape = [chunk.shape[i] + sum(s) for i, s in enumerate(chunk_pad_width)]
                chunk_op._pad_width = chunk_pad_width
                new_chunk = chunk_op.new_chunk([chunk], shape=shape, index=chunk.index)
                out_chunks.append(new_chunk)
            else:
                out_chunks.append(chunk)
        new_op = op.copy()
        nsplits = np.asarray(nsplits)
        for axis, axis_pad_width in enumerate(pad_width):
            nsplits[axis][0] += axis_pad_width[0]
            nsplits[axis][-1] += axis_pad_width[-1]

        return new_op.new_tensor(op.inputs, chunks=out_chunks, nsplits=nsplits, **op.outputs[0].params)

    @classmethod
    def _tile_other_modes(cls, op: 'TensorPad', length):
        inp = op.inputs[0]
        pad_width = np.asarray(op.pad_width)
        nsplits = [list(splits) for splits in inp.nsplits]
        out_chunk_shape = [s+1 if pad_width[axis][0] != 0 else s for axis, s in enumerate(inp.chunk_shape)]
        out_chunk_shape = [s+1 if pad_width[axis][1] != 0 else s for axis, s in enumerate(out_chunk_shape)]
        out_chunks = {}

        for chunk in inp.chunks:
            chunk_op = op.copy().reset_key()
            chunk_op._mode = None
            index = tuple([idx+1 if pad_width[axis][0] != 0 else idx for axis, idx in enumerate(chunk.index)])
            new_chunk = chunk_op.new_chunk([chunk], shape=chunk.shape, index=index)
            out_chunks[index] = new_chunk

        for axis, axis_pad_width in enumerate(pad_width):
            new_splits = nsplits[axis].copy()
            if axis_pad_width[0] != 0:
                cum_splits = np.cumsum(nsplits[axis])
                left_length = length[axis][0]
                if left_length is None or left_length > cum_splits[-1]:
                    left_length = cum_splits[-1]
                n_chunks = cum_splits.searchsorted(left_length) + 1
                for index in product(*[range(s) if i != axis else [0] for i, s in enumerate(out_chunk_shape)]):
                    if index in out_chunks:
                        continue
                    input_index = list(index)
                    input_index[axis] += 1
                    if not(tuple(input_index) in out_chunks):
                        continue
                    input_chunks = []
                    for ni in range(1, n_chunks+1):
                        input_index = list(index)
                        input_index[axis] += ni
                        input_chunks.append(out_chunks[tuple(input_index)])
                    chunk_op = op.copy().reset_key()
                    chunk_op._axis = axis
                    chunk_op._side = 0
                    chunk_pad_width = np.zeros_like(pad_width)
                    chunk_pad_width[axis][0] = axis_pad_width[0]
                    chunk_op._pad_width = chunk_pad_width
                    chunk_op._length = left_length
                    shape = list(input_chunks[0].shape)
                    shape[axis] = axis_pad_width[0]
                    new_chunk = chunk_op.new_chunk(input_chunks, shape=shape, index=index)
                    out_chunks[index] = new_chunk
                new_splits.insert(0, axis_pad_width[0])
            if axis_pad_width[1] != 0:
                cum_splits = np.cumsum(nsplits[axis][::-1])
                right_length = length[axis][1]
                if right_length is None or right_length > cum_splits[-1]:
                    right_length = cum_splits[-1]
                n_chunks = cum_splits.searchsorted(right_length) + 1
                for index in product(*[range(s) if i != axis else [s-1] for i, s in enumerate(out_chunk_shape)]):
                    if index in out_chunks:
                        continue
                    input_index = list(index)
                    input_index[axis] -= 1
                    if not(tuple(input_index) in out_chunks):
                        continue
                    input_chunks = []
                    for ni in range(1, n_chunks+1):
                        input_index = list(index)
                        input_index[axis] -= ni
                        input_chunks.insert(0, out_chunks[tuple(input_index)])
                    chunk_op = op.copy().reset_key()
                    chunk_op._axis = axis
                    chunk_op._side =  -1
                    chunk_pad_width = np.zeros_like(pad_width)
                    chunk_pad_width[axis][1] = axis_pad_width[1]
                    chunk_op._pad_width = chunk_pad_width
                    chunk_op._length = right_length
                    shape = list(input_chunks[0].shape)
                    shape[axis] = axis_pad_width[1]
                    new_chunk = chunk_op.new_chunk(input_chunks, shape=shape, index=index)
                    out_chunks[index] = new_chunk
                new_splits.append(axis_pad_width[1])
            nsplits[axis] = new_splits
        new_op = op.copy()
        out_chunks = list(out_chunks.values())
        return new_op.new_tensor(op.inputs, chunks=out_chunks, nsplits=nsplits, **op.outputs[0].params)

    @classmethod
    def tile(cls, op: 'TensorPad'):
        if op.mode in cls.basic_modes:
            return cls._tile_basic_modes(op)
        if op.mode in cls.stat_modes:
            length = op.pad_kwargs.get('stat_length', None)
            length = _as_pairs(length, op.inputs[0].ndim, as_index=True)
            return cls._tile_other_modes(op, length)
        if op.mode in cls.reflect_modes:
            length = np.asarray(op.pad_width) + 1
            return cls._tile_other_modes(op, length)

    @classmethod
    def execute(cls, ctx, op: 'TensorPad'):
        mode = op.mode
        if mode is None:
            inp = ctx[op.inputs[0].key]
            res = inp
            ctx[op.outputs[0].key] = res
        elif mode in cls.basic_modes:
            inp = ctx[op.inputs[0].key]
            pad_width = op.pad_width
            res = np.pad(inp, pad_width, op.mode, **op.pad_kwargs)
            ctx[op.outputs[0].key] = res
        elif mode in cls.stat_modes:
            axis = op.axis
            inp = [ctx[inp.key] for inp in op.inputs]
            inp = np.concatenate(inp, axis=axis)
            ndim = inp.ndim
            res = np.pad(inp, op.pad_width, op.mode, stat_length=op.length)
            if op.side == 0:
                width = op.pad_width[axis][0]
                res_slice = [slice(0, width) if i == axis else slice(None) for i in range(ndim)]
            if op.side == -1:
                width = op.pad_width[axis][1]
                res_slice = [slice(-width, None) if i == axis else slice(None) for i in range(ndim)]
            res = res[tuple(res_slice)]
            ctx[op.outputs[0].key] = res
        elif mode in cls.reflect_modes:
            axis = op.axis
            inp = [ctx[inp.key] for inp in op.inputs]
            inp = np.concatenate(inp, axis=axis)
            ndim = inp.ndim
            res = np.pad(inp, op.pad_width, op.mode, **op.pad_kwargs)
            if op.side == 0:
                width = op.pad_width[axis][0]
                res_slice = [slice(0, width) if i == axis else slice(None) for i in range(ndim)]
            if op.side == -1:
                width = op.pad_width[axis][1]
                res_slice = [slice(-width, None) if i == axis else slice(None) for i in range(ndim)]
            res = res[tuple(res_slice)]
            ctx[op.outputs[0].key] = res

    def __call__(self, array, shape):
        return self.new_tensor([array], shape=shape)


def pad(array, pad_width, mode='constant', **kwagrs):
    if mode == 'wrap' or callable(mode):
        raise NotImplementedError('Input mode has not been supported') # pragma: no cover

    array = astensor(array)
    pad_width = np.asarray(pad_width)

    if not pad_width.dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.') # pragma: no cover
    pad_width = _as_pairs(pad_width, array.ndim, as_index=True)

    shape = tuple(s + sum(pad_width[i]) for i, s in enumerate(array.shape))
    op = TensorPad(pad_width=pad_width, mode=mode, pad_kwargs=kwagrs, dtype=array.dtype)
    return op(array, shape)
