import numpy as np

from ...serialization.serializables import AnyField, KeyField
from ..datasource import tensor as astensor
from ..operands import TensorHasInput, TensorOperandMixin
from ..utils import filter_inputs


def _as_pairs(x, ndim, as_index=False):
    if x is None:
        return ((None, None),) * ndim

    x = np.array(x)
    if as_index:
        x = np.round(x).astype(np.intp, copy=False)

    if x.ndim < 3:
        if x.size == 1:
            x = x.ravel()
            if as_index and x < 0:
                raise ValueError("index can't contain negative values")
            return ((x[0], x[0]),) * ndim

        if x.size == 2 and x.shape != (2, 1):
            x = x.ravel()
            if as_index and (x[0] < 0 or x[1] < 0):
                raise ValueError("index can't contain negative values")
            return ((x[0], x[1]),) * ndim

    if as_index and x.min() < 0:
        raise ValueError("index can't contain negative values")

    return np.broadcast_to(x, (ndim, 2)).tolist()


class TensorPad(TensorHasInput, TensorOperandMixin):
    _pad_width = AnyField('pad_width')
    _mode = AnyField('mode')
    _pad_kwargs = AnyField('pad_kwargs')
    _output_slice = AnyField('output_slice')
    _input = KeyField('input')

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
    def output_slice(self):
        return self._output_slice

    @classmethod
    def tile(cls, op: 'TensorPad'):
        inp = op.inputs[0]
        pad_width = np.asarray(op.pad_width)
        chunk_shape = inp.chunk_shape
        nsplits = inp.nsplits
        out_chunks = []

        if 'stat_length' in op.pad_kwargs:
            values_length = _as_pairs(op.pad_kwargs['stat_length'], inp.ndim)
        elif op.mode in ['reflect']:
            values_length = pad_width + 1
        else:
            values_length = pad_width

        for chunk in inp.chunks:
            if any([chunk.index[axis] in [0, shape-1] for axis, shape in enumerate(chunk_shape)]):
                chunk_op = op.copy().reset_key()
                chunk_pad_width = np.zeros_like(pad_width)
                input_slice = [slice(0, 0)] * inp.ndim
                output_slice = [slice(None)] * inp.ndim
                for axis, shape in enumerate(chunk_shape):
                    if chunk.index[axis] == 0:
                        pw, vl = pad_width[axis][0], values_length[axis][0]
                        chunk_pad_width[axis][0] = pw
                        input_slice[axis] = slice(0, max(vl, chunk.shape[axis]))
                        output_slice[axis] = slice(0, pw + chunk.shape[axis])

                    elif chunk.index[axis] == shape - 1:
                        pw, vl = pad_width[axis][1], values_length[axis][1]
                        chunk_pad_width[axis][1] = pw
                        input_slice[axis] = slice(- max(vl, chunk.shape[axis]), None)
                        output_slice[axis] = slice(- pw - chunk.shape[axis], None)
                    else:
                        start = np.sum(nsplits[axis][:chunk.index[axis]])
                        stop = start + chunk.shape[axis]
                        input_slice[axis] = slice(start, stop)
                shape = [chunk.shape[i] + sum(s) for i, s in enumerate(chunk_pad_width)]
                chunk_op._pad_width = chunk_pad_width
                chunk_op._output_slice = output_slice
                chunk_input = inp[tuple(input_slice)]
                new_chunk = chunk_op.new_chunk([chunk_input], shape=shape, index=chunk.index)
                out_chunks.append(new_chunk)
            else:
                out_chunks.append(chunk)

        new_op = op.copy()
        nsplits = np.asarray(nsplits)
        for axis, axis_pad_with in enumerate(pad_width):
            nsplits[axis][0] += axis_pad_with[0]
            nsplits[axis][-1] += axis_pad_with[-1]

        return new_op.new_tensor(op.inputs, chunks=out_chunks, nsplits=nsplits, **op.outputs[0].params)

    @classmethod
    def execute(cls, ctx, op: 'TensorPad'):
        inp = ctx[op.inputs[0].key]
        pad_width = op.pad_width
        res = np.pad(inp, pad_width, op.mode, **op.pad_kwargs)
        ctx[op.outputs[0].key] = res[tuple(op.output_slice)]

    def __call__(self, array, pad_width, mode, pad_kwargs, shape):
        return self.new_tensor(filter_inputs([array, pad_width, mode, pad_kwargs]), shape=shape)


def pad(array, pad_width, mode='constant', **kwagrs):
    if mode == 'wrap' or callable(mode):
        raise NotImplementedError('Input mode has not been supported')

    array = astensor(array)
    pad_width = np.asarray(pad_width)

    if not pad_width.dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')
    pad_width = _as_pairs(pad_width, array.ndim, as_index=True)

    shape = tuple(s + sum(pad_width[i]) for i, s in enumerate(array.shape))
    op = TensorPad(pad_width=pad_width, mode=mode, pad_kwargs=kwagrs, dtype=array.dtype)
    return op(array, pad_width, mode, kwagrs, shape)
