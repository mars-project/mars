import numpy as np

from ...core import ENTITY_TYPE
from ... import opcodes as OperandDef
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
    #_op_type = OperandDef.PAD

    _pad_width = AnyField('pad_width')
    _mode = AnyField('mode')
    _pad_kwargs = AnyField('pad_kwargs')
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

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs[1:])
        if isinstance(self._pad_width, ENTITY_TYPE):
            self._pad_width = next(inputs_iter)
        if isinstance(self._mode, ENTITY_TYPE):
            self._mode = next(inputs_iter)
        if isinstance(self._pad_kwargs, ENTITY_TYPE):
            self._pad_kwargs = next(inputs_iter)

    @classmethod
    def tile(cls, op: 'TensorPad'):
        inp = op.inputs[0]
        chunk_shape = inp.chunk_shape
        out_chunks = []
        pad_width = op.pad_width
        for chunk in inp.chunks:
            chunk_op = op.copy().reset_key()
            chunk_pad_width = np.asarray(pad_width)
            for i in range(len(chunk_shape)):
                if chunk.index[i] != 0:
                    chunk_pad_width[i][0] = 0
                if chunk.index[i] != chunk_shape[i] - 1:
                    chunk_pad_width[i][1] = 0
            shape = [chunk.shape[i] + sum(s) for i, s in enumerate(chunk_pad_width)]
            chunk_op._pad_width = chunk_pad_width
            new_chunk = chunk_op.new_chunk([chunk], shape=shape, index=chunk.index)
            out_chunks.append(new_chunk)
        new_op = op.copy()
        nsplits = np.asarray(inp.nsplits)
        for i, axis_pad_with in enumerate(pad_width):
            nsplits[i][0] += axis_pad_with[0]
            nsplits[i][-1] += axis_pad_with[-1]

        return new_op.new_tensor(op.inputs, chunks=out_chunks, nsplits=nsplits, **op.outputs[0].params)

    @classmethod
    def execute(cls, ctx, op: 'TensorPad'):
        inp = ctx[op.input.key]
        pad_width = op.pad_width

        if np.sum(pad_width) > 0:
            res = np.pad(inp, pad_width, op.mode, **op.pad_kwargs)
            ctx[op.outputs[0].key] = res
        else:
            ctx[op.outputs[0].key] = inp

    def __call__(self, array, pad_width, mode, pad_kwargs, shape):
        return self.new_tensor(filter_inputs([array, pad_width, mode, pad_kwargs]), shape=shape)


def pad(array, pad_width, mode='constant', **kwagrs):
    array = astensor(array)

    pad_width = np.asarray(pad_width)
    if not pad_width.dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')
    pad_width = _as_pairs(pad_width, array.ndim, as_index=True)

    shape = tuple(s + sum(pad_width[i]) for i, s in enumerate(array.shape))
    op = TensorPad(pad_width=pad_width, mode=mode, pad_kwargs=kwagrs, dtype=array.dtype)
    return op(array, pad_width, mode, kwagrs, shape)
