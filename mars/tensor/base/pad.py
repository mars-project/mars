from ...core import ENTITY_TYPE
from ... import opcodes as OperandDef
from ...serialization.serializables import AnyField, KeyField
from ..datasource import tensor as astensor
from ..operands import TensorHasInput, TensorOperandMixin
from ..utils import filter_inputs

# temporary import
import mars.tensor as mt
import numpy as np


def _as_pairs(x, ndim, as_index=False):
    if x is None:
        return ((None, None),) * ndim

    x = np.array(x)
    if as_index:
        x = np.round(x).astype(np.intp, copy=False)

    if x.ndim < 3:
        if x.size == 1:
            # x was supplied as a single value
            x = x.ravel()  # Ensure x[0] works for x.ndim == 0, 1, 2
            if as_index and x < 0:
                raise ValueError("index can't contain negative values")
            return ((x[0], x[0]),) * ndim

        if x.size == 2 and x.shape != (2, 1):
            x = x.ravel()  # Ensure x[0], x[1] works
            if as_index and (x[0] < 0 or x[1] < 0):
                raise ValueError("index can't contain negative values")
            return ((x[0], x[1]),) * ndim

    if as_index and x.min() < 0:
        raise ValueError("index can't contain negative values")

    # Converting the array with `tolist` seems to improve performance
    # when iterating and indexing the result (see usage in `pad`)
    return np.broadcast_to(x, (ndim, 2)).tolist()


class TensorPad(TensorHasInput, TensorOperandMixin):
    #_op_type = OperandDef.PAD

    _pad_width = AnyField('pad_width')
    _mode = AnyField('mode')
    _kwargs = AnyField('kwargs')
    _input = KeyField('input')
    _kwargs = AnyField('kwargs')
    def __init__(self, pad_width=None, mode=None, **kwargs):
        super().__init__(_pad_width=pad_width, _mode=mode, _kwargs=kwargs)

    @property
    def pad_width(self):
        return self._pad_width

    @property
    def mode(self):
        return self._mode

    @property
    def kwargs(self):
        return self._kwargs

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs[1:])
        if isinstance(self._pad_width, ENTITY_TYPE):
            self._pad_width = next(inputs_iter)
        if isinstance(self._mode, ENTITY_TYPE):
            self._mode = next(inputs_iter)
        if isinstance(self._kwargs, ENTITY_TYPE):
            self._kwargs = next(inputs_iter)

    @classmethod
    def tile(cls, op: 'TensorPad'):
        inp = op.inputs[0]
        chunk_shape = inp.chunk_shape
        out_chunks = []
        pad_width = op.pad_width
        for chunk in inp.chunks:
            #print(pad_width)
            chunk_op = op.copy().reset_key()
            chunk_pad_width = np.zeros_like(pad_width, dtype=int)
            #print(chunk.index, chunk_op.pad_width)
            for i in range(len(chunk_shape)):
                if chunk.index[i] == 0:
                    chunk_pad_width[i][0] = pad_width[i][0]
                if chunk.index[i] == chunk_shape[i] - 1:
                    chunk_pad_width[i][1] = pad_width[i][1]

            shape = [chunk.shape[i] + sum(s) for i, s in enumerate(chunk_op.pad_width)]
            chunk_op._pad_width = chunk_pad_width.tolist()
            new_chunk = chunk_op.new_chunk([chunk], shape=shape, index=chunk.index)
            out_chunks.append(new_chunk)
        out = op.outputs[0]
        new_op = op.copy()
        return new_op.new_tensor(op.inputs, shape=out.shape, order=out.order, chunks=out_chunks, nsplits=inp.nsplits)

    @classmethod
    def execute(cls, ctx, op: 'TensorPad'):
        inp = ctx[op.input.key]
        pad_width = op.pad_width
        mode = op.mode
        kwargs = op.kwargs
        print(op.input.index, op.pad_width)
        #print(f'{op.inputs[0].index} & {chunk_data}')
        res = np.pad(inp, pad_width, mode, **kwargs)
        ctx[op.outputs[0].key] = res

    def __call__(self, array, pad_width, mode, shape, **kwargs):
        return self.new_tensor(filter_inputs([array, pad_width, mode, kwargs]), shape=shape)


def pad(array, pad_width, mode='constant', **kwagrs):
    array = astensor(array)

    pad_width = np.asarray(pad_width)
    if not pad_width.dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')
    pad_width = _as_pairs(pad_width, array.ndim, as_index=True)

    shape = tuple(s + sum(pad_width[i]) for i, s in enumerate(array.shape))
    op = TensorPad(pad_width=pad_width, mode=mode, **kwagrs)
    return op(array, pad_width, mode, shape, **kwagrs)
