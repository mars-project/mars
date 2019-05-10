#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from .... import opcodes as OperandDef
from ....serialize import AnyField
from ....config import options
from ..utils import decide_chunk_sizes
from .core import TensorNoInput


class TensorArange(TensorNoInput):
    _op_type_ = OperandDef.TENSOR_ARANGE

    _start = AnyField('start')
    _stop = AnyField('stop')
    _step = AnyField('step')

    def __init__(self, start=None, stop=None, step=None, dtype=None, gpu=None, **kw):
        if dtype is not None:
            dtype = np.dtype(dtype)
        elif stop is not None and step is not None:
            dtype = np.dtype(dtype) if dtype is not None else np.arange(0, type(stop)(1), step).dtype
        super(TensorArange, self).__init__(_start=start, _stop=stop, _step=step,
                                           _dtype=dtype, _gpu=gpu, **kw)

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def step(self):
        return self._step

    def to_chunk_op(self, *args):
        op = self.copy().reset_key()
        start, stop, step = args
        op._start = start
        op._stop = stop
        op._step = step
        return op

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]

        chunk_length = tensor.extra_params.raw_chunk_size or options.tensor.chunk_size
        chunk_length = decide_chunk_sizes(tensor.shape, chunk_length, tensor.dtype.itemsize)

        start, stop, step = tensor.op.start, tensor.op.stop, tensor.op.step  # noqa: F841

        out_chunks = []
        n_elem = 0
        for i, cs in enumerate(chunk_length[0]):
            chunk_start = start + n_elem * step
            chunk_stop = start + (n_elem + cs) * step
            chunk_size = max(int(np.ceil((chunk_stop - chunk_start) / step)), 0)
            if chunk_size > cs:
                chunk_stop -= step
            chunk_shape = (cs,)
            chunk_idx = (i,)
            chunk_op = op.to_chunk_op(chunk_start, chunk_stop, step)
            out_chunk = chunk_op.new_chunk(None, shape=chunk_shape, index=chunk_idx)
            n_elem += cs
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape, chunks=out_chunks,
                                  nsplits=chunk_length)


def arange(*args, **kwargs):
    """
    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
    but returns a tensor rather than a list.

    When using a non-integer step, such as 0.1, the results will often not
    be consistent.  It is better to use ``linspace`` for these cases.

    Parameters
    ----------
    start : number, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : number, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : dtype
        The type of the output tensor.  If `dtype` is not given, infer the data
        type from the other input arguments.
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default

    Returns
    -------
    arange : Tensor
        Tensor of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.

    See Also
    --------
    linspace : Evenly spaced numbers with careful handling of endpoints.
    ogrid: Tensors of evenly spaced numbers in N-dimensions.
    mgrid: Grid-shaped tensors of evenly spaced numbers in N-dimensions.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.arange(3).execute()
    array([0, 1, 2])
    >>> mt.arange(3.0).execute()
    array([ 0.,  1.,  2.])
    >>> mt.arange(3,7).execute()
    array([3, 4, 5, 6])
    >>> mt.arange(3,7,2).execute()
    array([3, 5])
    """
    kw_args = [kwargs.get('start'), kwargs.get('stop'), kwargs.get('step')]
    kw_def = any(arg is not None for arg in kw_args)
    dtype = None
    if not kw_def:
        if len(args) == 1:
            start = 0
            stop = args[0]
            step = 1
        elif len(args) == 2:
            start = args[0]
            stop = args[1]
            step = 1
        elif len(args) == 3:
            start, stop, step = args
        elif len(args) == 4:
            start, stop, step, dtype = args
            dtype = np.dtype(dtype)
        else:
            raise TypeError("Required argument 'start' (pos 1) not found")
    else:
        names = 'start', 'stop', 'step'
        for i, arg in enumerate(args):
            if kw_args[i] is not None:
                raise TypeError("Argument given by name ('{0}') and position ({1})".format(names[i], i))
            kw_args[i] = arg
        start, stop, step = kw_args

    if dtype is None:
        if 'dtype' in kwargs:
            dtype = np.dtype(kwargs['dtype'])
        else:
            dtype = np.arange(0, type(stop)(1), step).dtype

    start, stop = dtype.type(start), dtype.type(stop)
    if dtype == np.datetime64 and not start:
        raise ValueError('arange requires both a start and a stop for Mars datetime64 ranges')
    if dtype == np.datetime64:
        span = np.array([stop - start])
        span[0] = step
        step = span[0]
        dtype = np.dtype(stop.dtype)
    else:
        step = dtype.type(step)
    size = max(int(np.ceil(np.true_divide(stop - start, step))), 0)

    op = TensorArange(start, stop, step, dtype=dtype, gpu=kwargs.get('gpu', False))
    shape = (size,)
    return op(shape, chunk_size=kwargs.pop('chunk_size', None))
