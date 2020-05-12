#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from ...core import ExecutableTuple
from ...serialize import Int64Field, BoolField, AnyField
from ...config import options
from ..utils import decide_chunk_sizes
from .core import TensorNoInput
from ..array_utils import create_array


class TensorLinspace(TensorNoInput):
    _op_type_ = OperandDef.TENSOR_LINSPACE

    _start = AnyField('start')
    _stop = AnyField('stop')
    _num = Int64Field('num')
    _endpoint = BoolField('endpoint')

    def __init__(self, start=None, stop=None, num=None, endpoint=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(np.linspace(0, 1, 1).dtype if dtype is None else dtype)
        super().__init__(_start=start, _stop=stop, _num=num, _endpoint=endpoint, _dtype=dtype,
                         _gpu=gpu, **kw)

    def to_chunk_op(self, *args):
        start, stop, num, endpoint = args
        op = self.copy().reset_key()
        op._start = start
        op._stop = stop
        op._num = num
        op._endpoint = endpoint
        return op

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]

        chunk_length = tensor.extra_params.raw_chunk_size or options.chunk_size
        chunk_length = decide_chunk_sizes(tensor.shape, chunk_length, tensor.dtype.itemsize)

        start, stop, num, endpoint = \
            tensor.op.start, tensor.op.stop, tensor.op.num, tensor.op.endpoint
        if num > 1:
            step = float(stop - start) / (num if not endpoint else num - 1)
        else:
            step = 0.0

        chunks = []
        chunk_start = start
        nsplit = []
        for i, cs in enumerate(chunk_length[0]):
            chunk_stop = chunk_start + (cs - 1) * step
            chunk_op = op.to_chunk_op(chunk_start, chunk_stop, cs, True)
            chunk_shape = (cs,)
            chunk_idx = (i,)
            chunk = chunk_op.new_chunk(None, shape=chunk_shape, index=chunk_idx)
            chunks.append(chunk)
            nsplit.append(cs)
            chunk_start = chunk_start + cs * step

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape, chunks=chunks,
                                  nsplits=(tuple(nsplit),))

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def num(self):
        return self._num

    @property
    def endpoint(self):
        return self._endpoint

    @classmethod
    def execute(cls, ctx, op):
        ctx[op.outputs[0].key] = create_array(op)(
            'linspace', op.start, op.stop, num=op.num,
            endpoint=op.endpoint, dtype=op.dtype)


def linspace(start, stop, num=50, endpoint=True, retstep=False,
             dtype=None, gpu=False, chunk_size=None):
    """
    Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop`].

    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : scalar
        The starting value of the sequence.
    stop : scalar
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.
    dtype : dtype, optional
        The type of the output tensor.  If `dtype` is not given, infer the data
        type from the other input arguments.
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension

    Returns
    -------
    samples : Tensor
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float, optional
        Only returned if `retstep` is True

        Size of spacing between samples.


    See Also
    --------
    arange : Similar to `linspace`, but uses a step size (instead of the
             number of samples).
    logspace : Samples uniformly distributed in log space.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.linspace(2.0, 3.0, num=5).execute()
    array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
    >>> mt.linspace(2.0, 3.0, num=5, endpoint=False).execute()
    array([ 2. ,  2.2,  2.4,  2.6,  2.8])
    >>> mt.linspace(2.0, 3.0, num=5, retstep=True).execute()
    (array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 8
    >>> y = mt.zeros(N)
    >>> x1 = mt.linspace(0, 10, N, endpoint=True)
    >>> x2 = mt.linspace(0, 10, N, endpoint=False)
    >>> plt.plot(x1.execute(), y.execute(), 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(x2.execute(), y.execute() + 0.5, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.ylim([-0.5, 1])
    (-0.5, 1)
    >>> plt.show()

    """
    num = int(num)

    op = TensorLinspace(start, stop, num, endpoint, dtype=dtype, gpu=gpu)
    shape = (num,)
    ret = op(shape, chunk_size=chunk_size)

    if not retstep:
        return ret

    if num > 1:
        step = float(stop - start) / (num if not endpoint else num - 1)
    else:
        step = np.nan

    return ExecutableTuple([ret, step])
