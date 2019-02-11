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

import itertools
from collections import Iterable

import numpy as np

from .... import operands
from ....config import options
from ....operands.random import State
from ....compat import irange, izip
from ..utils import decide_chunk_sizes, random_state_data
from .core import TensorRandomOperandMixin


class TensorDirichlet(operands.Dirichlet, TensorRandomOperandMixin):
    __slots__ = '_alpha', '_size'

    def __init__(self, alpha=None, state=None, size=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorDirichlet, self).__init__(_alpha=alpha, _state=state, _size=size,
                                              _dtype=dtype, _gpu=gpu, **kw)

    def _get_shape(self, shapes):
        shape = super(TensorDirichlet, self)._get_shape(shapes)
        return shape + (len(self._alpha),)

    def __call__(self, chunk_size=None):
        return self.new_tensor(None, None, raw_chunk_size=chunk_size)

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]
        chunk_size = tensor.params.raw_chunk_size or options.tensor.chunk_size
        nsplits = decide_chunk_sizes(tensor.shape[:-1], chunk_size, tensor.dtype.itemsize)
        nsplits += ((len(op.alpha),),)

        idxes = list(itertools.product(*[irange(len(s)) for s in nsplits]))
        states = random_state_data(len(idxes), op.state.random_state) \
            if op.state is not None else [None] * len(idxes)

        out_chunks = []
        for state, idx, shape in izip(states, idxes, itertools.product(*nsplits)):
            inputs = [inp.cix[idx] for inp in op.inputs]
            state = State(np.random.RandomState(state)) \
                if state is not None else None
            size = shape[:-1]

            chunk_op = op.copy().reset_key()
            chunk_op._state = state
            chunk_op._size = size
            out_chunk = chunk_op.new_chunk(inputs, shape, index=idx)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, tensor.shape,
                                  chunks=out_chunks, nsplits=nsplits)


def dirichlet(random_state, alpha, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from the Dirichlet distribution.

    Draw `size` samples of dimension k from a Dirichlet distribution. A
    Dirichlet-distributed random variable can be seen as a multivariate
    generalization of a Beta distribution. Dirichlet pdf is the conjugate
    prior of a multinomial in Bayesian inference.

    Parameters
    ----------
    alpha : array
        Parameter of the distribution (k dimension for sample of
        dimension k).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    samples : Tensor
        The drawn samples, of shape (size, alpha.ndim).

    Raises
    -------
    ValueError
        If any value in alpha is less than or equal to zero

    Notes
    -----
    .. math:: X \approx \prod_{i=1}^{k}{x^{\alpha_i-1}_i}

    Uses the following property for computation: for each dimension,
    draw a random sample y_i from a standard gamma generator of shape
    `alpha_i`, then
    :math:`X = \frac{1}{\sum_{i=1}^k{y_i}} (y_1, \ldots, y_n)` is
    Dirichlet distributed.

    References
    ----------
    .. [1] David McKay, "Information Theory, Inference and Learning
           Algorithms," chapter 23,
           http://www.inference.phy.cam.ac.uk/mackay/
    .. [2] Wikipedia, "Dirichlet distribution",
           http://en.wikipedia.org/wiki/Dirichlet_distribution

    Examples
    --------
    Taking an example cited in Wikipedia, this distribution can be used if
    one wanted to cut strings (each of initial length 1.0) into K pieces
    with different lengths, where each piece had, on average, a designated
    average length, but allowing some variation in the relative sizes of
    the pieces.

    >>> import mars.tensor as mt

    >>> s = mt.random.dirichlet((10, 5, 3), 20).transpose()

    >>> import matplotlib.pyplot as plt

    >>> plt.barh(range(20), s[0].execute())
    >>> plt.barh(range(20), s[1].execute(), left=s[0].execute(), color='g')
    >>> plt.barh(range(20), s[2].execute(), left=(s[0]+s[1]).execute(), color='r')
    >>> plt.title("Lengths of Strings")
    """
    if isinstance(alpha, Iterable):
        alpha = tuple(alpha)
    else:
        raise TypeError('`alpha` should be an array')
    if dtype is None:
        dtype = np.random.RandomState().dirichlet(alpha, size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorDirichlet(state=random_state._state, alpha=alpha, size=size, gpu=gpu, dtype=dtype)
    return op(chunk_size=chunk_size)
