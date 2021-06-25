#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from collections.abc import Iterable

import numpy as np

from ... import opcodes as OperandDef
from ...serialization.serializables import TupleField
from ...config import options
from ..utils import decide_chunk_sizes, gen_random_seeds
from .core import TensorRandomOperandMixin, TensorDistribution


class TensorDirichlet(TensorDistribution, TensorRandomOperandMixin):
    _op_type_ = OperandDef.RAND_DIRICHLET

    _fields_ = '_alpha', '_size'
    _alpha = TupleField('alpha')
    _func_name = 'dirichlet'

    def __init__(self, alpha=None, size=None, dtype=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_alpha=alpha, _size=size, dtype=dtype, **kw)

    @property
    def alpha(self):
        return self._alpha

    def _calc_shape(self, shapes):
        shape = super()._calc_shape(shapes)
        return shape + (len(self._alpha),)

    def __call__(self, chunk_size=None):
        return self.new_tensor(None, None, raw_chunk_size=chunk_size)

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]
        chunk_size = tensor.extra_params.raw_chunk_size or options.chunk_size
        nsplits = decide_chunk_sizes(tensor.shape[:-1], chunk_size, tensor.dtype.itemsize)
        nsplits += ((len(op.alpha),),)

        idxes = list(itertools.product(*[range(len(s)) for s in nsplits]))
        seeds = gen_random_seeds(len(idxes), np.random.RandomState(op.seed))

        out_chunks = []
        for seed, idx, shape in zip(seeds, idxes, itertools.product(*nsplits)):
            inputs = [inp.cix[idx] for inp in op.inputs]
            size = shape[:-1]

            chunk_op = op.copy().reset_key()
            chunk_op._state = None
            chunk_op._seed = seed
            chunk_op._size = size
            out_chunk = chunk_op.new_chunk(inputs, shape=shape, index=idx)
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
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorDirichlet(seed=seed, alpha=alpha, size=size, gpu=gpu, dtype=dtype)
    return op(chunk_size=chunk_size)
