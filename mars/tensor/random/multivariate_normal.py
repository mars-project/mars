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

import itertools

import numpy as np

from ... import opcodes as OperandDef
from ...serialize import NDArrayField, StringField, Float64Field
from ...config import options
from ..utils import decide_chunk_sizes, gen_random_seeds
from ..array_utils import array_module, device
from .core import TensorRandomOperandMixin, TensorDistribution, CHUNK_TYPE


class TensorMultivariateNormal(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_mean', '_cov', '_size', '_check_valid', '_tol'
    _op_type_ = OperandDef.RAND_MULTIVARIATE_NORMAL

    _mean = NDArrayField('mean')
    _cov = NDArrayField('cov')
    _check_valid = StringField('check_valid')
    _tol = Float64Field('tol')
    _func_name = 'multivariate_normal'

    def __init__(self, mean=None, cov=None, size=None, check_valid=None, tol=None,
                 state=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_mean=mean, _cov=cov, _size=size, _check_valid=check_valid,
                         _tol=tol, _state=state, _dtype=dtype, _gpu=gpu, **kw)

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov

    @property
    def check_valid(self):
        return self._check_valid

    @property
    def tol(self):
        return self._tol

    def __call__(self, chunk_size=None):
        N = self._mean.size
        if self._size is None:
            shape = (N,)
        else:
            try:
                shape = tuple(self._size) + (N,)
            except TypeError:
                shape = (self._size, N)

        return self.new_tensor(None, shape, raw_chunk_size=chunk_size)

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]
        chunk_size = tensor.extra_params.raw_chunk_size or options.chunk_size
        nsplits = decide_chunk_sizes(tensor.shape[:-1], chunk_size, tensor.dtype.itemsize) + ((tensor.shape[-1],),)

        mean_chunk = op.mean.chunks[0] if hasattr(op.mean, 'chunks') else op.mean
        cov_chunk = op.cov.chunks[0] if hasattr(op.cov, 'chunks') else op.cov

        idxes = list(itertools.product(*[range(len(s)) for s in nsplits]))
        seeds = gen_random_seeds(len(idxes), op.state)

        out_chunks = []
        for seed, out_idx, shape in zip(seeds, idxes, itertools.product(*nsplits)):
            chunk_op = op.copy().reset_key()
            chunk_op._state = None
            chunk_op._seed = seed
            chunk_op._size = shape[:-1]
            out_chunk = chunk_op.new_chunk([mean_chunk, cov_chunk], shape=shape, index=out_idx)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, tensor.shape,
                                  chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        xp = array_module(op.gpu)
        if xp is np:
            device_id = -1
        else:
            device_id = op.device or 0

        with device(device_id):
            rs = xp.random.RandomState(op.seed)

            args = []
            for k in op.args:
                val = getattr(op, k, None)
                if isinstance(val, CHUNK_TYPE):
                    args.append(ctx[val.key])
                else:
                    args.append(val)
            mean, cov = args[:2]
            kw = {}
            if args[2] is not None:
                kw['size'] = args[2]
            if args[3] is not None:
                kw['check_valid'] = args[3]
            if args[4] is not None:
                kw['tol'] = args[4]

            try:
                res = rs.multivariate_normal(mean, cov, **kw)
                if xp is not np:
                    ctx[op.outputs[0].key] = xp.asarray(res)
                else:
                    ctx[op.outputs[0].key] = res
            except AttributeError:
                if xp is not np:
                    # cupy cannot generate data, fallback to numpy first
                    rs = np.random.RandomState(op.seed)
                    res = rs.multivariate_normal(mean, cov, **kw)
                    ctx[op.outputs[0].key] = xp.asarray(res)
                else:
                    raise


def multivariate_normal(random_state, mean, cov, size=None, check_valid=None, tol=None,
                        chunk_size=None, gpu=None, dtype=None):
    """
    Draw random samples from a multivariate normal distribution.

    The multivariate normal, multinormal or Gaussian distribution is a
    generalization of the one-dimensional normal distribution to higher
    dimensions.  Such a distribution is specified by its mean and
    covariance matrix.  These parameters are analogous to the mean
    (average or "center") and variance (standard deviation, or "width,"
    squared) of the one-dimensional normal distribution.

    Parameters
    ----------
    mean : 1-D array_like, of length N
        Mean of the N-dimensional distribution.
    cov : 2-D array_like, of shape (N, N)
        Covariance matrix of the distribution. It must be symmetric and
        positive-semidefinite for proper sampling.
    size : int or tuple of ints, optional
        Given a shape of, for example, ``(m,n,k)``, ``m*n*k`` samples are
        generated, and packed in an `m`-by-`n`-by-`k` arrangement.  Because
        each sample is `N`-dimensional, the output shape is ``(m,n,k,N)``.
        If no shape is specified, a single (`N`-D) sample is returned.
    check_valid : { 'warn', 'raise', 'ignore' }, optional
        Behavior when the covariance matrix is not positive semidefinite.
    tol : float, optional
        Tolerance when checking the singular values in covariance matrix.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor
        The drawn samples, of shape *size*, if that was provided.  If not,
        the shape is ``(N,)``.

        In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
        value drawn from the distribution.

    Notes
    -----
    The mean is a coordinate in N-dimensional space, which represents the
    location where samples are most likely to be generated.  This is
    analogous to the peak of the bell curve for the one-dimensional or
    univariate normal distribution.

    Covariance indicates the level to which two variables vary together.
    From the multivariate normal distribution, we draw N-dimensional
    samples, :math:`X = [x_1, x_2, ... x_N]`.  The covariance matrix
    element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`.
    The element :math:`C_{ii}` is the variance of :math:`x_i` (i.e. its
    "spread").

    Instead of specifying the full covariance matrix, popular
    approximations include:

      - Spherical covariance (`cov` is a multiple of the identity matrix)
      - Diagonal covariance (`cov` has non-negative elements, and only on
        the diagonal)

    This geometrical property can be seen in two dimensions by plotting
    generated data-points:

    >>> mean = [0, 0]
    >>> cov = [[1, 0], [0, 100]]  # diagonal covariance

    Diagonal covariance means that points are oriented along x or y-axis:

    >>> import matplotlib.pyplot as plt
    >>> import mars.tensor as mt
    >>> x, y = mt.random.multivariate_normal(mean, cov, 5000).T
    >>> plt.plot(x.execute(), y.execute(), 'x')
    >>> plt.axis('equal')
    >>> plt.show()

    Note that the covariance matrix must be positive semidefinite (a.k.a.
    nonnegative-definite). Otherwise, the behavior of this method is
    undefined and backwards compatibility is not guaranteed.

    References
    ----------
    .. [1] Papoulis, A., "Probability, Random Variables, and Stochastic
           Processes," 3rd ed., New York: McGraw-Hill, 1991.
    .. [2] Duda, R. O., Hart, P. E., and Stork, D. G., "Pattern
           Classification," 2nd ed., New York: Wiley, 2001.

    Examples
    --------
    >>> mean = (1, 2)
    >>> cov = [[1, 0], [0, 1]]
    >>> x = mt.random.multivariate_normal(mean, cov, (3, 3))
    >>> x.shape
    (3, 3, 2)

    The following is probably true, given that 0.6 is roughly twice the
    standard deviation:

    >>> list(((x[0,0,:] - mean) < 0.6).execute())
    [True, True]
    """
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    if mean.ndim != 1:
        raise ValueError('mean must be 1 dimensional')
    if cov.ndim != 2:
        raise ValueError('cov must be 1 dimensional')
    if len(set(mean.shape + cov.shape)) != 1:
        raise ValueError('mean and cov must have same length')

    if dtype is None:
        small_kw = {}
        if check_valid:
            small_kw['check_valid'] = check_valid
        if tol:
            small_kw['tol'] = tol
        dtype = np.random.multivariate_normal(mean, cov, size=(0,), **small_kw).dtype

    size = random_state._handle_size(size)
    op = TensorMultivariateNormal(mean=mean, cov=cov, size=size, check_valid=check_valid,
                                  tol=tol, state=random_state.to_numpy(), gpu=gpu, dtype=dtype)
    return op(chunk_size=chunk_size)
