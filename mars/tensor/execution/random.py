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

from ..core import CHUNK_TYPE
from .. import random
from .array import array_module


_RANDOM_OP_TO_NP_METHOD = {
    random.TensorRand: 'rand',
    random.TensorRandn: 'randn',
    random.TensorRandint: 'randint',
    random.TensorRandomIntegers: 'random_integers',
    random.TensorRandomSample: 'random_sample',
    random.TensorChoice: 'choice',
    random.TensorBeta: 'beta',
    random.TensorBinomial: 'binomial',
    random.TensorChisquare: 'chisquare',
    random.TensorDirichlet: 'dirichlet',
    random.TensorExponential: 'exponential',
    random.TensorF: 'f',
    random.TensorGamma: 'gamma',
    random.TensorGeometric: 'geometric',
    random.TensorGumbel: 'gumbel',
    random.TensorHypergeometric: 'hypergeometric',
    random.TensorLaplace: 'laplace',
    random.TensorLogistic: 'logistic',
    random.TensorLognormal: 'lognormal',
    random.TensorLogseries: 'logseries',
    random.TensorMultinomial: 'multinomial',
    random.TensorMultivariateNormal: 'multivariate_normal',
    random.TensorNegativeBinomial: 'negative_binomial',
    random.TensorNoncentralChisquare: 'noncentral_chisquare',
    random.TensorNoncentralF: 'noncentral_f',
    random.TensorNormal: 'normal',
    random.TensorPareto: 'pareto',
    random.TensorPoisson: 'poisson',
    random.TensorRandomPower: 'power',
    random.TensorRayleigh: 'rayleigh',
    random.TensorStandardCauchy: 'standard_cauchy',
    random.TensorStandardExponential: 'standard_exponential',
    random.TensorStandardGamma: 'standard_gamma',
    random.TensorStandardNormal: 'standard_normal',
    random.TensorStandardT: 'standard_t',
    random.TensorTriangular: 'triangular',
    random.TensorUniform: 'uniform',
    random.TensorVonmises: 'vonmises',
    random.TensorWald: 'wald',
    random.TensorWeibull: 'weibull',
    random.TensorZipf: 'zipf',
}


def _rand(ctx, chunk):
    xp = array_module(chunk.op.gpu)
    if chunk.op.state:
        rs = chunk.op.state.random_state
    else:
        if xp == np:
            rs = xp.random.RandomState()
        else:
            rs = xp.random
    get_val = lambda x: ctx[x.key] if isinstance(x, CHUNK_TYPE) else x

    method_name = _RANDOM_OP_TO_NP_METHOD[type(chunk.op)]
    try:
        if method_name in ('rand', 'randn'):
            try:
                res = getattr(rs, method_name)(*chunk.op.size, dtype=chunk.op.dtype)
            except TypeError:
                res = getattr(rs, method_name)(*chunk.op.size)
        elif method_name == 'randint':
            try:
                res = rs.randint(get_val(chunk.op.low), get_val(chunk.op.high), size=chunk.op.size,
                                 dtype=chunk.op.dtype)
            except TypeError:
                res = rs.randint(get_val(chunk.op.low), get_val(chunk.op.high), size=chunk.op.size)
        else:
            try:
                res = getattr(rs, method_name)(*(get_val(getattr(chunk.op, arg)) for arg in chunk.op.args),
                                               dtype=chunk.op.dtype)
            except TypeError:
                res = getattr(rs, method_name)(*(get_val(getattr(chunk.op, arg)) for arg in chunk.op.args))
        if hasattr(res, 'dtype') and res.dtype != chunk.op.dtype:
            res = res.astype(chunk.op.dtype)
        if xp != np:
            with xp.cuda.Device(chunk.op.device or 0):
                ctx[chunk.key] = xp.asarray(res)
        else:
            ctx[chunk.key] = res
    except AttributeError:
        if xp != np:
            if not chunk.op.state:
                rs = np.random.RandomState()
            if method_name in ('rand', 'randn'):
                try:
                    res = getattr(rs, method_name)(*chunk.op.size, dtype=chunk.op.dtype)
                except TypeError:
                    res = getattr(rs, method_name)(*chunk.op.size)
            else:
                try:
                    res = getattr(rs, method_name)(*(get_val(getattr(chunk.op, arg)) for arg in chunk.op.args),
                                                   dtype=chunk.op.dtype)
                except TypeError:
                    res = getattr(rs, method_name)(*(get_val(getattr(chunk.op, arg)) for arg in chunk.op.args))
            if res.dtype != chunk.op.dtype:
                res = res.astype(chunk.op.dtype)
            with xp.cuda.Device(chunk.op.device or 0):
                ctx[chunk.key] = xp.asarray(res)
        else:
            raise


def _multivariate_normal(ctx, chunk):
    xp = array_module(chunk.op.gpu)
    if chunk.op.state:
        rs = chunk.op.state.random_state
    else:
        rs = xp.random.RandomState()

    args = []
    for k in chunk.op.args:
        val = getattr(chunk.op, k, None)
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
        if xp != np:
            with xp.cuda.Device(chunk.op.device or 0):
                ctx[chunk.key] = xp.asarray(res)
        else:
            ctx[chunk.key] = res
    except AttributeError:
        if xp != np:
            if not chunk.op.state:
                rs = np.random.RandomState()
            res = rs.multivariate_normal(mean, cov, **kw)
            with xp.cuda.Device(chunk.op.device or 0):
                ctx[chunk.key] = xp.asarray(res)
        else:
            raise


def _distribution(ctx, chunk):
    xp = array_module(chunk.op.gpu)
    if chunk.op.state:
        rs = chunk.op.state.random_state
    else:
        rs = xp.random.RandomState()

    args = []
    for k in chunk.op.args:
        val = getattr(chunk.op, k, None)
        if isinstance(val, CHUNK_TYPE):
            args.append(ctx[val.key])
        else:
            args.append(val)

    method_name = _RANDOM_OP_TO_NP_METHOD[type(chunk.op)]
    try:
        res = getattr(rs, method_name)(*args)
        if xp != np:
            with xp.cuda.Device(chunk.op.device or 0):
                ctx[chunk.key] = xp.asarray(res)
        else:
            ctx[chunk.key] = res
    except AttributeError:
        if xp != np:
            if not chunk.op.state:
                rs = np.random.RandomState()
            res = getattr(rs, method_name)(*args)
            with xp.cuda.Device(chunk.op.device or 0):
                ctx[chunk.key] = xp.asarray(res)
        else:
            raise


def _sparse_randint(ctx, chunk):
    from ...lib.sparse import SparseNDArray
    from ...lib.sparse.core import cps, sps

    xp = array_module(chunk.op.gpu)
    if chunk.op.state:
        rs = chunk.op.state.random_state
    else:
        rs = None

    if chunk.ndim > 2:
        raise NotImplementedError

    low = 1 if chunk.op.low == 0 else chunk.op.low

    rs = rs or xp.random
    size = int(np.ceil(np.prod(chunk.shape) * chunk.op.density))
    xps = cps if chunk.op.gpu else sps
    ij = xp.empty((2, size))
    ij[0] = rs.randint(chunk.shape[0], size=size)
    ij[1] = rs.randint(chunk.shape[1], size=size)
    data = rs.randint(low, chunk.op.high, size=size).astype(chunk.op.dtype)
    m = xps.coo_matrix((data, ij), chunk.shape).tocsr()
    m.data[m.data >= chunk.op.high] = chunk.op.high - 1

    # scipy.sparse is too slow, we remove the precise version due to the performance
    # m = sps.random(*chunk.shape, density=chunk.op.density, format='csr')
    # m.data = (rs or xp.random).randint(low, chunk.op.high, size=m.data.size)\
    #     .astype(chunk.op.dtype)

    ctx[chunk.key] = SparseNDArray(m)


def _randint(ctx, chunk):
    if chunk.issparse():
        return _sparse_randint(ctx, chunk)

    return _rand(ctx, chunk)


def _random_estimate_size(ctx, chunk):
    if not chunk.is_sparse() or not getattr(chunk.op, '_density', None):
        raise NotImplementedError
    # use density to estimate real memory usage
    nbytes = int(chunk.nbytes * getattr(chunk.op, '_density'))
    ctx[chunk.key] = (nbytes, nbytes)


def register_random_handler():
    from ...executor import register
    from ..expressions.random.core import TensorSimpleRandomData, TensorDistribution

    register(TensorSimpleRandomData, _rand)
    register(TensorDistribution, _distribution)
    register(random.TensorMultivariateNormal, _multivariate_normal)
    register(random.TensorRandint, _randint, _random_estimate_size)
