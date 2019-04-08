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

from collections import namedtuple

import numpy as np

from .core import Operand
from .. import opcodes as OperandDef
from ..serialize import TupleField, AnyField, Int64Field, BoolField, \
    KeyField, NDArrayField, StringField, Float64Field, ValueType
from ..compat import zip_longest
from ..utils import tokenize


class RandomOperand(Operand):
    _state = TupleField('state', on_serialize=lambda x: tuple(x) if x is not None else x,
                        on_deserialize=lambda x: State(x) if x is not None else x)

    @property
    def state(self):
        return getattr(self, '_state', None)

    def __setattr__(self, attr, value):
        if attr == '_state' and value is not None and not isinstance(value, State):
            value = State(value)
        super(RandomOperand, self).__setattr__(attr, value)

    @property
    def args(self):
        return [slot for slot in self.__slots__
                if slot not in set(RandomOperand.__slots__)]

    def _update_key(self):
        args = tuple(getattr(self, k, None) for k in self._keys_)
        if self.state is None:
            args += (np.random.random(),)
        self._key = tokenize(type(self), *args)


STATE = namedtuple('State', 'label keys pos has_gauss cached_gaussian')


class State(STATE):
    def __new__(cls, random_state=None, *args, **kwargs):
        if isinstance(random_state, np.random.RandomState) and not kwargs:
            return super(State, cls).__new__(cls, *random_state.get_state())
        elif isinstance(random_state, tuple) and not kwargs:
            return super(State, cls).__new__(cls, *random_state)
        elif args:
            return super(State, cls).__new__(cls, *((random_state,) + args))
        elif not kwargs:
            return super(State, cls).__new__(cls, *random_state)
        else:
            assert random_state is None
            return super(State, cls).__new__(cls, **kwargs)

    def __eq__(self, other):
        if not isinstance(other, tuple):
            return False

        for it, other_it in zip_longest(self, other):
            if isinstance(it, np.ndarray):
                if not np.array_equal(it, other_it):
                    return False
            elif it != other_it:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def random_state(self):
        rs = np.random.RandomState()
        s = (self.label, self.keys, self.pos, self.has_gauss, self.cached_gaussian)
        rs.set_state(s)

        return rs


class SimpleRandomData(RandomOperand):
    __slots__ = ()


class Rand(SimpleRandomData):
    _op_type_ = OperandDef.RAND_RAND

    _size = TupleField('size', ValueType.int64)

    @property
    def size(self):
        return self._size


class Randn(SimpleRandomData):
    _op_type_ = OperandDef.RAND_RANDN

    _size = TupleField('size')

    @property
    def size(self):
        return self._size


class Randint(SimpleRandomData):
    __slots__ = '_low', '_high', '_density', '_size'
    _op_type_ = OperandDef.RAND_RANDINT

    _low = Int64Field('low')
    _high = Int64Field('high')
    _density = Float64Field('density')
    _size = TupleField('size', ValueType.int64)

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def density(self):
        return self._density

    @property
    def size(self):
        return self._size


class RandomIntegers(SimpleRandomData):
    __slots__ = '_low', '_high', '_size'
    _op_type_ = OperandDef.RAND_RANDOM_INTEGERS

    _low = Int64Field('low')
    _high = Int64Field('high')
    _size = TupleField('size', ValueType.int64)

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def size(self):
        return self._size


class RandomSample(SimpleRandomData):
    __slots__ = '_size',
    _op_type_ = OperandDef.RAND_RANDOM_SAMPLE

    _size = TupleField('size', ValueType.int64)

    @property
    def size(self):
        return self._size


class Random(SimpleRandomData):
    __slots__ = '_size',
    _op_type_ = OperandDef.RAND_RANDOM

    _size = TupleField('size', ValueType.int64)

    @property
    def size(self):
        return self._size


class Ranf(SimpleRandomData):
    __slots__ = '_size',
    _op_type_ = OperandDef.RAND_RANF

    _size = TupleField('size', ValueType.int64)

    @property
    def size(self):
        return self._size


class Sample(SimpleRandomData):
    __slots__ = '_size',
    _op_type_ = OperandDef.RAND_SAMPLE

    _size = TupleField('size', ValueType.int64)

    @property
    def size(self):
        return self._size


class Choice(SimpleRandomData):
    __slots__ = '_a', '_size', '_replace', '_p'
    _op_type_ = OperandDef.RAND_CHOICE

    _a = AnyField('a')
    _size = TupleField('size', ValueType.int64)
    _replace = BoolField('replace')
    _p = KeyField('p')

    @property
    def a(self):
        return self._a

    @property
    def size(self):
        return self._size

    @property
    def replace(self):
        return self._replace

    @property
    def p(self):
        return self._p


class Distribution(RandomOperand):
    __slots__ = ()


class Beta(Distribution):
    __slots__ = '_a', '_b', '_size'
    _op_type_ = OperandDef.RAND_BETA

    _a = AnyField('a')
    _b = AnyField('b')
    _size = TupleField('size', ValueType.int64)

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def size(self):
        return self._size


class Binomial(Distribution):
    __slots__ = '_n', '_p', '_size'
    _op_type_ = OperandDef.RAND_BINOMIAL

    _n = AnyField('n')
    _p = AnyField('p')
    _size = TupleField('size', ValueType.int64)

    @property
    def n(self):
        return self._n

    @property
    def p(self):
        return self._p

    @property
    def size(self):
        return self._size


class Chisquare(Distribution):
    __slots__ = '_df', '_size'
    _op_type_ = OperandDef.RAND_CHISQUARE

    _df = AnyField('df')
    _size = TupleField('size', ValueType.int64)

    @property
    def df(self):
        return self._df

    @property
    def size(self):
        return self._size


class Dirichlet(Distribution):
    __slots__ = '_alpha', '_size'
    _op_type_ = OperandDef.RAND_DIRICHLET

    _alpha = TupleField('alpha')
    _size = TupleField('size', ValueType.int64)

    @property
    def alpha(self):
        return self._alpha

    @property
    def size(self):
        return self._size


class Exponential(Distribution):
    __slots__ = '_scale', '_size'
    _op_type_ = OperandDef.RAND_EXPONENTIAL

    _scale = AnyField('scale')
    _size = TupleField('size', ValueType.int64)

    @property
    def scale(self):
        return self._scale

    @property
    def size(self):
        return self._size


class F(Distribution):
    __slots__ = '_dfnum', '_dfden', '_size'
    _op_type_ = OperandDef.RAND_F

    _dfnum = AnyField('dfnum')
    _dfden = AnyField('dfden')
    _size = TupleField('size', ValueType.int64)

    @property
    def dfnum(self):
        return self._dfnum

    @property
    def dfden(self):
        return self._dfden

    @property
    def size(self):
        return self._size


class Gamma(Distribution):
    __slots__ = '_shape', '_scale', '_size'
    _op_type_ = OperandDef.RAND_GAMMA

    _shape = AnyField('shape')
    _scale = AnyField('scale')
    _size = TupleField('size', ValueType.int64)

    @property
    def shape(self):
        return self._shape

    @property
    def scale(self):
        return self._scale

    @property
    def size(self):
        return self._size


class Geometric(Distribution):
    __slots__ = '_p', '_size'
    _op_type_ = OperandDef.RAND_GEOMETRIC

    _p = AnyField('p')
    _size = TupleField('size', ValueType.int64)

    @property
    def p(self):
        return self._p

    @property
    def size(self):
        return self._size


class Gumbel(Distribution):
    __slots__ = '_loc', '_scale', '_size'
    _op_type_ = OperandDef.RAND_GUMBEL

    _loc = AnyField('loc')
    _scale = AnyField('scale')
    _size = TupleField('size', ValueType.int64)

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    @property
    def size(self):
        return self._size


class Hypergeometric(Distribution):
    __slots__ = '_ngood', '_nbad', '_nsample', '_size'
    _op_type_ = OperandDef.RAND_HYPERGEOMETRIC

    _ngood = AnyField('ngood')
    _nbad = AnyField('nbad')
    _nsample = AnyField('nsample')
    _size = TupleField('size', ValueType.int64)

    @property
    def ngood(self):
        return self._ngood

    @property
    def nbad(self):
        return self._nbad

    @property
    def nsample(self):
        return self._nsample

    @property
    def size(self):
        return self._size


class Laplace(Distribution):
    __slots__ = '_loc', '_scale', '_size'
    _op_type_ = OperandDef.RAND_LAPLACE

    _loc = AnyField('loc')
    _scale = AnyField('scale')
    _size = TupleField('size', ValueType.int64)

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    @property
    def size(self):
        return self._size


class Logistic(Distribution):
    __slots__ = '_loc', '_scale', '_size'
    _op_type_ = OperandDef.RAND_LOGISTIC

    _loc = AnyField('loc')
    _scale = AnyField('scale')
    _size = TupleField('size', ValueType.int64)

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    @property
    def size(self):
        return self._size


class Lognormal(Distribution):
    __slots__ = '_mean', '_sigma', '_size'
    _op_type_ = OperandDef.RAND_LOGNORMAL

    _mean = AnyField('mean')
    _sigma = AnyField('sigma')
    _size = TupleField('size', ValueType.int64)

    @property
    def mean(self):
        return self._mean

    @property
    def sigma(self):
        return self._sigma

    @property
    def size(self):
        return self._size


class Logseries(Distribution):
    __slots__ = '_p', '_size'
    _op_type_ = OperandDef.RAND_LOGSERIES

    _p = AnyField('p')
    _size = TupleField('size', ValueType.int64)

    @property
    def p(self):
        return self._p

    @property
    def size(self):
        return self._size


class Multinomial(Distribution):
    __slots__ = '_n', '_pvals', '_size'
    _op_type_ = OperandDef.RAND_MULTINOMIAL

    _n = Int64Field('n')
    _pvals = TupleField('pvals', ValueType.float64)
    _size = TupleField('size', ValueType.int64)

    @property
    def n(self):
        return self._n

    @property
    def pvals(self):
        return self._pvals

    @property
    def size(self):
        return self._size


class MultivariateNormal(Distribution):
    __slots__ = '_mean', '_cov', '_size', '_check_valid', '_tol'
    _op_type_ = OperandDef.RAND_MULTIVARIATE_NORMAL

    _mean = NDArrayField('mean')
    _cov = NDArrayField('cov')
    _size = TupleField('size', ValueType.int64)
    _check_valid = StringField('check_valid')
    _tol = Float64Field('tol')

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov

    @property
    def size(self):
        return self._size

    @property
    def check_valid(self):
        return self._check_valid

    @property
    def tol(self):
        return self._tol


class NegativeBinomial(Distribution):
    __slots__ = '_n', '_p', '_size'
    _op_type_ = OperandDef.RAND_NEGATIVE_BINOMIAL

    _n = AnyField('n')
    _p = AnyField('p')
    _size = TupleField('size', ValueType.int64)

    @property
    def n(self):
        return self._n

    @property
    def p(self):
        return self._p

    @property
    def size(self):
        return self._size


class NoncentralChisquare(Distribution):
    __slots__ = '_df', '_nonc', '_size'
    _op_type_ = OperandDef.RAND_NONCENTRAL_CHISQURE

    _df = AnyField('df')
    _nonc = AnyField('nonc')
    _size = TupleField('size', ValueType.int64)

    @property
    def df(self):
        return self._df

    @property
    def nonc(self):
        return self._nonc

    @property
    def size(self):
        return self._size


class NoncentralF(Distribution):
    __slots__ = '_dfnum', '_dfden', '_nonc', '_size'
    _op_type_ = OperandDef.RAND_NONCENTRAL_F

    _dfnum = AnyField('dfnum')
    _dfden = AnyField('dfden')
    _nonc = AnyField('nonc')
    _size = TupleField('size', ValueType.int64)

    @property
    def dfnum(self):
        return self._dfnum

    @property
    def dfden(self):
        return self._dfden

    @property
    def nonc(self):
        return self._nonc

    @property
    def size(self):
        return self._size


class Normal(Distribution):
    __slots__ = '_loc', '_scale', '_size'
    _op_type_ = OperandDef.RAND_NORMAL

    _loc = AnyField('loc')
    _scale = AnyField('scale')
    _size = TupleField('size', ValueType.int64)

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    @property
    def size(self):
        return self._size


class Pareto(Distribution):
    __slots__ = '_a', '_size'
    _op_type_ = OperandDef.RAND_PARETO

    _a = AnyField('a')
    _size = TupleField('size', ValueType.int64)

    @property
    def a(self):
        return self._a

    @property
    def size(self):
        return self._size


class Poisson(Distribution):
    __slots__ = '_lam', '_size'
    _op_type_ = OperandDef.RAND_POSSION

    _lam = AnyField('lam')
    _size = TupleField('size', ValueType.int64)

    @property
    def lam(self):
        return self._lam

    @property
    def size(self):
        return self._size


class RandomPower(Distribution):
    __slots__ = '_a', '_size'
    _op_type_ = OperandDef.RAND_POWER

    _a = AnyField('a')
    _size = TupleField('size', ValueType.int64)

    @property
    def a(self):
        return self._a

    @property
    def size(self):
        return self._size


class Rayleigh(Distribution):
    __slots__ = '_scale', '_size'
    _op_type_ = OperandDef.RAND_RAYLEIGH

    _scale = AnyField('scale')
    _size = TupleField('size', ValueType.int64)

    @property
    def scale(self):
        return self._scale

    @property
    def size(self):
        return self._size


class StandardCauchy(Distribution):
    __slots__ = '_size',
    _op_type_ = OperandDef.RAND_STANDARD_CAUCHY

    _size = TupleField('size', ValueType.int64)

    @property
    def size(self):
        return self._size


class StandardExponential(Distribution):
    __slots__ = '_size',
    _op_type_ = OperandDef.RAND_STANDARD_EXPONENTIAL

    _size = TupleField('size', ValueType.int64)

    @property
    def size(self):
        return self._size


class StandardGamma(Distribution):
    __slots__ = '_shape', '_size'
    _op_type_ = OperandDef.RAND_STANDARD_GAMMMA

    _shape = AnyField('shape')
    _size = TupleField('size', ValueType.int64)

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size


class StandardNormal(Distribution):
    __slots__ = '_size',
    _op_type_ = OperandDef.RAND_STANDARD_NORMAL

    _size = TupleField('size', ValueType.int64)

    @property
    def size(self):
        return self._size


class StandardT(Distribution):
    __slots__ = '_df', '_size'
    _op_type_ = OperandDef.RAND_STANDARD_T

    _df = AnyField('df')
    _size = TupleField('size', ValueType.int64)

    @property
    def df(self):
        return self._df

    @property
    def size(self):
        return self._size


class Triangular(Distribution):
    __slots__ = '_left', '_mode', '_right', '_size'
    _op_type_ = OperandDef.RAND_TRIANGULAR

    _left = AnyField('left')
    _mode = AnyField('mode')
    _right = AnyField('right')
    _size = TupleField('size', ValueType.int64)

    @property
    def left(self):
        return self._left

    @property
    def mode(self):
        return self._mode

    @property
    def right(self):
        return self._right

    @property
    def size(self):
        return self._size


class Uniform(Distribution):
    __slots__ = '_low', '_high', '_size'
    _op_type_ = OperandDef.RAND_UNIFORM

    _low = AnyField('low')
    _high = AnyField('high')
    _size = TupleField('size', ValueType.int64)

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def size(self):
        return self._size


class Vonmises(Distribution):
    __slots__ = '_mu', '_kappa', '_size'
    _op_type_ = OperandDef.RAND_VONMISES

    _mu = AnyField('mu')
    _kappa = AnyField('kappa')
    _size = TupleField('size', ValueType.int64)

    @property
    def mu(self):
        return self._mu

    @property
    def kappa(self):
        return self._kappa

    @property
    def size(self):
        return self._size


class Wald(Distribution):
    __slots__ = '_mean', '_scale', '_size'
    _op_type_ = OperandDef.RAND_WALD

    _mean = AnyField('mean')
    _scale = AnyField('scale')
    _size = TupleField('size', ValueType.int64)

    @property
    def mean(self):
        return self._mean

    @property
    def scale(self):
        return self._scale

    @property
    def size(self):
        return self._size


class Weibull(Distribution):
    __slots__ = '_a', '_size'
    _op_type_ = OperandDef.RAND_WEIBULL

    _a = AnyField('a')
    _size = TupleField('size', ValueType.int64)

    @property
    def a(self):
        return self._a

    @property
    def size(self):
        return self._size


class Zipf(Distribution):
    __slots__ = '_a', '_size'
    _op_type_ = OperandDef.RAND_ZIPF

    _a = AnyField('a')
    _size = TupleField('size', ValueType.int64)

    @property
    def a(self):
        return self._a

    @property
    def size(self):
        return self._size
