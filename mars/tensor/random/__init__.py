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

from .core import RandomState, _random_state
from .rand import rand, TensorRand
from .randn import randn, TensorRandn
from .randint import randint, TensorRandint
from .random_integers import random_integers, TensorRandomIntegers
from .random_sample import random_sample, TensorRandomSample
from .choice import choice, TensorChoice
from .bytes import bytes
from .beta import beta, TensorBeta
from .binomial import binomial, TensorBinomial
from .chisquare import chisquare, TensorChisquare
from .dirichlet import dirichlet, TensorDirichlet
from .exponential import exponential, TensorExponential
from .f import f, TensorF
from .gamma import gamma, TensorGamma
from .geometric import geometric, TensorGeometric
from .gumbel import gumbel, TensorGumbel
from .hypergeometric import hypergeometric, TensorHypergeometric
from .laplace import laplace, TensorLaplace
from .logistic import logistic, TensorLogistic
from .lognormal import lognormal, TensorLognormal
from .logseries import logseries, TensorLogseries
from .multinomial import multinomial, TensorMultinomial
from .multivariate_normal import multivariate_normal, TensorMultivariateNormal
from .negative_binomial import negative_binomial, TensorNegativeBinomial
from .noncentral_chisquare import noncentral_chisquare, TensorNoncentralChisquare
from .noncentral_f import noncentral_f, TensorNoncentralF
from .normal import normal, TensorNormal
from .pareto import pareto, TensorPareto
from .poisson import poisson, TensorPoisson
from .power import power, TensorRandomPower
from .rayleigh import rayleigh, TensorRayleigh
from .standard_cauchy import standard_cauchy, TensorStandardCauchy
from .standard_exponential import standard_exponential, TensorStandardExponential
from .standard_gamma import standard_gamma, TensorStandardGamma
from .standard_normal import standard_normal, TensorStandardNormal
from .standard_t import standard_t, TensorStandardT
from .triangular import triangular, TensorTriangular
from .uniform import uniform, TensorUniform
from .vonmises import vonmises, TensorVonmises
from .wald import wald, TensorWald
from .weibull import weibull, TensorWeibull
from .zipf import zipf, TensorZipf
from .permutation import permutation, TensorPermutation
from .shuffle import shuffle


def _install():
    setattr(RandomState, 'rand', rand)
    setattr(RandomState, 'randn', randn)
    setattr(RandomState, 'randint', randint)
    setattr(RandomState, 'random_integers', random_integers)
    setattr(RandomState, 'random_sample', random_sample)
    setattr(RandomState, 'ranf', random_sample)
    setattr(RandomState, 'random', random_sample)
    setattr(RandomState, 'sample', random_sample)
    setattr(RandomState, 'choice', choice)
    setattr(RandomState, 'bytes', bytes)
    setattr(RandomState, 'beta', beta)
    setattr(RandomState, 'binomial', binomial)
    setattr(RandomState, 'chisquare', chisquare)
    setattr(RandomState, 'dirichlet', dirichlet)
    setattr(RandomState, 'exponential', exponential)
    setattr(RandomState, 'f', f)
    setattr(RandomState, 'gamma', gamma)
    setattr(RandomState, 'geometric', geometric)
    setattr(RandomState, 'gumbel', gumbel)
    setattr(RandomState, 'hypergeometric', hypergeometric)
    setattr(RandomState, 'laplace', laplace)
    setattr(RandomState, 'logistic', logistic)
    setattr(RandomState, 'lognormal', lognormal)
    setattr(RandomState, 'logseries', logseries)
    setattr(RandomState, 'multinomial', multinomial)
    setattr(RandomState, 'multivariate_normal', multivariate_normal)
    setattr(RandomState, 'negative_binomial', negative_binomial)
    setattr(RandomState, 'noncentral_chisquare', noncentral_chisquare)
    setattr(RandomState, 'noncentral_f', noncentral_f)
    setattr(RandomState, 'normal', normal)
    setattr(RandomState, 'pareto', pareto)
    setattr(RandomState, 'poisson', poisson)
    setattr(RandomState, 'power', power)
    setattr(RandomState, 'rayleigh', rayleigh)
    setattr(RandomState, 'standard_cauchy', standard_cauchy)
    setattr(RandomState, 'standard_exponential', standard_exponential)
    setattr(RandomState, 'standard_gamma', standard_gamma)
    setattr(RandomState, 'standard_normal', standard_normal)
    setattr(RandomState, 'standard_t', standard_t)
    setattr(RandomState, 'triangular', triangular)
    setattr(RandomState, 'uniform', uniform)
    setattr(RandomState, 'vonmises', vonmises)
    setattr(RandomState, 'wald', wald)
    setattr(RandomState, 'weibull', weibull)
    setattr(RandomState, 'zipf', zipf)
    setattr(RandomState, 'permutation', permutation)
    setattr(RandomState, 'shuffle', shuffle)


_install()
del _install


seed = _random_state.seed

rand = _random_state.rand
randn = _random_state.randn
randint = _random_state.randint
random_integers = _random_state.random_integers
random_sample = _random_state.random_sample
random = _random_state.random
ranf = _random_state.ranf
sample = _random_state.sample
choice = _random_state.choice
bytes = _random_state.bytes

permutation = _random_state.permutation
shuffle = _random_state.shuffle

beta = _random_state.beta
binomial = _random_state.binomial
chisquare = _random_state.chisquare
dirichlet = _random_state.dirichlet
exponential = _random_state.exponential
f = _random_state.f
gamma = _random_state.gamma
geometric = _random_state.geometric
gumbel = _random_state.gumbel
hypergeometric = _random_state.hypergeometric
laplace = _random_state.laplace
logistic = _random_state.logistic
lognormal = _random_state.lognormal
logseries = _random_state.logseries
multinomial = _random_state.multinomial
multivariate_normal = _random_state.multivariate_normal
negative_binomial = _random_state.negative_binomial
noncentral_chisquare = _random_state.noncentral_chisquare
noncentral_f = _random_state.noncentral_f
normal = _random_state.normal
pareto = _random_state.pareto
poisson = _random_state.poisson
power = _random_state.power
rayleigh = _random_state.rayleigh
standard_cauchy = _random_state.standard_cauchy
standard_exponential = _random_state.standard_exponential
standard_gamma = _random_state.standard_gamma
standard_normal = _random_state.standard_normal
standard_t = _random_state.standard_t
triangular = _random_state.triangular
uniform = _random_state.uniform
vonmises = _random_state.vonmises
wald = _random_state.wald
weibull = _random_state.weibull
zipf = _random_state.zipf
