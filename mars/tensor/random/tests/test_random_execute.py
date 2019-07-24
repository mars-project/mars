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

import unittest

import numpy as np

from mars.executor import Executor
from mars.tensor.random.core import State
from mars.tensor.datasource import tensor as from_ndarray
from mars.lib.sparse.core import issparse
from mars import tensor


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

    def testRandExecution(self):
        arr = tensor.random.rand(10, 20, chunk_size=3, dtype='f4')
        res = self.executor.execute_tensor(arr, concat=True)[0]
        self.assertEqual(res.shape, (10, 20))
        self.assertTrue(np.all(res < 1))
        self.assertTrue(np.all(res > 0))
        self.assertEqual(res.dtype, np.float32)

    def testRandnExecution(self):
        arr = tensor.random.randn(10, 20, chunk_size=3)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (10, 20))

        arr = tensor.random.randn(10, 20, chunk_size=5).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).randn(5, 5)))

    def testRandintExecution(self):
        size_executor = Executor(sync_provider_type=Executor.SyncProviderType.MOCK)

        arr = tensor.random.randint(0, 2, size=(10, 30), chunk_size=3)
        size_res = size_executor.execute_tensor(arr, mock=True)
        self.assertEqual(arr.nbytes, sum(tp[0] for tp in size_res))

        res = self.executor.execute_tensor(arr, concat=True)[0]
        self.assertEqual(res.shape, (10, 30))
        self.assertTrue(np.all(res >= 0))
        self.assertTrue(np.all(res < 2))

    def testRandomIntegersExecution(self):
        arr = tensor.random.random_integers(0, 10, size=(10, 20), chunk_size=3)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (10, 20))

        arr = tensor.random.random_integers(0, 10, size=(10, 20), chunk_size=5).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            np.testing.assert_equal(res, np.random.RandomState(0).random_integers(0, 10, size=(5, 5)))

    def testRandomSampleExecution(self):
        arr = tensor.random.random_sample(size=(10, 20), chunk_size=3)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (10, 20))

        arr = tensor.random.random_sample(size=(10, 20), chunk_size=5).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).random_sample(size=(5, 5))))

    def testRandomExecution(self):
        arr = tensor.random.random(size=(10, 20), chunk_size=3)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (10, 20))

        arr = tensor.random.random(size=(10, 20), chunk_size=5).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).random_sample(size=(5, 5))))

    def testRandfExecution(self):
        arr = tensor.random.ranf(size=(10, 20), chunk_size=3)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (10, 20))

        arr = tensor.random.ranf(size=(10, 20), chunk_size=5).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).random_sample(size=(5, 5))))

    def testSampleExecution(self):
        arr = tensor.random.sample(size=(10, 20), chunk_size=3)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (10, 20))

        arr = tensor.random.sample(size=(10, 20), chunk_size=5).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).random_sample(size=(5, 5))))

    def testChoiceExecution(self):
        arr = tensor.random.choice(5, size=3, chunk_size=1)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (3,))

        arr = tensor.random.choice(5, size=(15,), chunk_size=5).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).choice(5, size=(5,))))

        arr = tensor.random.choice([1, 4, 9], size=3, chunk_size=1)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (3,))

        arr = tensor.random.choice([1, 4, 9], size=(15,), chunk_size=5).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).choice([1, 4, 9], size=(5,))))

        with self.assertRaises(ValueError):
            tensor.random.choice([1, 3, 4], size=5, replace=False, chunk_size=2)

        arr = tensor.random.choice([1, 4, 9], size=3, replace=False, chunk_size=1)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (3,))

        arr = tensor.random.choice([1, 4, 9], size=(3,), replace=False, chunk_size=1).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(
                np.array_equal(res, np.random.RandomState(0).choice([1, 4, 9], size=(1,), replace=False)))

        arr = tensor.random.choice([1, 4, 9], size=3, p=[.2, .5, .3], chunk_size=1)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (3,))

        arr = tensor.random.choice([1, 4, 9], size=(15,), chunk_size=5, p=[.2, .5, .3]).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(
                np.array_equal(res, np.random.RandomState(0).choice([1, 4, 9], size=(5,),
                                                                    p=[.2, .5, .3])))

    def testSparseRandintExecution(self):
        size_executor = Executor(sync_provider_type=Executor.SyncProviderType.MOCK)

        arr = tensor.random.randint(1, 2, size=(30, 50), density=.1, chunk_size=10, dtype='f4')
        size_res = size_executor.execute_tensor(arr, mock=True)
        self.assertAlmostEqual(arr.nbytes * 0.1, sum(tp[0] for tp in size_res))

        res = self.executor.execute_tensor(arr, concat=True)[0]
        self.assertTrue(issparse(res))
        self.assertEqual(res.shape, (30, 50))
        self.assertTrue(np.all(res.data >= 1))
        self.assertTrue(np.all(res.data < 2))
        self.assertAlmostEqual((res >= 1).toarray().sum(), 30 * 50 * .1, delta=20)

    def testBetaExecute(self):
        arr = tensor.random.beta(1, 2, chunk_size=2).tiles()
        arr.chunks[0].op._state = State(np.random.RandomState(0))

        self.assertEqual(self.executor.execute_tensor(arr)[0], np.random.RandomState(0).beta(1, 2))

        arr = tensor.random.beta([1, 2], [3, 4], chunk_size=2).tiles()
        arr.chunks[0].op._state = State(np.random.RandomState(0))

        self.assertTrue(np.array_equal(self.executor.execute_tensor(arr)[0],
                                       np.random.RandomState(0).beta([1, 2], [3, 4])))

        arr = tensor.random.beta([[2, 3]], from_ndarray([[4, 6], [5, 2]], chunk_size=2),
                                 chunk_size=1, size=(3, 2, 2)).tiles()
        for c in arr.chunks:
            c.op._state = State(np.random.RandomState(0))

        res = self.executor.execute_tensor(arr, concat=True)[0]

        self.assertEqual(res[0, 0, 0], np.random.RandomState(0).beta(2, 4))
        self.assertEqual(res[0, 0, 1], np.random.RandomState(0).beta(3, 6))
        self.assertEqual(res[0, 1, 0], np.random.RandomState(0).beta(2, 5))
        self.assertEqual(res[0, 1, 1], np.random.RandomState(0).beta(3, 2))

        arr = tensor.random.RandomState(0).beta([[3, 4]], [[1], [2]], chunk_size=1)
        tensor.random.seed(0)
        arr2 = tensor.random.beta([[3, 4]], [[1], [2]], chunk_size=1)

        self.assertTrue(np.array_equal(self.executor.execute_tensor(arr, concat=True)[0],
                                       self.executor.execute_tensor(arr2, concat=True)[0]))

    def testBinomialExecute(self):
        arr = tensor.random.binomial(10, .5, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.binomial(10, .5, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).binomial(10, .5, 10)))

    def testChisquareExecute(self):
        arr = tensor.random.chisquare(2, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.chisquare(2, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).chisquare(2, 10)))

    def testDirichletExecute(self):
        arr = tensor.random.dirichlet((10, 5, 3), 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100, 3))

        arr = tensor.random.dirichlet((10, 5, 3), 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).dirichlet((10, 5, 3), 10)))

    def testExponentialExecute(self):
        arr = tensor.random.exponential(1.0, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.exponential(1.0, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).exponential(1.0, 10)))

    def testFExecute(self):
        arr = tensor.random.f(1.0, 2.0, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.f(1.0, 2.0, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).f(1.0, 2.0, 10)))

    def testGammaExecute(self):
        arr = tensor.random.gamma(1.0, 2.0, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.gamma(1.0, 2.0, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).gamma(1.0, 2.0, 10)))

    def testGeometricExecution(self):
        arr = tensor.random.geometric(1.0, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.geometric(1.0, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).geometric(1.0, 10)))

    def testGumbelExecution(self):
        arr = tensor.random.gumbel(.5, 1.0, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.gumbel(.5, 1.0, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).gumbel(.5, 1.0, 10)))

    def testHypergeometricExecution(self):
        arr = tensor.random.hypergeometric(10, 20, 15, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.hypergeometric(10, 20, 15, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).hypergeometric(10, 20, 15, 10)))

    def testLaplaceExecution(self):
        arr = tensor.random.laplace(.5, 1.0, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.laplace(.5, 1.0, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).laplace(.5, 1.0, 10)))

    def testLogisticExecution(self):
        arr = tensor.random.logistic(.5, 1.0, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.logistic(.5, 1.0, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            np.testing.assert_equal(res, np.random.RandomState(0).logistic(.5, 1.0, 10))

    def testLognormalExecution(self):
        arr = tensor.random.lognormal(.5, 1.0, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.lognormal(.5, 1.0, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).lognormal(.5, 1.0, 10)))

    def testLogseriesExecution(self):
        arr = tensor.random.logseries(.5, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.logseries(.5, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).logseries(.5, 10)))

    def testMultinomialExecution(self):
        arr = tensor.random.multinomial(10, [.2, .5, .3], 100, chunk_size=10)
        self.assertEqual(arr.shape, (100, 3))
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100, 3))

        arr = tensor.random.multinomial(10, [.2, .5, .3], 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).multinomial(10, [.2, .5, .3], 10)))

    def testMultivariateNormalExecution(self):
        arr = tensor.random.multivariate_normal([1, 2], [[1, 0], [0, 1]], 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100, 2))

        arr = tensor.random.multivariate_normal([1, 2], [[1, 0], [0, 1]], 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).multivariate_normal(
                [1, 2], [[1, 0], [0, 1]], 10)))

    def testNegativeBinomialExecution(self):
        arr = tensor.random.negative_binomial(5, 1.0, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.negative_binomial(5, 1.0, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).negative_binomial(5, 1.0, 10)))

    def testNoncentralChisquareExecution(self):
        arr = tensor.random.noncentral_chisquare(.5, 1.0, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.noncentral_chisquare(.5, 1.0, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).noncentral_chisquare(.5, 1.0, 10)))

    def testNoncentralFExecution(self):
        arr = tensor.random.noncentral_f(1.5, 1.0, 1.1, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.noncentral_f(1.5, 1.0, 1.1, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).noncentral_f(1.5, 1.0, 1.1, 10)))

    def testNormalExecute(self):
        arr = tensor.random.normal(10, 1.0, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.normal(10, 1.0, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).normal(10, 1.0, 10)))

    def testParetoExecute(self):
        arr = tensor.random.pareto(1.0, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.pareto(1.0, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).pareto(1.0, 10)))

    def testPoissonExecute(self):
        arr = tensor.random.poisson(1.0, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.poisson(1.0, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).poisson(1.0, 10)))

    def testPowerExecute(self):
        arr = tensor.random.power(1.0, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.power(1.0, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).power(1.0, 10)))

    def testRayleighExecute(self):
        arr = tensor.random.rayleigh(1.0, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.rayleigh(1.0, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).rayleigh(1.0, 10)))

    def testStandardCauchyExecute(self):
        arr = tensor.random.standard_cauchy(100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.standard_cauchy(100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).standard_cauchy(10)))

    def testStandardExponentialExecute(self):
        arr = tensor.random.standard_exponential(100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.standard_exponential(100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).standard_exponential(10)))

    def testStandardGammaExecute(self):
        arr = tensor.random.standard_gamma(.1, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.standard_gamma(.1, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).standard_gamma(.1, 10)))

    def testStandardNormalExecute(self):
        arr = tensor.random.standard_normal(100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.standard_normal(100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).standard_normal(10)))

    def testStandardTExecute(self):
        arr = tensor.random.standard_t(.1, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.standard_t(.1, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).standard_t(.1, 10)))

    def testTriangularExecute(self):
        arr = tensor.random.triangular(.1, .2, .3, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.triangular(.1, .2, .3, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).triangular(.1, .2, .3, 10)))

    def testUniformExecute(self):
        arr = tensor.random.uniform(.1, .2, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.uniform(.1, .2, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).uniform(.1, .2, 10)))

    def testVonmisesExecute(self):
        arr = tensor.random.vonmises(.1, .2, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.vonmises(.1, .2, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).vonmises(.1, .2, 10)))

    def testWaldExecute(self):
        arr = tensor.random.wald(.1, .2, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.wald(.1, .2, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).wald(.1, .2, 10)))

    def testWeibullExecute(self):
        arr = tensor.random.weibull(.1, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.weibull(.1, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).weibull(.1, 10)))

    def testZipfExecute(self):
        arr = tensor.random.zipf(1.1, 100, chunk_size=10)
        self.assertEqual(self.executor.execute_tensor(arr, concat=True)[0].shape, (100,))

        arr = tensor.random.zipf(1.1, 100, chunk_size=10).tiles()
        for chunk in arr.chunks:
            chunk.op._state = State(np.random.RandomState(0))

        for res in self.executor.execute_tensor(arr):
            self.assertTrue(np.array_equal(res, np.random.RandomState(0).zipf(1.1, 10)))
