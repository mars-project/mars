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

import unittest

from mars.session import new_session

try:
    import sklearn

    from sklearn.datasets import load_iris
    from sklearn.utils import gen_batches
    from sklearn.utils._testing import assert_array_almost_equal, assert_allclose
except ImportError:
    raise
    sklearn = None

from mars.tests.core import ExecutorForTest
from mars import tensor as mt
if sklearn:
    from mars.learn.preprocessing import MinMaxScaler, minmax_scale


def assert_correct_incr(i, batch_start, batch_stop, n, chunk_size,
                        n_samples_seen):
    if batch_stop != n:
        assert (i + 1) * chunk_size == n_samples_seen
    else:
        assert (i * chunk_size + (batch_stop - batch_start) ==
                n_samples_seen)


def _check_dim_1axis(a):
    return mt.asarray(a).shape[0]


@unittest.skipIf(sklearn is None, 'scikit-learn not installed')
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.executor = self.session._sess._executor = \
            ExecutorForTest('numpy', storage=self.session._sess._context)

        rng = mt.random.RandomState(0)
        self.n_features = n_features = 30
        self.n_samples = n_samples = 1000
        offsets = rng.uniform(-1, 1, size=n_features)
        scales = rng.uniform(1, 10, size=n_features)
        self.X_2d = X_2d = rng.randn(n_samples, n_features) * scales + offsets
        self.X_1row = X_1row = X_2d[0, :].reshape(1, n_features)
        self.X_1col = X_1col = X_2d[:, 0].reshape(n_samples, 1)
        self.X_list_1row = X_1row.to_numpy().tolist()
        self.X_list_1col = X_1col.to_numpy().tolist()

        self.iris = mt.tensor(load_iris().data)

    def tearDown(self) -> None:
        self.session._sess._executor = self._old_executor

    def testMinMaxScalerPartialFit(self):
        # Test if partial_fit run over many batches of size 1 and 50
        # gives the same results as fit
        X = self.X_2d
        n = X.shape[0]

        for chunk_size in [50, n, n + 42]:
            # Test mean at the end of the process
            scaler_batch = MinMaxScaler().fit(X)

            scaler_incr = MinMaxScaler()
            for batch in gen_batches(self.n_samples, chunk_size):
                scaler_incr = scaler_incr.partial_fit(X[batch])

            assert_array_almost_equal(scaler_batch.data_min_,
                                      scaler_incr.data_min_)
            assert_array_almost_equal(scaler_batch.data_max_,
                                      scaler_incr.data_max_)
            assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
            assert_array_almost_equal(scaler_batch.data_range_,
                                      scaler_incr.data_range_)
            assert_array_almost_equal(scaler_batch.scale_, scaler_incr.scale_)
            assert_array_almost_equal(scaler_batch.min_, scaler_incr.min_)

            # Test std after 1 step
            batch0 = slice(0, chunk_size)
            scaler_batch = MinMaxScaler().fit(X[batch0])
            scaler_incr = MinMaxScaler().partial_fit(X[batch0])

            assert_array_almost_equal(scaler_batch.data_min_,
                                      scaler_incr.data_min_)
            assert_array_almost_equal(scaler_batch.data_max_,
                                      scaler_incr.data_max_)
            assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
            assert_array_almost_equal(scaler_batch.data_range_,
                                      scaler_incr.data_range_)
            assert_array_almost_equal(scaler_batch.scale_, scaler_incr.scale_)
            assert_array_almost_equal(scaler_batch.min_, scaler_incr.min_)

            # Test std until the end of partial fits, and
            _ = MinMaxScaler().fit(X)
            scaler_incr = MinMaxScaler()  # Clean estimator
            for i, batch in enumerate(gen_batches(self.n_samples, chunk_size)):
                scaler_incr = scaler_incr.partial_fit(X[batch])
                assert_correct_incr(i, batch_start=batch.start,
                                    batch_stop=batch.stop, n=n,
                                    chunk_size=chunk_size,
                                    n_samples_seen=scaler_incr.n_samples_seen_)

    def testMinMaxScalerIris(self):
        X = self.iris
        scaler = MinMaxScaler()
        # default params
        X_trans = scaler.fit_transform(X)
        assert_array_almost_equal(X_trans.min(axis=0), 0)
        assert_array_almost_equal(X_trans.max(axis=0), 1)
        X_trans_inv = scaler.inverse_transform(X_trans)
        assert_array_almost_equal(X, X_trans_inv)

        # not default params: min=1, max=2
        scaler = MinMaxScaler(feature_range=(1, 2))
        X_trans = scaler.fit_transform(X)
        assert_array_almost_equal(X_trans.min(axis=0), 1)
        assert_array_almost_equal(X_trans.max(axis=0), 2)
        X_trans_inv = scaler.inverse_transform(X_trans)
        assert_array_almost_equal(X, X_trans_inv)

        # min=-.5, max=.6
        scaler = MinMaxScaler(feature_range=(-.5, .6))
        X_trans = scaler.fit_transform(X)
        assert_array_almost_equal(X_trans.min(axis=0), -.5)
        assert_array_almost_equal(X_trans.max(axis=0), .6)
        X_trans_inv = scaler.inverse_transform(X_trans)
        assert_array_almost_equal(X, X_trans_inv)

        # raises on invalid range
        scaler = MinMaxScaler(feature_range=(2, 1))
        with self.assertRaises(ValueError):
            scaler.fit(X)

    def testMinMaxScalerZeroVarianceFeatures(self):
        # Check min max scaler on toy data with zero variance features
        X = [[0., 1., +0.5],
             [0., 1., -0.1],
             [0., 1., +1.1]]

        X_new = [[+0., 2., 0.5],
                 [-1., 1., 0.0],
                 [+0., 1., 1.5]]

        # default params
        scaler = MinMaxScaler()
        X_trans = scaler.fit_transform(X)
        X_expected_0_1 = [[0., 0., 0.5],
                          [0., 0., 0.0],
                          [0., 0., 1.0]]
        assert_array_almost_equal(X_trans, X_expected_0_1)
        X_trans_inv = scaler.inverse_transform(X_trans)
        assert_array_almost_equal(X, X_trans_inv)

        X_trans_new = scaler.transform(X_new)
        X_expected_0_1_new = [[+0., 1., 0.500],
                              [-1., 0., 0.083],
                              [+0., 0., 1.333]]
        assert_array_almost_equal(X_trans_new, X_expected_0_1_new, decimal=2)

        # not default params
        scaler = MinMaxScaler(feature_range=(1, 2))
        X_trans = scaler.fit_transform(X)
        X_expected_1_2 = [[1., 1., 1.5],
                          [1., 1., 1.0],
                          [1., 1., 2.0]]
        assert_array_almost_equal(X_trans, X_expected_1_2)

        # function interface
        X_trans = minmax_scale(X)
        assert_array_almost_equal(X_trans, X_expected_0_1)
        X_trans = minmax_scale(X, feature_range=(1, 2))
        assert_array_almost_equal(X_trans, X_expected_1_2)

    def testMinmaxScaleAxis1(self):
        X = self.iris
        X_trans = minmax_scale(X, axis=1)
        assert_array_almost_equal(mt.min(X_trans, axis=1), 0)
        assert_array_almost_equal(mt.max(X_trans, axis=1), 1)

    def testMinMaxScaler1d(self):
        # Test scaling of dataset along single axis
        for X in [self.X_1row, self.X_1col, self.X_list_1row, self.X_list_1row]:

            scaler = MinMaxScaler(copy=True)
            X_scaled = scaler.fit(X).transform(X)

            if isinstance(X, list):
                X = mt.array(X)  # cast only after scaling done

            if _check_dim_1axis(X) == 1:
                assert_array_almost_equal(X_scaled.min(axis=0),
                                          mt.zeros(self.n_features))
                assert_array_almost_equal(X_scaled.max(axis=0),
                                          mt.zeros(self.n_features))
            else:
                assert_array_almost_equal(X_scaled.min(axis=0), .0)
                assert_array_almost_equal(X_scaled.max(axis=0), 1.)
            assert scaler.n_samples_seen_ == X.shape[0]

            # check inverse transform
            X_scaled_back = scaler.inverse_transform(X_scaled)
            assert_array_almost_equal(X_scaled_back, X)

        # Constant feature
        X = mt.ones((5, 1))
        scaler = MinMaxScaler()
        X_scaled = scaler.fit(X).transform(X)
        assert X_scaled.min().to_numpy() >= 0.
        assert X_scaled.max().to_numpy() <= 1.
        assert scaler.n_samples_seen_ == X.shape[0]

        # Function interface
        X_1d = self.X_1row.ravel()
        min_ = X_1d.min()
        max_ = X_1d.max()
        assert_array_almost_equal((X_1d - min_) / (max_ - min_),
                                  minmax_scale(X_1d, copy=True))

    def testMinmaxScalerClip(self):
        for feature_range in [(0, 1), (-10, 10)]:
            # test behaviour of the paramter 'clip' in MinMaxScaler
            X = self.iris
            scaler = MinMaxScaler(feature_range=feature_range, clip=True).fit(X)
            X_min, X_max = mt.min(X, axis=0), mt.max(X, axis=0)
            X_test = [mt.r_[X_min[:2] - 10, X_max[2:] + 10]]
            X_transformed = scaler.transform(X_test)
            assert_allclose(
                X_transformed,
                [[feature_range[0], feature_range[0],
                  feature_range[1], feature_range[1]]])
