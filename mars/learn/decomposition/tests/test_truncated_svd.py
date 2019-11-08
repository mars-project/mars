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

import mars.tensor as mt

import numpy as np
try:
    import sklearn
    import scipy.sparse as sp
    from sklearn.utils import  check_random_state
    from sklearn.utils.testing import assert_array_almost_equal, assert_array_less

    from mars.learn.decomposition.truncated_svd import TruncatedSVD
except ImportError:
    sklearn = None


@unittest.skipIf(sklearn is None, 'scikit-learn not installed')
class Test(unittest.TestCase):
    def setUp(self):
        # Make an X that looks somewhat like a small tf-idf matrix.
        # XXX newer versions of SciPy >0.16 have scipy.sparse.rand for this.
        shape = 60, 55
        n_samples, n_features = shape
        rng = check_random_state(42)
        X = rng.randint(-100, 20, np.product(shape)).reshape(shape)
        X = sp.csr_matrix(np.maximum(X, 0), dtype=np.float64)
        X.data[:] = 1 + np.log(X.data)
        self.X = X
        self.Xdense = X.A
        self.n_samples = n_samples
        self.n_features = n_features

    def test_attributes(self):
        for n_components in (10, 25, 41):
            tsvd = TruncatedSVD(n_components).fit(self.X)
            self.assertEqual(tsvd.n_components, n_components)
            self.assertEqual(tsvd.components_.shape, (n_components, self.n_features))

    def test_too_many_components(self):
        for n_components in (self.n_features, self.n_features + 1):
            tsvd = TruncatedSVD(n_components=n_components, algorithm='randomized')
            with self.assertRaises(ValueError):
                tsvd.fit(self.X)

    def test_sparse_formats(self):
        tsvd = TruncatedSVD(n_components=11)
        Xtrans = tsvd.fit_transform(self.Xdense)
        self.assertEqual(Xtrans.shape, (self.n_samples, 11))
        Xtrans = tsvd.transform(self.Xdense)
        self.assertEqual(Xtrans.shape, (self.n_samples, 11))

    def test_inverse_transform(self):
        # We need a lot of components for the reconstruction to be "almost
        # equal" in all positions. XXX Test means or sums instead?
        tsvd = TruncatedSVD(n_components=52, random_state=42, algorithm='randomized')
        Xt = tsvd.fit_transform(self.X)
        Xinv = tsvd.inverse_transform(Xt)
        assert_array_almost_equal(Xinv.fetch(), self.Xdense, decimal=1)

    def test_integers(self):
        Xint = self.X.astype(np.int64)
        tsvd = TruncatedSVD(n_components=6)
        Xtrans = tsvd.fit_transform(Xint)
        self.assertEqual(Xtrans.shape, (self.n_samples, tsvd.n_components))

    def test_explained_variance(self):
        # Test sparse data
        svd_r_10_sp = TruncatedSVD(10, algorithm="randomized", random_state=42)
        svd_r_20_sp = TruncatedSVD(20, algorithm="randomized", random_state=42)
        X_trans_r_10_sp = svd_r_10_sp.fit_transform(self.X)
        X_trans_r_20_sp = svd_r_20_sp.fit_transform(self.X)

        # Test dense data
        svd_r_10_de = TruncatedSVD(10, algorithm="randomized", random_state=42)
        svd_r_20_de = TruncatedSVD(20, algorithm="randomized", random_state=42)
        X_trans_r_10_de = svd_r_10_de.fit_transform(self.X.toarray())
        X_trans_r_20_de = svd_r_20_de.fit_transform(self.X.toarray())

        # helper arrays for tests below
        svds = (svd_r_10_sp, svd_r_20_sp,  svd_r_10_de, svd_r_20_de)
        svds_trans = (
            (svd_r_10_sp, X_trans_r_10_sp),
            (svd_r_20_sp, X_trans_r_20_sp),
            (svd_r_10_de, X_trans_r_10_de),
            (svd_r_20_de, X_trans_r_20_de),
        )
        svds_10_v_20 = (
            (svd_r_10_sp, svd_r_20_sp),
            (svd_r_10_de, svd_r_20_de),
        )
        svds_sparse_v_dense = (
            (svd_r_10_sp, svd_r_10_de),
            (svd_r_20_sp, svd_r_20_de),
        )

        # Assert the 1st component is equal
        for svd_10, svd_20 in svds_10_v_20:
            assert_array_almost_equal(
                svd_10.explained_variance_ratio_.execute(),
                svd_20.explained_variance_ratio_[:10].execute(),
                decimal=4,
            )

        # Assert that 20 components has higher explained variance than 10
        for svd_10, svd_20 in svds_10_v_20:
            self.assertGreater(
                svd_20.explained_variance_ratio_.sum().execute(),
                svd_10.explained_variance_ratio_.sum().execute(),
            )

        # Assert that all the values are greater than 0
        for svd in svds:
            assert_array_less(0.0, svd.explained_variance_ratio_.execute())

        # Assert that total explained variance is less than 1
        for svd in svds:
            assert_array_less(svd.explained_variance_ratio_.sum().execute(), 1.0)

        # Compare sparse vs. dense
        for svd_sparse, svd_dense in svds_sparse_v_dense:
            assert_array_almost_equal(svd_sparse.explained_variance_ratio_.execute(),
                                      svd_dense.explained_variance_ratio_.execute())

        # Test that explained_variance is correct
        for svd, transformed in svds_trans:
            total_variance = mt.var(self.X.toarray(), axis=0).sum().execute()
            variances = mt.var(transformed, axis=0)
            true_explained_variance_ratio = variances / total_variance

            assert_array_almost_equal(
                svd.explained_variance_ratio_.execute(),
                true_explained_variance_ratio.execute(),
            )

    def test_singular_values(self):
        # Check that the TruncatedSVD output has the correct singular values

        # Set the singular values and see what we get back
        rng = np.random.RandomState(0)
        n_samples = 100
        n_features = 110

        X = rng.randn(n_samples, n_features)

        rpca = TruncatedSVD(n_components=3, algorithm='randomized',
                            random_state=rng)
        X_rpca = rpca.fit_transform(X)

        X_rpca /= mt.sqrt(mt.sum(X_rpca**2.0, axis=0))
        X_rpca[:, 0] *= 3.142
        X_rpca[:, 1] *= 2.718

        X_hat_rpca = mt.dot(X_rpca, rpca.components_)
        rpca.fit(X_hat_rpca)
        assert_array_almost_equal(rpca.singular_values_.execute(), [3.142, 2.718, 1.0], 14)
