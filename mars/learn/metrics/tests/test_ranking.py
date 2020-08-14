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

import re
import unittest

import numpy as np
try:
    import sklearn
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics.tests.test_ranking import make_prediction, _auc
    from sklearn.exceptions import UndefinedMetricWarning
    from sklearn.utils import check_random_state
    from sklearn.utils._testing import assert_warns
except ImportError:  # pragma: no cover
    sklearn = None
import pytest

from mars import tensor as mt
from mars.learn.metrics import roc_curve, auc
from mars.tests.core import TestBase


@unittest.skipIf(sklearn is None, 'scikit-learn not installed')
class Test(TestBase):
    def testRocCurve(self):
        for drop in [True, False]:
            # Test Area under Receiver Operating Characteristic (ROC) curve
            y_true, _, probas_pred = make_prediction(binary=True)
            expected_auc = _auc(y_true, probas_pred)

            fpr, tpr, thresholds = roc_curve(y_true, probas_pred,
                                             drop_intermediate=drop).execute().fetch()
            roc_auc = auc(fpr, tpr).to_numpy()
            np.testing.assert_array_almost_equal(roc_auc, expected_auc, decimal=2)
            np.testing.assert_almost_equal(roc_auc, roc_auc_score(y_true, probas_pred))
            assert fpr.shape == tpr.shape
            assert fpr.shape == thresholds.shape

    def testRocCurveEndPoints(self):
        # Make sure that roc_curve returns a curve start at 0 and ending and
        # 1 even in corner cases
        rng = np.random.RandomState(0)
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = rng.randint(3, size=100)
        fpr, tpr, thr = roc_curve(y_true, y_pred, drop_intermediate=True).fetch()
        assert fpr[0] == 0
        assert fpr[-1] == 1
        assert fpr.shape == tpr.shape
        assert fpr.shape == thr.shape

    def testRocReturnsConsistency(self):
        # Test whether the returned threshold matches up with tpr
        # make small toy dataset
        y_true, _, probas_pred = make_prediction(binary=True)
        fpr, tpr, thresholds = roc_curve(y_true, probas_pred).fetch()

        # use the given thresholds to determine the tpr
        tpr_correct = []
        for t in thresholds:
            tp = np.sum((probas_pred >= t) & y_true)
            p = np.sum(y_true)
            tpr_correct.append(1.0 * tp / p)

        # compare tpr and tpr_correct to see if the thresholds' order was correct
        np.testing.assert_array_almost_equal(tpr, tpr_correct, decimal=2)
        assert fpr.shape == tpr.shape
        assert fpr.shape == thresholds.shape

    def testRocCurveMulti(self):
        # roc_curve not applicable for multi-class problems
        y_true, _, probas_pred = make_prediction(binary=False)

        with self.assertRaises(ValueError):
            roc_curve(y_true, probas_pred)

    def testRocCurveConfidence(self):
        # roc_curve for confidence scores
        y_true, _, probas_pred = make_prediction(binary=True)

        fpr, tpr, thresholds = roc_curve(y_true, probas_pred - 0.5)
        roc_auc = auc(fpr, tpr).fetch()
        np.testing.assert_array_almost_equal(roc_auc, 0.90, decimal=2)
        assert fpr.shape == tpr.shape
        assert fpr.shape == thresholds.shape

    def testRocCurveHard(self):
        # roc_curve for hard decisions
        y_true, pred, probas_pred = make_prediction(binary=True)

        # always predict one
        trivial_pred = np.ones(y_true.shape)
        fpr, tpr, thresholds = roc_curve(y_true, trivial_pred)
        roc_auc = auc(fpr, tpr).fetch()
        np.testing.assert_array_almost_equal(roc_auc, 0.50, decimal=2)
        assert fpr.shape == tpr.shape
        assert fpr.shape == thresholds.shape

        # always predict zero
        trivial_pred = np.zeros(y_true.shape)
        fpr, tpr, thresholds = roc_curve(y_true, trivial_pred)
        roc_auc = auc(fpr, tpr).fetch()
        np.testing.assert_array_almost_equal(roc_auc, 0.50, decimal=2)
        assert fpr.shape == tpr.shape
        assert fpr.shape == thresholds.shape

        # hard decisions
        fpr, tpr, thresholds = roc_curve(y_true, pred)
        roc_auc = auc(fpr, tpr).fetch()
        np.testing.assert_array_almost_equal(roc_auc, 0.78, decimal=2)
        assert fpr.shape == tpr.shape
        assert fpr.shape == thresholds.shape

    def testRocCurveOneLabel(self):
        y_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        y_pred = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        # assert there are warnings
        w = UndefinedMetricWarning
        fpr, tpr, thresholds = assert_warns(w, roc_curve, y_true, y_pred)
        # all true labels, all fpr should be nan
        np.testing.assert_array_equal(fpr.fetch(), np.full(len(thresholds), np.nan))
        assert fpr.shape == tpr.shape
        assert fpr.shape == thresholds.shape

        # assert there are warnings
        fpr, tpr, thresholds = assert_warns(w, roc_curve,
                                            [1 - x for x in y_true],
                                            y_pred)
        # all negative labels, all tpr should be nan
        np.testing.assert_array_equal(tpr.fetch(), np.full(len(thresholds), np.nan))
        assert fpr.shape == tpr.shape
        assert fpr.shape == thresholds.shape

    def testRocCurveDropIntermediate(self):
        # Test that drop_intermediate drops the correct thresholds
        y_true = [0, 0, 0, 0, 1, 1]
        y_score = [0., 0.2, 0.5, 0.6, 0.7, 1.0]
        tpr, fpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
        np.testing.assert_array_almost_equal(thresholds.fetch(), [2., 1., 0.7, 0.])

        # Test dropping thresholds with repeating scores
        y_true = [0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 1]
        y_score = [0., 0.1, 0.6, 0.6, 0.7, 0.8, 0.9,
                   0.6, 0.7, 0.8, 0.9, 0.9, 1.0]
        tpr, fpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
        np.testing.assert_array_almost_equal(thresholds.fetch(),
                                             [2.0, 1.0, 0.9, 0.7, 0.6, 0.])

    def testRocCurveFprTprIncreasing(self):
        # Ensure that fpr and tpr returned by roc_curve are increasing.
        # Construct an edge case with float y_score and sample_weight
        # when some adjacent values of fpr and tpr are actually the same.
        y_true = [0, 0, 1, 1, 1]
        y_score = [0.1, 0.7, 0.3, 0.4, 0.5]
        sample_weight = np.repeat(0.2, 5)
        fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
        assert ((mt.diff(fpr) < 0).sum() == 0).to_numpy()
        assert ((mt.diff(tpr) < 0).sum() == 0).to_numpy()

    def testAuc(self):
        # Test Area Under Curve (AUC) computation
        x = [0, 1]
        y = [0, 1]
        np.testing.assert_array_almost_equal(auc(x, y).fetch(), 0.5)
        x = [1, 0]
        y = [0, 1]
        np.testing.assert_array_almost_equal(auc(x, y).fetch(), 0.5)
        x = [1, 0, 0]
        y = [0, 1, 1]
        np.testing.assert_array_almost_equal(auc(x, y).fetch(), 0.5)
        x = [0, 1]
        y = [1, 1]
        np.testing.assert_array_almost_equal(auc(x, y).fetch(), 1)
        x = [0, 0.5, 1]
        y = [0, 0.5, 1]
        np.testing.assert_array_almost_equal(auc(x, y).fetch(), 0.5)

    def testAucErrors(self):
        # Incompatible shapes
        with self.assertRaises(ValueError):
            auc([0.0, 0.5, 1.0], [0.1, 0.2])

        # Too few x values
        with self.assertRaises(ValueError):
            auc([0.0], [0.1])

        # x is not in order
        x = [2, 1, 3, 4]
        y = [5, 6, 7, 8]
        error_message = f"x is neither increasing nor decreasing : {np.array(x)}"
        with pytest.raises(ValueError, match=re.escape(error_message)):
            auc(x, y)

    def testBinaryClfCurveMulticlassError(self):
        rng = check_random_state(404)
        y_true = rng.randint(0, 3, size=10)
        y_pred = rng.rand(10)
        msg = "multiclass format is not supported"

        with pytest.raises(ValueError, match=msg):
            roc_curve(y_true, y_pred)
