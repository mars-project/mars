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
import unittest

import numpy as np
import pandas as pd
import pytest
try:
    import scipy.sparse as sps
except ImportError:  # pragma: no cover
    sps = None

from mars import dataframe as md
from mars import tensor as mt
from mars.dataframe.core import DATAFRAME_TYPE
from mars.learn.model_selection import train_test_split
from mars.lib.sparse import SparseNDArray
from mars.tests.core import TestBase


class Test(TestBase):
    def setUp(self) -> None:
        self.ctx, self.executor = self._create_test_context()
        self.ctx.__enter__()

    def tearDown(self) -> None:
        self.ctx.__exit__(None, None, None)

    def testTrainTestSplitErrors(self):
        self.assertRaises(ValueError, train_test_split)

        self.assertRaises(ValueError, train_test_split, range(3), train_size=1.1)

        self.assertRaises(ValueError, train_test_split, range(3), test_size=0.6,
                          train_size=0.6)
        self.assertRaises(ValueError, train_test_split, range(3),
                          test_size=np.float32(0.6), train_size=np.float32(0.6))
        self.assertRaises(ValueError, train_test_split, range(3),
                          test_size="wrong_type")
        self.assertRaises(ValueError, train_test_split, range(3), test_size=2,
                          train_size=4)
        self.assertRaises(TypeError, train_test_split, range(3),
                          some_argument=1.1)
        self.assertRaises(ValueError, train_test_split, range(3), range(42))
        self.assertRaises(ValueError, train_test_split, range(10),
                          shuffle=False, stratify=True)

        with pytest.raises(ValueError,
                           match=r'train_size=11 should be either positive and '
                                 r'smaller than the number of samples 10 or a '
                                 r'float in the \(0, 1\) range'):
            train_test_split(range(10), train_size=11, test_size=1)

    def testTrainTestSplitInvalidSizes1(self):
        for train_size, test_size in [
                (1.2, 0.8),
                (1., 0.8),
                (0.0, 0.8),
                (-.2, 0.8),
                (0.8, 1.2),
                (0.8, 1.),
                (0.8, 0.),
                (0.8, -.2)]:
            with pytest.raises(ValueError,
                               match=r'should be .* in the \(0, 1\) range'):
                train_test_split(range(10), train_size=train_size, test_size=test_size)

    def testTrainTestSplitInvalidSizes2(self):
        for train_size, test_size in [
                (-10, 0.8),
                (0, 0.8),
                (11, 0.8),
                (0.8, -10),
                (0.8, 0),
                (0.8, 11)]:
            with pytest.raises(ValueError,
                               match=r'should be .* in the \(0, 1\) range'):
                train_test_split(range(10), train_size=train_size, test_size=test_size)

    def testTrainTestSplit(self):
        X = np.arange(100).reshape((10, 10))
        y = np.arange(10)

        # simple test
        split = train_test_split(X, y, test_size=None, train_size=.5)
        X_train, X_test, y_train, y_test = split
        assert len(y_test) == len(y_train)
        # test correspondence of X and y
        np.testing.assert_array_equal(X_train[:, 0], y_train * 10)
        np.testing.assert_array_equal(X_test[:, 0], y_test * 10)

        # allow nd-arrays
        X_4d = np.arange(10 * 5 * 3 * 2).reshape(10, 5, 3, 2)
        y_3d = np.arange(10 * 7 * 11).reshape(10, 7, 11)
        split = train_test_split(X_4d, y_3d)
        assert split[0].shape == (7, 5, 3, 2)
        assert split[1].shape == (3, 5, 3, 2)
        assert split[2].shape == (7, 7, 11)
        assert split[3].shape == (3, 7, 11)

        # test unshuffled split
        y = np.arange(10)
        for test_size in [2, 0.2]:
            train, test = train_test_split(y, shuffle=False, test_size=test_size)
            np.testing.assert_array_equal(test, [8, 9])
            np.testing.assert_array_equal(train, [0, 1, 2, 3, 4, 5, 6, 7])

    def testTrainTestSplitDataFrame(self):
        X = np.ones(10)
        types = [pd.DataFrame, md.DataFrame]
        for InputFeatureType in types:
            # X dataframe
            X_df = InputFeatureType(X)
            X_train, X_test = train_test_split(X_df)
            assert isinstance(X_train, DATAFRAME_TYPE)
            assert isinstance(X_test, DATAFRAME_TYPE)

    @unittest.skipIf(sps is None, 'scipy not installed')
    def testTrainTestSplitSparse(self):
        # check that train_test_split converts scipy sparse matrices
        # to csr, as stated in the documentation
        X = np.arange(100).reshape((10, 10))
        sparse_types = [sps.csr_matrix, sps.csc_matrix, sps.coo_matrix]
        for InputFeatureType in sparse_types:
            X_s = InputFeatureType(X)
            for x in (X_s, mt.tensor(X_s, chunk_size=(2, 5))):
                X_train, X_test = train_test_split(x)
                assert isinstance(X_train.fetch(), SparseNDArray)
                assert isinstance(X_test.fetch(), SparseNDArray)

    def testTrainTestplitListInput(self):
        # Check that when y is a list / list of string labels, it works.
        X = np.ones(7)
        y1 = ['1'] * 4 + ['0'] * 3
        y2 = np.hstack((np.ones(4), np.zeros(3)))
        y3 = y2.tolist()

        for stratify in (False,):
            X_train1, X_test1, y_train1, y_test1 = train_test_split(
                X, y1, stratify=y1 if stratify else None, random_state=0)
            X_train2, X_test2, y_train2, y_test2 = train_test_split(
                X, y2, stratify=y2 if stratify else None, random_state=0)
            X_train3, X_test3, y_train3, y_test3 = train_test_split(
                X, y3, stratify=y3 if stratify else None, random_state=0)

            np.testing.assert_equal(X_train1, X_train2)
            np.testing.assert_equal(y_train2, y_train3)
            np.testing.assert_equal(X_test1, X_test3)
            np.testing.assert_equal(y_test3, y_test2)

    def testMixiedInputTypeTrainTestSplit(self):
        rs = np.random.RandomState(0)
        df_raw = pd.DataFrame(rs.rand(10, 4))
        df = md.DataFrame(df_raw, chunk_size=5)
        X, y = df.iloc[:, :-1], df.iloc[:, -1]

        for x_to_tensor, y_to_tensor in itertools.product(range(1), range(1)):
            x = X
            if x_to_tensor:
                x = mt.tensor(x)
            yy = y
            if y_to_tensor:
                yy = mt.tensor(yy)

            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
            self.assertIsInstance(x_train, type(x))
            self.assertIsInstance(x_test, type(x))
            self.assertIsInstance(y_train, type(yy))
            self.assertIsInstance(y_test, type(yy))
