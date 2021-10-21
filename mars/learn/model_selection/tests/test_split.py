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

import numpy as np
import pandas as pd
import pytest

try:
    import scipy.sparse as sps
except ImportError:  # pragma: no cover
    sps = None

from .... import dataframe as md
from .... import tensor as mt
from ....dataframe.core import DATAFRAME_TYPE
from ....lib.sparse import SparseNDArray
from ...utils.validation import _num_samples
from .. import train_test_split, KFold


def test_train_test_split_errors(setup):
    pytest.raises(ValueError, train_test_split)

    pytest.raises(ValueError, train_test_split, range(3), train_size=1.1)

    pytest.raises(ValueError, train_test_split, range(3), test_size=0.6, train_size=0.6)
    pytest.raises(
        ValueError,
        train_test_split,
        range(3),
        test_size=np.float32(0.6),
        train_size=np.float32(0.6),
    )
    pytest.raises(ValueError, train_test_split, range(3), test_size="wrong_type")
    pytest.raises(ValueError, train_test_split, range(3), test_size=2, train_size=4)
    pytest.raises(TypeError, train_test_split, range(3), some_argument=1.1)
    pytest.raises(ValueError, train_test_split, range(3), range(42))
    pytest.raises(ValueError, train_test_split, range(10), shuffle=False, stratify=True)

    with pytest.raises(
        ValueError,
        match=r"train_size=11 should be either positive and "
        r"smaller than the number of samples 10 or a "
        r"float in the \(0, 1\) range",
    ):
        train_test_split(range(10), train_size=11, test_size=1)


def test_train_test_split_invalid_sizes1(setup):
    for train_size, test_size in [
        (1.2, 0.8),
        (1.0, 0.8),
        (0.0, 0.8),
        (-0.2, 0.8),
        (0.8, 1.2),
        (0.8, 1.0),
        (0.8, 0.0),
        (0.8, -0.2),
    ]:
        with pytest.raises(ValueError, match=r"should be .* in the \(0, 1\) range"):
            train_test_split(range(10), train_size=train_size, test_size=test_size)


def test_train_test_split_invalid_sizes2(setup):
    for train_size, test_size in [
        (-10, 0.8),
        (0, 0.8),
        (11, 0.8),
        (0.8, -10),
        (0.8, 0),
        (0.8, 11),
    ]:
        with pytest.raises(ValueError, match=r"should be .* in the \(0, 1\) range"):
            train_test_split(range(10), train_size=train_size, test_size=test_size)


def test_train_test_split(setup):
    X = np.arange(100).reshape((10, 10))
    y = np.arange(10)

    # simple test
    split = train_test_split(X, y, test_size=None, train_size=0.5)
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


def test_train_test_split_dataframe(setup):
    X = np.ones(10)
    types = [pd.DataFrame, md.DataFrame]
    for InputFeatureType in types:
        # X dataframe
        X_df = InputFeatureType(X)
        X_train, X_test = train_test_split(X_df)
        assert isinstance(X_train, DATAFRAME_TYPE)
        assert isinstance(X_test, DATAFRAME_TYPE)


@pytest.mark.skipif(sps is None, reason="scipy not installed")
def test_train_test_split_sparse(setup):
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


def test_train_testplit_list_input(setup):
    # Check that when y is a list / list of string labels, it works.
    X = np.ones(7)
    y1 = ["1"] * 4 + ["0"] * 3
    y2 = np.hstack((np.ones(4), np.zeros(3)))
    y3 = y2.tolist()

    for stratify in (False,):
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y1, stratify=y1 if stratify else None, random_state=0
        )
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y2, stratify=y2 if stratify else None, random_state=0
        )
        X_train3, X_test3, y_train3, y_test3 = train_test_split(
            X, y3, stratify=y3 if stratify else None, random_state=0
        )

        np.testing.assert_equal(X_train1, X_train2)
        np.testing.assert_equal(y_train2, y_train3)
        np.testing.assert_equal(X_test1, X_test3)
        np.testing.assert_equal(y_test3, y_test2)


def test_mixied_input_type_train_test_split(setup):
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

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=0, run_kwargs={"extra_config": {"check_nsplits": False}}
        )
        assert isinstance(x_train, type(x))
        assert isinstance(x_test, type(x))
        assert isinstance(y_train, type(yy))
        assert isinstance(y_test, type(yy))


def test_kfold_valueerrors():
    X1 = np.array([[1, 2], [3, 4], [5, 6]])
    # Check that errors are raised if there is not enough samples
    with pytest.raises(ValueError):
        next(KFold(4).split(X1))

    # Error when number of folds is <= 1
    with pytest.raises(ValueError):
        KFold(0)
    with pytest.raises(ValueError):
        KFold(1)

    # When n_splits is not integer:
    with pytest.raises(ValueError):
        KFold(1.5)
    with pytest.raises(ValueError):
        KFold(2.0)

    # When shuffle is not  a bool:
    with pytest.raises(TypeError):
        KFold(n_splits=4, shuffle=None)


def check_valid_split(train, test, n_samples=None):
    # Use python sets to get more informative assertion failure messages
    train = train.execute().to_numpy()
    test = test.execute().to_numpy()
    train, test = set(train), set(test)

    # Train and test split should not overlap
    assert train.intersection(test) == set()

    if n_samples is not None:
        # Check that the union of train an test split cover all the indices
        assert train.union(test) == set(range(n_samples))


def check_cv_coverage(cv, X, y, groups, expected_n_splits):
    n_samples = _num_samples(X)
    # Check that a all the samples appear at least once in a test fold
    assert cv.get_n_splits(X, y, groups) == expected_n_splits

    collected_test_samples = set()
    iterations = 0
    for train, test in cv.split(X, y, groups):
        check_valid_split(train, test, n_samples=n_samples)
        iterations += 1
        collected_test_samples.update(test.execute().to_numpy())

    # Check that the accumulated test samples cover the whole dataset
    assert iterations == expected_n_splits
    if n_samples is not None:
        assert collected_test_samples == set(range(n_samples))


def test_kfold_indices(setup):
    # Check all indices are returned in the test folds
    X1 = np.ones(18)
    kf = KFold(3)
    check_cv_coverage(kf, X1, y=None, groups=None, expected_n_splits=3)

    # Check all indices are returned in the test folds even when equal-sized
    # folds are not possible
    X2 = np.ones(17)
    kf = KFold(3)
    check_cv_coverage(kf, X2, y=None, groups=None, expected_n_splits=3)

    # Check if get_n_splits returns the number of folds
    assert 5 == KFold(5).get_n_splits(X2)


def test_kfold_no_shuffle(setup):
    # Manually check that KFold preserves the data ordering on toy datasets
    X2 = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

    splits = KFold(2).split(X2[:-1])
    train, test = next(splits)
    np.testing.assert_array_equal(test.execute().fetch(), [0, 1])
    np.testing.assert_array_equal(train.execute().fetch(), [2, 3])

    train, test = next(splits)
    np.testing.assert_array_equal(test.execute().fetch(), [2, 3])
    np.testing.assert_array_equal(train.execute().fetch(), [0, 1])

    splits = KFold(2).split(X2)
    train, test = next(splits)
    np.testing.assert_array_equal(test.execute().fetch(), [0, 1, 2])
    np.testing.assert_array_equal(train.execute().fetch(), [3, 4])

    train, test = next(splits)
    np.testing.assert_array_equal(test.execute().fetch(), [3, 4])
    np.testing.assert_array_equal(train.execute().fetch(), [0, 1, 2])


def test_kfold_balance(setup):
    # Check that KFold returns folds with balanced sizes
    for i in range(11, 17):
        kf = KFold(5).split(X=np.ones(i))
        sizes = [len(test) for _, test in kf]

        assert (np.max(sizes) - np.min(sizes)) <= 1
        assert np.sum(sizes) == i


def test_shuffle_kfold(setup):
    # Check the indices are shuffled properly
    kf = KFold(3)
    kf2 = KFold(3, shuffle=True, random_state=0)
    kf3 = KFold(3, shuffle=True, random_state=1)

    X = mt.ones(300)

    all_folds = np.zeros(300)
    for (tr1, te1), (tr2, te2), (tr3, te3) in zip(
        kf.split(X), kf2.split(X), kf3.split(X)
    ):
        for tr_a, tr_b in itertools.combinations((tr1, tr2, tr3), 2):
            # Assert that there is no complete overlap
            tr_a = tr_a.execute().fetch()
            tr_b = tr_b.execute().fetch()
            assert len(np.intersect1d(tr_a, tr_b)) != len(tr1)

        # Set all test indices in successive iterations of kf2 to 1
        all_folds[te2.execute().fetch()] = 1

    # Check that all indices are returned in the different test folds
    assert sum(all_folds) == 300
