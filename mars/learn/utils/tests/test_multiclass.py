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

from itertools import product

import numpy as np
import pytest
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.multiclass import (
    is_multilabel as sklearn_is_multilabel,
    type_of_target as sklearn_type_of_target,
)

from .... import tensor as mt
from ..multiclass import is_multilabel, unique_labels, type_of_target


EXAMPLES = {
    "multilabel-indicator": [
        # valid when the data is formatted as sparse or dense, identified
        # by CSR format when the testing takes place
        csr_matrix(np.random.RandomState(42).randint(2, size=(10, 10))),
        [[0, 1], [1, 0]],
        [[0, 1]],
        csr_matrix(np.array([[0, 1], [1, 0]])),
        csr_matrix(np.array([[0, 1], [1, 0]], dtype=bool)),
        csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.int8)),
        csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.uint8)),
        csr_matrix(np.array([[0, 1], [1, 0]], dtype=float)),
        csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float32)),
        csr_matrix(np.array([[0, 0], [0, 0]])),
        csr_matrix(np.array([[0, 1]])),
        # Only valid when data is dense
        [[-1, 1], [1, -1]],
        np.array([[-1, 1], [1, -1]]),
        np.array([[-3, 3], [3, -3]]),
        _NotAnArray(np.array([[-3, 3], [3, -3]])),
    ],
    "multiclass": [
        [1, 0, 2, 2, 1, 4, 2, 4, 4, 4],
        np.array([1, 0, 2]),
        np.array([1, 0, 2], dtype=np.int8),
        np.array([1, 0, 2], dtype=np.uint8),
        np.array([1, 0, 2], dtype=float),
        np.array([1, 0, 2], dtype=np.float32),
        np.array([[1], [0], [2]]),
        _NotAnArray(np.array([1, 0, 2])),
        [0, 1, 2],
        ["a", "b", "c"],
        np.array(["a", "b", "c"]),
        np.array(["a", "b", "c"], dtype=object),
        np.array(["a", "b", "c"], dtype=object),
    ],
    "multiclass-multioutput": [
        [[1, 0, 2, 2], [1, 4, 2, 4]],
        [["a", "b"], ["c", "d"]],
        np.array([[1, 0, 2, 2], [1, 4, 2, 4]]),
        np.array([[1, 0, 2, 2], [1, 4, 2, 4]], dtype=np.int8),
        np.array([[1, 0, 2, 2], [1, 4, 2, 4]], dtype=np.uint8),
        np.array([[1, 0, 2, 2], [1, 4, 2, 4]], dtype=float),
        np.array([[1, 0, 2, 2], [1, 4, 2, 4]], dtype=np.float32),
        np.array([["a", "b"], ["c", "d"]]),
        np.array([["a", "b"], ["c", "d"]]),
        np.array([["a", "b"], ["c", "d"]], dtype=object),
        np.array([[1, 0, 2]]),
        _NotAnArray(np.array([[1, 0, 2]])),
    ],
    "binary": [
        [0, 1],
        [1, 1],
        [],
        [0],
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1]),
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool),
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.int8),
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.uint8),
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=float),
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.float32),
        np.array([[0], [1]]),
        _NotAnArray(np.array([[0], [1]])),
        [1, -1],
        [3, 5],
        ["a"],
        ["a", "b"],
        ["abc", "def"],
        np.array(["abc", "def"]),
        ["a", "b"],
        np.array(["abc", "def"], dtype=object),
    ],
    "continuous": [
        [1e-5],
        [0, 0.5],
        np.array([[0], [0.5]]),
        np.array([[0], [0.5]], dtype=np.float32),
    ],
    "continuous-multioutput": [
        np.array([[0, 0.5], [0.5, 0]]),
        np.array([[0, 0.5], [0.5, 0]], dtype=np.float32),
        np.array([[0, 0.5]]),
    ],
    "unknown": [
        [[]],
        [()],
        # sequence of sequences that weren't supported even before deprecation
        np.array([np.array([]), np.array([1, 2, 3])], dtype=object),
        # [np.array([]), np.array([1, 2, 3])], # deprecated in numpy v1.24
        [{1, 2, 3}, {1, 2}],
        [frozenset([1, 2, 3]), frozenset([1, 2])],
        # and also confusable as sequences of sequences
        [{0: "a", 1: "b"}, {0: "a"}],
        # empty second dimension
        np.array([[], []]),
        # 3d
        np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
    ],
}

NON_ARRAY_LIKE_EXAMPLES = [
    {1, 2, 3},
    {0: "a", 1: "b"},
    {0: [5], 1: [5]},
    "abc",
    frozenset([1, 2, 3]),
    None,
]


def test_unique_labels(setup):
    # Empty iterable
    with pytest.raises(ValueError):
        unique_labels()

    # Multiclass problem
    assert_array_equal(unique_labels(range(10)), np.arange(10))
    assert_array_equal(unique_labels(np.arange(10)), np.arange(10))
    assert_array_equal(unique_labels([4, 0, 2]), np.array([0, 2, 4]))

    # Multilabel indicator
    assert_array_equal(
        unique_labels(np.array([[0, 0, 1], [1, 0, 1], [0, 0, 0]])), np.arange(3)
    )

    assert_array_equal(unique_labels(np.array([[0, 0, 1], [0, 0, 0]])), np.arange(3))

    # Several arrays passed
    assert_array_equal(unique_labels([4, 0, 2], range(5)), np.arange(5))
    assert_array_equal(unique_labels((0, 1, 2), (0,), (2, 1)), np.arange(3))

    # Border line case with binary indicator matrix
    with pytest.raises(ValueError):
        unique_labels([4, 0, 2], np.ones((5, 5))).execute()
    with pytest.raises(ValueError):
        unique_labels(np.ones((5, 4)), np.ones((5, 5))).execute()

    assert_array_equal(unique_labels(np.ones((4, 5)), np.ones((5, 5))), np.arange(5))


def test_unique_labels_non_specific(setup):
    # Test unique_labels with a variety of collected examples

    # Smoke test for all supported format
    for format in ["binary", "multiclass", "multilabel-indicator"]:
        for y in EXAMPLES[format]:
            unique_labels(y).execute()

    # We don't support those format at the moment
    for example in NON_ARRAY_LIKE_EXAMPLES:
        with pytest.raises(ValueError):
            unique_labels(example).execute()

    for y_type in [
        "unknown",
        "continuous",
        "continuous-multioutput",
        "multiclass-multioutput",
    ]:
        for example in EXAMPLES[y_type]:
            with pytest.raises(ValueError):
                unique_labels(example).execute()


def test_unique_labels_mixed_types(setup):
    # Mix with binary or multiclass and multilabel
    mix_clf_format = product(
        EXAMPLES["multilabel-indicator"], EXAMPLES["multiclass"] + EXAMPLES["binary"]
    )

    for y_multilabel, y_multiclass in mix_clf_format:
        with pytest.raises(ValueError):
            unique_labels(y_multiclass, y_multilabel).execute()
        with pytest.raises(ValueError):
            unique_labels(y_multilabel, y_multiclass).execute()

    with pytest.raises(ValueError):
        unique_labels([[1, 2]], [["a", "d"]]).execute()

    with pytest.raises(ValueError):
        unique_labels(["1", 2]).execute()

    with pytest.raises(ValueError):
        unique_labels([["1", 2], [1, 3]]).execute()

    with pytest.raises(ValueError):
        unique_labels([["1", "2"], [2, 3]]).execute()


def test_is_multilabel(setup):
    raws = [
        [[1, 2]],
        [0, 1, 0, 1],
        # [[1], [0, 2], []], # deprecated in numpy v1.24
        np.array([[1, 0], [0, 0]]),
        np.array([[1], [0], [0]]),
        np.array([[1, 0, 0]]),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        sps.csr_matrix([[1, 0], [0, 1]]),
    ]

    for raw in raws:
        assert is_multilabel(raw).to_numpy() == sklearn_is_multilabel(raw)

    t = mt.tensor(raws[3], chunk_size=1)
    assert is_multilabel(t).to_numpy() == sklearn_is_multilabel(raws[3])


def test_type_of_target(setup):
    raws = [
        np.array([[0, 1], [0, 0]]),  # multilabel
        np.random.randint(2, size=(5, 3, 3)),  # ndim > 2, unknown
        np.array([[]]),  # ndim == 2, shape[1] == 0, unknown
        np.array([[1, 2], [1, 2]]),
        np.array([1, 2, 3]),
        np.array([0.1, 0.2, 3]),
        np.array([[0.1, 0.2, 3]]),
        np.array([[1.0, 0.2]]),
        np.array([[1.0, 2.0, 3]]),
        np.array([[1, 2]]),
        np.array([1, 2]),
        np.array([["a"], ["b"]], dtype=object),
        [[1, 2]],
        [],  # empty list
    ]

    for raw in raws:
        assert type_of_target(raw).to_numpy() == sklearn_type_of_target(raw)

    t = mt.tensor(raws[0], chunk_size=1)
    assert type_of_target(t).to_numpy() == sklearn_type_of_target(raws[0])

    with pytest.raises(ValueError):
        type_of_target("sth")
