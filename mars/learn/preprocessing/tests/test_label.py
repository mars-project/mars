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

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.preprocessing._label import _inverse_binarize_thresholding
from sklearn.preprocessing._label import _inverse_binarize_multiclass
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.multiclass import type_of_target

from .... import tensor as mt
from .. import LabelBinarizer, label_binarize, LabelEncoder


def test_label_binarizer(setup):
    # one-class case defaults to negative label
    # For dense case:
    inp = ["pos", "pos", "pos", "pos"]
    lb = LabelBinarizer(sparse_output=False)
    expected = np.array([[0, 0, 0, 0]]).T
    got = lb.fit_transform(inp)
    assert_array_equal(lb.classes_, ["pos"])
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)

    # For sparse case:
    lb = LabelBinarizer(sparse_output=True)
    got = lb.fit_transform(inp)
    assert got.issparse()
    assert_array_equal(lb.classes_, ["pos"])
    assert_array_equal(expected, got.fetch().toarray())
    assert_array_equal(lb.inverse_transform(got.todense()), inp)

    lb = LabelBinarizer(sparse_output=False)
    # two-class case
    inp = ["neg", "pos", "pos", "neg"]
    expected = np.array([[0, 1, 1, 0]]).T
    got = lb.fit_transform(inp)
    assert_array_equal(lb.classes_, ["neg", "pos"])
    assert_array_equal(expected, got)

    to_invert = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    assert_array_equal(lb.inverse_transform(to_invert), inp)

    # multi-class case
    inp = ["spam", "ham", "eggs", "ham", "0"]
    expected = np.array(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    )
    got = lb.fit_transform(inp)
    assert_array_equal(lb.classes_, ["0", "eggs", "ham", "spam"])
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)


def test_label_binarizer_set_label_encoding(setup):
    lb = LabelBinarizer(neg_label=-2, pos_label=0)

    # two-class case with pos_label=0
    inp = np.array([0, 1, 1, 0])
    expected = np.array([[-2, 0, 0, -2]]).T
    got = lb.fit_transform(mt.tensor(inp))
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)

    lb = LabelBinarizer(neg_label=-2, pos_label=2)

    # multi-class case
    inp = np.array([3, 2, 1, 2, 0])
    expected = np.array(
        [
            [-2, -2, -2, +2],
            [-2, -2, +2, -2],
            [-2, +2, -2, -2],
            [-2, -2, +2, -2],
            [+2, -2, -2, -2],
        ]
    )
    got = lb.fit_transform(inp)
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)


@ignore_warnings
def test_label_binarizer_errors(setup):
    # Check that invalid arguments yield ValueError
    one_class = np.array([0, 0, 0, 0])
    lb = LabelBinarizer().fit(one_class)

    multi_label = [(2, 3), (0,), (0, 2)]
    with pytest.raises(ValueError):
        lb.transform(multi_label)

    lb = LabelBinarizer()
    with pytest.raises(ValueError):
        lb.transform([])
    with pytest.raises(ValueError):
        lb.inverse_transform([])

    with pytest.raises(ValueError):
        LabelBinarizer(neg_label=2, pos_label=1)
    with pytest.raises(ValueError):
        LabelBinarizer(neg_label=2, pos_label=2)

    with pytest.raises(ValueError):
        LabelBinarizer(neg_label=1, pos_label=2, sparse_output=True)

    # Sequence of seq type should raise ValueError
    y_seq_of_seqs = [[], [1, 2], [3], [0, 1, 3], [2]]
    with pytest.raises(ValueError):
        LabelBinarizer().fit_transform(y_seq_of_seqs)

    # Fail on multioutput data
    with pytest.raises(ValueError):
        LabelBinarizer().fit(np.array([[1, 3], [2, 1]]))
    with pytest.raises(ValueError):
        label_binarize(np.array([[1, 3], [2, 1]]), classes=[1, 2, 3])


def test_label_binarize_with_class_order(setup):
    out = label_binarize([1, 6], classes=[1, 2, 4, 6])
    expected = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    assert_array_equal(out, expected)

    # Modified class order
    out = label_binarize([1, 6], classes=[1, 6, 4, 2])
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    assert_array_equal(out, expected)

    out = label_binarize([0, 1, 2, 3], classes=[3, 2, 0, 1])
    expected = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]])
    assert_array_equal(out, expected)


def toarray(a):
    if hasattr(a, "toarray"):
        a = a.toarray()
    return a


def check_binarized_results(y, classes, pos_label, neg_label, expected):
    for sparse_output in [True, False]:
        if (pos_label == 0 or neg_label != 0) and sparse_output:
            with pytest.raises(ValueError):
                label_binarize(
                    y,
                    classes=classes,
                    neg_label=neg_label,
                    pos_label=pos_label,
                    sparse_output=sparse_output,
                )
            continue

        # check label_binarize
        binarized = label_binarize(
            y,
            classes=classes,
            neg_label=neg_label,
            pos_label=pos_label,
            sparse_output=sparse_output,
        )
        binarized = binarized.fetch()
        if hasattr(binarized, "raw"):
            binarized = binarized.raw
        assert_array_equal(toarray(binarized), expected)
        assert sp.issparse(binarized) == sparse_output

        # check inverse
        y_type = type_of_target(y)
        if y_type == "multiclass":
            inversed = _inverse_binarize_multiclass(binarized, classes=classes)

        else:
            inversed = _inverse_binarize_thresholding(
                binarized.copy(),  # https://github.com/mars-project/mars/issues/3268
                output_type=y_type,
                classes=classes,
                threshold=((neg_label + pos_label) / 2.0),
            )

        assert_array_equal(toarray(inversed), toarray(y))

        # Check label binarizer
        lb = LabelBinarizer(
            neg_label=neg_label, pos_label=pos_label, sparse_output=sparse_output
        )
        binarized = lb.fit_transform(y)
        assert_array_equal(toarray(binarized), expected)
        assert binarized.issparse() == sparse_output
        inverse_output = lb.inverse_transform(binarized)
        assert_array_equal(toarray(inverse_output), toarray(y))
        assert inverse_output.issparse() == sp.issparse(y)


def test_label_binarize_binary(setup):
    y = [0, 1, 0]
    classes = [0, 1]
    pos_label = 2
    neg_label = -1
    expected = np.array([[2, -1], [-1, 2], [2, -1]])[:, 1].reshape((-1, 1))

    check_binarized_results(y, classes, pos_label, neg_label, expected)

    # Binary case where sparse_output = True will not result in a ValueError
    y = [0, 1, 0]
    classes = [0, 1]
    pos_label = 3
    neg_label = 0
    expected = np.array([[3, 0], [0, 3], [3, 0]])[:, 1].reshape((-1, 1))

    check_binarized_results(y, classes, pos_label, neg_label, expected)


def test_label_binarize_multiclass(setup):
    y = [0, 1, 2]
    classes = [0, 1, 2]
    pos_label = 2
    neg_label = 0
    expected = 2 * np.eye(3)

    check_binarized_results(y, classes, pos_label, neg_label, expected)

    with pytest.raises(ValueError):
        label_binarize(
            y, classes=classes, neg_label=-1, pos_label=pos_label, sparse_output=True
        )


def test_label_binarize_multilabel(setup):
    y_ind = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]])
    classes = [0, 1, 2]
    pos_label = 2
    neg_label = 0
    expected = pos_label * y_ind
    y_sparse = [sp.csr_matrix(y_ind)]

    for y in [y_ind] + y_sparse:
        check_binarized_results(y, classes, pos_label, neg_label, expected)

    with pytest.raises(ValueError):
        label_binarize(
            y, classes=classes, neg_label=-1, pos_label=pos_label, sparse_output=True
        )


def test_invalid_input_label_binarize(setup):
    with pytest.raises(ValueError):
        label_binarize([0, 2], classes=[0, 2], pos_label=0, neg_label=1)
    with pytest.raises(ValueError, match="continuous target data is not "):
        label_binarize([1.2, 2.7], classes=[0, 1])
    with pytest.raises(ValueError, match="mismatch with the labels"):
        label_binarize([[1, 3]], classes=[1, 2, 3])


@pytest.mark.parametrize(
    "values, classes, unknown",
    [
        (
            np.array([2, 1, 3, 1, 3], dtype="int64"),
            np.array([1, 2, 3], dtype="int64"),
            np.array([4], dtype="int64"),
        ),
        (
            np.array(["b", "a", "c", "a", "c"], dtype=object),
            np.array(["a", "b", "c"], dtype=object),
            np.array(["d"], dtype=object),
        ),
        (
            np.array(["b", "a", "c", "a", "c"]),
            np.array(["a", "b", "c"]),
            np.array(["d"]),
        ),
    ],
    ids=["int64", "object", "str"],
)
def test_label_encoder(setup, values, classes, unknown):
    # Test LabelEncoder's transform, fit_transform and
    # inverse_transform methods
    values_t = mt.tensor(values)

    le = LabelEncoder()
    le.fit(values_t)
    assert_array_equal(le.classes_.fetch(), classes)
    assert_array_equal(le.transform(values_t).fetch(), [1, 0, 2, 0, 2])
    assert_array_equal(le.inverse_transform(mt.tensor([1, 0, 2, 0, 2])).fetch(), values)

    le = LabelEncoder()
    ret = le.fit_transform(values)
    assert_array_equal(ret.fetch(), [1, 0, 2, 0, 2])

    with pytest.raises(ValueError, match="unseen labels"):
        le.transform(unknown)


def test_label_encoder_missing_values_numeric(setup):
    values = np.array([3, 1, np.nan, 5, 3, np.nan], dtype=float)
    values_t = mt.tensor(values)
    le = LabelEncoder()
    assert_array_equal(le.fit_transform(values_t).fetch(), [1, 0, 3, 2, 1, 3])


def test_label_encoder_negative_ints(setup):
    le = LabelEncoder()
    le.fit(mt.tensor([1, 1, 4, 5, -1, 0]))
    assert_array_equal(le.classes_, [-1, 0, 1, 4, 5])
    assert_array_equal(
        le.transform(mt.tensor([0, 1, 4, 4, 5, -1, -1])), [1, 2, 3, 3, 4, 0, 0]
    )
    assert_array_equal(
        le.inverse_transform(mt.tensor([1, 2, 3, 3, 4, 0, 0])), [0, 1, 4, 4, 5, -1, -1]
    )
    with pytest.raises(ValueError):
        le.transform(mt.tensor([0, 6]))


@pytest.mark.parametrize("dtype", ["str", "object"])
def test_label_encoder_str_bad_shape(setup, dtype):
    le = LabelEncoder()
    le.fit(mt.tensor(np.array(["apple", "orange"], dtype=dtype)))
    msg = "should be a 1d array"
    with pytest.raises(ValueError, match=msg):
        le.transform("apple")


def test_label_encoder_errors(setup):
    # Check that invalid arguments yield ValueError
    le = LabelEncoder()
    with pytest.raises(ValueError):
        le.transform([])
    with pytest.raises(ValueError):
        le.inverse_transform([])

    # Fail on unseen labels
    le = LabelEncoder()
    le.fit(mt.tensor([1, 2, 3, -1, 1]))
    msg = "contains previously unseen labels"
    with pytest.raises(ValueError, match=msg):
        le.inverse_transform(mt.tensor([-2]))
    with pytest.raises(ValueError, match=msg):
        le.inverse_transform(mt.tensor([-2, -3, -4]))

    # Fail on inverse_transform("")
    msg = r"should be a 1d array.+shape \(\)"
    with pytest.raises(ValueError, match=msg):
        le.inverse_transform("")


@pytest.mark.parametrize(
    "values",
    [
        np.array([2, 1, 3, 1, 3], dtype="int64"),
        np.array(["b", "a", "c", "a", "c"], dtype=object),
        np.array(["b", "a", "c", "a", "c"]),
    ],
    ids=["int64", "object", "str"],
)
def test_label_encoder_empty_array(setup, values):
    values_t = mt.tensor(values)

    le = LabelEncoder()
    le.fit(values_t)
    # test empty transform
    transformed = le.transform(mt.array([]))
    assert_array_equal(np.array([]), transformed)
    # test empty inverse transform
    inverse_transformed = le.inverse_transform([])
    assert_array_equal(np.array([]), inverse_transformed)
