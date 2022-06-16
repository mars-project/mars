#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from ....core import tile
from ....serialization import serialize, deserialize
from ....serialization.serializables import Serializable
from ...datasource import tensor as from_ndarray
from .. import (
    beta,
    rand,
    choice,
    multivariate_normal,
    randint,
    randn,
    permutation,
    TensorPermutation,
    shuffle,
    RandomState,
)
from ..core import RandomStateField


class ObjWithRandomStateField(Serializable):
    random_state = RandomStateField("random_state")


@pytest.mark.parametrize("rs", [None, np.random.RandomState()])
def test_serial_random_state_field(rs):
    res = deserialize(*serialize(ObjWithRandomStateField(rs)))
    if rs is None:
        assert res.random_state is None
    else:
        original_state = rs.get_state()
        new_state = res.random_state.get_state()
        assert original_state[0] == new_state[0]
        np.testing.assert_array_equal(original_state[1], new_state[1])


def test_random():
    arr = rand(2, 3)

    assert arr.dtype is not None

    arr = tile(beta(1, 2, chunk_size=2))

    assert arr.shape == ()
    assert len(arr.chunks) == 1
    assert arr.chunks[0].shape == ()
    assert arr.chunks[0].op.dtype == np.dtype("f8")

    arr = tile(beta([1, 2], [3, 4], chunk_size=2))

    assert arr.shape == (2,)
    assert len(arr.chunks) == 1
    assert arr.chunks[0].shape == (2,)
    assert arr.chunks[0].op.dtype == np.dtype("f8")

    arr = tile(
        beta(
            [[2, 3]],
            from_ndarray([[4, 6], [5, 2]], chunk_size=2),
            chunk_size=1,
            size=(3, 2, 2),
        )
    )

    assert arr.shape == (3, 2, 2)
    assert len(arr.chunks) == 12
    assert arr.chunks[0].op.dtype == np.dtype("f8")


def test_same_key():
    assert RandomState(0).rand(10).key == RandomState(0).rand(10).key


def test_choice():
    t = choice(5, chunk_size=1)
    assert t.shape == ()
    t = tile(t)
    assert t.nsplits == ()
    assert len(t.chunks) == 1

    t = choice(5, 3, chunk_size=1)
    assert t.shape == (3,)
    t = tile(t)
    assert t.nsplits == ((1, 1, 1),)

    t = choice(5, 3, replace=False)
    assert t.shape == (3,)

    with pytest.raises(ValueError):
        choice(-1)

    # a should be 1-d
    with pytest.raises(ValueError):
        choice(np.random.rand(2, 2))

    # p sum != 1
    with pytest.raises(ValueError):
        choice(np.random.rand(3), p=[0.2, 0.2, 0.2])

    # p should b 1-d
    with pytest.raises(ValueError):
        choice(np.random.rand(3), p=[[0.2, 0.6, 0.2]])

    # replace=False, choice size cannot be greater than a.size
    with pytest.raises(ValueError):
        choice(np.random.rand(10), 11, replace=False)

    # replace=False, choice size cannot be greater than a.size
    with pytest.raises(ValueError):
        choice(np.random.rand(10), (3, 4), replace=False)


def test_multivariate_normal():
    mean = [0, 0]
    cov = [[1, 0], [0, 100]]

    t = multivariate_normal(mean, cov, 5000, chunk_size=500)
    assert t.shape == (5000, 2)
    assert t.op.size == (5000,)

    t = tile(t)
    assert t.nsplits == ((500,) * 10, (2,))
    assert len(t.chunks) == 10
    c = t.chunks[0]
    assert c.shape == (500, 2)
    assert c.op.size == (500,)


def test_randint():
    arr = tile(randint(1, 2, size=(10, 9), dtype="f8", density=0.01, chunk_size=2))

    assert arr.shape == (10, 9)
    assert len(arr.chunks) == 25
    assert arr.chunks[0].shape == (2, 2)
    assert arr.chunks[0].op.dtype == np.float64
    assert arr.chunks[0].op.low == 1
    assert arr.chunks[0].op.high == 2
    assert arr.chunks[0].op.density == 0.01


def test_unexpected_key():
    with pytest.raises(ValueError):
        rand(10, 10, chunks=5)

    with pytest.raises(ValueError):
        randn(10, 10, chunks=5)


def test_permutation():
    x = permutation(10)

    assert x.shape == (10,)
    assert isinstance(x.op, TensorPermutation)

    x = tile(x)

    assert len(x.chunks) == 1
    assert isinstance(x.chunks[0].op, TensorPermutation)

    arr = from_ndarray([1, 4, 9, 12, 15], chunk_size=2)
    x = permutation(arr)

    assert x.shape == (5,)
    assert isinstance(x.op, TensorPermutation)

    x = tile(x)
    arr = tile(arr)

    assert len(x.chunks) == 3
    assert np.isnan(x.chunks[0].shape[0])
    assert x.chunks[0].inputs[0].inputs[0].inputs[0].key == arr.chunks[0].data.key

    arr = rand(3, 3, chunk_size=2)
    x = permutation(arr)

    assert x.shape == (3, 3)
    assert isinstance(x.op, TensorPermutation)

    x = tile(x)
    arr = tile(arr)

    assert len(x.chunks) == 4
    assert np.isnan(x.chunks[0].shape[0])
    assert x.chunks[0].shape[1] == 2
    assert x.cix[0, 0].op.seed == x.cix[0, 1].op.seed
    assert (
        x.cix[0, 0].inputs[0].inputs[0].inputs[0].op.seed
        == x.cix[1, 0].inputs[0].inputs[0].inputs[0].op.seed
    )

    with pytest.raises(np.AxisError):
        pytest.raises(permutation("abc"))


def test_shuffle():
    with pytest.raises(TypeError):
        shuffle("abc")

    x = rand(10, 10, chunk_size=2)
    shuffle(x)
    assert isinstance(x.op, TensorPermutation)

    x = rand(10, 10, chunk_size=2)
    shuffle(x, axis=1)
    assert isinstance(x.op, TensorPermutation)
    assert x.op.axis == 1
