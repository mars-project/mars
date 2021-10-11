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

import sys

import pytest
import numpy as np

from ....deploy.oscar.local import new_cluster
from ..supervisor.core import MutableTensor


_is_windows = sys.platform.lower().startswith('win')


@pytest.fixture
async def create_cluster():
    client = await new_cluster(n_worker=2, n_cpu=2)
    async with client:
        yield client


@pytest.mark.skipif(_is_windows, reason="FIXME")
@pytest.mark.asyncio
async def test_mutable_tensor_actor(create_cluster):
    session = create_cluster.session

    tensor_useless: MutableTensor = await session.create_mutable_tensor(  # noqa: F841
            shape=(100, 100, 100), dtype=np.int64,
            chunk_size=(10, 10, 10), default_value=100)

    tensor: MutableTensor = await session.create_mutable_tensor(
        shape=(100, 100, 100), dtype=np.int64,
        chunk_size=(10, 10, 10), name="mytensor", default_value=100)

    # non exists
    with pytest.raises(ValueError):
        tensor = await session.get_mutable_tensor("notensor")

    # create with duplicate name
    with pytest.raises(ValueError):
        tensor = await session.create_mutable_tensor(
            shape=(100, 100, 100), dtype=np.int64,
            chunk_size=(10, 10, 10), name="mytensor", default_value=100)

    tensor1: MutableTensor = await session.get_mutable_tensor("mytensor")

    expected = np.full((100, 100, 100), fill_value=100)
    xs = await tensor1[:]
    np.testing.assert_array_equal(expected, xs)

    await tensor.write(slice(None, None, None), 1)
    expected[:] = 1
    xs = await tensor1[:]
    np.testing.assert_array_equal(expected, xs)

    await tensor.write((11, 2, 3), 2)
    expected[11, 2, 3] = 2
    xs = await tensor1[11, 2, 3]
    assert expected[11, 2, 3] == xs

    # TODO: real fancy index not supported yet, as `TensorConcatenate` involved
    #
    # await tensor.write(([11, 2, 3, 50], [14, 5, 6, 50], [17, 8, 9, 50]), 3)
    # expected[[11, 2, 3, 50], [14, 5, 6, 50], [17, 8, 9, 50]] = 3
    # xs = await tensor1[:]
    # np.testing.assert_array_equal(expected, xs)

    await tensor.write((slice(2, 12, 3), slice(5, 15, None), slice(8, 50, 9)), 4)
    expected[2:12:3, 5:15, 8:50:9] = 4
    xs = await tensor1[:]
    np.testing.assert_array_equal(expected, xs)

    sealed = await tensor.seal()
    info = await session.execute(sealed)
    await info
    value = await session.fetch(sealed)
    np.testing.assert_array_equal(expected, value)
