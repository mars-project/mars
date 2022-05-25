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

import asyncio
import uuid
import sys

import pytest
import numpy as np

from ....deploy.oscar.local import new_cluster
from ....deploy.oscar.session import AsyncSession, SyncSession
from ..core import MutableTensor
from ..utils import normalize_timestamp


_is_windows = sys.platform.lower().startswith("win")


@pytest.fixture
async def create_cluster():
    client = await new_cluster(n_worker=2, n_cpu=2, web=True)
    async with client:
        yield client


@pytest.mark.skipif(_is_windows, reason="FIXME")
@pytest.mark.parametrize(
    "session_type",
    ["async_session", "async_web_session", "sync_session", "sync_web_session"],
)
@pytest.mark.asyncio
async def test_mutable_tensor(create_cluster, session_type):
    is_web = "web" in session_type
    is_async = "async" in session_type

    if is_web:
        session_id = str(uuid.uuid4())
        session = await AsyncSession.init(create_cluster.web_address, session_id)
    else:
        session = create_cluster.session
    if not is_async:
        session = SyncSession.from_isolated_session(session)

    tensor_useless: MutableTensor = session.create_mutable_tensor(  # noqa: F841
        shape=(10, 30, 50), dtype=np.int64, default_value=100, chunk_size=(20, 20, 20)
    )
    if is_async:
        tensor_useless = await tensor_useless

    tensor: MutableTensor = session.create_mutable_tensor(
        shape=(10, 30, 50),
        dtype=np.int64,
        name="mytensor",
        default_value=100,
        chunk_size=(20, 20, 20),
    )
    if is_async:
        tensor = await tensor

    assert tensor.shape == (10, 30, 50)
    assert tensor.dtype == np.int64
    assert tensor.name == "mytensor"
    assert tensor.default_value == 100

    assert tensor_useless.name != tensor.name

    # non exists
    with pytest.raises(ValueError):
        tensor1 = session.get_mutable_tensor("notensor")
        if is_async:
            tensor1 = await tensor1

    # create with duplicate name
    with pytest.raises(ValueError):
        tensor2 = session.create_mutable_tensor(
            shape=(10, 30, 50),
            dtype=np.int64,
            name="mytensor",
            default_value=100,
            chunk_size=(20, 20, 20),
        )
        if is_async:
            tensor2 = await tensor2

    tensor3: MutableTensor = session.get_mutable_tensor("mytensor")
    if is_async:
        tensor3 = await tensor3
    assert tensor3.shape == (10, 30, 50)
    assert tensor3.dtype == np.int64
    assert tensor3.name == "mytensor"
    assert tensor3.default_value == 100

    # test using read/write

    expected = np.full((10, 30, 50), fill_value=100)
    xs = await tensor3.read(slice(None, None, None))
    np.testing.assert_array_equal(expected, xs)

    await tensor.write(slice(None, None, None), 1)
    expected[:] = 1
    xs = await tensor3.read(slice(None, None, None))
    np.testing.assert_array_equal(expected, xs)

    await tensor.write((9, 2, 3), 2)
    expected[9, 2, 3] = 2
    xs = await tensor3.read((9, 2, 3))
    assert expected[9, 2, 3] == xs

    await tensor.write((slice(2, 9, 3), slice(5, 15, None), slice(8, 50, 9)), 4)
    expected[2:9:3, 5:15, 8:50:9] = 4
    xs = await tensor3.read(slice(None, None, None))
    np.testing.assert_array_equal(expected, xs)

    # test using __getitem__/__setitem__

    # reset
    tensor[:] = 100

    expected = np.full((10, 30, 50), fill_value=100)
    xs = tensor3[:]
    np.testing.assert_array_equal(expected, xs)

    tensor[:] = 1
    expected[:] = 1
    xs = tensor3[:]
    np.testing.assert_array_equal(expected, xs)

    tensor[9, 2, 3] = 2
    expected[9, 2, 3] = 2
    xs = tensor3[9, 2, 3]
    assert expected[9, 2, 3] == xs

    tensor[2:19:3, 5:15, 8:50:9] = 4
    expected[2:19:3, 5:15, 8:50:9] = 4
    xs = tensor3[:]
    np.testing.assert_array_equal(expected, xs)

    # seal

    if is_async:
        sealed = await tensor.seal()
        info = await session.execute(sealed)
        await info
        value = await session.fetch(sealed)
    else:
        sealed = await tensor.seal()
        session.execute(sealed)
        value = session.fetch(sealed)
    np.testing.assert_array_equal(expected, value)

    # non exists after sealed
    with pytest.raises(ValueError):
        await tensor.seal()
    with pytest.raises(ValueError):
        await tensor3.seal()

    # TODO: real fancy index not supported yet, as `TensorConcatenate` involved
    #
    # await tensor.write(([11, 2, 3, 50], [14, 5, 6, 50], [17, 8, 9, 50]), 3)
    # expected[[11, 2, 3, 50], [14, 5, 6, 50], [17, 8, 9, 50]] = 3
    # xs = await tensor1[:]
    # np.testing.assert_array_equal(expected, xs)


@pytest.mark.skipif(_is_windows, reason="FIXME")
@pytest.mark.parametrize(
    "session_type",
    ["async_session", "async_web_session", "sync_session", "sync_web_session"],
)
@pytest.mark.asyncio
async def test_mutable_tensor_timestamp(create_cluster, session_type):
    is_web = "web" in session_type
    is_async = "async" in session_type

    if is_web:
        session_id = str(uuid.uuid4())
        session = await AsyncSession.init(create_cluster.web_address, session_id)
    else:
        session = create_cluster.session
    if not is_async:
        session = SyncSession.from_isolated_session(session)

    tensor: MutableTensor = session.create_mutable_tensor(
        shape=(2, 4), dtype=np.int64, default_value=0, chunk_size=(1, 3)
    )
    if is_async:
        tensor = await tensor

    assert tensor.shape == (2, 4)
    assert tensor.dtype == np.int64
    assert tensor.default_value == 0

    t0 = normalize_timestamp()
    await asyncio.sleep(5)
    t1 = normalize_timestamp()

    # write with earlier timestamp
    await tensor.write((slice(0, 2, 1), slice(0, 2, 1)), 1, timestamp=t1)

    # read staled value
    actual = await tensor.read(slice(None, None, None), t0)
    expected = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(expected, actual)

    # read current value
    actual = await tensor.read(slice(None, None, None), t1)
    expected = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
    np.testing.assert_array_equal(expected, actual)

    # read new value
    t2 = normalize_timestamp()
    actual = await tensor.read(slice(None, None, None), t2)
    expected = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
    np.testing.assert_array_equal(expected, actual)

    # read latest value
    actual = await tensor.read(slice(None, None, None))
    expected = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
    np.testing.assert_array_equal(expected, actual)

    # seal on staled value
    if is_async:
        sealed = await tensor.seal(timestamp=t0)
        info = await session.execute(sealed)
        await info
        actual = await session.fetch(sealed)
    else:
        sealed = await tensor.seal(timestamp=t0)
        session.execute(sealed)
        actual = session.fetch(sealed)
    expected = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(expected, actual)

    # non exists after sealed
    with pytest.raises(ValueError):
        await tensor.seal()
