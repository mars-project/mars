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

import pytest

from ...tests.core import mock
from ..backends.router import Router
from ..backends.core import ActorCaller
from ..errors import ServerClosed


@pytest.mark.asyncio
@mock.patch.object(Router, "get_client")
async def test_send_when_close(fake_get_client):
    class FakeClient:
        def __init__(self):
            self.closed = False
            self.send_num = 0
            self._messages = asyncio.Queue()
            self.dest_address = "test"

        async def send(self, message):
            await self._messages.put(message)
            self.send_num += 1
            if self.send_num >= 3:
                raise ConnectionError("test")

        async def recv(self, *args, **kwargs):
            await asyncio.sleep(3)
            res = await self._messages.get()
            return res

        async def close(self):
            self.closed = True

    fake_client = FakeClient()
    fake_get_client.side_effect = lambda *args, **kwargs: fake_client

    class FakeMessage:
        def __init__(self, id_num):
            self.message_id = id_num

    caller = ActorCaller()

    router = Router(
        external_addresses=["test1"],
        local_address="test2",
    )
    futures = []
    for index in range(2):
        futures.append(
            await caller.call(
                router=router,
                dest_address="test1",
                message=FakeMessage(index),
                wait=False,
            )
        )

    with pytest.raises(ServerClosed):
        # Just wait _list run.
        await asyncio.sleep(1)
        await caller.call(
            router=router, dest_address="test1", message=FakeMessage(2), wait=False
        )

    res0 = await futures[0]
    assert res0.message_id == 0

    with pytest.raises(ServerClosed):
        await futures[1]
