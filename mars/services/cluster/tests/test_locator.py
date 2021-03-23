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

import asyncio

import pytest

import mars.oscar as mo
from mars.services.cluster.locator import SupervisorLocatorActor
from mars.tests.core import mock


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool('127.0.0.1', n_process=0)
    await pool.start()
    yield pool
    await pool.stop()


async def _changing_watch_supervisors(self):
    idx = 0
    while True:
        try:
            await asyncio.sleep(0.1)
            # make sure supervisors are different every time
            idx = (idx + 2) % len(self._supervisors)
            yield self._supervisors[idx:idx + 2]
        except asyncio.CancelledError:
            break


@pytest.mark.asyncio
async def test_fixed_locator(actor_pool):
    addresses = ['1.2.3.4:1234', '1.2.3.4:1235',
                 '1.2.3.4:1236', '1.2.3.4:1237']
    locator_ref = await mo.create_actor(
        SupervisorLocatorActor, 'fixed', ','.join(addresses),
        address=actor_pool.external_address)

    assert await locator_ref.get_supervisor('mock_name') in addresses

    dbl_addrs = await locator_ref.get_supervisor('mock_name', 2)
    assert len(dbl_addrs) == 2
    assert all(addr in addresses for addr in dbl_addrs)

    await mo.destroy_actor(locator_ref)


@pytest.mark.asyncio
@mock.patch('mars.services.cluster.backends.FixedClusterBackend.watch_supervisors',
            new=_changing_watch_supervisors)
async def test_changing_locator(actor_pool):
    addresses = ['1.2.3.4:1234', '1.2.3.4:1235',
                 '1.2.3.4:1236', '1.2.3.4:1237']
    locator_ref = await mo.create_actor(
        SupervisorLocatorActor, 'fixed', ','.join(addresses),
        address=actor_pool.external_address)

    assert (await locator_ref.watch_supervisors_by_keys(['mock_name']))[0] in addresses
    assert (await locator_ref.watch_supervisors_by_keys(['mock_name']))[0] in addresses

    assert all(addr in addresses for addr in await locator_ref.watch_supervisors())

    await mo.destroy_actor(locator_ref)
