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

import os
import pytest
import numpy as np

import mars.tensor as mt
from mars.core.session import get_default_session
from mars.deploy.oscar.local import new_cluster
from mars.services.fault_injection.api import FaultInjectionAPI

CONFIG_FILE = os.path.join(
        os.path.dirname(__file__), 'fault_injection_config.yml')


@pytest.fixture
async def fault_cluster(request):
    param = getattr(request, "param", {})
    start_method = os.environ.get('POOL_START_METHOD', None)
    client = await new_cluster(subprocess_start_method=start_method,
                               config=CONFIG_FILE,
                               n_worker=2,
                               n_cpu=2)
    async with client:
        fault_injection_api = await FaultInjectionAPI.create(client.session.address)
        await fault_injection_api.set_options(param["options"])
        yield client


@pytest.mark.parametrize("fault_cluster",
                         [{"options": {"fault_count": 1}}],
                         indirect=True)
@pytest.mark.asyncio
async def test_invalid_fault_options(fault_cluster):
    fault_injection_api = await FaultInjectionAPI.create(fault_cluster.session.address)
    with pytest.raises(ValueError, match='invalid_key'):
        await fault_injection_api.set_options({"invalid_key": 1})


@pytest.mark.parametrize("fault_cluster",
                         [{"options": {"fault_count": 1}}],
                         indirect=True)
@pytest.mark.asyncio
async def test_fault_inject_subtask_processor(fault_cluster):
    session = get_default_session()

    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    # TODO(fyrestone): We can use b.execute() as
    with pytest.raises(RuntimeError, match='Fault Injection'):
        info = await session.execute(b)
        await info

    info = await session.execute(b)
    await info
    assert info.result() is None
    assert info.exception() is None

    r = await session.fetch(b)
    np.testing.assert_array_equal(r[0], raw + 1)
