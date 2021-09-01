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

import os
import uuid
import pytest
import numpy as np

from .... import tensor as mt
from ..local import new_cluster
from ..session import get_default_async_session

CONFIG_FILE = os.path.join(
        os.path.dirname(__file__), 'fault_injection_config.yml')


class FaultInjectionManager:
    name = str(uuid.uuid4())

    def __init__(self):
        self._fault_count = 1

    def on_execute_operand(self):
        if self._fault_count > 0:
            self._fault_count -= 1
            return True
        return False


@pytest.fixture
async def fault_cluster():
    start_method = os.environ.get('POOL_START_METHOD', None)
    client = await new_cluster(subprocess_start_method=start_method,
                               config=CONFIG_FILE,
                               n_worker=2,
                               n_cpu=2)
    async with client:
        await client.session.create_remote_object(
            client.session.session_id,
            FaultInjectionManager.name,
            FaultInjectionManager)
        assert client.session.get_remote_object(
            client.session.session_id,
            FaultInjectionManager.name) is not None
        yield client
        client.session.destroy_remote_object(
            client.session.session_id,
            FaultInjectionManager.name)


@pytest.mark.asyncio
async def test_fault_inject_subtask_processor(fault_cluster):
    extra_config = {'fault_injection_manager_name': FaultInjectionManager.name}
    session = get_default_async_session()

    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    # TODO(fyrestone): We can use b.execute() when the issue
    # https://github.com/mars-project/mars/issues/2165 is fixed
    with pytest.raises(RuntimeError, match='Fault Injection'):
        info = await session.execute(b, extra_config=extra_config)
        await info

    info = await session.execute(b, extra_config=extra_config)
    await info
    assert info.result() is None
    assert info.exception() is None

    r = await session.fetch(b)
    np.testing.assert_array_equal(r, raw + 1)
