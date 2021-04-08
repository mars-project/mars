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

import numpy as np
import pytest

import mars.tensor as mt
from mars.deploy.oscar.local import new_cluster
from mars.tests.core import CONFIG_TEST_FILE


@pytest.mark.asyncio
async def test_execute():
    client = await new_cluster(subprocess_start_method='spawn',
                               config=CONFIG_TEST_FILE,
                               n_cpu=2)

    async with client:
        raw = np.random.RandomState(0).rand(10, 10)
        a = mt.tensor(raw, chunk_size=5)
        b = a + 1

        info = await client.session.execute(b)
        await info
        assert info.progress() == 1
        np.testing.assert_equal(raw + 1, (await client.session.fetch(b))[0])
