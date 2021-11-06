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

import pytest

from ....utils import lazy_import
from ..ray import new_cluster
from ..tests import test_local
from .modules.utils import (  # noqa: F401; pylint: disable=unused-variable
    cleanup_third_party_modules_output,
    get_output_filenames,
)

ray = lazy_import('ray')


@pytest.fixture
async def speculative_cluster():
    client = await new_cluster('test_cluster',
                               worker_num=10,
                               worker_cpu=2,
                               worker_mem=2 * 1024 ** 3,
                               supervisor_mem=100 * 1024 ** 2,
                               config={
                                   'scheduling': {
                                       'speculation': {
                                           'enabled': True,
                                           'interval': 1,
                                           'threshold': 0.3,
                                           'min_task_runtime': 2,
                                           'multiplier': 1.5
                                       },
                                       # used to kill hanged subtask to release slot.
                                       'subtask_cancel_timeout': 0.1
                                   }
                               })
    async with client:
        yield client


@pytest.mark.timeout(timeout=180)
@pytest.mark.parametrize('ray_large_cluster', [{'num_nodes': 2}], indirect=True)
@pytest.mark.asyncio
async def test_task_speculation_execution(ray_large_cluster, speculative_cluster):
    await test_local.test_task_speculation_execution(speculative_cluster)
