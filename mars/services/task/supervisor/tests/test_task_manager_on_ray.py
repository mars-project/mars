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

from ..... import oscar as mo
from .....oscar.backends.ray.utils import placement_group_info_to_addresses
from .....tests.core import require_ray
from .....utils import lazy_import
from ..manager import TaskConfigurationActor

ray = lazy_import("ray")


@require_ray
@pytest.mark.asyncio
async def test_task_manager_creation(ray_start_regular):
    mo.setup_cluster(
        address_to_resources=placement_group_info_to_addresses(
            "test_cluster", [{"CPU": 2}]
        )
    )
    # the pool is an ActorHandle, it does not have an async context.
    pool = await mo.create_actor_pool(
        "ray://test_cluster/0/0", n_process=2, labels=[None] + ["numa-0"] * 2
    )
    assert pool

    # create configuration
    await mo.create_actor(
        TaskConfigurationActor,
        dict(),
        dict(),
        uid=TaskConfigurationActor.default_uid(),
        address="ray://test_cluster/0/0",
    )

    configuration_ref = await mo.actor_ref(
        TaskConfigurationActor.default_uid(), address="ray://test_cluster/0/0"
    )
    await configuration_ref.get_config()
