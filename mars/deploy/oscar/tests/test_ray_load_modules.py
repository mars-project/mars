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

import numpy as np
import pytest

from .... import tensor as mt
from ....tests.core import require_ray
from ....utils import lazy_import
from ..ray import (
    _load_config,
    new_cluster,
)
from ..session import get_default_session, new_session
from .modules.utils import (  # noqa: F401  # pylint: disable=unused-variable
    cleanup_third_party_modules_output,
    get_output_filenames,
)

ray = lazy_import("ray")

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "local_test_with_ray_config.yml")
CONFIG_THIRD_PARTY_MODULES_TEST_FILE = os.path.join(
    os.path.dirname(__file__), "ray_test_with_third_parity_modules_config.yml"
)


@pytest.fixture
async def create_cluster(request):
    param = getattr(request, "param", {})
    ray_config = _load_config(CONFIG_FILE)
    ray_config.update(param.get("config", {}))
    client = await new_cluster(
        supervisor_mem=1 * 1024**3,
        worker_num=2,
        worker_cpu=2,
        worker_mem=1 * 1024**3,
        config=ray_config,
    )
    async with client:
        yield client, param


@require_ray
@pytest.mark.parametrize(
    "config_exception",
    [
        [set(), pytest.raises(TypeError, match="set")],
        [
            {"supervisor": ["not_exists_for_supervisor"]},
            pytest.raises(ModuleNotFoundError, match="not_exists_for_supervisor"),
        ],
        [
            {"worker": ["not_exists_for_worker"]},
            pytest.raises(ModuleNotFoundError, match="not_exists_for_worker"),
        ],
    ],
)
@pytest.mark.asyncio
async def test_load_third_party_modules(ray_start_regular, config_exception):
    third_party_modules_config, expected_exception = config_exception
    config = _load_config()

    config["third_party_modules"] = third_party_modules_config
    with expected_exception:
        await new_cluster(
            worker_num=1,
            worker_cpu=1,
            worker_mem=1 * 1024**3,
            config=config,
        )


@require_ray
@pytest.mark.parametrize(
    "create_cluster",
    [
        {
            "config": {
                "third_party_modules": {
                    "worker": ["mars.deploy.oscar.tests.modules.replace_op"]
                },
            },
        }
    ],
    indirect=True,
)
@pytest.mark.asyncio
def test_load_third_party_modules2(ray_start_regular, create_cluster):
    client = create_cluster[0]
    assert client.session
    session = new_session(address=client.address)
    with session:
        raw = np.random.RandomState(0).rand(10, 10)
        a = mt.tensor(raw, chunk_size=5)
        b = a + 1
        b.execute(show_progress=False)
        result = b.fetch()

        np.testing.assert_equal(raw - 1, result)

    assert get_default_session() is None


@require_ray
@pytest.mark.asyncio
async def test_load_third_party_modules_from_config(
    ray_start_regular, cleanup_third_party_modules_output  # noqa: F811
):
    client = await new_cluster(
        supervisor_mem=1 * 1024**3,
        worker_num=1,
        worker_cpu=1,
        worker_mem=1 * 1024**3,
        config=CONFIG_THIRD_PARTY_MODULES_TEST_FILE,
    )
    async with client:
        # 1 supervisor, 1 worker main pools, 1 worker sub pools.
        assert len(get_output_filenames()) == 3
