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

import copy
import os

import numpy as np
import pandas as pd
import pytest

from .... import tensor as mt
from .... import dataframe as md
from ....tests.core import DICT_NOT_EMPTY, require_ray
from ....utils import lazy_import
from ..local import new_cluster
from ..session import new_session
from ..tests import test_local
from ..tests.session import new_test_session
from .modules.utils import (  # noqa: F401; pylint: disable=unused-variable
    cleanup_third_party_modules_output,
    get_output_filenames,
)

ray = lazy_import("ray")

EXPECT_PROFILING_STRUCTURE = {
    "supervisor": {
        "general": {
            "optimize": 0.0005879402160644531,
            "stage_*": {
                "tile(*)": 0.008243083953857422,
                "gen_subtask_graph(*)": 0.012202978134155273,
                "run": 0.27870702743530273,
                "total": 0.30318617820739746,
            },
            "total": 0.30951380729675293,
        },
        "serialization": {},
        "most_calls": DICT_NOT_EMPTY,
        "slow_calls": DICT_NOT_EMPTY,
        "band_subtasks": {},
        "slow_subtasks": {},
    }
}
EXPECT_PROFILING_STRUCTURE_NO_SLOW = copy.deepcopy(EXPECT_PROFILING_STRUCTURE)
EXPECT_PROFILING_STRUCTURE_NO_SLOW["supervisor"]["slow_calls"] = {}


@pytest.mark.parametrize(indirect=True)
@pytest.fixture
async def create_cluster(request):
    start_method = os.environ.get("POOL_START_METHOD", None)
    client = await new_cluster(
        subprocess_start_method=start_method,
        backend="ray",
        n_worker=2,
        n_cpu=2,
        use_uvloop=False,
    )
    async with client:
        assert client.session.client is not None
        yield client, {}


@require_ray
@pytest.mark.parametrize("backend", ["ray"])
@pytest.mark.parametrize("_new_session", [new_session, new_test_session])
def test_new_session_backend(ray_start_regular_shared2, _new_session, backend):
    test_local.test_new_session_backend(_new_session, backend)


@require_ray
@pytest.mark.parametrize(
    "config",
    [
        [
            {
                "enable_profiling": {
                    "slow_calls_duration_threshold": 0,
                    "slow_subtasks_duration_threshold": 0,
                }
            },
            EXPECT_PROFILING_STRUCTURE,
        ],
        [
            {
                "enable_profiling": {
                    "slow_calls_duration_threshold": 1000,
                    "slow_subtasks_duration_threshold": 1000,
                }
            },
            EXPECT_PROFILING_STRUCTURE_NO_SLOW,
        ],
        [{}, {}],
    ],
)
@pytest.mark.asyncio
async def test_execute(ray_start_regular_shared2, create_cluster, config):
    await test_local.test_execute(create_cluster, config)


@require_ray
@pytest.mark.asyncio
async def test_iterative_tiling(ray_start_regular_shared2, create_cluster):
    await test_local.test_iterative_tiling(create_cluster)


@require_ray
@pytest.mark.parametrize("config", [{"backend": "ray"}])
def test_sync_execute(config):
    test_local.test_sync_execute(config)


@require_ray
@pytest.mark.asyncio
async def test_session_get_progress(ray_start_regular_shared2, create_cluster):
    await test_local.test_session_get_progress(create_cluster)


@require_ray
@pytest.mark.asyncio
async def test_execute_describe(ray_start_regular_shared2, create_cluster):
    # `describe` contains multiple shuffle.
    await test_local.test_execute_describe(create_cluster)


@require_ray
@pytest.mark.asyncio
async def test_shuffle(ray_start_regular_shared2, create_cluster):
    arr = np.random.RandomState(0).rand(31, 27)
    t1 = mt.tensor(arr, chunk_size=10).reshape(27, 31)
    t1.op.extra_params["_reshape_with_shuffle"] = True
    np.testing.assert_almost_equal(arr.reshape(27, 31), t1.to_numpy())

    np.testing.assert_equal(
        mt.bincount(mt.arange(5, 10)).to_numpy(), np.bincount(np.arange(5, 10))
    )

    # `RayExecutionContext.get_chunk_meta` not supported, skip dataframe.groupby
    # df = md.DataFrame(mt.random.rand(300, 4, chunk_size=100), columns=list("abcd"))
    # df["a"], df["b"] = (df["a"] * 5).astype(int), (df["b"] * 2).astype(int)
    # df.groupby(["a", "b"]).apply(lambda pdf: pdf.sum()).execute()
