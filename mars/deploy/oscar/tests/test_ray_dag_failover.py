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


import operator
from functools import reduce

import pandas as pd
import pytest

import mars
from .... import dataframe as md
from .... import tensor as mt
from ....tests.core import require_ray
from ....utils import lazy_import

ray = lazy_import("ray")
try:
    from ray.exceptions import ObjectReconstructionFailedMaxAttemptsExceededError
except ImportError:  # pragma: no cover
    ObjectReconstructionFailedMaxAttemptsExceededError = None


@require_ray
@pytest.mark.parametrize(
    "ray_large_cluster",
    [{"num_nodes": 0}],
    indirect=True,
)
@pytest.mark.parametrize("reconstruction_enabled", [True, False])
@pytest.mark.skipif(
    ObjectReconstructionFailedMaxAttemptsExceededError is None,
    reason="Not support ObjectReconstructionFailedMaxAttemptsExceededError",
)
def test_basic_object_reconstruction(
    ray_large_cluster, reconstruction_enabled, stop_mars
):
    config = {
        "num_heartbeats_timeout": 10,
        "raylet_heartbeat_period_milliseconds": 200,
        "object_timeout_milliseconds": 200,
    }
    # Workaround to reset the config to the default value.
    if not reconstruction_enabled:
        config["lineage_pinning_enabled"] = False
        subtask_max_retries = 0
    else:
        subtask_max_retries = 1

    cluster = ray_large_cluster
    # Head node with no resources.
    cluster.add_node(
        num_cpus=0,
        _system_config=config,
        enable_object_reconstruction=reconstruction_enabled,
    )
    ray.init(address=cluster.address)
    # Node to place the initial object.
    node_to_kill = cluster.add_node(num_cpus=1, object_store_memory=10**8)
    mars.new_session(
        backend="ray",
        config={"scheduling.subtask_max_retries": subtask_max_retries},
        default=True,
    )
    cluster.wait_for_nodes()

    df = md.DataFrame(mt.random.RandomState(0).rand(2_000_000, 1, chunk_size=1_000_000))
    df.execute()
    # this will submit new ray tasks
    df2 = df.map_chunk(lambda pdf: pdf * 2).execute()
    executed_infos = df2.fetch_infos(fields=["object_refs"])
    object_refs = reduce(operator.concat, executed_infos["object_refs"])
    head5 = df2.head(5).to_pandas()

    cluster.remove_node(node_to_kill, allow_graceful=False)
    node_to_kill = cluster.add_node(num_cpus=1, object_store_memory=10**8)

    # use a dependent_task to avoid fetch lost objects to local
    @ray.remote
    def dependent_task(x):
        return x

    if reconstruction_enabled:
        ray.get([dependent_task.remote(ref) for ref in object_refs])
        new_head5 = df2.head(5).to_pandas()
        pd.testing.assert_frame_equal(head5, new_head5)
    else:
        with pytest.raises(ray.exceptions.RayTaskError):
            df2.head(5).to_pandas()
        with pytest.raises(ray.exceptions.ObjectLostError):
            ray.get(object_refs)

    # Losing the object a second time will cause reconstruction to fail because
    # we have reached the max task retries.
    cluster.remove_node(node_to_kill, allow_graceful=False)
    cluster.add_node(num_cpus=1, object_store_memory=10**8)

    if reconstruction_enabled:
        with pytest.raises(ObjectReconstructionFailedMaxAttemptsExceededError):
            ray.get(object_refs)
    else:
        with pytest.raises(ray.exceptions.ObjectLostError):
            ray.get(object_refs)
