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
import subprocess

import pytest

from mars.serialization.ray import register_ray_serializers, unregister_ray_serializers
from mars.oscar.backends.router import Router
from mars.oscar.backends.ray.communication import RayServer
from mars.utils import lazy_import

ray = lazy_import('ray')


@pytest.fixture
def ray_start_regular(request):
    param = getattr(request, "param", {})
    if not param.get('enable', True):
        yield
    else:
        register_ray_serializers()
        try:
            yield ray.init(num_cpus=20)
        finally:
            ray.shutdown()
            unregister_ray_serializers()
            Router.set_instance(None)
            RayServer.clear()
            if 'COV_CORE_SOURCE' in os.environ:
                # Remove this when https://github.com/ray-project/ray/issues/16802 got fixed
                subprocess.check_call(["ray", "stop", "--force"])


@pytest.fixture
def ray_large_cluster():
    try:
        from ray.cluster_utils import Cluster
    except ModuleNotFoundError:
        from ray._private.cluster_utils import Cluster
    cluster = Cluster()
    remote_nodes = []
    num_nodes = 3
    for i in range(num_nodes):
        remote_nodes.append(cluster.add_node(num_cpus=10))
        if len(remote_nodes) == 1:
            ray.init(address=cluster.address)
    register_ray_serializers()
    try:
        yield
    finally:
        unregister_ray_serializers()
        Router.set_instance(None)
        RayServer.clear()
        ray.shutdown()
        cluster.shutdown()
        if 'COV_CORE_SOURCE' in os.environ:
            # Remove this when https://github.com/ray-project/ray/issues/16802 got fixed
            subprocess.check_call(["ray", "stop", "--force"])


__all__ = ['ray_start_regular', 'ray_large_cluster']
