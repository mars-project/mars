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
import collections

import pytest

import mars.oscar as mo
from mars.tests.core import require_ray
from ..utils import (
    address_to_placement_group_bundle,
    placement_group_bundle_to_address,
    addresses_to_placement_group_info,
    placement_group_info_to_addresses,
)

TEST_PLACEMENT_GROUP_NAME = 'test_placement_group'
TEST_PLACEMENT_GROUP_BUNDLES = [{"CPU": 3}, {"CPU": 5}, {"CPU": 7}]
TEST_ADDRESS_TO_RESOURCES = placement_group_info_to_addresses(TEST_PLACEMENT_GROUP_NAME,
                                                              TEST_PLACEMENT_GROUP_BUNDLES)


class DummyActor(mo.Actor):
    def __init__(self, index):
        super().__init__()
        self._index = index

    def getppid(self):
        return os.getppid()

    def index(self):
        return self._index


@require_ray
@pytest.mark.asyncio
class Test:
    def setup_method(self, _):
        import ray
        from ray.cluster_utils import Cluster
        self.cluster = Cluster()
        remote_nodes = []
        num_nodes = 3
        for i in range(num_nodes):
            remote_nodes.append(self.cluster.add_node(num_cpus=10))
            if len(remote_nodes) == 1:
                ray.init(address=self.cluster.address)
        mo.setup_cluster(address_to_resources=TEST_ADDRESS_TO_RESOURCES)

    def teardown_method(self, _):
        import ray
        ray.shutdown()
        self.cluster.shutdown()

    def test_address_to_placement_group_bundle(self):
        # Missing bundle index.
        with pytest.raises(ValueError):
            address_to_placement_group_bundle("ray://bundle_name")
        # Extra path is not allowed.
        with pytest.raises(ValueError):
            address_to_placement_group_bundle("ray://bundle_name/0/")
        # The scheme is not ray
        with pytest.raises(ValueError):
            address_to_placement_group_bundle("http://bundle_name/0")
        # The bundle index is not an int string.
        with pytest.raises(ValueError):
            address_to_placement_group_bundle("ray://abc/def")
        pg_name, bundle_index = address_to_placement_group_bundle("ray://bundle_name/0")
        assert pg_name == "bundle_name"
        assert bundle_index == 0
        pg_name, bundle_index = address_to_placement_group_bundle("ray://127.0.0.1/1")
        assert pg_name == "127.0.0.1"
        assert bundle_index == 1
        pg_name, bundle_index = address_to_placement_group_bundle("ray://127.0.0.1%2F2")
        assert pg_name == "127.0.0.1"
        assert bundle_index == 2
        pg_name, bundle_index = address_to_placement_group_bundle("ray://")
        assert pg_name == ""
        assert bundle_index == -1

    def test_addresses_to_placement_group_info(self):
        # Missing bundle index 1
        with pytest.raises(ValueError):
            addresses_to_placement_group_info({"ray://127.0.0.1/0": {"CPU": 1},
                                               "ray://127.0.0.1/2": {"CPU": 1}})
        # The bundle index is not starts from 0
        with pytest.raises(ValueError):
            addresses_to_placement_group_info({"ray://127.0.0.1/1": {"CPU": 1}})
        pg_name, bundles = addresses_to_placement_group_info({"ray://127.0.0.1/0": {"CPU": 1}})
        assert pg_name == "127.0.0.1"
        assert bundles == [{"CPU": 1}]
        pg_name, bundles = addresses_to_placement_group_info({"ray://127.0.0.1/4": {"CPU": 4},
                                                              "ray://127.0.0.1/2": {"CPU": 2},
                                                              "ray://127.0.0.1/1": {"CPU": 1},
                                                              "ray://127.0.0.1/3": {"CPU": 3},
                                                              "ray://127.0.0.1/0": {"CPU": 0}})
        assert pg_name == "127.0.0.1"
        assert bundles == [{"CPU": 0}, {"CPU": 1}, {"CPU": 2}, {"CPU": 3}, {"CPU": 4}]
        pg_name, bundles = addresses_to_placement_group_info(TEST_ADDRESS_TO_RESOURCES)
        assert pg_name == TEST_PLACEMENT_GROUP_NAME
        assert bundles == TEST_PLACEMENT_GROUP_BUNDLES

    async def test_create_actor_in_placement_group(self):
        actor_refs = []
        for i, r in enumerate(TEST_PLACEMENT_GROUP_BUNDLES):
            for _ in range(r["CPU"]):
                address = placement_group_bundle_to_address(TEST_PLACEMENT_GROUP_NAME, i)
                actor_ref = await mo.create_actor(DummyActor, i, address=address)
                actor_refs.append(actor_ref)
        results = []
        for actor_ref in actor_refs:
            ppid = await actor_ref.getppid()
            index = await actor_ref.index()
            results.append((ppid, index))

        counter = collections.Counter(results)
        assert len(counter) == len(TEST_PLACEMENT_GROUP_BUNDLES)
        assert sorted(counter.values()) == sorted(r["CPU"] for r in TEST_PLACEMENT_GROUP_BUNDLES)
