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

import logging
from numbers import Number
from typing import Dict

from ....utils import lazy_import
from ...driver import BaseActorDriver
from .pool import RayMainPool
from .utils import process_placement_to_address, addresses_to_placement_group_info

ray = lazy_import("ray")
logger = logging.getLogger(__name__)


class RayActorDriver(BaseActorDriver):
    _cluster_info = dict()

    @classmethod
    def setup_cluster(cls, address_to_resources: Dict[str, Dict[str, Number]]):
        logger.info("Setup cluster with %s", address_to_resources)
        pg_name, bundles = addresses_to_placement_group_info(address_to_resources)
        logger.info("Creating placement group %s with bundles %s.", pg_name, bundles)
        pg = ray.util.placement_group(name=pg_name,
                                      bundles=bundles,
                                      strategy="SPREAD")
        ray.get(pg.ready())
        cluster_info = {
            'address_to_resources': address_to_resources,
            'pg_name': pg_name,
            'pg_group': pg,
            'main_pool_handles': []  # Hold actor_handle to avoid actor being freed.
        }
        logger.info("Create placement group success.")
        for index, bundle_spec in enumerate(bundles):
            address = process_placement_to_address(pg_name, index, process_index=0)
            actor_handle = ray.remote(RayMainPool).options(
                num_cpus=0,  # main pool doesn't do horse work, mark it use no cpu.
                name=address, placement_group=pg, placement_group_bundle_index=index).remote()
            n_process = bundle_spec["CPU"]
            ray.get(actor_handle.start.remote(address, n_process))
            cluster_info['main_pool_handles'].append(actor_handle)
        logger.info('Create cluster success.')
        cls._cluster_info = cluster_info

    @classmethod
    def stop_cluster(cls):
        logger.info('Stopping cluster %s.', cls._cluster_info)
        pg_name = cls._cluster_info['pg_name']
        pg = cls._cluster_info['pg_group']
        for index, bundle_spec in enumerate(pg.bundle_specs):
            n_process = int(bundle_spec["CPU"])
            for process_index in range(n_process):
                address = process_placement_to_address(pg_name, index, process_index=process_index)
                try:
                    ray.kill(ray.get_actor(address))
                except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                    pass
        ray.util.remove_placement_group(pg)
        cls._cluster_info = dict()
        logger.info('Stopped cluster %s.', pg_name)
