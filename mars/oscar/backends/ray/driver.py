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

import logging
import os
from numbers import Number
from typing import Dict

from ....serialization.ray import register_ray_serializers, unregister_ray_serializers
from ....utils import lazy_import
from ...driver import BaseActorDriver
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
        create_pg_timeout = 10
        done, _ = ray.wait([pg.ready()], timeout=create_pg_timeout)
        if not done:
            raise Exception(f'''Can't create placement group in {create_pg_timeout} seconds''')
        cluster_info = {
            'address_to_resources': address_to_resources,
            'pg_name': pg_name,
            'pg_group': pg,
            'main_pool_handles': []  # Hold actor_handle to avoid actor being freed.
        }
        logger.info("Create placement group success.")
        cls._cluster_info = cluster_info
        register_ray_serializers()

    @classmethod
    def stop_cluster(cls):
        logger.info('Stopping cluster %s.', cls._cluster_info)
        pg_name = cls._cluster_info['pg_name']
        pg = cls._cluster_info['pg_group']
        for index, bundle_spec in enumerate(pg.bundle_specs):
            n_process = int(bundle_spec["CPU"]) + 1
            for process_index in range(n_process):
                address = process_placement_to_address(pg_name, index, process_index=process_index)
                try:
                    if 'COV_CORE_SOURCE' in os.environ:  # pragma: no cover
                        # must clean up first, or coverage info lost
                        ray.get(ray.get_actor(address).cleanup.remote())
                    ray.kill(ray.get_actor(address))
                except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                    pass
        ray.util.remove_placement_group(pg)
        cls._cluster_info = dict()
        unregister_ray_serializers()
        logger.info('Stopped cluster %s.', pg_name)
