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
import logging
import os
from numbers import Number
from typing import Dict

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
        # Note: Deep copy the dict to keep the origin values, because `bundles`
        # returned by `addresses_to_placement_group_info()` will be modified
        # by `ray.util.placement_group()`
        original_address_to_resources = copy.deepcopy(address_to_resources)
        pg_name, bundles = addresses_to_placement_group_info(address_to_resources)
        logger.info("Creating placement group %s with bundles %s.", pg_name, bundles)
        pg = ray.util.placement_group(name=pg_name, bundles=bundles, strategy="SPREAD")
        create_pg_timeout = 120
        done, _ = ray.wait([pg.ready()], timeout=create_pg_timeout)
        if not done:  # pragma: no cover
            raise Exception(
                f"""Can't create placement group {pg.bundle_specs} in {create_pg_timeout} seconds"""
            )
        cluster_info = {
            "original_address_to_resources": original_address_to_resources,
            "address_to_resources": address_to_resources,
            "pg_name": pg_name,
            "pg_group": pg,
            "main_pool_handles": [],  # Hold actor_handle to avoid actor being freed.
        }
        logger.info("Create placement group success.")
        cls._cluster_info = cluster_info

    @classmethod
    def stop_cluster(cls):
        logger.info("Stopping cluster %s.", cls._cluster_info)
        if not cls._cluster_info:  # pragma: no cover
            return
        pg_name = cls._cluster_info["pg_name"]
        pg = cls._cluster_info["pg_group"]
        for index, bundle_spec in enumerate(pg.bundle_specs):
            # Main pool took a process.
            # If supervisor is created in the same node with worker, it will take a process too.
            n_process = int(bundle_spec["CPU"]) + 2
            for process_index in reversed(range(n_process)):
                address = process_placement_to_address(
                    pg_name, index, process_index=process_index
                )
                try:
                    ray_actor = ray.get_actor(address)
                    if "COV_CORE_SOURCE" in os.environ:  # pragma: no cover
                        # must clean up first, or coverage info lost.
                        # must save the local reference until this is fixed:
                        # https://github.com/ray-project/ray/issues/7815
                        ray.get(ray_actor.cleanup.remote())
                    ray.kill(ray_actor, no_restart=True)
                    while True:
                        try:
                            ray.get(ray_actor.wait.remote(30))
                            logger.warning(
                                "Waiting actor %s to be killed.", ray_actor
                            )  # pragma: no cover
                        except ray.exceptions.RayActorError:
                            break
                except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                    pass
        ray.util.remove_placement_group(pg)
        cls._cluster_info = dict()
        logger.info("Stopped cluster %s.", pg_name)
