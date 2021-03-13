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

from .utils import addresses_to_placement_group_info
from ...driver import BaseActorDriver
from ....utils import lazy_import

ray = lazy_import("ray")
logger = logging.getLogger(__name__)


class RayActorDriver(BaseActorDriver):
    @classmethod
    def setup_cluster(cls, address_to_resources: Dict[str, Dict[str, Number]]):
        logger.info("Setup cluster with %s", address_to_resources)
        pg_name, bundles = addresses_to_placement_group_info(address_to_resources)
        logger.info("Creating placement group %s with bundles %s.", pg_name, bundles)
        # TODO(fyrestone): We should destroy the placement group when current job is dropped.
        pg = ray.util.placement_group(name=pg_name,
                                      bundles=bundles,
                                      strategy="SPREAD",
                                      lifetime="detached")
        ray.get(pg.ready())
        logger.info("Create placement group success.")
