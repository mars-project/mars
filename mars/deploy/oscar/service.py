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
from typing import List, Dict, Union

from ...resource import Resource
from ...services import start_services, stop_services, NodeRole

logger = logging.getLogger(__name__)


async def start_supervisor(
    address: str,
    lookup_address: str = None,
    modules: Union[List, str, None] = None,
    config: Dict = None,
    web: Union[str, bool] = "auto",
):
    logger.debug("Starting Mars supervisor at %s", address)
    lookup_address = lookup_address or address
    backend = config["cluster"].get("backend", "fixed")
    if backend == "fixed" and config["cluster"].get("lookup_address") is None:
        config["cluster"]["lookup_address"] = lookup_address
    if web:
        # try to append web to services
        config["services"].append("web")
    if modules:
        config["modules"] = modules
    try:
        await start_services(NodeRole.SUPERVISOR, config, address=address)
        logger.debug("Mars supervisor started at %s", address)
    except ImportError:
        if web == "auto":
            config["services"] = [
                service for service in config["services"] if service != "web"
            ]
            await start_services(NodeRole.SUPERVISOR, config, address=address)
            logger.debug("Mars supervisor started at %s", address)
            return False
        else:  # pragma: no cover
            raise
    else:
        return bool(web)


async def stop_supervisor(address: str, config: Dict = None):
    await stop_services(NodeRole.SUPERVISOR, address=address, config=config)


async def start_worker(
    address: str,
    lookup_address: str,
    band_to_resource: Dict[str, Resource],
    modules: Union[List, str, None] = None,
    config: Dict = None,
    mark_ready: bool = True,
):
    logger.debug("Starting Mars worker at %s", address)
    backend = config["cluster"].get("backend", "fixed")
    if backend == "fixed" and config["cluster"].get("lookup_address") is None:
        config["cluster"]["lookup_address"] = lookup_address
    if config["cluster"].get("resource") is None:
        config["cluster"]["resource"] = band_to_resource
    if any(
        band_name.startswith("gpu-") for band_name in band_to_resource
    ):  # pragma: no cover
        if "cuda" not in config["storage"]["backends"]:
            config["storage"]["backends"].append("cuda")
    if modules:
        config["modules"] = modules
    await start_services(
        NodeRole.WORKER, config, address=address, mark_ready=mark_ready
    )
    logger.debug("Mars worker started at %s", address)


async def stop_worker(address: str, config: Dict = None):
    await stop_services(NodeRole.WORKER, address=address, config=config)
