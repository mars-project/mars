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

import asyncio
import enum
import logging
import os
import posixpath
from urllib.parse import urlparse, unquote

from ....utils import lazy_import, lazy_import_on_load

ray = lazy_import("ray")

logger = logging.getLogger(__name__)


def get_placement_group(pg_name):  # pragma: no cover
    return ray.util.get_placement_group(pg_name)


def process_address_to_placement(address):
    """
    Parameters
    ----------
    address: str
        The address of an actor pool which running in a ray actor. It's also
        the name of the ray actor. address ex: ray://${pg_name}/${bundle_index}/${process_index}

    Returns
    -------
    tuple
        A tuple consisting of placement group name, bundle index, process index.
    """
    name, parts = _address_to_placement(address)
    if not parts or len(parts) != 2:
        raise ValueError(
            f"Only bundle index and process index path are allowed in ray "
            f"address {address} but got {parts}."
        )
    bundle_index, process_index = parts
    return name, int(bundle_index), int(process_index)


def node_address_to_placement(address):
    """
    Parameters
    ----------
    address : str
        The address of a node. ex: ray://${pg_name}/${bundle_index}

    Returns
    -------
    tuple
        A tuple consisting of placement group name, bundle index.
    """
    name, parts = _address_to_placement(address)
    if not parts or len(parts) != 1:
        raise ValueError(
            f"Only bundle index path is allowed in ray address {address} but got {parts}"
        )
    bundle_index = parts[0]
    return name, int(bundle_index)


def _address_to_placement(address):
    """

    Parameters
    ----------
    address : str
        The address of a node or an actor pool which running in a ray actor.

    Returns
    -------
    tuple
        A tuple consisting of placement group name, bundle index, process index.
    """
    parsed_url = urlparse(unquote(address))
    if parsed_url.scheme != "ray":
        raise ValueError(f"The address scheme is not ray: {address}")
    # os.path.split will not handle backslashes (\) correctly,
    # so we use the posixpath.
    parts = []
    if parsed_url.netloc:
        tmp = parsed_url.path
        while tmp and tmp != "/":
            tmp2, item = posixpath.split(tmp)
            parts.append(item)
            if tmp2 != tmp:
                tmp = tmp2
            else:
                parts.append(tmp2)
                break
    parts = list(reversed(parts))
    return parsed_url.netloc, parts


def process_placement_to_address(
    pg_name: str, bundle_index: int, process_index: int = 0
):
    return f"ray://{pg_name}/{bundle_index}/{process_index}"


def node_placement_to_address(pg_name, bundle_index):
    return f"ray://{pg_name}/{bundle_index}"


def addresses_to_placement_group_info(address_to_resources):
    bundles = {}
    pg_name = None
    for address, bundle_resources in address_to_resources.items():
        name, bundle_index = node_address_to_placement(address)
        if pg_name is None:
            pg_name = name
        else:
            if name != pg_name:
                raise ValueError(
                    "All addresses should have consistent placement group names."
                )
        bundles[bundle_index] = bundle_resources
    sorted_bundle_keys = sorted(bundles.keys())
    if sorted_bundle_keys != list(range(len(address_to_resources))):
        raise ValueError("The addresses contains invalid bundle.")
    bundles = [bundles[k] for k in sorted_bundle_keys]
    if not pg_name:
        raise ValueError("Can't find a valid placement group name.")
    return pg_name, bundles


def placement_group_info_to_addresses(pg_name, bundles):
    addresses = {}
    for bundle_index, bundle_resources in enumerate(bundles):
        address = node_placement_to_address(pg_name, bundle_index)
        addresses[address] = bundle_resources
    return addresses


async def kill_and_wait(
    actor_handle: "ray.actor.ActorHandle", no_restart=False, timeout: float = 30
):
    if "COV_CORE_SOURCE" in os.environ:  # pragma: no cover
        try:
            # must clean up first, or coverage info lost
            await actor_handle.cleanup.remote()
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except
            pass
    r = actor_handle.wait.remote(timeout)
    ray.kill(actor_handle, no_restart=no_restart)
    ready, _ = await asyncio.wait([r], timeout=timeout)
    if ready:
        try:
            await r
        except ray.exceptions.RayActorError:
            return  # We expect a RayActorError, it indicated that the actor is died.
    raise Exception(
        f"The actor {actor_handle} is not died after ray.kill {timeout} seconds."
    )


@lazy_import_on_load(ray)
def _patch_event_security():
    global ray

    if ray and not hasattr(ray, "report_event"):  # pragma: no cover
        # lower version of ray doesn't support event

        class EventSeverity(enum.Enum):
            INFO = 0
            WARNING = 1
            ERROR = 2
            FATAL = 3

        def _report_event(severity, label, message):
            logger.warning(
                "severity: %s, label: %s, message: %s.", severity, label, message
            )

        import ray

        ray.EventSeverity = EventSeverity
        ray.report_event = _report_event


def report_event(severity, label, message):
    if ray and ray.is_initialized():
        severity = (
            getattr(ray.EventSeverity, severity)
            if isinstance(severity, str)
            else severity
        )
        ray.report_event(severity, label, message)
