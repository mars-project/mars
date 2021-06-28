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
import posixpath
from urllib.parse import urlparse, unquote

from ....utils import lazy_import

ray = lazy_import('ray')

logger = logging.getLogger(__name__)


def get_placement_group(pg_name):  # pragma: no cover
    if hasattr(ray.util, "get_placement_group"):
        return ray.util.get_placement_group(pg_name)
    else:
        logger.warning("Current installed ray version doesn't support named placement group. "
                       "Actor will be created on arbitrary node randomly.")
        return None


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
        raise ValueError(f"Only bundle index and process index path are allowed in ray "
                         f"address {address} but got {parts}.")
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
        raise ValueError(f"Only bundle index path is allowed in ray address {address} but got {parts}")
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


def process_placement_to_address(pg_name: str, bundle_index: int, process_index: int = 0):
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
                raise ValueError("All addresses should have consistent placement group names.")
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
