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

from urllib.parse import urlparse, unquote
import posixpath


def address_to_placement_group_bundle(address):
    parsed_url = urlparse(unquote(address))
    if parsed_url.scheme != "ray":
        raise ValueError(f"The address scheme is not ray: {address}")
    # os.path.split will not handle backslashes (\) correctly,
    # so we use the posixpath.
    parts = []
    if parsed_url.netloc:
        tmp = parsed_url.path
        while tmp and tmp != "/":
            tmp, item = posixpath.split(tmp)
            parts.append(item)
    if parts and len(parts) != 1:
        raise ValueError(f"Only bundle index path is allowed in ray address {address}.")
    name, bundle_index = parsed_url.netloc, parts[-1] if parts else ""
    if bool(name) != bool(bundle_index):
        raise ValueError(f"Missing placement group name or bundle index from address {address}")
    if name and bundle_index:
        return name, int(bundle_index)
    else:
        return name, -1


def placement_group_bundle_to_address(pg_name, bundle_index):
    return f"ray://{pg_name}/{bundle_index}"


def addresses_to_placement_group_info(address_to_resources):
    bundles = {}
    pg_name = None
    for address, bundle_resources in address_to_resources.items():
        name, bundle_index = address_to_placement_group_bundle(address)
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
        address = placement_group_bundle_to_address(pg_name, bundle_index)
        addresses[address] = bundle_resources
    return addresses
