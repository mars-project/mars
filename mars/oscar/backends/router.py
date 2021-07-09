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

import threading
from typing import Dict, List, Tuple, Type, Any, Optional

from .communication import get_client_type, Client


class Router:
    """
    Router provides mapping from external address to internal address.
    """
    __slots__ = '_curr_external_addresses', '_local_mapping', \
                '_mapping', '_cache_local'

    _instance: "Router" = None

    @staticmethod
    def set_instance(router: Optional["Router"]):
        # Default router is set when an actor pool started
        Router._instance = router

    @staticmethod
    def get_instance() -> "Router":
        return Router._instance

    @staticmethod
    def get_instance_or_empty() -> "Router":
        return Router._instance or Router(list(), None)

    def __init__(self,
                 external_addresses: List[str],
                 local_address: Optional[str],
                 mapping: Dict[str, str] = None):
        self._curr_external_addresses = external_addresses
        self._local_mapping = dict()
        for addr in self._curr_external_addresses:
            self._local_mapping[addr] = local_address
        if mapping is None:
            mapping = dict()
        self._mapping = mapping
        self._cache_local = threading.local()

    @property
    def _cache(self) -> Dict[Tuple[str, Any], Client]:
        try:
            return self._cache_local.cache
        except AttributeError:
            cache = self._cache_local.cache = dict()
            return cache

    def set_mapping(self, mapping: Dict[str, str]):
        self._mapping = mapping

    def add_router(self, router: "Router"):
        self._curr_external_addresses.extend(router._curr_external_addresses)
        self._local_mapping.update(router._local_mapping)
        self._mapping.update(router._mapping)

    def remove_router(self, router: "Router"):
        for external_address in router._curr_external_addresses:
            try:
                self._curr_external_addresses.remove(external_address)
            except ValueError:
                pass
        for addr in router._local_mapping:
            self._local_mapping.pop(addr, None)
        for addr in router._mapping:
            self._mapping.pop(addr, None)

    @property
    def external_address(self):
        if self._curr_external_addresses:
            return self._curr_external_addresses[0]

    def get_internal_address(self, external_address: str) -> str:
        if external_address in self._curr_external_addresses:
            # local address, use dummy address
            return self._local_mapping.get(external_address)
        # try to lookup inner address from address mapping
        return self._mapping.get(external_address)

    async def get_client(self,
                         external_address: str,
                         from_who: Any = None,
                         cached: bool = True,
                         **kw) -> Client:
        if cached and (external_address, from_who) in self._cache:
            cached_client = self._cache[external_address, from_who]
            if cached_client.closed:
                # closed before, ignore it
                del self._cache[external_address, from_who]
            else:
                return cached_client

        address = self.get_internal_address(external_address)
        if address is None:
            # no inner address, just use external address
            address = external_address
        client_type: Type[Client] = get_client_type(address)
        local_address = self._curr_external_addresses[0] \
            if self._curr_external_addresses else None
        client = await client_type.connect(
            address, local_address=local_address, **kw)
        if cached:
            self._cache[external_address, from_who] = client
        return client
