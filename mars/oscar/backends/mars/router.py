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

from typing import Dict, List, Tuple, Type, Any, Optional

from .communication import get_client_type, Client

LOCAL_ADDRESS = 'dummy://'


class Router:
    """
    Router provides mapping from external address to internal address.
    """
    __slots__ = '_curr_external_addresses', '_mapping', '_cache'

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
        return Router._instance or _empty_router

    def __init__(self,
                 external_addresses: List[str],
                 mapping: Dict[str, str] = None):
        self._curr_external_addresses = external_addresses
        if mapping is None:
            mapping = dict()
        self._mapping = mapping
        self._cache: Dict[Tuple[str, Any], Client] = dict()

    def set_mapping(self, mapping: Dict[str, str]):
        self._mapping = mapping

    @property
    def external_address(self):
        if self._curr_external_addresses:
            return self._curr_external_addresses[0]

    def get_internal_address(self, external_address: str) -> str:
        if external_address in self._curr_external_addresses:
            # local address, use dummy address
            return LOCAL_ADDRESS
        # try to lookup inner address from address mapping
        return self._mapping.get(external_address)

    async def get_client(self,
                         external_address: str,
                         from_who: Any = None,
                         cached: bool = True,
                         **kw) -> Client:
        if cached and (external_address, from_who) in self._cache:
            return self._cache[external_address, from_who]

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


_empty_router = Router(list())
