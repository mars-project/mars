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

from typing import Dict, Type
from urllib.parse import urlparse

from .base import Client, Server

_scheme_to_client_types: Dict[str, Type[Client]] = dict()
_scheme_to_server_types: Dict[str, Type[Server]] = dict()


def register_client(client_type: Type[Client]):
    _scheme_to_client_types[client_type.scheme] = client_type
    return client_type


def register_server(server_type: Type[Server]):
    _scheme_to_server_types[server_type.scheme] = server_type
    return server_type


def _check_scheme(scheme: str, types: Dict):
    if scheme == '':
        scheme = None
    if scheme not in types:  # pragma: no cover
        raise ValueError(f'address illegal, address scheme '
                         f'should be one of '
                         f'{", ".join(types)}, '
                         f'got {scheme}')
    return scheme


def get_scheme(address: str) -> str:
    if '://' not in address:
        scheme = None
    else:
        scheme = urlparse(address).scheme
    return scheme


def get_client_type(address: str) -> Type[Client]:
    scheme = _check_scheme(get_scheme(address), _scheme_to_client_types)
    return _scheme_to_client_types[scheme]


def get_server_type(address: str) -> Type[Server]:
    scheme = _check_scheme(get_scheme(address), _scheme_to_server_types)
    return _scheme_to_server_types[scheme]


def gen_local_address(process_index: int) -> str:
    return f'dummy://{process_index}'
