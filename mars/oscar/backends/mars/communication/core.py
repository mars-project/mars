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

from urllib.parse import urlparse
from typing import Dict, Type

from .base import Client


_scheme_to_client_types: Dict[str, Type[Client]] = dict()


def register_client(client_type: Type[Client]):
    _scheme_to_client_types[client_type.scheme] = client_type
    return client_type


def get_client_type(address: str):
    scheme = urlparse(address).scheme
    if scheme == '':
        scheme = None
    if scheme not in _scheme_to_client_types:  # pragma: no cover
        raise ValueError(f'address illegal, address scheme '
                         f'should be one of '
                         f'{", ".join(_scheme_to_client_types)}, '
                         f'got {scheme}')
    return _scheme_to_client_types[scheme]
