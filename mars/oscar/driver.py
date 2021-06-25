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

from abc import ABC, abstractmethod
from numbers import Number
from typing import Dict, Type


class BaseActorDriver(ABC):
    @classmethod
    @abstractmethod
    def setup_cluster(cls, address_to_resources: Dict[str, Dict[str, Number]]):
        """
        Setup cluster according to given resources,
        resources is a dict, e.g. {'CPU': 3, 'GPU': 1}

        Parameters
        ----------
        address_to_resources: dict
            resources that required for each node.
        """
        pass


_backend_driver_cls: Dict[str, Type[BaseActorDriver]] = dict()


def register_backend_driver(scheme: str, cls: Type[BaseActorDriver]):
    assert issubclass(cls, BaseActorDriver)
    _backend_driver_cls[scheme] = cls
