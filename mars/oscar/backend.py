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
from typing import Dict, Type

from .context import register_backend_context
from .driver import register_backend_driver


__all__ = ["BaseActorBackend", "register_backend", "get_backend"]


class BaseActorBackend(ABC):
    @staticmethod
    @abstractmethod
    def name():
        pass

    @staticmethod
    @abstractmethod
    def get_context_cls():
        pass

    @staticmethod
    async def create_actor_pool(
        address: str,
        n_process: int = None,
        **kwargs
    ):
        pass

    @staticmethod
    @abstractmethod
    def get_driver_cls():
        pass


_scheme_to_backend_cls: Dict[str, Type[BaseActorBackend]] = dict()


def register_backend(backend_cls: Type[BaseActorBackend]):
    _scheme_to_backend_cls[backend_cls.name()] = backend_cls
    register_backend_context(backend_cls.name(), backend_cls.get_context_cls())
    register_backend_driver(backend_cls.name(), backend_cls.get_driver_cls())
    return backend_cls


def get_backend(name):
    return _scheme_to_backend_cls[name]
