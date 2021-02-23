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

from abc import ABC, abstractmethod
from typing import Type

from .context import register_backend_context
from .driver import register_backend_driver


__all__ = ["BaseActorBackend", "register_backend"]


class BaseActorBackend(ABC):
    # allocate strategy is for Mars backend only
    support_allocate_strategy = False

    @staticmethod
    @abstractmethod
    def name():
        pass

    @staticmethod
    @abstractmethod
    def get_context_cls():
        pass

    @staticmethod
    @abstractmethod
    def get_driver_cls():
        pass


def register_backend(backend_cls: Type[BaseActorBackend]):
    register_backend_context(backend_cls.name(), backend_cls.get_context_cls())
    register_backend_driver(backend_cls.name(), backend_cls.get_driver_cls())
    return backend_cls
