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

from random import choice
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

from ...utils import implements
from ..core import ActorRef
from ..errors import NoIdleSlot
from .config import ActorPoolConfig
from .message import _MessageBase


allocated_value = Tuple["AllocateStrategy", Optional[_MessageBase]]
allocated_values = Dict[Optional[ActorRef], allocated_value]
allocated_type = Dict[str, allocated_values]


class AllocateStrategy(ABC):
    __slots__ = ()

    @abstractmethod
    def get_allocated_address(self,
                              config: ActorPoolConfig,
                              allocated: allocated_type) -> str:
        """
        Get external address where the actor allocated to.

        Parameters
        ----------
        config: ActorPoolConfig
            Actor pool config.
        allocated:
            Already allocated of actor and its strategy.

        Returns
        -------
        allocated_address: str
            External address to allocate.
        """


class AddressSpecified(AllocateStrategy):
    __slots__ = 'address',

    def __init__(self, address):
        self.address = address

    @implements(AllocateStrategy.get_allocated_address)
    def get_allocated_address(self,
                              config: ActorPoolConfig,
                              allocated: allocated_type) -> str:
        return self.address


class MainPool(AllocateStrategy):
    __slots__ = ()

    @implements(AllocateStrategy.get_allocated_address)
    def get_allocated_address(self,
                              config: ActorPoolConfig,
                              allocated: allocated_type) -> str:
        # allocate to main process
        main_process_index = config.get_process_indexes()[0]
        return config.get_external_address(main_process_index)


class Random(AllocateStrategy):
    __slots__ = ()

    @implements(AllocateStrategy.get_allocated_address)
    def get_allocated_address(self,
                              config: ActorPoolConfig,
                              allocated: allocated_type) -> str:
        return choice(config.get_external_addresses())


class RandomSubPool(AllocateStrategy):
    __slots__ = ()

    @implements(AllocateStrategy.get_allocated_address)
    def get_allocated_address(self,
                              config: ActorPoolConfig,
                              allocated: allocated_type) -> str:
        return choice(config.get_external_addresses()[1:])


class ProcessIndex(AllocateStrategy):
    __slots__ = 'process_index',

    def __init__(self, process_index: int):
        self.process_index = process_index

    @implements(AllocateStrategy.get_allocated_address)
    def get_allocated_address(self,
                              config: ActorPoolConfig,
                              allocated: allocated_type) -> str:
        actual_process_index = config.get_process_indexes()[self.process_index]
        return config.get_pool_config(actual_process_index)['external_address'][0]


class RandomLabel(AllocateStrategy):
    __slots__ = 'label',

    def __init__(self, label):
        self.label = label

    @implements(AllocateStrategy.get_allocated_address)
    def get_allocated_address(self,
                              config: ActorPoolConfig,
                              allocated: allocated_type) -> str:
        return choice(config.get_external_addresses(label=self.label))


class IdleLabel(AllocateStrategy):
    __slots__ = 'label', 'mark'

    def __init__(self, label, mark):
        self.label = label
        self.mark = mark

    def __hash__(self):
        return hash((type(self), self.label, self.mark))

    def __eq__(self, other):
        return isinstance(other, IdleLabel) and \
               self.label == other.label and \
               self.mark == other.mark

    @implements(AllocateStrategy.get_allocated_address)
    def get_allocated_address(self,
                              config: ActorPoolConfig,
                              allocated: allocated_type) -> str:
        addresses = config.get_external_addresses(label=self.label)
        for addr in addresses:
            occupied = False
            for strategy, _ in allocated.get(addr, dict()).values():
                if strategy == self:
                    occupied = True
                    break
            if not occupied:
                return addr
        raise NoIdleSlot(f'No idle slot for creating actor '
                         f'with label {self.label}, mark {self.mark}')
