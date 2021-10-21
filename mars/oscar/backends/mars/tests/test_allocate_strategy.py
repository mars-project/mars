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

import pytest

from .... import create_actor_ref
from ....errors import NoIdleSlot
from ...allocate_strategy import (
    AddressSpecified,
    MainPool,
    RandomSubPool,
    Random,
    RandomLabel,
    IdleLabel,
)
from ...config import ActorPoolConfig

config = ActorPoolConfig()
config.add_pool_conf(0, "main", "unixsocket:///0", "127.0.0.1:1111")
config.add_pool_conf(1, "test", "unixsocket:///1", "127.0.0.1:1112")
config.add_pool_conf(2, "test2", "unixsocket:///2", "127.0.0.1:1113")
config.add_pool_conf(3, "test", "unixsocket:///3", "127.0.0.1:1114")


def test_address_specified():
    addr = "127.0.0.1:1112"
    strategy = AddressSpecified(addr)
    assert strategy.get_allocated_address(config, dict()) == addr


def test_main_pool():
    strategy = MainPool()
    assert strategy.get_allocated_address(config, dict()) == "127.0.0.1:1111"


def test_random():
    strategy = Random()
    addresses = config.get_external_addresses()
    assert strategy.get_allocated_address(config, dict()) in addresses


def test_random_sub_pool():
    strategy = RandomSubPool()
    addresses = config.get_external_addresses()[1:]
    assert strategy.get_allocated_address(config, dict()) in addresses


def test_random_label():
    strategy = RandomLabel("test")
    addresses = config.get_external_addresses(label="test")
    assert len(addresses) == 2
    assert strategy.get_allocated_address(config, dict()) in addresses


def test_idle_label():
    strategy = IdleLabel("test", "my_mark")
    addresses = config.get_external_addresses(label="test")
    assert len(addresses) == 2
    allocated = {
        addresses[0]: {create_actor_ref(addresses[0], b"id1"): (strategy, None)}
    }
    assert strategy.get_allocated_address(config, allocated) == addresses[1]

    strategy2 = IdleLabel("test", "my_mark")
    allocated = {
        addresses[0]: {
            create_actor_ref(addresses[0], b"id1"): (strategy, None),
            create_actor_ref(addresses[0], b"id2"): (RandomLabel("test"), None),
        },
        addresses[1]: {create_actor_ref(addresses[1], b"id3"): (strategy2, None)},
    }
    with pytest.raises(NoIdleSlot):
        strategy2.get_allocated_address(config, allocated)
