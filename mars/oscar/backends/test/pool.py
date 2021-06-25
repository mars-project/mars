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

import asyncio
import multiprocessing
from typing import List, Dict

from ..config import ActorPoolConfig
from ..communication import gen_local_address, get_server_type, DummyServer
from ..mars.pool import MainActorPool, SubActorPool
from ..pool import ActorPoolType


class TestMainActorPool(MainActorPool):
    @classmethod
    def get_external_addresses(
            cls, address: str, n_process: int = None, ports: List[int] = None):
        if '://' in address:
            address = address.split('://', 1)[1]
        return super().get_external_addresses(address, n_process=n_process,
                                              ports=ports)

    @classmethod
    def gen_internal_address(cls, process_index: int, external_address: str = None) -> str:
        return f'dummy://{process_index}'

    @classmethod
    async def start_sub_pool(
            cls,
            actor_pool_config: ActorPoolConfig,
            process_index: int,
            start_method: str = None):
        started = multiprocessing.Event()
        return asyncio.create_task(
            cls._create_sub_pool(actor_pool_config, process_index, started))

    @classmethod
    async def _create_sub_pool(
            cls,
            actor_config: ActorPoolConfig,
            process_index: int,
            started: multiprocessing.Event):
        pool = await TestSubActorPool.create({
            'actor_pool_config': actor_config,
            'process_index': process_index
        })
        await pool.start()
        started.set()
        await pool.join()

    async def kill_sub_pool(self, process: multiprocessing.Process,
                            force: bool = False):
        process.cancel()

    async def is_sub_pool_alive(self, process: multiprocessing.Process):
        return not process.cancelled()


class TestSubActorPool(SubActorPool):
    @classmethod
    async def create(cls, config: Dict) -> ActorPoolType:
        kw = dict()
        cls._parse_config(config, kw)
        process_index: int = kw['process_index']
        actor_pool_config = kw['config']
        external_addresses = \
            actor_pool_config.get_pool_config(process_index)['external_address']

        def handle_channel(channel):
            return pool.on_new_channel(channel)

        # create servers
        create_server_tasks = []
        for addr in set(external_addresses + [gen_local_address(process_index)]):
            server_type = get_server_type(addr)
            task = asyncio.create_task(
                server_type.create(dict(address=addr,
                                        handle_channel=handle_channel)))
            create_server_tasks.append(task)
        await asyncio.gather(*create_server_tasks)
        kw['servers'] = [f.result() for f in create_server_tasks]

        # create pool
        pool = cls(**kw)
        return pool

    async def stop(self):
        # do not close dummy server
        self._servers = [s for s in self._servers[:-1]
                         if not isinstance(s, DummyServer)]
        await super().stop()
