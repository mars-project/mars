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
from ..mars.pool import MainActorPool, SubActorPool, SubpoolStatus
from ..pool import ActorPoolType


class TestMainActorPool(MainActorPool):
    @classmethod
    def get_external_addresses(
        cls, address: str, n_process: int = None, ports: List[int] = None
    ):
        if "://" in address:
            address = address.split("://", 1)[1]
        return super().get_external_addresses(address, n_process=n_process, ports=ports)

    @classmethod
    def gen_internal_address(
        cls, process_index: int, external_address: str = None
    ) -> str:
        return f"dummy://{process_index}"

    @classmethod
    async def start_sub_pool(
        cls,
        actor_pool_config: ActorPoolConfig,
        process_index: int,
        start_method: str = None,
    ):
        status_queue = multiprocessing.Queue()
        return (
            asyncio.create_task(
                cls._create_sub_pool(actor_pool_config, process_index, status_queue)
            ),
            status_queue,
        )

    @classmethod
    async def wait_sub_pools_ready(cls, create_pool_tasks: List[asyncio.Task]):
        addresses = []
        tasks = []
        for t in create_pool_tasks:
            pool_task, queue = await t
            tasks.append(pool_task)
            status = await asyncio.to_thread(queue.get)
            addresses.append(status.external_addresses)
        return tasks, addresses

    @classmethod
    async def _create_sub_pool(
        cls,
        actor_config: ActorPoolConfig,
        process_index: int,
        status_queue: multiprocessing.Queue,
    ):
        pool = await TestSubActorPool.create(
            {"actor_pool_config": actor_config, "process_index": process_index}
        )
        await pool.start()
        status_queue.put(
            SubpoolStatus(status=0, external_addresses=[pool.external_address])
        )
        actor_config.reset_pool_external_address(process_index, [pool.external_address])
        await pool.join()

    async def kill_sub_pool(
        self, process: multiprocessing.Process, force: bool = False
    ):
        process.cancel()

    async def is_sub_pool_alive(self, process: multiprocessing.Process):
        return not process.cancelled()


class TestSubActorPool(SubActorPool):
    @classmethod
    async def create(cls, config: Dict) -> ActorPoolType:
        kw = dict()
        cls._parse_config(config, kw)
        process_index: int = kw["process_index"]
        actor_pool_config = kw["config"]  # type: ActorPoolConfig
        external_addresses = actor_pool_config.get_pool_config(process_index)[
            "external_address"
        ]

        def handle_channel(channel):
            return pool.on_new_channel(channel)

        # create servers
        external_address_set = set(external_addresses)
        create_server_tasks = []
        is_external_server = []
        for addr in set(external_addresses + [gen_local_address(process_index)]):
            server_type = get_server_type(addr)
            task = asyncio.create_task(
                server_type.create(dict(address=addr, handle_channel=handle_channel))
            )
            create_server_tasks.append(task)
            is_external_server.append(addr in external_address_set)

        await asyncio.gather(*create_server_tasks)
        kw["servers"] = servers = [f.result() for f in create_server_tasks]

        new_external_addresses = [
            server.address
            for server, is_ext in zip(servers, is_external_server)
            if is_ext
        ]

        if set(external_addresses) != set(new_external_addresses):
            external_addresses = new_external_addresses
            actor_pool_config.reset_pool_external_address(
                process_index, external_addresses
            )
            cls._update_kw_addresses(actor_pool_config, process_index, kw)

        # set default router
        # actor context would be able to use exact client
        cls._set_global_router(kw["router"])

        # create pool
        pool = cls(**kw)
        return pool

    async def stop(self):
        # do not close dummy server
        self._servers = [
            s for s in self._servers[:-1] if not isinstance(s, DummyServer)
        ]
        await super().stop()
