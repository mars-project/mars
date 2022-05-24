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
import atexit
import os
import sys
import logging
from concurrent.futures import Future as SyncFuture
from typing import Dict, List, Union

import numpy as np

from ... import oscar as mo
from ...core.entrypoints import init_extension_entrypoints
from ...lib.aio import get_isolation, stop_isolation
from ...resource import cpu_count, cuda_count, mem_total
from ...services import NodeRole
from ...services.task.execution.api import ExecutionConfig
from ...typing import ClusterType, ClientType
from ..utils import get_third_party_modules_from_config, load_config
from .pool import create_supervisor_actor_pool, create_worker_actor_pool
from .service import (
    start_supervisor,
    start_worker,
    stop_supervisor,
    stop_worker,
)
from .session import AbstractSession, _new_session, ensure_isolation_created

logger = logging.getLogger(__name__)

_is_exiting_future = SyncFuture()
atexit.register(
    lambda: _is_exiting_future.set_result(0) if not _is_exiting_future.done() else None
)
atexit.register(stop_isolation)

# The default config file.
DEFAULT_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.yml"
)


def _load_config(config: Union[str, Dict] = None):
    return load_config(config, default_config_file=DEFAULT_CONFIG_FILE)


async def new_cluster_in_isolation(
    address: str = "0.0.0.0",
    n_worker: int = 1,
    n_cpu: Union[int, str] = "auto",
    mem_bytes: Union[int, str] = "auto",
    cuda_devices: Union[List[int], str] = "auto",
    subprocess_start_method: str = None,
    backend: str = None,
    config: Union[str, Dict] = None,
    web: bool = True,
    timeout: float = None,
    n_supervisor_process: int = 0,
) -> ClientType:
    cluster = LocalCluster(
        address,
        n_worker,
        n_cpu,
        mem_bytes,
        cuda_devices,
        subprocess_start_method,
        backend,
        config,
        web,
        n_supervisor_process,
    )
    await cluster.start()
    return await LocalClient.create(cluster, timeout)


async def new_cluster(
    address: str = "0.0.0.0",
    n_worker: int = 1,
    n_cpu: Union[int, str] = "auto",
    mem_bytes: Union[int, str] = "auto",
    cuda_devices: Union[List[int], str] = "auto",
    subprocess_start_method: str = None,
    backend: str = None,
    config: Union[str, Dict] = None,
    web: bool = True,
    loop: asyncio.AbstractEventLoop = None,
    use_uvloop: Union[bool, str] = "auto",
    n_supervisor_process: int = 0,
) -> ClientType:
    coro = new_cluster_in_isolation(
        address,
        n_worker=n_worker,
        n_cpu=n_cpu,
        mem_bytes=mem_bytes,
        cuda_devices=cuda_devices,
        subprocess_start_method=subprocess_start_method,
        backend=backend,
        config=config,
        web=web,
        n_supervisor_process=n_supervisor_process,
    )
    isolation = ensure_isolation_created(dict(loop=loop, use_uvloop=use_uvloop))
    fut = asyncio.run_coroutine_threadsafe(coro, isolation.loop)
    client = await asyncio.wrap_future(fut)
    client.session.as_default()
    return client


async def stop_cluster(cluster: ClusterType):
    isolation = get_isolation()
    coro = cluster.stop()
    await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, isolation.loop))


class LocalCluster:
    def __init__(
        self: ClusterType,
        address: str = "0.0.0.0",
        n_worker: int = 1,
        n_cpu: Union[int, str] = "auto",
        mem_bytes: Union[int, str] = "auto",
        cuda_devices: Union[List[int], List[List[int]], str] = "auto",
        subprocess_start_method: str = None,
        backend: str = None,
        config: Union[str, Dict] = None,
        web: Union[bool, str] = "auto",
        n_supervisor_process: int = 0,
    ):
        # load third party extensions.
        init_extension_entrypoints()
        # auto choose the subprocess_start_method.
        if subprocess_start_method is None:
            subprocess_start_method = (
                "spawn" if sys.platform == "win32" else "forkserver"
            )
        self._address = address
        self._n_worker = n_worker
        self._n_cpu = cpu_count() if n_cpu == "auto" else n_cpu
        self._mem_bytes = mem_total() if mem_bytes == "auto" else mem_bytes
        self._cuda_devices = self._get_cuda_devices(cuda_devices, n_worker)
        self._subprocess_start_method = subprocess_start_method
        self._config = load_config(config, default_config_file=DEFAULT_CONFIG_FILE)
        execution_config = ExecutionConfig.from_config(self._config, backend=backend)
        self._backend = execution_config.backend
        self._web = web
        self._n_supervisor_process = n_supervisor_process

        execution_config.merge_from(
            ExecutionConfig.from_params(
                backend=self._backend,
                n_worker=self._n_worker,
                n_cpu=self._n_cpu,
                mem_bytes=self._mem_bytes,
                cuda_devices=self._cuda_devices,
            )
        )

        self._bands_to_resource = execution_config.get_deploy_band_resources()
        self._supervisor_pool = None
        self._worker_pools = []
        self._exiting_check_task = None

        self.supervisor_address = None
        self.web_address = None

    @staticmethod
    def _get_cuda_devices(cuda_devices, n_worker):
        if cuda_devices == "auto":
            total = cuda_count()
            all_devices = np.arange(total)
            return [list(arr) for arr in np.array_split(all_devices, n_worker)]

        else:  # pragma: no cover
            if isinstance(cuda_devices[0], int):
                assert n_worker == 1
                return [cuda_devices]
            else:
                assert len(cuda_devices) == n_worker
                return cuda_devices

    @property
    def backend(self):
        return self._backend

    @property
    def external_address(self):
        return self._supervisor_pool.external_address

    async def start(self):
        await self._start_supervisor_pool()
        await self._start_worker_pools()
        # start service
        await self._start_service()

        if self._web:
            from ...services.web.supervisor import WebActor

            web_actor = await mo.actor_ref(
                WebActor.default_uid(), address=self.supervisor_address
            )
            self.web_address = await web_actor.get_web_address()
            logger.warning("Web service started at %s", self.web_address)

        self._exiting_check_task = asyncio.create_task(self._check_exiting())

    async def _check_exiting(self):
        await asyncio.wrap_future(_is_exiting_future)
        await self.stop()

    async def _start_supervisor_pool(self):
        supervisor_modules = get_third_party_modules_from_config(
            self._config, NodeRole.SUPERVISOR
        )
        self._supervisor_pool = await create_supervisor_actor_pool(
            self._address,
            n_process=self._n_supervisor_process,
            modules=supervisor_modules,
            subprocess_start_method=self._subprocess_start_method,
            metrics=self._config.get("metrics", {}),
        )
        self.supervisor_address = self._supervisor_pool.external_address

    async def _start_worker_pools(self):
        worker_modules = get_third_party_modules_from_config(
            self._config, NodeRole.WORKER
        )
        for band_to_resource in self._bands_to_resource:
            worker_pool = await create_worker_actor_pool(
                self._address,
                band_to_resource,
                modules=worker_modules,
                subprocess_start_method=self._subprocess_start_method,
                metrics=self._config.get("metrics", {}),
            )
            self._worker_pools.append(worker_pool)

    async def _start_service(self):
        self._web = await start_supervisor(
            self.supervisor_address, config=self._config, web=self._web
        )
        for worker_pool, band_to_resource in zip(
            self._worker_pools, self._bands_to_resource
        ):
            await start_worker(
                worker_pool.external_address,
                self.supervisor_address,
                band_to_resource,
                config=self._config,
            )

    async def stop(self):
        from .session import SessionAPI

        # delete all sessions
        session_api = await SessionAPI.create(self._supervisor_pool.external_address)
        await session_api.delete_all_sessions()

        for worker_pool in self._worker_pools:
            await stop_worker(worker_pool.external_address, self._config)
        await stop_supervisor(self._supervisor_pool.external_address, self._config)
        for worker_pool in self._worker_pools:
            await worker_pool.stop()
        await self._supervisor_pool.stop()
        AbstractSession.reset_default()
        self._exiting_check_task.cancel()


class LocalClient:
    def __init__(self: ClientType, cluster: ClusterType, session: AbstractSession):
        self._cluster = cluster
        self.session = session

    @classmethod
    async def create(
        cls,
        cluster: LocalCluster,
        timeout: float = None,
    ) -> ClientType:
        session = await _new_session(
            cluster.external_address,
            backend=cluster.backend,
            default=True,
            timeout=timeout,
        )
        client = LocalClient(cluster, session)
        session.client = client
        return client

    @property
    def web_address(self):
        return self._cluster.web_address

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.stop()

    async def stop(self):
        await stop_cluster(self._cluster)
