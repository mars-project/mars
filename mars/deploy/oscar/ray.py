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
import itertools
import logging
import os
import time
from typing import Union, Dict, List, Optional, AsyncGenerator

from ... import oscar as mo
from ...core.entrypoints import init_extension_entrypoints
from ...metrics import init_metrics
from ...oscar.backends.ray.driver import RayActorDriver
from ...oscar.backends.ray.utils import (
    process_placement_to_address,
    node_placement_to_address,
    process_address_to_placement,
)
from ...oscar.backends.ray.pool import RayPoolState
from ...oscar.errors import ReconstructWorkerError
from ...resource import Resource
from ...services.cluster.backends.base import (
    register_cluster_backend,
    AbstractClusterBackend,
)
from ...services import NodeRole
from ...utils import lazy_import
from ..utils import (
    load_config,
    get_third_party_modules_from_config,
)
from .service import start_supervisor, start_worker, stop_supervisor, stop_worker
from .session import (
    _new_session,
    new_session,
    AbstractSession,
    ensure_isolation_created,
)
from .pool import create_supervisor_actor_pool, create_worker_actor_pool

ray = lazy_import("ray")
logger = logging.getLogger(__name__)

# The default config file.
DEFAULT_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "rayconfig.yml"
)
# The default value for supervisor standalone (not share node with worker).
DEFAULT_SUPERVISOR_STANDALONE = False
# The default value for supervisor sub pool count.
DEFAULT_SUPERVISOR_SUB_POOL_NUM = 0


def _load_config(config: Union[str, Dict] = None):
    return load_config(config, default_config_file=DEFAULT_CONFIG_FILE)


@register_cluster_backend
class RayClusterBackend(AbstractClusterBackend):
    name = "ray"

    def __init__(self, lookup_address: str, cluster_state_ref):
        self._supervisors = [n.strip() for n in lookup_address.split(",")]
        self._cluster_state_ref = cluster_state_ref

    @classmethod
    async def create(
        cls, node_role: NodeRole, lookup_address: str, pool_address: str
    ) -> "RayClusterBackend":
        try:
            ref = await mo.create_actor(
                ClusterStateActor,
                uid=ClusterStateActor.default_uid(),
                address=lookup_address,
            )
        except mo.ActorAlreadyExist:  # pragma: no cover
            ref = await mo.actor_ref(
                ClusterStateActor.default_uid(), address=lookup_address
            )
        return cls(lookup_address, ref)

    async def watch_supervisors(self) -> AsyncGenerator[List[str], None]:
        yield self._supervisors

    async def get_supervisors(self, filter_ready: bool = True) -> List[str]:
        return self._supervisors

    async def new_worker(self, worker_address):
        return await self._cluster_state_ref.new_worker(worker_address)

    async def request_worker(
        self, worker_cpu: int = None, worker_mem: int = None, timeout: int = None
    ) -> str:
        return await self._cluster_state_ref.request_worker(
            worker_cpu, worker_mem, timeout
        )

    async def release_worker(self, address: str):
        return await self._cluster_state_ref.release_worker(address)

    async def reconstruct_worker(self, address: str):
        return await self._cluster_state_ref.reconstruct_worker(address)

    def get_cluster_state_ref(self):
        return self._cluster_state_ref


class ClusterStateActor(mo.StatelessActor):
    def __init__(self):
        self._worker_cpu, self._worker_mem, self._config = None, None, None
        self._pg_name, self._band_to_resource, self._worker_modules = None, None, None
        self._pg_counter = itertools.count()
        self._worker_count = 0
        self._workers = {}
        self._releasing_tasks = {}
        self._reconstructing_tasks = {}

    async def __post_create__(self):
        self._pg_name, _, _ = process_address_to_placement(self.address)

    def set_config(self, worker_cpu, worker_mem, config):
        self._worker_cpu, self._worker_mem, self._config = (
            worker_cpu,
            worker_mem,
            config,
        )
        # TODO(chaokunyang) Support gpu
        self._band_to_resource = {
            "numa-0": Resource(num_cpus=self._worker_cpu, mem_bytes=self._worker_mem)
        }
        self._worker_modules = get_third_party_modules_from_config(
            self._config, NodeRole.WORKER
        )

    async def request_worker(
        self, worker_cpu: int = None, worker_mem: int = None, timeout: int = None
    ) -> Optional[str]:
        worker_cpu = worker_cpu or self._worker_cpu
        worker_mem = worker_mem or self._worker_mem
        bundle = {
            "CPU": worker_cpu,
            # "memory": worker_mem or self._worker_mem
        }
        band_to_resource = {
            "numa-0": Resource(num_cpus=worker_cpu, mem_bytes=worker_mem)
        }
        start_time = time.time()
        logger.info("Start to request worker with resource %s.", bundle)
        # TODO rescale ray placement group instead of creating new placement group
        pg_name = f"{self._pg_name}_{next(self._pg_counter)}"
        pg = ray.util.placement_group(name=pg_name, bundles=[bundle], strategy="SPREAD")
        create_pg_timeout = timeout or 120
        try:
            await asyncio.wait_for(pg.ready(), timeout=create_pg_timeout)
        except asyncio.CancelledError:  # pragma: no cover
            logger.warning(
                "Request worker with placement group %s in %s seconds canceled.",
                pg.bundle_specs,
                create_pg_timeout,
            )
            ray.util.remove_placement_group(pg)
            return None
        except asyncio.TimeoutError:
            logger.warning(
                "Request worker failed, "
                "can not create placement group %s in %s seconds.",
                pg.bundle_specs,
                create_pg_timeout,
            )
            ray.util.remove_placement_group(pg)
            return None
        logger.info(
            "Creating placement group %s took %.4f seconds",
            pg.bundle_specs,
            time.time() - start_time,
        )
        worker_address = process_placement_to_address(pg_name, 0, 0)
        worker_pool = await self.create_worker(worker_address)
        await self.start_worker(worker_address, band_to_resource=band_to_resource)
        logger.info(
            "Request worker %s succeeds in %.4f seconds",
            worker_address,
            time.time() - start_time,
        )
        self._workers[worker_address] = (worker_pool, pg)
        return worker_address

    async def create_worker(self, worker_address):
        start_time = time.time()
        worker_pool = await create_worker_actor_pool(
            worker_address,
            self._band_to_resource,
            modules=self._worker_modules,
            metrics=self._config.get("metrics", {}),
        )
        logger.info(
            "Create worker node %s succeeds in %.4f seconds.",
            worker_address,
            time.time() - start_time,
        )
        return worker_pool

    async def start_worker(self, worker_address, band_to_resource=None):
        self._worker_count += 1
        start_time = time.time()
        band_to_resource = band_to_resource or self._band_to_resource
        await start_worker(
            worker_address, self.address, band_to_resource, config=self._config
        )
        worker_pool = ray.get_actor(worker_address)
        await worker_pool.mark_service_ready.remote()
        logger.info(
            "Start services on worker %s succeeds in %.4f seconds.",
            worker_address,
            time.time() - start_time,
        )
        return worker_pool

    async def release_worker(self, address: str):
        logger.info("Start to release worker %s", address)
        task = self._reconstructing_tasks.get(address)
        if task is not None:
            task.cancel()

        task = self._releasing_tasks.get(address)
        if task is not None:
            logger.info("Waiting for releasing worker %s", address)
            return await task

        async def _release_worker():
            await stop_worker(address, self._config)
            pool, pg = self._workers.pop(address)
            await pool.actor_pool.remote("stop")
            if "COV_CORE_SOURCE" in os.environ:  # pragma: no cover
                try:
                    # must clean up first, or coverage info lost
                    await pool.cleanup.remote()
                except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                    pass
            ray.kill(pool.main_pool)
            ray.util.remove_placement_group(pg)
            logger.info("Released worker %s", address)

        task = asyncio.create_task(_release_worker())
        task.add_done_callback(lambda _: self._releasing_tasks.pop(address, None))
        self._releasing_tasks[address] = task
        return await task

    async def reconstruct_worker(self, address: str):
        task = self._releasing_tasks.get(address)
        if task is not None:
            raise ReconstructWorkerError(
                f"Can't reconstruct releasing worker {address}"
            )

        task = self._reconstructing_tasks.get(address)
        if task is not None:
            logger.info("Waiting for reconstruct worker %s", address)
            return await task

        async def _reconstruct_worker():
            logger.info("Reconstruct worker %s", address)
            actor = ray.get_actor(address)
            # set `max_retries=-1` to make task pending when actor is restarting
            state = await actor.state.options(max_retries=-1).remote()
            if state == RayPoolState.SERVICE_READY:
                logger.info("Worker %s is service ready.")
                return

            if state == RayPoolState.INIT:
                await actor.start.remote()
            else:
                assert state == RayPoolState.POOL_READY

            start_time = time.time()
            await start_worker(
                address, self.address, self._band_to_resource, config=self._config
            )
            await actor.mark_service_ready.remote()
            logger.info(
                "Start services on worker %s succeeds in %.4f seconds.",
                address,
                time.time() - start_time,
            )

        task = asyncio.create_task(_reconstruct_worker())
        task.add_done_callback(lambda _: self._reconstructing_tasks.pop(address, None))
        self._reconstructing_tasks[address] = task
        return await task


async def new_cluster(
    cluster_name: str = None,
    supervisor_mem: int = 1 * 1024**3,
    worker_num: int = 1,
    worker_cpu: int = 2,
    worker_mem: int = 2 * 1024**3,
    config: Union[str, Dict] = None,
    **kwargs,
):
    cluster_name = cluster_name or f"ray-cluster-{int(time.time())}"
    if not ray.is_initialized():
        logger.warning("Ray is not started, start the local ray cluster by `ray.init`.")
        # add 16 logical cpus for other computing in ray.
        ray.init(num_cpus=16 + worker_num * worker_cpu)
    ensure_isolation_created(kwargs)
    if kwargs:  # pragma: no cover
        raise TypeError(f"new_cluster got unexpected arguments: {list(kwargs)}")
    n_supervisor_process = kwargs.get(
        "n_supervisor_process", DEFAULT_SUPERVISOR_SUB_POOL_NUM
    )
    cluster = RayCluster(
        cluster_name,
        supervisor_mem,
        worker_num,
        worker_cpu,
        worker_mem,
        config,
        n_supervisor_process=n_supervisor_process,
    )
    try:
        await cluster.start()
        return await RayClient.create(cluster)
    except Exception as ex:  # pragma: no cover
        # cleanup the cluster if failed.
        try:
            await cluster.stop()
        except Exception as stop_ex:
            raise stop_ex from ex
        raise ex


def new_cluster_in_ray(**kwargs):
    isolation = ensure_isolation_created(kwargs)
    coro = new_cluster(**kwargs)
    fut = asyncio.run_coroutine_threadsafe(coro, isolation.loop)
    client = fut.result()
    client.session.as_default()
    return client


new_cluster_in_ray.__doc__ = new_cluster.__doc__


def new_ray_session(
    address: str = None,
    session_id: str = None,
    default: bool = True,
    **new_cluster_kwargs,
) -> AbstractSession:
    """

    Parameters
    ----------
    address: str
        mars web server address.
    session_id: str
        session id. If not specified, will be generated automatically.
    default: bool
        whether set the session as default session.
    new_cluster_kwargs:
        See `new_cluster` arguments.
    """
    client = None
    if not address:
        client = new_cluster_in_ray(**new_cluster_kwargs)
        session_id = session_id or client.session.session_id
        address = client.address
    session = new_session(
        address=address, session_id=session_id, backend="mars", default=default
    )
    session._ray_client = client
    if default:
        # SyncSession set isolated_session as default session instead.
        AbstractSession.default._ray_client = client
    return session


class RayCluster:
    _supervisor_pool: "ray.actor.ActorHandle"
    _worker_pools: List["ray.actor.ActorHandle"]

    def __init__(
        self,
        cluster_name: str,
        supervisor_mem: int = 1 * 1024**3,
        worker_num: int = 1,
        worker_cpu: int = 2,
        worker_mem: int = 4 * 1024**3,
        config: Union[str, Dict] = None,
        n_supervisor_process: int = DEFAULT_SUPERVISOR_SUB_POOL_NUM,
    ):
        # load third party extensions.
        init_extension_entrypoints()
        self._cluster_name = cluster_name
        self._supervisor_mem = supervisor_mem
        self._n_supervisor_process = n_supervisor_process
        self._worker_num = worker_num
        self._worker_cpu = worker_cpu
        self._worker_mem = worker_mem
        # load config file to dict.
        self._config = load_config(config, default_config_file=DEFAULT_CONFIG_FILE)
        self.supervisor_address = None
        # Hold actor handles to avoid being freed
        self._supervisor_pool = None
        self._worker_addresses = []
        self._worker_pools = []
        self._stopped = False
        self._cluster_backend = None
        self.web_address = None

    async def start(self):
        logging.basicConfig(
            format=ray.ray_constants.LOGGER_FORMAT, level=logging.INFO, force=True
        )
        logger.info("Start cluster with config %s", self._config)
        # init metrics to guarantee metrics use in driver
        metric_configs = self._config.get("metrics", {})
        metric_backend = metric_configs.get("backend")
        init_metrics(metric_backend, config=metric_configs.get(metric_backend))
        address_to_resources = dict()
        supervisor_standalone = (
            self._config.get("cluster", {})
            .get("ray", {})
            .get("supervisor", {})
            .get("standalone", DEFAULT_SUPERVISOR_STANDALONE)
        )
        supervisor_sub_pool_num = (
            self._config.get("cluster", {})
            .get("ray", {})
            .get("supervisor", {})
            .get("sub_pool_num", self._n_supervisor_process)
        )
        from ...storage.ray import support_specify_owner

        if not support_specify_owner():  # pragma: no cover
            logger.warning(
                "Current installed ray version does not support specify owner, "
                "autoscale may not work."
            )
            # config['scheduling']['autoscale']['enabled'] = False
        self.supervisor_address = process_placement_to_address(self._cluster_name, 0, 0)
        if "cluster" not in self._config:  # pragma: no cover
            self._config["cluster"] = dict()
        self._config["cluster"]["lookup_address"] = self.supervisor_address
        address_to_resources[node_placement_to_address(self._cluster_name, 0)] = {
            "CPU": 1,
            # "memory": self._supervisor_mem,
        }
        worker_addresses = []
        if supervisor_standalone:
            for worker_index in range(1, self._worker_num + 1):
                worker_address = process_placement_to_address(
                    self._cluster_name, worker_index, 0
                )
                worker_addresses.append(worker_address)
                worker_node_address = node_placement_to_address(
                    self._cluster_name, worker_index
                )
                address_to_resources[worker_node_address] = {
                    "CPU": self._worker_cpu,
                    # "memory": self._worker_mem,
                }
        else:
            for worker_index in range(self._worker_num):
                worker_process_index = (
                    supervisor_sub_pool_num + 1 if worker_index == 0 else 0
                )
                worker_address = process_placement_to_address(
                    self._cluster_name, worker_index, worker_process_index
                )
                worker_addresses.append(worker_address)
                worker_node_address = node_placement_to_address(
                    self._cluster_name, worker_index
                )
                address_to_resources[worker_node_address] = {
                    "CPU": self._worker_cpu,
                    # "memory": self._worker_mem,
                }
        mo.setup_cluster(address_to_resources)

        # third party modules from config
        supervisor_modules = get_third_party_modules_from_config(
            self._config, NodeRole.SUPERVISOR
        )

        # create supervisor actor pool
        supervisor_pool_coro = asyncio.create_task(
            create_supervisor_actor_pool(
                self.supervisor_address,
                n_process=supervisor_sub_pool_num,
                main_pool_cpus=0,
                sub_pool_cpus=0,
                modules=supervisor_modules,
                metrics=self._config.get("metrics", {}),
            )
        )
        worker_pools = [
            asyncio.create_task(
                create_worker_actor_pool(
                    addr,
                    {
                        "numa-0": Resource(
                            num_cpus=self._worker_cpu, mem_bytes=self._worker_mem
                        )
                    },
                    modules=get_third_party_modules_from_config(
                        self._config, NodeRole.WORKER
                    ),
                    metrics=self._config.get("metrics", {}),
                )
            )
            for addr in worker_addresses
        ]
        self._supervisor_pool = await supervisor_pool_coro
        logger.info("Create supervisor on node %s succeeds.", self.supervisor_address)
        self._cluster_backend = await RayClusterBackend.create(
            NodeRole.WORKER, self.supervisor_address, self.supervisor_address
        )
        cluster_state_ref = self._cluster_backend.get_cluster_state_ref()
        await self._cluster_backend.get_cluster_state_ref().set_config(
            self._worker_cpu, self._worker_mem, self._config
        )
        # start service
        await start_supervisor(self.supervisor_address, config=self._config)
        logger.info(
            "Start services on supervisor %s succeeds.", self.supervisor_address
        )
        await self._supervisor_pool.mark_service_ready.remote()
        worker_pools = await asyncio.gather(*worker_pools)
        logger.info("Create %s workers succeeds.", len(worker_pools))
        await asyncio.gather(
            *[cluster_state_ref.start_worker(addr) for addr in worker_addresses]
        )
        logger.info("Start services on %s workers succeeds.", len(worker_addresses))
        for worker_address, worker_pool in zip(worker_addresses, worker_pools):
            self._worker_addresses.append(worker_address)
            self._worker_pools.append(worker_pool)

        from ...services.web.supervisor import WebActor

        web_actor = await mo.actor_ref(
            WebActor.default_uid(), address=self.supervisor_address
        )
        self.web_address = await web_actor.get_web_address()
        logger.warning("Web service started at %s", self.web_address)

    async def stop(self):
        if not self._stopped:
            try:
                for worker_address in self._worker_addresses:
                    await stop_worker(worker_address, self._config)
                for pool in self._worker_pools:
                    await pool.actor_pool.remote("stop")
                if self._supervisor_pool is not None:
                    await stop_supervisor(self.supervisor_address, self._config)
                    await self._supervisor_pool.actor_pool.remote("stop")
            finally:
                AbstractSession.reset_default()
                RayActorDriver.stop_cluster()
            self._stopped = True


class RayClient:
    def __init__(self, cluster: RayCluster, session: AbstractSession):
        self._cluster = cluster
        self._address = cluster.supervisor_address
        self._session = session
        # hold ray cluster by client to avoid actor handle out-of-scope
        session._ray_client = self

    @classmethod
    async def create(cls, cluster: RayCluster) -> "RayClient":
        session = await _new_session(cluster.supervisor_address, default=True)
        client = RayClient(cluster, session)
        AbstractSession.default._ray_client = client
        return client

    @property
    def address(self):
        return self._session.address

    @property
    def session(self):
        return self._session

    @property
    def web_address(self):
        return self._cluster.web_address

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self._stop()

    def stop(self):
        isolation = ensure_isolation_created({})
        fut = asyncio.run_coroutine_threadsafe(self._stop(), isolation.loop)
        return fut.result()

    async def _stop(self):
        await self._cluster.stop()
        AbstractSession.reset_default()
