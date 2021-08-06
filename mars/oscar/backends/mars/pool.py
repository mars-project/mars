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
import concurrent.futures as futures
import logging.config
import multiprocessing
import os
import signal
import sys
from typing import List

from ....utils import get_next_port
from ..config import ActorPoolConfig
from ..message import CreateActorMessage
from ..pool import MainActorPoolBase, SubActorPoolBase, _register_message_handler


_is_windows: bool = sys.platform.startswith('win')

if sys.version_info[:2] == (3, 9):
    # fix for Python 3.9, see https://bugs.python.org/issue43517
    if sys.platform == 'win32':
        from multiprocessing import popen_spawn_win32 as popen_spawn
    else:
        from multiprocessing import popen_spawn_posix as popen_spawn
    from multiprocessing import popen_forkserver, popen_fork, synchronize
    _ = popen_spawn, popen_forkserver, popen_fork, synchronize
elif sys.version_info[:2] == (3, 6):  # pragma: no cover
    # define kill method for multiprocessing
    if not _is_windows:
        def _mp_kill(self):
            try:
                os.kill(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                if self.wait(timeout=0.1) is None:
                    raise
    else:
        def _mp_kill(self):
            self.terminate()

    from multiprocessing.process import BaseProcess
    BaseProcess.kill = _mp_kill

logger = logging.getLogger(__name__)


@_register_message_handler
class MainActorPool(MainActorPoolBase):

    @classmethod
    def get_external_addresses(
            cls, address: str, n_process: int = None, ports: List[int] = None):
        """Get socket address for every process"""
        if ':' in address:
            host, port = address.split(':', 1)
            port = int(port)
            if ports:
                if len(ports) != n_process:
                    raise ValueError(f'`ports` specified, but its count '
                                     f'is not equal to `n_process`, '
                                     f'number of ports: {len(ports)}, '
                                     f'n_process: {n_process}')
                sub_ports = ports
            else:
                sub_ports = [get_next_port() for _ in range(n_process)]
        else:
            host = address
            if ports and len(ports) != n_process + 1:
                # ports specified, the first of which should be main port
                raise ValueError(f'`ports` specified, but its count '
                                 f'is not equal to `n_process` + 1, '
                                 f'number of ports: {len(ports)}, '
                                 f'n_process + 1: {n_process + 1}')
            elif not ports:
                ports = [get_next_port() for _ in range(n_process + 1)]
            port = ports[0]
            sub_ports = ports[1:]
        return [f'{host}:{port}' for port in [port] + sub_ports]

    @classmethod
    def gen_internal_address(cls, process_index: int, external_address: str = None) -> str:
        if hasattr(asyncio, 'start_unix_server'):
            return f'unixsocket:///{process_index}'
        else:
            return external_address

    @classmethod
    async def start_sub_pool(
            cls,
            actor_pool_config: ActorPoolConfig,
            process_index: int,
            start_method: str = None):

        def start_pool_in_process():
            ctx = multiprocessing.get_context(method=start_method)
            started = ctx.Event()
            process = ctx.Process(
                target=cls._start_sub_pool,
                args=(actor_pool_config, process_index, started),
                name=f'MarsActorPool{process_index}',
            )
            process.daemon = True
            process.start()
            # wait for sub actor pool to finish starting
            started.wait()
            return process

        loop = asyncio.get_running_loop()
        executor = futures.ThreadPoolExecutor(1)
        create_pool_task = loop.run_in_executor(executor, start_pool_in_process)
        return await create_pool_task

    @classmethod
    def _start_sub_pool(
            cls,
            actor_config: ActorPoolConfig,
            process_index: int,
            started: multiprocessing.Event):
        if not _is_windows:
            try:
                # register coverage hooks on SIGTERM
                from pytest_cov.embed import cleanup_on_sigterm
                if 'COV_CORE_SOURCE' in os.environ:  # pragma: no branch
                    cleanup_on_sigterm()
            except ImportError:  # pragma: no cover
                pass

        conf = actor_config.get_pool_config(process_index)
        suspend_sigint = conf['suspend_sigint']
        if suspend_sigint:
            signal.signal(signal.SIGINT, lambda *_: None)

        logging_conf = conf['logging_conf'] or {}
        if logging_conf.get('file'):
            logging.config.fileConfig(logging_conf['file'])
        elif logging_conf.get('level'):
            logging.basicConfig(
                level=logging_conf['level'], format=logging_conf.get('format')
            )

        use_uvloop = conf['use_uvloop']
        if use_uvloop:
            import uvloop
            asyncio.set_event_loop(uvloop.new_event_loop())
        else:
            asyncio.set_event_loop(asyncio.new_event_loop())

        coro = cls._create_sub_pool(actor_config, process_index, started)
        asyncio.run(coro)

    @classmethod
    async def _create_sub_pool(
            cls,
            actor_config: ActorPoolConfig,
            process_index: int,
            started: multiprocessing.Event):
        try:
            env = actor_config.get_pool_config(process_index)['env']
            if env:
                os.environ.update(env)
            pool = await SubActorPool.create({
                'actor_pool_config': actor_config,
                'process_index': process_index
            })
            await pool.start()
        finally:
            started.set()
        await pool.join()

    async def kill_sub_pool(self, process: multiprocessing.Process,
                            force: bool = False):
        if 'COV_CORE_SOURCE' in os.environ and not force and not _is_windows:  # pragma: no cover
            # must shutdown gracefully, or coverage info lost
            try:
                os.kill(process.pid, signal.SIGINT)
            except OSError:  # pragma: no cover
                pass
            process.terminate()
            wait_pool = futures.ThreadPoolExecutor(1)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(wait_pool, process.join, 3)
        process.kill()
        await asyncio.to_thread(process.join, 5)

    async def is_sub_pool_alive(self, process: multiprocessing.Process):
        return process.is_alive()

    async def recover_sub_pool(self, address: str):
        process_index = self._config.get_process_index(address)
        # process dead, restart it
        # remember always use spawn to recover sub pool
        self.sub_processes[address] = await self.__class__.start_sub_pool(
            self._config, process_index, 'spawn')

        if self._auto_recover == 'actor':
            # need to recover all created actors
            for _, message in self._allocated_actors[address].values():
                create_actor_message: CreateActorMessage = message
                await self.call(address, create_actor_message)


@_register_message_handler
class SubActorPool(SubActorPoolBase):
    pass
