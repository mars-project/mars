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

import asyncio
import concurrent.futures as futures
import os
import multiprocessing
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, List, Type

from ....utils import implements, get_next_port
from ...api import Actor
from ...core import ActorRef
from ...errors import ActorAlreadyExist, ActorNotExist
from ...utils import create_actor_ref
from .allocate_strategy import allocated_type, AddressSpecified
from .communication import Channel, Server, \
    get_server_type, gen_internal_address
from .core import result_message_type, ActorCaller
from .config import ActorPoolConfig
from .message import _MessageBase, new_message_id, DEFAULT_PROTOCOL, MessageType, \
    ResultMessage, ErrorMessage, CreateActorMessage, HasActorMessage, \
    DestroyActorMessage, ActorRefMessage, SendMessage, TellMessage, \
    ControlMessage, ControlMessageType
from .router import Router, LOCAL_ADDRESS


class _ErrorProcessor:
    def __init__(self, message_id: bytes, protocol):
        self._message_id = message_id
        self._protocol = protocol
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.result is None:
            self.result = ErrorMessage(
                self._message_id, exc_type, exc_val, exc_tb,
                protocol=self._protocol
            )
            return True


def _register_message_handler(pool_type: Type["AbstractActorPool"]):
    pool_type._message_handler = dict()
    for message_type, handler in [
        (MessageType.create_actor, pool_type.create_actor),
        (MessageType.destroy_actor, pool_type.destroy_actor),
        (MessageType.has_actor, pool_type.has_actor),
        (MessageType.actor_ref, pool_type.actor_ref),
        (MessageType.send, pool_type.send),
        (MessageType.tell, pool_type.tell),
        (MessageType.control, pool_type.handle_control_command)
    ]:
        pool_type._message_handler[message_type] = handler
    return pool_type


class AbstractActorPool(ABC):
    __slots__ = 'process_index', 'label', 'external_address', 'internal_address', 'env', \
                '_servers', '_router', '_config', '_stopped', '_actors', '_caller'

    def __init__(self,
                 process_index: int,
                 label: str,
                 external_address: str,
                 internal_address: str,
                 env: Dict,
                 router: Router,
                 config: ActorPoolConfig,
                 servers: List[Server]):
        self.process_index = process_index
        self.label = label
        self.external_address = external_address
        self.internal_address = internal_address
        self.env = env
        self._router = router
        self._config = config
        self._servers = servers

        self._stopped = asyncio.Event()

        # actor id -> actor
        self._actors: Dict[bytes, Actor] = dict()

        # manage async actor callers
        self._caller = ActorCaller()

    @property
    def router(self):
        return self._router

    @abstractmethod
    async def create_actor(self,
                           message: CreateActorMessage) -> result_message_type:
        """
        Create an actor.

        Parameters
        ----------
        message: CreateActorMessage
            message to create an actor.

        Returns
        -------
        result_message
            result or error message.
        """

    @abstractmethod
    async def has_actor(self,
                        message: HasActorMessage) -> ResultMessage:
        """
        Check if an actor exists or not.

        Parameters
        ----------
        message: HasActorMessage
            message

        Returns
        -------
        result_message
            result message contains if an actor exists or not.
        """

    @abstractmethod
    async def destroy_actor(self,
                            message: DestroyActorMessage) -> result_message_type:
        """
        Destroy an actor.

        Parameters
        ----------
        message: DestroyActorMessage
            message to destroy an actor.

        Returns
        -------
        result_message
            result or error message.
        """

    @abstractmethod
    async def actor_ref(self,
                        message: ActorRefMessage) -> result_message_type:
        """
        Get an actor's ref.

        Parameters
        ----------
        message: ActorRefMessage
            message to get an actor's ref.

        Returns
        -------
        result_message
            result or error message.
        """

    @abstractmethod
    async def send(self,
                   message: SendMessage) -> result_message_type:
        """
        Send a message to some actor.

        Parameters
        ----------
        message: SendMessage
            Message to send.

        Returns
        -------
        result_message
            result or error message.
        """

    @abstractmethod
    async def tell(self,
                   message: TellMessage) -> result_message_type:
        """
        Tell message to some actor.

        Parameters
        ----------
        message: TellMessage
            Message to tell.

        Returns
        -------
        result_message
            result or error message.
        """

    async def handle_control_command(self,
                                     message: ControlMessage) -> result_message_type:
        """
        Handle control command.

        Parameters
        ----------
        message: ControlMessage
            Control message.

        Returns
        -------
        result_message
            result or error message.
        """
        with _ErrorProcessor(message.message_id,
                             protocol=message.protocol) as processor:
            content = True
            if message.control_message_type == ControlMessageType.stop:
                await self.stop()
            elif message.control_message_type == ControlMessageType.sync_config:
                actor_pool_config: ActorPoolConfig = message.content
                self._config = actor_pool_config
                self._router.set_mapping(
                    actor_pool_config.external_to_internal_address_map)
            elif message.control_message_type == ControlMessageType.get_config:
                content = self._config
            else:  # pragma: no cover
                raise TypeError(f'Unable to handle control message '
                                f'with type {message.control_message_type}')
            processor.result = ResultMessage(
                message.message_id, content,
                protocol=message.protocol)

        return processor.result

    async def process_message(self,
                              message: _MessageBase,
                              channel: Channel):
        handler = self._message_handler[message.message_type]
        result = await handler(self, message)
        await channel.send(result)

    async def call(self,
                   dest_address: str,
                   message: _MessageBase) -> result_message_type:
        return await self._caller.call(self._router, dest_address, message)

    @staticmethod
    def _parse_config(config: Dict) -> Dict:
        kw = dict()
        actor_pool_config: ActorPoolConfig = config.pop('actor_pool_config')
        kw['config'] = actor_pool_config
        kw['process_index'] = process_index = config.pop('process_index')
        curr_pool_config = actor_pool_config.get_pool_config(process_index)
        kw['label'] = curr_pool_config['label']
        external_addresses = curr_pool_config['external_address']
        kw['external_address'] = external_addresses[0]
        kw['internal_address'] = curr_pool_config['internal_address']
        kw['router'] = Router(external_addresses,
                              actor_pool_config.external_to_internal_address_map)
        kw['env'] = curr_pool_config['env']

        if config:  # pragma: no cover
            raise TypeError(f'Creating pool got unexpected '
                            f'arguments: {",".join(config)}')

        return kw

    @classmethod
    @abstractmethod
    async def create(cls, config: Dict) -> "AbstractActorPool":
        """
        Create an actor pool.

        Parameters
        ----------
        config: Dict
            configurations.

        Returns
        -------
        actor_pool:
            Actor pool.
        """

    async def start(self):
        start_servers = [server.start() for server in self._servers]
        await asyncio.wait(start_servers)

    async def join(self, timeout: float = None):
        wait_stopped = asyncio.create_task(self._stopped.wait())

        try:
            await asyncio.wait_for(wait_stopped, timeout=timeout)
        except (futures.TimeoutError, asyncio.TimeoutError):
            wait_stopped.cancel()

    async def stop(self):
        try:
            stop_tasks = []
            # stop all servers
            stop_tasks.extend([server.stop() for server in self._servers])
            # stop all clients
            stop_tasks.append(self._caller.stop())
            await asyncio.gather(*stop_tasks)
        finally:
            self._stopped.set()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    async def on_new_channel(self, channel: Channel):
        while not self._stopped.is_set():
            try:
                message = await channel.recv()
            except EOFError:
                # no data to read, check channel
                await channel.close()
                return
            asyncio.create_task(self.process_message(message, channel))
            await asyncio.sleep(0)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


class ActorPoolBase(AbstractActorPool, metaclass=ABCMeta):
    __slots__ = ()

    @implements(AbstractActorPool.create_actor)
    async def create_actor(self,
                           message: CreateActorMessage) -> result_message_type:
        with _ErrorProcessor(message.message_id,
                             message.protocol) as processor:
            actor_id = message.actor_id
            if actor_id in self._actors:
                raise ActorAlreadyExist(f'Actor {actor_id} already exist, '
                                        f'cannot create')

            actor = message.actor_cls(*message.args, **message.kwargs)
            actor.uid = actor_id
            actor.address = address = self.external_address
            self._actors[actor_id] = actor
            await actor.__post_create__()

            result = ActorRef(address, actor_id)
            # ensemble result message
            processor.result = ResultMessage(message.message_id, result,
                                             protocol=message.protocol)
        return processor.result

    @implements(AbstractActorPool.has_actor)
    async def has_actor(self,
                        message: HasActorMessage) -> ResultMessage:
        result = ResultMessage(message.message_id,
                               message.actor_ref.uid in self._actors,
                               protocol=message.protocol)
        return result

    @implements(AbstractActorPool.destroy_actor)
    async def destroy_actor(self,
                            message: DestroyActorMessage) -> result_message_type:
        with _ErrorProcessor(message.message_id,
                             message.protocol) as processor:
            actor_id = message.actor_ref.uid
            try:
                actor = self._actors[actor_id]
            except KeyError:
                raise ActorNotExist(f'Actor {actor_id} does not exist')
            await actor.__pre_destroy__()
            del self._actors[actor_id]

            processor.result = ResultMessage(message.message_id,
                                             actor_id,
                                             protocol=message.protocol)
        return processor.result

    @implements(AbstractActorPool.actor_ref)
    async def actor_ref(self,
                        message: ActorRefMessage) -> result_message_type:
        with _ErrorProcessor(message.message_id,
                             message.protocol) as processor:
            actor_id = message.actor_ref.uid
            if actor_id not in self._actors:
                raise ActorNotExist(f'Actor {actor_id} does not exist')
            result = ResultMessage(message.message_id,
                                   ActorRef(self.external_address, actor_id),
                                   protocol=message.protocol)
            processor.result = result
        return processor.result

    @implements(AbstractActorPool.send)
    async def send(self,
                   message: SendMessage) -> result_message_type:
        with _ErrorProcessor(message.message_id,
                             message.protocol) as processor:
            actor_id = message.actor_ref.uid
            if actor_id not in self._actors:
                raise ActorNotExist(f'Actor {actor_id} does not exist')
            result = await self._actors[actor_id].__on_receive__(message.content)
            processor.result = ResultMessage(message.message_id, result,
                                             protocol=message.protocol)
        return processor.result

    @implements(AbstractActorPool.tell)
    async def tell(self,
                   message: TellMessage) -> result_message_type:
        with _ErrorProcessor(message.message_id,
                             message.protocol) as processor:
            actor_id = message.actor_ref.uid
            if actor_id not in self._actors:
                raise ActorNotExist(f'Actor {actor_id} does not exist')
            call = self._actors[actor_id].__on_receive__(message.content)
            # asynchronously run, tell does not care about result
            asyncio.create_task(call)
            await asyncio.sleep(0)
            processor.result = ResultMessage(message.message_id, None,
                                             protocol=message.protocol)
        return processor.result

    @classmethod
    @implements(AbstractActorPool.create)
    async def create(cls, config: Dict) -> "AbstractActorPool":
        config = config.copy()
        kw = cls._parse_config(config)
        process_index = kw['process_index']
        actor_pool_config = kw['config']
        external_addresses = \
            actor_pool_config.get_pool_config(process_index)['external_address']
        internal_address = kw['internal_address']

        # set default router
        # actor context would be able to use exact client
        Router.set_instance(kw['router'])

        def handle_channel(channel):
            return pool.on_new_channel(channel)

        # create servers
        create_server_tasks = []
        for addr in external_addresses + [internal_address, LOCAL_ADDRESS]:
            server_type = get_server_type(addr)
            task = asyncio.create_task(
                server_type.create(dict(address=addr,
                                        handle_channel=handle_channel)))
            create_server_tasks.append(task)
        done, _ = await asyncio.wait(create_server_tasks)
        kw['servers'] = [f.result() for f in done]

        # create pool
        pool = cls(**kw)
        return pool


@_register_message_handler
class SubActorPool(ActorPoolBase):
    __slots__ = '_main_address',

    def __init__(self,
                 process_index: int,
                 label: str,
                 external_address: str,
                 internal_address: str,
                 env: Dict,
                 router: Router,
                 config: ActorPoolConfig,
                 servers: List[Server],
                 main_address: str):
        super().__init__(process_index, label,
                         external_address,
                         internal_address,
                         env, router, config, servers)
        self._main_address = main_address

    async def notify_main_pool_to_destroy(self, message: DestroyActorMessage):
        await self.call(self._main_address, message)

    @implements(AbstractActorPool.destroy_actor)
    async def destroy_actor(self,
                            message: DestroyActorMessage) -> result_message_type:
        result = await super().destroy_actor(message)
        if isinstance(result, ResultMessage) and not message.from_main:
            # sync back to main actor pool
            await self.notify_main_pool_to_destroy(message)
        return result

    @staticmethod
    def _parse_config(config: Dict) -> Dict:
        kw = AbstractActorPool._parse_config(config)
        kw['main_address'] = \
            kw['config'].get_pool_config(0)['external_address'][0]
        return kw


async def _create_sub_pool(
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


def _start_sub_pool(
        actor_config: ActorPoolConfig,
        process_index: int,
        started: multiprocessing.Event):
    coro = _create_sub_pool(actor_config, process_index, started)
    asyncio.run(coro)


def _start_sub_pool_in_process(
        actor_config: ActorPoolConfig,
        process_index: int,
        start_method: str = None
):
    if start_method is not None:
        multiprocessing.set_start_method(start_method)
    started = multiprocessing.Event()
    process = multiprocessing.Process(
        target=_start_sub_pool,
        args=(actor_config, process_index, started),
        name=f'MarsActorPool{process_index}'
    )
    process.daemon = True
    process.start()
    # wait for sub actor pool to finish starting
    started.wait()
    return process


@_register_message_handler
class MainActorPool(ActorPoolBase):
    __slots__ = '_allocated_actors', '_sub_processes'

    def __init__(self,
                 process_index: int,
                 label: str,
                 external_address: str,
                 internal_address: str,
                 env: Dict,
                 router: Router,
                 config: ActorPoolConfig,
                 servers: List[Server]):
        super().__init__(process_index, label, external_address,
                         internal_address, env, router, config, servers)
        self._allocated_actors: allocated_type = \
            {addr: dict() for addr in self._config.get_external_addresses()}
        self._sub_processes: Dict[str, multiprocessing.Process] = dict()

    def attach_sub_process(self,
                           external_address: str,
                           process: multiprocessing.Process):
        self._sub_processes[external_address] = process

    @implements(AbstractActorPool.create_actor)
    async def create_actor(self,
                           message: CreateActorMessage) -> result_message_type:
        with _ErrorProcessor(message_id=message.message_id,
                             protocol=message.protocol) as processor:
            allocate_strategy = message.allocate_strategy
            # get allocated address according to corresponding strategy
            address = allocate_strategy.get_allocated_address(
                self._config, self._allocated_actors)
            if address == self.external_address:
                # creating actor on main actor pool
                result = await super().create_actor(message)
                if isinstance(result, ResultMessage):
                    self._allocated_actors[self.external_address][result.result] = \
                        (allocate_strategy, message)
                processor.result = result
            else:
                # creating actor on sub actor pool
                # rewrite allocate strategy to AddressSpecified
                new_allocate_strategy = AddressSpecified(address)
                new_create_actor_message = CreateActorMessage(
                    message.message_id, message.actor_cls,
                    message.actor_id, message.args, message.kwargs,
                    allocate_strategy=new_allocate_strategy,
                    scoped_message_ids=message.scoped_message_ids,
                    protocol=message.protocol
                )
                result = await self.call(address, new_create_actor_message)
                if isinstance(result, ResultMessage):
                    self._allocated_actors[address][result.result] = \
                        (allocate_strategy, message)
                processor.result = result

        return processor.result

    @implements(AbstractActorPool.has_actor)
    async def has_actor(self,
                        message: HasActorMessage) -> ResultMessage:
        actor_ref = message.actor_ref
        # lookup allocated
        for address, item in self._allocated_actors.items():
            ref = create_actor_ref(address, actor_ref.uid)
            if ref in item:
                return ResultMessage(
                    message.message_id, True,
                    protocol=message.protocol)

        return ResultMessage(message.message_id, False,
                             protocol=message.protocol)

    @implements(AbstractActorPool.destroy_actor)
    async def destroy_actor(self,
                            message: DestroyActorMessage) -> result_message_type:
        actor_ref_message = ActorRefMessage(
            message.message_id, message.actor_ref,
            protocol=message.protocol)
        result = await self.actor_ref(actor_ref_message)
        if not isinstance(result, ResultMessage):
            return result
        real_actor_ref = result.result
        if real_actor_ref.address == self.external_address:
            del self._actors[real_actor_ref.uid]
            del self._allocated_actors[self.external_address][real_actor_ref]
            return ResultMessage(message.message_id, real_actor_ref.uid,
                                 protocol=message.protocol)
        # remove allocated actor ref
        self._allocated_actors[real_actor_ref.address].pop(real_actor_ref, None)
        new_destroy_message = DestroyActorMessage(
            message.message_id, real_actor_ref, from_main=True,
            protocol=message.protocol)
        return await self.call(real_actor_ref.address, new_destroy_message)

    @implements(AbstractActorPool.send)
    async def send(self,
                   message: SendMessage) -> result_message_type:
        if message.actor_ref.uid in self._actors:
            return await super().send(message)
        actor_ref_message = ActorRefMessage(
            message.message_id, message.actor_ref,
            protocol=message.protocol)
        result = await self.actor_ref(actor_ref_message)
        if not isinstance(result, ResultMessage):
            return result
        actor_ref = result.result
        new_send_message = SendMessage(
            message.message_id, actor_ref, message.content,
            scoped_message_ids=message.scoped_message_ids,
            protocol=message.protocol)
        return await self.call(actor_ref.address, new_send_message)

    @implements(AbstractActorPool.tell)
    async def tell(self,
                   message: TellMessage) -> result_message_type:
        if message.actor_ref.uid in self._actors:
            return await super().tell(message)
        actor_ref_message = ActorRefMessage(
            message.message_id, message.actor_ref,
            protocol=message.protocol)
        result = await self.actor_ref(actor_ref_message)
        if not isinstance(result, ResultMessage):
            return result
        actor_ref = result.result
        new_tell_message = TellMessage(
            message.message_id, actor_ref, message.content,
            scoped_message_ids=message.scoped_message_ids,
            protocol=message.protocol)
        return await self.call(actor_ref.address, new_tell_message)

    @implements(AbstractActorPool.actor_ref)
    async def actor_ref(self,
                        message: ActorRefMessage) -> result_message_type:
        actor_ref = message.actor_ref
        if actor_ref.address == self.external_address and \
                actor_ref.uid in self._actors:
            return ResultMessage(
                message.message_id, actor_ref,
                protocol=message.protocol)

        # lookup allocated
        for address, item in self._allocated_actors.items():
            ref = create_actor_ref(address, actor_ref.uid)
            if ref in item:
                return ResultMessage(
                    message.message_id, ref,
                    protocol=message.protocol)

        with _ErrorProcessor(message.message_id,
                             protocol=message.protocol) as processor:
            raise ActorNotExist(f'Actor {actor_ref.uid} does not exist')

        return processor.result

    @classmethod
    @implements(AbstractActorPool.create)
    async def create(cls, config: Dict) -> "MainActorPool":
        config = config.copy()
        start_method = config.pop('start_method', None)
        if 'process_index' not in config:
            config['process_index'] = 0
        curr_process_index = config.get('process_index')

        tasks = []

        # create sub actor pools
        actor_pool_config: ActorPoolConfig = config.get('actor_pool_config')
        n_sub_pool = actor_pool_config.n_pool - 1
        if n_sub_pool > 0:
            executor = futures.ThreadPoolExecutor(n_sub_pool)
            loop = asyncio.get_running_loop()
            process_indexes = actor_pool_config.get_process_indexes()
            for process_index in process_indexes:
                if process_index == curr_process_index:
                    continue
                create_task_pool = loop.run_in_executor(
                    executor, _start_sub_pool_in_process,
                    actor_pool_config, process_index, start_method)
                await asyncio.sleep(0)
                tasks.append(create_task_pool)

        # create main actor pool
        create_task = asyncio.create_task(super().create(config))
        tasks.append(create_task)

        # wait for all pools
        await asyncio.gather(*tasks)
        pool: MainActorPool = await create_task
        addresses = actor_pool_config.get_external_addresses()[1:]
        processes = [await t for t in tasks[:-1]]
        assert len(addresses) == len(processes)
        for addr, proc in zip(addresses, processes):
            pool.attach_sub_process(addr, proc)
        return pool

    async def _stop_sub_pool(self,
                             address: str,
                             process: multiprocessing.Process):
        stop_message = ControlMessage(
            new_message_id(), ControlMessageType.stop,
            None, protocol=DEFAULT_PROTOCOL)
        message = await self.call(address, stop_message)
        if isinstance(message, ErrorMessage):
            raise message.error.with_traceback(message.traceback)
        # terminate process
        process.terminate()

    async def _stop_sub_pools(self):
        to_stop_processes: Dict[str, multiprocessing.Process] = dict()
        for address, process in self._sub_processes.items():
            if not process.is_alive():
                return
            to_stop_processes[address] = process

        tasks = []
        for address, process in to_stop_processes.items():
            tasks.append(self._stop_sub_pool(address, process))
        await asyncio.wait(tasks)

    @implements(AbstractActorPool.stop)
    async def stop(self):
        await self._stop_sub_pools()
        await super().stop()


async def create_actor_pool(address: str,
                            n_process: int = None,
                            labels: List[str] = None,
                            ports: List[int] = None,
                            envs: List[Dict] = None,
                            subprocess_start_method: str = None) -> MainActorPool:
    n_process = n_process or multiprocessing.cpu_count()
    if labels and len(labels) != n_process + 1:
        raise ValueError(f'`labels` should be of size {n_process + 1}, '
                         f'got {len(labels)}')
    if envs and len(envs) != n_process:
        raise ValueError(f'`envs` should be of size {n_process}, '
                         f'got {len(envs)}')
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
        ports = [get_next_port() for _ in range(n_process + 1)]
        port = ports[0]
        sub_ports = ports[1:]

    actor_pool_config = ActorPoolConfig()
    # add main config
    actor_pool_config.add_pool_conf(
        0, labels[0] if labels else None,
        gen_internal_address(0), f'{host}:{port}')
    # add sub configs
    for i in range(n_process):
        actor_pool_config.add_pool_conf(
            i + 1, labels[i + 1] if labels else None,
            gen_internal_address(i + 1), f'{host}:{sub_ports[i]}',
            env=envs[i] if envs else None
        )

    pool = await MainActorPool.create({
        'actor_pool_config': actor_pool_config,
        'start_method': subprocess_start_method
    })
    await pool.start()
    return pool
