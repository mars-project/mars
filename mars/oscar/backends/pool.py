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
import contextlib
import itertools
import multiprocessing
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, List, Type, TypeVar, Coroutine, Callable, Union, Optional

from ...utils import implements, to_binary
from ...utils import lazy_import
from ..api import Actor
from ..core import ActorRef
from ..errors import ActorAlreadyExist, ActorNotExist, ServerClosed, CannotCancelTask
from ..utils import create_actor_ref
from .allocate_strategy import allocated_type, AddressSpecified
from .communication import Channel, Server, \
    get_server_type, gen_local_address
from .config import ActorPoolConfig
from .core import result_message_type, ActorCaller
from .message import _MessageBase, new_message_id, DEFAULT_PROTOCOL, MessageType, \
    ResultMessage, ErrorMessage, CreateActorMessage, HasActorMessage, \
    DestroyActorMessage, ActorRefMessage, SendMessage, TellMessage, \
    CancelMessage, ControlMessage, ControlMessageType
from .router import Router

ray = lazy_import("ray")


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
        (MessageType.cancel, pool_type.cancel),
        (MessageType.control, pool_type.handle_control_command)
    ]:
        pool_type._message_handler[message_type] = handler
    return pool_type


class AbstractActorPool(ABC):
    __slots__ = 'process_index', 'label', 'external_address', 'internal_address', 'env', \
                '_servers', '_router', '_config', '_stopped', '_actors', '_caller', '_process_messages'

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

        # states
        # actor id -> actor
        self._actors: Dict[bytes, Actor] = dict()
        # message id -> future
        self._process_messages: Dict[bytes, asyncio.Future] = dict()

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

    @abstractmethod
    async def cancel(self,
                     message: CancelMessage) -> result_message_type:
        """
        Cancel message that sent

        Parameters
        ----------
        message: CancelMessage
            Cancel message.

        Returns
        -------
        result_message
            result or error message
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
                # remove router from global one
                global_router = Router.get_instance()
                global_router.remove_router(self._router)
                # update router
                self._router.set_mapping(
                    actor_pool_config.external_to_internal_address_map)
                # update global router
                global_router.add_router(self._router)
            elif message.control_message_type == ControlMessageType.get_config:
                if message.content == 'main_pool_address':
                    main_process_index = self._config.get_process_indexes()[0]
                    content = \
                        self._config.get_pool_config(main_process_index)['external_address'][0]
                else:
                    content = self._config
            else:  # pragma: no cover
                raise TypeError(f'Unable to handle control message '
                                f'with type {message.control_message_type}')
            processor.result = ResultMessage(
                message.message_id, content,
                protocol=message.protocol)

        return processor.result

    @contextlib.contextmanager
    def _run_coro(self, message_id: bytes, coro: Coroutine):
        future = asyncio.create_task(coro)
        self._process_messages[message_id] = future
        try:
            yield future
        finally:
            self._process_messages.pop(message_id, None)

    async def process_message(self,
                              message: _MessageBase,
                              channel: Channel):
        handler = self._message_handler[message.message_type]
        with _ErrorProcessor(message.message_id,
                             message.protocol) as processor:
            with self._run_coro(message.message_id, handler(self, message)) as future:
                processor.result = await future
        await channel.send(processor.result)

    async def call(self,
                   dest_address: str,
                   message: _MessageBase) -> result_message_type:
        return await self._caller.call(self._router, dest_address, message)

    @staticmethod
    def _parse_config(config: Dict, kw: Dict) -> Dict:
        actor_pool_config: ActorPoolConfig = config.pop('actor_pool_config')
        kw['config'] = actor_pool_config
        kw['process_index'] = process_index = config.pop('process_index')
        curr_pool_config = actor_pool_config.get_pool_config(process_index)
        kw['label'] = curr_pool_config['label']
        external_addresses = curr_pool_config['external_address']
        kw['external_address'] = external_addresses[0]
        kw['internal_address'] = curr_pool_config['internal_address']
        kw['router'] = Router(external_addresses,
                              gen_local_address(process_index),
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
        if self._stopped.is_set():
            raise RuntimeError('pool has been stopped, cannot start again')
        start_servers = [server.start() for server in self._servers]
        await asyncio.gather(*start_servers)

    async def join(self, timeout: float = None):
        wait_stopped = asyncio.create_task(self._stopped.wait())

        try:
            await asyncio.wait_for(wait_stopped, timeout=timeout)
        except (futures.TimeoutError, asyncio.TimeoutError):  # pragma: no cover
            wait_stopped.cancel()

    async def stop(self):
        try:
            # clean global router
            Router.get_instance().remove_router(self._router)

            stop_tasks = []
            # stop all servers
            stop_tasks.extend([server.stop() for server in self._servers])
            # stop all clients
            stop_tasks.append(self._caller.stop())
            await asyncio.gather(*stop_tasks)

            self._servers = []
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
            with self._run_coro(message.message_id,
                                actor.__post_create__()) as future:
                await future

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
            with self._run_coro(message.message_id,
                                actor.__pre_destroy__()) as future:
                await future
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
            actor_id = to_binary(message.actor_ref.uid)
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
            coro = self._actors[actor_id].__on_receive__(message.content)
            with self._run_coro(message.message_id, coro) as future:
                result = await future
            processor.result = ResultMessage(message.message_id, result,
                                             protocol=message.protocol)
        return processor.result

    @implements(AbstractActorPool.tell)
    async def tell(self,
                   message: TellMessage) -> result_message_type:
        with _ErrorProcessor(message.message_id,
                             message.protocol) as processor:
            actor_id = message.actor_ref.uid
            if actor_id not in self._actors:  # pragma: no cover
                raise ActorNotExist(f'Actor {actor_id} does not exist')
            call = self._actors[actor_id].__on_receive__(message.content)
            # asynchronously run, tell does not care about result
            asyncio.create_task(call)
            await asyncio.sleep(0)
            processor.result = ResultMessage(message.message_id, None,
                                             protocol=message.protocol)
        return processor.result

    @implements(AbstractActorPool.cancel)
    async def cancel(self,
                     message: CancelMessage) -> result_message_type:
        with _ErrorProcessor(message.message_id,
                             message.protocol) as processor:
            future = self._process_messages.get(message.cancel_message_id)
            if future is None:  # pragma: no cover
                raise CannotCancelTask('Task not exists, maybe it is done '
                                       'or cancelled already')
            future.cancel()
            processor.result = ResultMessage(message.message_id, True,
                                             protocol=message.protocol)
        return processor.result

    @staticmethod
    def _set_global_router(router: Router):
        # be cautious about setting global router
        # for instance, multiple main pool may be created in the same process

        # get default router or create an empty one
        default_router = Router.get_instance_or_empty()
        Router.set_instance(default_router)
        # append this router to global
        default_router.add_router(router)

    @classmethod
    @implements(AbstractActorPool.create)
    async def create(cls, config: Dict) -> "ActorPoolType":
        config = config.copy()
        kw = dict()
        cls._parse_config(config, kw)
        process_index: int = kw['process_index']
        actor_pool_config = kw['config']
        external_addresses = \
            actor_pool_config.get_pool_config(process_index)['external_address']
        internal_address = kw['internal_address']

        # set default router
        # actor context would be able to use exact client
        cls._set_global_router(kw['router'])

        def handle_channel(channel):
            return pool.on_new_channel(channel)

        # create servers
        create_server_tasks = []
        for addr in set(external_addresses + [internal_address, gen_local_address(process_index)]):
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


ActorPoolType = TypeVar('ActorPoolType', bound=AbstractActorPool)
MainActorPoolType = TypeVar('MainActorPoolType', bound='MainActorPoolBase')
SubProcessHandle = Union[multiprocessing.Process, 'ray.actor.ActorHandle']


class SubActorPoolBase(ActorPoolBase):
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

    async def notify_main_pool_to_destroy(self, message: DestroyActorMessage):  # pragma: no cover
        await self.call(self._main_address, message)

    @implements(AbstractActorPool.actor_ref)
    async def actor_ref(self,
                        message: ActorRefMessage) -> result_message_type:
        result = await super().actor_ref(message)
        if isinstance(result, ErrorMessage):
            message.actor_ref.address = self._main_address
            result = await self.call(self._main_address, message)
        return result

    @implements(AbstractActorPool.destroy_actor)
    async def destroy_actor(self,
                            message: DestroyActorMessage) -> result_message_type:
        result = await super().destroy_actor(message)
        if isinstance(result, ResultMessage) and not message.from_main:
            # sync back to main actor pool
            await self.notify_main_pool_to_destroy(message)
        return result

    @staticmethod
    def _parse_config(config: Dict, kw: Dict) -> Dict:
        kw = AbstractActorPool._parse_config(config, kw)
        config: ActorPoolConfig = kw['config']
        main_process_index = config.get_process_indexes()[0]
        kw['main_address'] = \
            config.get_pool_config(main_process_index)['external_address'][0]
        return kw


class MainActorPoolBase(ActorPoolBase):
    __slots__ = '_allocated_actors', 'sub_actor_pool_manager', '_auto_recover', \
                '_monitor_task', '_on_process_down', '_on_process_recover'

    def __init__(self,
                 process_index: int,
                 label: str,
                 external_address: str,
                 internal_address: str,
                 env: Dict,
                 router: Router,
                 config: ActorPoolConfig,
                 servers: List[Server],
                 subprocess_start_method: str = None,
                 auto_recover: Union[str, bool] = 'actor',
                 on_process_down: Callable[[MainActorPoolType, str], None] = None,
                 on_process_recover: Callable[[MainActorPoolType, str], None] = None):
        super().__init__(process_index, label, external_address,
                         internal_address, env, router, config, servers)
        self._subprocess_start_method = subprocess_start_method

        # auto recovering
        self._auto_recover = auto_recover
        self._monitor_task: Optional[asyncio.Task] = None
        self._on_process_down = on_process_down
        self._on_process_recover = on_process_recover

        # states
        self._allocated_actors: allocated_type = \
            {addr: dict() for addr in self._config.get_external_addresses()}

        self.sub_processes: Dict[str, SubProcessHandle] = dict()

    _process_index_gen = itertools.count()

    @classmethod
    def next_process_index(cls):
        return next(cls._process_index_gen)

    @property
    def _sub_processes(self):
        return self.sub_processes

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
                        (allocate_strategy, new_create_actor_message)
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
            await super().destroy_actor(message)
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
        actor_ref.uid = to_binary(actor_ref.uid)
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
            raise ActorNotExist(f'Actor {actor_ref.uid} does not exist in {actor_ref.address}')

        return processor.result

    @implements(AbstractActorPool.cancel)
    async def cancel(self,
                     message: CancelMessage) -> result_message_type:
        if message.address == self.external_address:
            # local message
            return await super().cancel(message)
        # redirect to sub pool
        return await self.call(message.address, message)

    @implements(AbstractActorPool.handle_control_command)
    async def handle_control_command(self,
                                     message: ControlMessage) -> result_message_type:
        with _ErrorProcessor(message.message_id, message.protocol) as processor:
            if message.address == self.external_address:
                if message.control_message_type == ControlMessageType.sync_config:
                    # sync config, need to notify all sub pools
                    tasks = []
                    for addr in self.sub_processes:
                        control_message = ControlMessage(
                            new_message_id(), addr,
                            message.control_message_type, message.content,
                            scoped_message_ids=message.scoped_message_ids,
                            protocol=message.protocol)
                        tasks.append(asyncio.create_task(self.call(addr, control_message)))
                    # call super
                    task = asyncio.create_task(super().handle_control_command(message))
                    tasks.append(task)
                    await asyncio.gather(*tasks)
                    processor.result = await task
                else:
                    processor.result = await super().handle_control_command(message)
            elif message.control_message_type == ControlMessageType.stop:
                timeout, force = message.content if message.content is not None \
                    else (None, None)
                await self.stop_sub_pool(
                    message.address,
                    self.sub_processes[message.address],
                    timeout=timeout,
                    force=force)
                processor.result = ResultMessage(message.message_id, True,
                                                 protocol=message.protocol)
            else:
                processor.result = await self.call(message.address, message)
        return processor.result

    @staticmethod
    def _parse_config(config: Dict, kw: Dict) -> Dict:
        kw['subprocess_start_method'] = config.pop('start_method', None)
        kw['auto_recover'] = config.pop('auto_recover', 'actor')
        kw['on_process_down'] = config.pop('on_process_down', None)
        kw['on_process_recover'] = config.pop('on_process_recover', None)
        kw = AbstractActorPool._parse_config(config, kw)
        return kw

    @classmethod
    @implements(AbstractActorPool.create)
    async def create(cls, config: Dict) -> MainActorPoolType:
        config = config.copy()
        actor_pool_config: ActorPoolConfig = config.get('actor_pool_config')
        start_method = config.get('start_method', None)
        if 'process_index' not in config:
            config['process_index'] = actor_pool_config.get_process_indexes()[0]
        curr_process_index = config.get('process_index')

        tasks = []
        # create sub actor pools
        n_sub_pool = actor_pool_config.n_pool - 1
        if n_sub_pool > 0:
            process_indexes = actor_pool_config.get_process_indexes()
            for process_index in process_indexes:
                if process_index == curr_process_index:
                    continue
                create_pool_task = asyncio.create_task(cls.start_sub_pool(
                    actor_pool_config, process_index, start_method))
                await asyncio.sleep(0)
                # await create_pool_task
                tasks.append(create_pool_task)

        processes = [await t for t in tasks]
        # create main actor pool
        pool: MainActorPoolType = await super().create(config)
        addresses = actor_pool_config.get_external_addresses()[1:]

        assert len(addresses) == len(processes), \
            f"addresses {addresses}, processes {processes}"
        for addr, proc in zip(addresses, processes):
            pool.attach_sub_process(addr, proc)
        return pool

    @implements(AbstractActorPool.start)
    async def start(self):
        await super().start()
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self.monitor_sub_pools())
            await asyncio.sleep(0)

    @implements(AbstractActorPool.stop)
    async def stop(self):
        await self.stop_sub_pools()
        await super().stop()
        if self._monitor_task and not self._monitor_task.done():
            await self._monitor_task
            self._monitor_task = None

    @classmethod
    @abstractmethod
    async def start_sub_pool(
            cls,
            actor_pool_config: ActorPoolConfig,
            process_index: int,
            start_method: str = None):
        """Start a sub actor pool"""

    def attach_sub_process(self,
                           external_address: str,
                           process: SubProcessHandle):
        self.sub_processes[external_address] = process

    async def stop_sub_pools(self):
        to_stop_processes: Dict[str, SubProcessHandle] = dict()
        for address, process in self.sub_processes.items():
            if not await self.is_sub_pool_alive(process):
                continue
            to_stop_processes[address] = process

        tasks = []
        for address, process in to_stop_processes.items():
            tasks.append(self.stop_sub_pool(address, process))
        await asyncio.gather(*tasks)

    async def stop_sub_pool(
            self,
            address: str,
            process: SubProcessHandle,
            timeout: float = None,
            force: bool = False):
        if force:
            await self.kill_sub_pool(process, force=True)
            return

        stop_message = ControlMessage(
            new_message_id(), address, ControlMessageType.stop,
            None, protocol=DEFAULT_PROTOCOL)
        try:
            if timeout is None:
                message = await self.call(address, stop_message)
                if isinstance(message, ErrorMessage):  # pragma: no cover
                    raise message.error.with_traceback(message.traceback)
            else:
                call = asyncio.create_task(self.call(address, stop_message))
                try:
                    await asyncio.wait_for(call, timeout)
                except (futures.TimeoutError, asyncio.TimeoutError):  # pragma: no cover
                    # timeout, just let kill to finish it
                    force = True
        except (ConnectionError, ServerClosed):  # pragma: no cover
            # process dead maybe, ignore it
            pass
        # kill process
        await self.kill_sub_pool(process, force=force)

    @abstractmethod
    async def kill_sub_pool(self, process: SubProcessHandle, force: bool = False):
        """Kill a sub actor pool"""

    @abstractmethod
    async def is_sub_pool_alive(self, process: SubProcessHandle):
        """
        Check whether sub pool process is alive
        Parameters
        ----------
        process : SubProcessHandle
            sub pool process handle
        Returns
        -------
        bool
        """

    def process_sub_pool_lost(self, address: str):
        if self._auto_recover in (False, 'process'):
            # process down, when not auto_recover
            # or only recover process, remove all created actors
            self._allocated_actors[address] = dict()

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

    async def monitor_sub_pools(self):
        try:
            while not self._stopped.is_set():
                for address in self.sub_processes:
                    process = self.sub_processes[address]
                    if not await self.is_sub_pool_alive(process):  # pragma: no cover
                        if self._on_process_down is not None:
                            self._on_process_down(self, address)
                        self.process_sub_pool_lost(address)
                        if self._auto_recover:
                            await self.recover_sub_pool(address)
                            if self._on_process_recover is not None:
                                self._on_process_recover(self, address)

                # check every half second
                await asyncio.sleep(.5)
        except asyncio.CancelledError:  # pragma: no cover
            # cancelled
            return

    @classmethod
    @abstractmethod
    def get_external_addresses(
            cls, address: str, n_process: int = None, ports: List[int] = None):
        """Returns external addresses for n pool processes"""

    @classmethod
    @abstractmethod
    def gen_internal_address(cls, process_index: int,
                             external_address: str = None) -> str:
        """Returns internal address for pool of specified process index"""


async def create_actor_pool(address: str,
                            pool_cls: Type[MainActorPoolType] = None,
                            n_process: int = None,
                            labels: List[str] = None,
                            ports: List[int] = None,
                            envs: List[Dict] = None,
                            subprocess_start_method: str = None,
                            auto_recover: Union[str, bool] = 'actor',
                            on_process_down: Callable[[MainActorPoolType, str], None] = None,
                            on_process_recover: Callable[[MainActorPoolType, str], None] = None) \
        -> MainActorPoolType:
    if n_process is None:
        n_process = multiprocessing.cpu_count()
    if labels and len(labels) != n_process + 1:
        raise ValueError(f'`labels` should be of size {n_process + 1}, '
                         f'got {len(labels)}')
    if envs and len(envs) != n_process:
        raise ValueError(f'`envs` should be of size {n_process}, '
                         f'got {len(envs)}')
    if auto_recover is True:
        auto_recover = 'actor'
    if auto_recover not in ('actor', 'process', False):
        raise ValueError(f'`auto_recover` should be one of "actor", "process", '
                         f'True or False, got {auto_recover}')
    external_addresses = pool_cls.get_external_addresses(address, n_process=n_process, ports=ports)
    actor_pool_config = ActorPoolConfig()
    # add main config
    main_process_index = pool_cls.next_process_index()
    actor_pool_config.add_pool_conf(
        main_process_index,
        labels[0] if labels else None,
        pool_cls.gen_internal_address(main_process_index, external_addresses[0]),
        external_addresses[0])
    # add sub configs
    for i in range(n_process):
        sub_process_index = pool_cls.next_process_index()
        actor_pool_config.add_pool_conf(
            sub_process_index,
            labels[i + 1] if labels else None,
            pool_cls.gen_internal_address(sub_process_index, external_addresses[i + 1]),
            external_addresses[i + 1],
            env=envs[i] if envs else None
        )

    pool: MainActorPoolType = await pool_cls.create({
        'actor_pool_config': actor_pool_config,
        'process_index': main_process_index,
        'start_method': subprocess_start_method,
        'auto_recover': auto_recover,
        'on_process_down': on_process_down,
        'on_process_recover': on_process_recover
    })
    await pool.start()
    return pool
