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

from urllib.parse import urlparse

from .core cimport ActorRef
from .utils cimport new_actor_id

cdef dict _backend_context_cls = dict()

cdef object _context = None


cdef class BaseActorContext:
    # allocate strategy is for Mars backend only
    support_allocate_strategy = False

    """
    Base class for actor context. Every backend need to implement
    actor context for their own.
    """
    async def create_actor(self, object actor_cls, *args, object uid=None, object address=None, **kwargs):
        """
        Stub method for creating an actor in current context.

        Parameters
        ----------
        actor_cls : Actor
            Actor class
        args : tuple
            args to be passed into actor_cls.__init__
        uid : identifier
            Actor identifier
        address : str
            Address to locate the actor
        kwargs : dict
            kwargs to be passed into actor_cls.__init__

        Returns
        -------
        ActorRef

        """
        raise NotImplementedError

    async def has_actor(self, ActorRef actor_ref):
        """
        Check if actor exists in current context

        Parameters
        ----------
        actor_ref : ActorRef
            Reference to an actor

        Returns
        -------
        bool
        """
        raise NotImplementedError

    async def destroy_actor(self, ActorRef actor_ref):
        """
        Destroy an actor by its reference

        Parameters
        ----------
        actor_ref : ActorRef
            Reference to an actor

        Returns
        -------
        bool
        """
        raise NotImplementedError

    async def kill_actor(self, ActorRef actor_ref):
        """
        Force to kill an actor, take care this is a dangerous operation,
        it may lead to the result that other actors are killed as well.
        Hence, unless you are knowing what you are doing and know how
        to recover possible effected actors, DO NOT USE this method!

        Parameters
        ----------
        actor_ref : ActorRef
            Reference to an actor

        Returns
        -------
        bool
        """

    async def send(self, ActorRef actor_ref, object message, bint wait_response=True):
        """
        Send a message to given actor by its reference

        Parameters
        ----------
        actor_ref : ActorRef
            Reference to an actor
        message : object
            Message to send to an actor, need to comply to Actor.__on_receive__
        wait_response : bool
            Whether to wait for responses from the actor.

        Returns
        -------
        object
        """
        raise NotImplementedError

    async def actor_ref(self, *args, **kwargs):
        """
        Create a reference to an actor

        Returns
        -------
        ActorRef
        """
        raise NotImplementedError

    async def wait_actor_pool_recovered(self, str address, str main_address = None):
        """
        Wait until an actor pool is recovered

        Parameters
        ----------
        address
            address of the actor pool
        main_address
            address of the main pool
        """
        raise NotImplementedError


cdef class ClientActorContext(BaseActorContext):
    """
    Default actor context. This context will keep references to other contexts
    given their protocol scheme (i.e., `ray://xxx`).
    """
    cdef dict _backend_contexts

    def __init__(self):
        self._backend_contexts = dict()

    cdef inline object _get_backend_context(self, object address):
        if address is None:
            raise ValueError('address has to be provided')
        if '://' not in address:
            scheme = None
        else:
            scheme = urlparse(address).scheme or None
        try:
            return self._backend_contexts[scheme]
        except KeyError:
            context = self._backend_contexts[scheme] = \
                _backend_context_cls[scheme]()
            return context

    def create_actor(self, object actor_cls, *args, object uid=None, object address=None, **kwargs):
        context = self._get_backend_context(address)
        uid = uid or new_actor_id()
        return context.create_actor(actor_cls, *args, uid=uid, address=address, **kwargs)

    def has_actor(self, ActorRef actor_ref):
        context = self._get_backend_context(actor_ref.address)
        return context.has_actor(actor_ref)

    def destroy_actor(self, ActorRef actor_ref):
        context = self._get_backend_context(actor_ref.address)
        return context.destroy_actor(actor_ref)

    def kill_actor(self, ActorRef actor_ref):
        context = self._get_backend_context(actor_ref.address)
        return context.kill_actor(actor_ref)

    def actor_ref(self, *args, **kwargs):
        from .utils import create_actor_ref
        actor_ref = create_actor_ref(*args, **kwargs)
        context = self._get_backend_context(actor_ref.address)
        return context.actor_ref(actor_ref)

    def send(self, ActorRef actor_ref, object message, bint wait_response=True):
        context = self._get_backend_context(actor_ref.address)
        return context.send(actor_ref, message, wait_response=wait_response)

    def wait_actor_pool_recovered(self, str address, str main_address = None):
        context = self._get_backend_context(address)
        return context.wait_actor_pool_recovered(address, main_address)


def register_backend_context(scheme, cls):
    assert issubclass(cls, BaseActorContext)
    _backend_context_cls[scheme] = cls


def get_context():
    """
    Get an actor context. If not in an actor environment,
    ClientActorContext will be used
    """
    global _context
    if _context is None:
        _context = ClientActorContext()
    return _context


def set_context(context):
    """
    Set default actor context to use
    """
    _context = context
