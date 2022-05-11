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
import functools
import inspect
import logging
from typing import Union, Dict, List

from .....core.context import Context
from .....utils import implements, lazy_import
from ....context import ThreadedServiceContext
from .fetcher import RayFetcher

ray = lazy_import("ray")
logger = logging.getLogger(__name__)


class RayRemoteObjectManager:
    """
    The remote object manager in task state actor.

    This should be a sync actor because ray call is out of order on the
    async actor: https://github.com/ray-project/ray/issues/19822
    """

    def __init__(self):
        self._loop = asyncio.get_event_loop()
        self._named_remote_objects = {}

    def create_remote_object(self, name: str, object_cls, *args, **kwargs):
        remote_object = object_cls(*args, **kwargs)
        self._named_remote_objects[name] = remote_object

    def destroy_remote_object(self, name: str):
        self._named_remote_objects.pop(name, None)

    def call_remote_object(self, name: str, attr: str, *args, **kwargs):
        remote_object = self._named_remote_objects[name]
        meth = getattr(remote_object, attr)
        async_meth = self._async_to_sync(self._loop, meth)
        return async_meth(*args, **kwargs)

    @staticmethod
    @functools.lru_cache(100)
    def _async_to_sync(loop, func):
        if inspect.iscoroutinefunction(func):

            def sync_wrapper(*args, **kwargs):
                co = func(*args, **kwargs)
                return loop.run_until_complete(co)

            return sync_wrapper
        else:
            return func


class _RayRemoteObjectWrapper:
    def __init__(self, task_state_actor: "ray.actor.ActorHandle", name: str):
        self._task_state_actor = task_state_actor
        self._name = name

    def __getattr__(self, attr):
        def wrap(*args, **kwargs):
            r = self._task_state_actor.call_remote_object.remote(
                self._name, attr, *args, **kwargs
            )
            return ray.get(r)

        return wrap


class _RayRemoteObjectContext:
    def __init__(
        self, actor_name_or_handle: Union[str, "ray.actor.ActorHandle"], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._actor_name_or_handle = actor_name_or_handle
        self._task_state_actor = None

    def _get_task_state_actor(self) -> "ray.actor.ActorHandle":
        if self._task_state_actor is None:
            if isinstance(self._actor_name_or_handle, ray.actor.ActorHandle):
                self._task_state_actor = self._actor_name_or_handle
            else:
                self._task_state_actor = ray.get_actor(self._actor_name_or_handle)
        return self._task_state_actor

    @implements(Context.create_remote_object)
    def create_remote_object(self, name: str, object_cls, *args, **kwargs):
        task_state_actor = self._get_task_state_actor()
        task_state_actor.create_remote_object.remote(name, object_cls, *args, **kwargs)
        return _RayRemoteObjectWrapper(task_state_actor, name)

    @implements(Context.get_remote_object)
    def get_remote_object(self, name: str):
        task_state_actor = self._get_task_state_actor()
        return _RayRemoteObjectWrapper(task_state_actor, name)

    @implements(Context.destroy_remote_object)
    def destroy_remote_object(self, name: str):
        task_state_actor = self._get_task_state_actor()
        task_state_actor.destroy_remote_object.remote(name)


# TODO(fyrestone): Implement more APIs for Ray.
class RayExecutionContext(_RayRemoteObjectContext, ThreadedServiceContext):
    """The context for tiling."""

    def __init__(self, task_context: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_context = task_context

    @implements(Context.get_chunks_result)
    def get_chunks_result(self, data_keys: List[str]) -> List:
        logger.info("Getting %s chunks result.", len(data_keys))
        object_refs = [self._task_context[key] for key in data_keys]
        result = ray.get(object_refs)
        logger.info("Got %s chunks result.", len(result))
        return result


# TODO(fyrestone): Implement more APIs for Ray.
class RayExecutionWorkerContext(_RayRemoteObjectContext, dict):
    """The context for executing operands."""

    @staticmethod
    def new_custom_log_dir():
        return None
