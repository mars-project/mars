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

import logging
from dataclasses import asdict
from typing import Dict, List, Callable

from .....core.context import Context
from .....storage.base import StorageLevel
from .....typing import ChunkType
from .....utils import implements, lazy_import, sync_to_async
from ....context import ThreadedServiceContext
from .config import RayExecutionConfig

ray = lazy_import("ray")
logger = logging.getLogger(__name__)


class RayRemoteObjectManager:
    """The remote object manager in task state actor."""

    def __init__(self):
        self._named_remote_objects = {}

    def create_remote_object(self, name: str, object_cls, *args, **kwargs):
        remote_object = object_cls(*args, **kwargs)
        self._named_remote_objects[name] = remote_object

    def destroy_remote_object(self, name: str):
        self._named_remote_objects.pop(name, None)

    async def call_remote_object(self, name: str, attr: str, *args, **kwargs):
        remote_object = self._named_remote_objects[name]
        meth = getattr(remote_object, attr)
        async_meth = sync_to_async(meth)
        return await async_meth(*args, **kwargs)


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
        self,
        get_or_create_actor: Callable[[], "ray.actor.ActorHandle"],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._get_or_create_actor = get_or_create_actor
        self._task_state_actor = None

    def _get_task_state_actor(self) -> "ray.actor.ActorHandle":
        # Get the RayTaskState actor, this is more clear and faster than wraps
        # the `get_or_create_actor` by lru_cache in __init__ because this method
        # is called as needed.
        if self._task_state_actor is None:
            self._task_state_actor = self._get_or_create_actor()
        return self._task_state_actor

    @implements(Context.create_remote_object)
    def create_remote_object(self, name: str, object_cls, *args, **kwargs):
        task_state_actor = self._get_task_state_actor()
        r = task_state_actor.create_remote_object.remote(
            name, object_cls, *args, **kwargs
        )
        # Make sure the actor is created. The remote object may not be created
        # when get_remote_object from worker because the callers of
        # create_remote_object and get_remote_object are not in the same worker.
        # Use sync Ray actor requires this `ray.get`, too.
        ray.get(r)
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

    def __init__(
        self,
        config: RayExecutionConfig,
        task_context: Dict,
        task_chunks_meta: Dict,
        worker_addresses: List[str],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._config = config
        self._task_context = task_context
        self._task_chunks_meta = task_chunks_meta
        self._worker_addresses = worker_addresses

    @implements(Context.get_chunks_result)
    def get_chunks_result(self, data_keys: List[str], fetch_only: bool = False) -> List:
        logger.info("Getting %s chunks result.", len(data_keys))
        object_refs = [self._task_context[key] for key in data_keys]
        result = ray.get(object_refs)
        logger.info("Got %s chunks result.", len(result))
        return result if not fetch_only else None

    @implements(Context.get_chunks_meta)
    def get_chunks_meta(
        self, data_keys: List[str], fields: List[str] = None, error="raise"
    ) -> List[Dict]:
        if not self._task_chunks_meta:
            result = self._call(
                self._get_chunks_meta_from_service(
                    data_keys, fields=fields, error=error
                )
            )
        else:
            result = [{}] * len(data_keys)
            missing_key_indexes = []
            missing_keys = []
            for idx, key in enumerate(data_keys):
                try:
                    chunk_meta = self._task_chunks_meta[key]
                except KeyError:
                    missing_key_indexes.append(idx)
                    missing_keys.append(key)
                else:
                    meta = asdict(chunk_meta)
                    meta = {f: meta.get(f) for f in fields}
                    result[idx] = meta
            if missing_keys:
                missing_meta = self._call(
                    self._get_chunks_meta_from_service(
                        missing_keys, fields=fields, error=error
                    )
                )
                for idx, meta in zip(missing_key_indexes, missing_meta):
                    result[idx] = meta
        return result

    async def _get_chunks_meta_from_service(
        self, data_keys: List[str], fields: List[str] = None, error="raise"
    ) -> List[Dict]:
        get_metas = [
            self._meta_api.get_chunk_meta.delay(data_key, fields=fields, error=error)
            for data_key in data_keys
        ]
        return await self._meta_api.get_chunk_meta.batch(*get_metas)

    @implements(Context.get_total_n_cpu)
    def get_total_n_cpu(self) -> int:
        # TODO(fyrestone): Support auto scaling.
        return self._config.get_n_cpu() * self._config.get_n_worker()

    @implements(Context.get_worker_addresses)
    def get_worker_addresses(self) -> List[str]:
        # Returns virtual worker addresses.
        return self._worker_addresses


# TODO(fyrestone): Implement more APIs for Ray.
class RayExecutionWorkerContext(_RayRemoteObjectContext, dict):
    """The context for executing operands."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_chunk = None

    @classmethod
    @implements(Context.new_custom_log_dir)
    def new_custom_log_dir(cls):
        logger.info(
            "%s does not support register_custom_log_path / new_custom_log_dir",
            cls.__name__,
        )
        return None

    @staticmethod
    @implements(Context.register_custom_log_path)
    def register_custom_log_path(
        session_id: str,
        tileable_op_key: str,
        chunk_op_key: str,
        worker_address: str,
        log_path: str,
    ):
        raise NotImplementedError

    @classmethod
    @implements(Context.set_progress)
    def set_progress(cls, progress: float):
        logger.info(
            "%s does not support set_running_operand_key / set_progress", cls.__name__
        )

    @staticmethod
    @implements(Context.set_running_operand_key)
    def set_running_operand_key(session_id: str, op_key: str):
        raise NotImplementedError

    @classmethod
    @implements(Context.get_storage_info)
    def get_storage_info(
        cls, address: str = None, level: StorageLevel = StorageLevel.MEMORY
    ):
        logger.info("%s does not support get_storage_info", cls.__name__)
        return {}

    def set_current_chunk(self, chunk: ChunkType):
        """Set current executing chunk."""
        self._current_chunk = chunk

    def get_current_chunk(self) -> ChunkType:
        """Set current executing chunk."""
        return self._current_chunk
