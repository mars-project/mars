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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Type, Union

from ....core import ChunkGraph, Chunk, TileContext
from ....resource import Resource
from ....typing import BandType
from ....utils import merge_dict
from ...subtask import SubtaskGraph, SubtaskResult


class ExecutionConfig:
    """
    The config for execution backends.

    This class should ONLY provide the APIs for the parts other than
    just the execution. Each backend may have a different implementation
    of the API.

    If some configuration is for a specific backend. They should be in
    the backend config. e.g. `get_mars_special_config()` should be in
    the `MarsExecutionConfig`.
    """

    name = None

    def __init__(self, execution_config: Dict):
        """
        An example of execution_config:
        {
            "backend": "mars",
            "mars": {
                "n_worker": 1,
                "n_cpu": 2,
                ...
            },
        }
        """
        self._execution_config = execution_config

    def merge_from(self, execution_config: "ExecutionConfig") -> "ExecutionConfig":
        assert isinstance(execution_config, ExecutionConfig)
        assert self.backend == execution_config.backend
        merge_dict(
            self._execution_config,
            execution_config.get_execution_config(),
        )
        return self

    @property
    def backend(self) -> str:
        """The backend from config."""
        return self._execution_config["backend"]

    def get_execution_config(self) -> Dict:
        """Get the execution config dict."""
        return self._execution_config

    @abstractmethod
    def get_deploy_band_resources(self) -> List[Dict[str, Resource]]:
        """Get the band resources for deployment."""

    @classmethod
    def from_config(cls, config: Dict, backend: str = None) -> "ExecutionConfig":
        """Construct an execution config instance from config."""
        execution_config = config["task"]["execution_config"]
        return cls.from_execution_config(execution_config, backend)

    @classmethod
    def from_execution_config(
        cls, execution_config: Union[Dict, "ExecutionConfig"], backend: str = None
    ) -> "ExecutionConfig":
        """Construct an execution config instance from execution config."""
        if isinstance(execution_config, ExecutionConfig):
            assert backend is None
            return execution_config
        if backend is not None:
            name = execution_config["backend"] = backend
        else:
            name = execution_config.setdefault("backend", "mars")
        config_cls = _name_to_config_cls[name]
        return config_cls(execution_config)

    @classmethod
    def from_params(
        cls,
        backend: str,
        n_worker: int,
        n_cpu: int,
        mem_bytes: int = 0,
        cuda_devices: List[List[int]] = None,
        **kwargs,
    ) -> "ExecutionConfig":
        """Construct an execution config instance from params."""
        execution_config = {
            "backend": backend,
            backend: dict(
                {
                    "n_worker": n_worker,
                    "n_cpu": n_cpu,
                    "mem_bytes": mem_bytes,
                    "cuda_devices": cuda_devices,
                },
                **kwargs,
            ),
        }
        return cls.from_execution_config(execution_config)


_name_to_config_cls: Dict[str, Type[ExecutionConfig]] = {}


def register_config_cls(config_cls: Type[ExecutionConfig]):
    _name_to_config_cls[config_cls.name] = config_cls
    return config_cls


@dataclass
class ExecutionChunkResult:
    meta: Dict  # The chunk meta for iterative tiling.
    context: Any  # The context info, e.g. ray.ObjectRef.


class TaskExecutor(ABC):
    name = None

    @classmethod
    @abstractmethod
    async def create(
        cls,
        config: Union[Dict, ExecutionConfig],
        *,
        session_id: str,
        address: str,
        task,
        tile_context: TileContext,
        **kwargs,
    ) -> "TaskExecutor":
        backend_config = ExecutionConfig.from_execution_config(config)
        executor_cls = _name_to_task_executor_cls[backend_config.backend]
        if executor_cls.create.__func__ is TaskExecutor.create.__func__:
            raise NotImplementedError(
                f"The {executor_cls} should implement the abstract classmethod `create`."
            )
        return await executor_cls.create(
            backend_config,
            session_id=session_id,
            address=address,
            task=task,
            tile_context=tile_context,
            **kwargs,
        )

    async def __aenter__(self):
        """Called when begin to execute the task."""

    @abstractmethod
    async def execute_subtask_graph(
        self,
        stage_id: str,
        subtask_graph: SubtaskGraph,
        chunk_graph: ChunkGraph,
        tile_context: TileContext,
        context: Any = None,
    ) -> Dict[Chunk, ExecutionChunkResult]:
        """Execute a subtask graph and returns result."""

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Called when finish the task."""

    @abstractmethod
    async def get_available_band_resources(self) -> Dict[BandType, Resource]:
        """Get available band resources."""

    @abstractmethod
    async def get_progress(self) -> float:
        """Get the execution progress."""

    @abstractmethod
    async def cancel(self):
        """Cancel execution."""

    # The following APIs are for compatible with mars backend, they
    # will be removed as soon as possible.
    async def set_subtask_result(self, subtask_result: SubtaskResult):
        """Set the subtask result."""

    def get_stage_processors(self):
        """Get stage processors."""


_name_to_task_executor_cls: Dict[str, Type[TaskExecutor]] = {}


def register_executor_cls(executor_cls: Type[TaskExecutor]):
    _name_to_task_executor_cls[executor_cls.name] = executor_cls
    return executor_cls


class Fetcher:
    """The data fetcher for execution backends."""

    name = None
    required_meta_keys = ()  # The required meta keys.

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    async def append(self, chunk_key: str, chunk_meta: Dict, conditions: List = None):
        """Append chunk key and related infos."""

    @abstractmethod
    async def get(self):
        """Get all the data of appended chunk keys."""

    @classmethod
    def create(cls, backend: str, **kwargs) -> "Fetcher":
        fetcher_cls = _name_to_fetcher_cls[backend]
        return fetcher_cls(**kwargs)


_name_to_fetcher_cls: Dict[str, Type[Fetcher]] = {}


def register_fetcher_cls(fetcher_cls: Type[Fetcher]):
    _name_to_fetcher_cls[fetcher_cls.name] = fetcher_cls
    return fetcher_cls
