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
import concurrent.futures
import logging
import random
import string
import threading
import warnings
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Coroutine, Dict, List, Optional, Tuple, Type, Union
from urllib.parse import urlparse
from weakref import ref

import numpy as np

from .config import options
from .core import TileableType
from .core.entrypoints import init_extension_entrypoints
from .lib.aio import (
    Isolation,
    get_isolation,
    new_isolation,
    stop_isolation,
)
from .services.mutable import MutableTensor
from .typing import ClientType
from .utils import implements, classproperty

logger = logging.getLogger(__name__)


@dataclass
class Progress:
    value: float = 0.0


@dataclass
class Profiling:
    result: dict = None


class ExecutionInfo:
    def __init__(
        self,
        aio_task: asyncio.Task,
        progress: Progress,
        profiling: Profiling,
        loop: asyncio.AbstractEventLoop,
        to_execute_tileables: List[TileableType],
    ):
        self._aio_task = aio_task
        self._progress = progress
        self._profiling = profiling
        self._loop = loop
        self._to_execute_tileables = [ref(t) for t in to_execute_tileables]

        self._future_local = threading.local()

    def _ensure_future(self):
        try:
            self._future_local.future
        except AttributeError:

            async def wait():
                return await self._aio_task

            self._future_local.future = fut = asyncio.run_coroutine_threadsafe(
                wait(), self._loop
            )
            self._future_local.aio_future = asyncio.wrap_future(fut)

    @property
    def loop(self):
        return self._loop

    @property
    def aio_task(self):
        return self._aio_task

    def progress(self) -> float:
        return self._progress.value

    @property
    def to_execute_tileables(self) -> List[TileableType]:
        return [t() for t in self._to_execute_tileables]

    def profiling_result(self) -> dict:
        return self._profiling.result

    def result(self, timeout=None):
        self._ensure_future()
        return self._future_local.future.result(timeout=timeout)

    def cancel(self):
        self._aio_task.cancel()

    def __getattr__(self, attr):
        self._ensure_future()
        return getattr(self._future_local.aio_future, attr)

    def __await__(self):
        self._ensure_future()
        return self._future_local.aio_future.__await__()

    def get_future(self):
        self._ensure_future()
        return self._future_local.aio_future


warning_msg = """
No session found, local session \
will be created in background, \
it may take a while before execution. \
If you want to new a local session by yourself, \
run code below:

```
import mars

mars.new_session()
```
"""


class AbstractSession(ABC):
    name = None
    _default = None
    _lock = threading.Lock()

    def __init__(self, address: str, session_id: str):
        self._address = address
        self._session_id = session_id
        self._closed = False

    @property
    def address(self):
        return self._address

    @property
    def session_id(self):
        return self._session_id

    def __eq__(self, other):
        return (
            isinstance(other, AbstractSession)
            and self._address == other.address
            and self._session_id == other.session_id
        )

    def __hash__(self):
        return hash((AbstractSession, self._address, self._session_id))

    def as_default(self) -> "AbstractSession":
        """
        Mark current session as default session.
        """
        AbstractSession._default = self
        return self

    @classmethod
    def reset_default(cls):
        AbstractSession._default = None

    @classproperty
    def default(self):
        return AbstractSession._default


class AbstractAsyncSession(AbstractSession, metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    async def init(
        cls, address: str, session_id: str, new: bool = True, **kwargs
    ) -> "AbstractSession":
        """
        Init a new session.

        Parameters
        ----------
        address : str
            Address.
        session_id : str
            Session ID.
        new : bool
            New a session.
        kwargs

        Returns
        -------
        session
        """

    async def destroy(self):
        """
        Destroy a session.
        """
        self.reset_default()
        self._closed = True

    @abstractmethod
    async def execute(self, *tileables, **kwargs) -> ExecutionInfo:
        """
        Execute tileables.

        Parameters
        ----------
        tileables
            Tileables.
        kwargs
        """

    @abstractmethod
    async def fetch(self, *tileables, **kwargs) -> list:
        """
        Fetch tileables' data.

        Parameters
        ----------
        tileables
            Tileables.

        Returns
        -------
        data
        """

    @abstractmethod
    async def _get_ref_counts(self) -> Dict[str, int]:
        """
        Get all ref counts

        Returns
        -------
        ref_counts
        """

    @abstractmethod
    async def fetch_tileable_op_logs(
        self,
        tileable_op_key: str,
        offsets: Union[Dict[str, List[int]], str, int],
        sizes: Union[Dict[str, List[int]], str, int],
    ) -> Dict:
        """
        Fetch logs given tileable op key.

        Parameters
        ----------
        tileable_op_key : str
            Tileable op key.
        offsets
            Chunk op key to offsets.
        sizes
            Chunk op key to sizes.

        Returns
        -------
        chunk_key_to_logs
        """

    @abstractmethod
    async def get_total_n_cpu(self):
        """
        Get number of cluster cpus.

        Returns
        -------
        number_of_cpu: int
        """

    @abstractmethod
    async def get_cluster_versions(self) -> List[str]:
        """
        Get versions used in current Mars cluster

        Returns
        -------
        version_list : list
            List of versions
        """

    @abstractmethod
    async def get_web_endpoint(self) -> Optional[str]:
        """
        Get web endpoint of current session

        Returns
        -------
        web_endpoint : str
            web endpoint
        """

    @abstractmethod
    async def create_remote_object(
        self, session_id: str, name: str, object_cls, *args, **kwargs
    ):
        """
        Create remote object

        Parameters
        ----------
        session_id : str
            Session ID.
        name : str
        object_cls
        args
        kwargs

        Returns
        -------
        actor_ref
        """

    @abstractmethod
    async def get_remote_object(self, session_id: str, name: str):
        """
        Get remote object.

        Parameters
        ----------
        session_id : str
            Session ID.
        name : str

        Returns
        -------
        actor_ref
        """

    @abstractmethod
    async def destroy_remote_object(self, session_id: str, name: str):
        """
        Destroy remote object.

        Parameters
        ----------
        session_id : str
            Session ID.
        name : str
        """

    @abstractmethod
    async def create_mutable_tensor(
        self,
        shape: tuple,
        dtype: Union[np.dtype, str],
        name: str = None,
        default_value: Union[int, float] = 0,
        chunk_size: Union[int, Tuple] = None,
    ):
        """
        Create a mutable tensor.

        Parameters
        ----------
        shape: tuple
            Shape of the mutable tensor.

        dtype: np.dtype or str
            Data type of the mutable tensor.

        name: str, optional
            Name of the mutable tensor, a random name will be used if not specified.

        default_value: optional
            Default value of the mutable tensor. Default is 0.

        chunk_size: int or tuple, optional
            Chunk size of the mutable tensor.

        Returns
        -------
            MutableTensor
        """

    @abstractmethod
    async def get_mutable_tensor(self, name: str):
        """
        Get a mutable tensor by name.

        Parameters
        ----------
        name: str
            Name of the mutable tensor to get.

        Returns
        -------
            MutableTensor
        """

    async def stop_server(self):
        """
        Stop server.
        """


class AbstractSyncSession(AbstractSession, metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def init(
        cls,
        address: str,
        session_id: str,
        backend: str = "mars",
        new: bool = True,
        **kwargs,
    ) -> "AbstractSession":
        """
        Init a new session.

        Parameters
        ----------
        address : str
            Address.
        session_id : str
            Session ID.
        backend : str
            Backend.
        new : bool
            New a session.
        kwargs

        Returns
        -------
        session
        """

    @abstractmethod
    def execute(
        self, tileable, *tileables, show_progress: Union[bool, str] = None, **kwargs
    ) -> Union[List[TileableType], TileableType, ExecutionInfo]:
        """
        Execute tileables.

        Parameters
        ----------
        tileable
            Tileable.
        tileables
            Tileables.
        show_progress
            If show progress.
        kwargs

        Returns
        -------
        result
        """

    @abstractmethod
    def fetch(self, *tileables, **kwargs) -> list:
        """
        Fetch tileables.

        Parameters
        ----------
        tileables
            Tileables.
        kwargs

        Returns
        -------
        fetched_data : list
        """

    @abstractmethod
    def fetch_infos(self, *tileables, fields, **kwargs) -> list:
        """
        Fetch infos of tileables.

        Parameters
        ----------
        tileables
            Tileables.
        fields
            List of fields
        kwargs

        Returns
        -------
        fetched_infos : list
        """

    @abstractmethod
    def decref(self, *tileables_keys):
        """
        Decref tileables.

        Parameters
        ----------
        tileables_keys : list
            Tileables' keys
        """

    @abstractmethod
    def _get_ref_counts(self) -> Dict[str, int]:
        """
        Get all ref counts

        Returns
        -------
        ref_counts
        """

    @abstractmethod
    def fetch_tileable_op_logs(
        self,
        tileable_op_key: str,
        offsets: Union[Dict[str, List[int]], str, int],
        sizes: Union[Dict[str, List[int]], str, int],
    ) -> Dict:
        """
        Fetch logs given tileable op key.

        Parameters
        ----------
        tileable_op_key : str
            Tileable op key.
        offsets
            Chunk op key to offsets.
        sizes
            Chunk op key to sizes.

        Returns
        -------
        chunk_key_to_logs
        """

    @abstractmethod
    def get_total_n_cpu(self):
        """
        Get number of cluster cpus.

        Returns
        -------
        number_of_cpu: int
        """

    @abstractmethod
    def get_cluster_versions(self) -> List[str]:
        """
        Get versions used in current Mars cluster

        Returns
        -------
        version_list : list
            List of versions
        """

    @abstractmethod
    def get_web_endpoint(self) -> Optional[str]:
        """
        Get web endpoint of current session

        Returns
        -------
        web_endpoint : str
            web endpoint
        """

    @abstractmethod
    def create_mutable_tensor(
        self,
        shape: tuple,
        dtype: Union[np.dtype, str],
        name: str = None,
        default_value: Union[int, float] = 0,
        chunk_size: Union[int, Tuple] = None,
    ):
        """
        Create a mutable tensor.

        Parameters
        ----------
        shape: tuple
            Shape of the mutable tensor.

        dtype: np.dtype or str
            Data type of the mutable tensor.

        name: str, optional
            Name of the mutable tensor, a random name will be used if not specified.

        default_value: optional
            Default value of the mutable tensor. Default is 0.

        chunk_size: int or tuple, optional
            Chunk size of the mutable tensor.

        Returns
        -------
            MutableTensor
        """

    @abstractmethod
    def get_mutable_tensor(self, name: str):
        """
        Get a mutable tensor by name.

        Parameters
        ----------
        name: str
            Name of the mutable tensor to get.

        Returns
        -------
            MutableTensor
        """

    def fetch_log(
        self,
        tileables: List[TileableType],
        offsets: List[int] = None,
        sizes: List[int] = None,
    ):
        from .core.custom_log import fetch

        return fetch(tileables, self, offsets=offsets, sizes=sizes)


def _delegate_to_isolated_session(func: Union[Callable, Coroutine]):
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def inner(session: "AsyncSession", *args, **kwargs):
            coro = getattr(session._isolated_session, func.__name__)(*args, **kwargs)
            fut = asyncio.run_coroutine_threadsafe(coro, session._loop)
            return await asyncio.wrap_future(fut)

    else:

        @wraps(func)
        def inner(session: "SyncSession", *args, **kwargs):
            coro = getattr(session._isolated_session, func.__name__)(*args, **kwargs)
            fut = asyncio.run_coroutine_threadsafe(coro, session._loop)
            return fut.result()

    return inner


_schemes_to_isolated_session_cls: Dict[
    Optional[str], Type["IsolatedAsyncSession"]
] = dict()


class IsolatedAsyncSession(AbstractAsyncSession):
    """
    Abstract class of isolated sessions which can be registered
    to different schemes
    """

    schemes = None

    @classmethod
    def register_schemes(cls, overwrite: bool = True):
        assert isinstance(cls.schemes, list)
        for scheme in cls.schemes:
            if overwrite or scheme not in _schemes_to_isolated_session_cls:
                _schemes_to_isolated_session_cls[scheme] = cls


def _get_isolated_session_cls(address: str) -> Type[IsolatedAsyncSession]:
    from .deploy.oscar.session import register_session_schemes

    register_session_schemes(overwrite=False)

    if ":/" not in address:
        url_scheme = None
    else:
        url_scheme = urlparse(address).scheme or None
    scheme_cls = _schemes_to_isolated_session_cls.get(url_scheme)
    if scheme_cls is None:
        raise ValueError(f"Address scheme {url_scheme} not supported")
    return scheme_cls


class AsyncSession(AbstractAsyncSession):
    def __init__(
        self,
        address: str,
        session_id: str,
        isolated_session: IsolatedAsyncSession,
        isolation: Isolation,
    ):
        super().__init__(address, session_id)

        self._isolated_session = _get_isolated_session(isolated_session)
        self._isolation = isolation
        self._loop = isolation.loop

    @classmethod
    def from_isolated_session(
        cls, isolated_session: IsolatedAsyncSession
    ) -> "AsyncSession":
        return cls(
            isolated_session.address,
            isolated_session.session_id,
            isolated_session,
            get_isolation(),
        )

    @property
    def client(self):
        return self._isolated_session.client

    @client.setter
    def client(self, client: ClientType):
        self._isolated_session.client = client

    @classmethod
    @implements(AbstractAsyncSession.init)
    async def init(
        cls,
        address: str,
        session_id: str,
        backend: str = "mars",
        new: bool = True,
        **kwargs,
    ) -> "AbstractSession":
        isolation = ensure_isolation_created(kwargs)
        coro = _get_isolated_session_cls(address).init(
            address, session_id, backend, new=new, **kwargs
        )
        fut = asyncio.run_coroutine_threadsafe(coro, isolation.loop)
        isolated_session = await asyncio.wrap_future(fut)
        return AsyncSession(address, session_id, isolated_session, isolation)

    def as_default(self) -> AbstractSession:
        AbstractSession._default = self._isolated_session
        return self

    @implements(AbstractAsyncSession.destroy)
    async def destroy(self):
        coro = self._isolated_session.destroy()
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self._loop))
        self.reset_default()

    @implements(AbstractAsyncSession.execute)
    @_delegate_to_isolated_session
    async def execute(self, *tileables, **kwargs) -> ExecutionInfo:
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.fetch)
    async def fetch(self, *tileables, **kwargs) -> list:
        coro = _fetch(*tileables, session=self._isolated_session, **kwargs)
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        )

    @implements(AbstractAsyncSession._get_ref_counts)
    @_delegate_to_isolated_session
    async def _get_ref_counts(self) -> Dict[str, int]:
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.fetch_tileable_op_logs)
    @_delegate_to_isolated_session
    async def fetch_tileable_op_logs(
        self,
        tileable_op_key: str,
        offsets: Union[Dict[str, List[int]], str, int],
        sizes: Union[Dict[str, List[int]], str, int],
    ) -> Dict:
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.get_total_n_cpu)
    @_delegate_to_isolated_session
    async def get_total_n_cpu(self):
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.get_cluster_versions)
    @_delegate_to_isolated_session
    async def get_cluster_versions(self) -> List[str]:
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.create_remote_object)
    @_delegate_to_isolated_session
    async def create_remote_object(
        self, session_id: str, name: str, object_cls, *args, **kwargs
    ):
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.get_remote_object)
    @_delegate_to_isolated_session
    async def get_remote_object(self, session_id: str, name: str):
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.destroy_remote_object)
    @_delegate_to_isolated_session
    async def destroy_remote_object(self, session_id: str, name: str):
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.create_mutable_tensor)
    async def create_mutable_tensor(
        self,
        shape: tuple,
        dtype: Union[np.dtype, str],
        name: str = None,
        default_value: Union[int, float] = 0,
        chunk_size: Union[int, Tuple] = None,
    ):
        tensor_info, mutable_api = await self._isolated_session.create_mutable_tensor(
            shape, dtype, name, default_value, chunk_size
        )
        return MutableTensor.create(tensor_info, mutable_api, self._loop)

    @implements(AbstractAsyncSession.get_mutable_tensor)
    async def get_mutable_tensor(self, name: str):
        tensor_info, mutable_api = await self._isolated_session.get_mutable_tensor(name)
        return MutableTensor.create(tensor_info, mutable_api, self._loop)

    @implements(AbstractAsyncSession.get_web_endpoint)
    @_delegate_to_isolated_session
    async def get_web_endpoint(self) -> Optional[str]:
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.stop_server)
    async def stop_server(self):
        coro = self._isolated_session.stop_server()
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self._loop))
        stop_isolation()


class ProgressBar:
    def __init__(self, show_progress):
        if not show_progress:
            self.progress_bar = None
        else:
            try:
                from tqdm.auto import tqdm
            except ImportError:
                if show_progress != "auto":  # pragma: no cover
                    raise ImportError("tqdm is required to show progress")
                else:
                    self.progress_bar = None
            else:
                self.progress_bar = tqdm(
                    total=100,
                    bar_format="{l_bar}{bar}| {n:6.2f}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                )

        self.last_progress: float = 0.0

    @property
    def show_progress(self) -> bool:
        return self.progress_bar is not None

    def __enter__(self):
        self.progress_bar.__enter__()

    def __exit__(self, *_):
        self.progress_bar.__exit__(*_)

    def update(self, progress: float):
        progress = min(progress, 100)
        last_progress = self.last_progress
        if self.progress_bar:
            incr = max(progress - last_progress, 0)
            self.progress_bar.update(incr)
        self.last_progress = max(last_progress, progress)


class SyncSession(AbstractSyncSession):
    _execution_pool = concurrent.futures.ThreadPoolExecutor(1)

    def __init__(
        self,
        address: str,
        session_id: str,
        isolated_session: IsolatedAsyncSession,
        isolation: Isolation,
    ):
        super().__init__(address, session_id)

        self._isolated_session = _get_isolated_session(isolated_session)
        self._isolation = isolation
        self._loop = isolation.loop

    @classmethod
    def from_isolated_session(
        cls, isolated_session: IsolatedAsyncSession
    ) -> "SyncSession":
        return cls(
            isolated_session.address,
            isolated_session.session_id,
            isolated_session,
            get_isolation(),
        )

    @classmethod
    def init(
        cls,
        address: str,
        session_id: str,
        backend: str = "mars",
        new: bool = True,
        **kwargs,
    ) -> "AbstractSession":
        isolation = ensure_isolation_created(kwargs)
        coro = _get_isolated_session_cls(address).init(
            address, session_id, backend, new=new, **kwargs
        )
        fut = asyncio.run_coroutine_threadsafe(coro, isolation.loop)
        isolated_session = fut.result()
        return SyncSession(address, session_id, isolated_session, isolation)

    def as_default(self) -> AbstractSession:
        AbstractSession._default = self._isolated_session
        return self

    @property
    def _session(self):
        return self._isolated_session

    def _new_cancel_event(self):
        async def new_event():
            return asyncio.Event()

        return asyncio.run_coroutine_threadsafe(new_event(), self._loop).result()

    @implements(AbstractSyncSession.execute)
    def execute(
        self,
        tileable,
        *tileables,
        show_progress: Union[bool, str] = None,
        warn_duplicated_execution: bool = None,
        **kwargs,
    ) -> Union[List[TileableType], TileableType, ExecutionInfo]:
        wait = kwargs.get("wait", True)
        if show_progress is None:
            show_progress = options.show_progress
        if warn_duplicated_execution is None:
            warn_duplicated_execution = options.warn_duplicated_execution
        to_execute_tileables = []
        for t in (tileable,) + tileables:
            to_execute_tileables.extend(t.op.outputs)

        cancelled = kwargs.get("cancelled")
        if cancelled is None:
            cancelled = kwargs["cancelled"] = self._new_cancel_event()

        coro = _execute(
            *set(to_execute_tileables),
            session=self._isolated_session,
            show_progress=show_progress,
            warn_duplicated_execution=warn_duplicated_execution,
            **kwargs,
        )
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            execution_info: ExecutionInfo = fut.result(
                timeout=self._isolated_session.timeout
            )
        except KeyboardInterrupt:  # pragma: no cover
            logger.warning("Cancelling running task")
            cancelled.set()
            fut.result()
            logger.warning("Cancel finished")

        if wait:
            return tileable if len(tileables) == 0 else [tileable] + list(tileables)
        else:
            aio_task = execution_info.aio_task

            async def run():
                await aio_task
                return tileable if len(tileables) == 0 else [tileable] + list(tileables)

            async def driver():
                return asyncio.create_task(run())

            new_aio_task = asyncio.run_coroutine_threadsafe(
                driver(), execution_info.loop
            ).result()
            new_execution_info = ExecutionInfo(
                new_aio_task,
                execution_info._progress,
                execution_info._profiling,
                execution_info.loop,
                to_execute_tileables,
            )
            return new_execution_info

    @implements(AbstractSyncSession.fetch)
    def fetch(self, *tileables, **kwargs) -> list:
        coro = _fetch(*tileables, session=self._isolated_session, **kwargs)
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    @implements(AbstractSyncSession.fetch_infos)
    def fetch_infos(self, *tileables, fields, **kwargs) -> list:
        coro = _fetch_infos(
            *tileables, fields=fields, session=self._isolated_session, **kwargs
        )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    @implements(AbstractSyncSession.decref)
    @_delegate_to_isolated_session
    def decref(self, *tileables_keys):
        pass  # pragma: no cover

    @implements(AbstractSyncSession._get_ref_counts)
    @_delegate_to_isolated_session
    def _get_ref_counts(self) -> Dict[str, int]:
        pass  # pragma: no cover

    @implements(AbstractSyncSession.fetch_tileable_op_logs)
    @_delegate_to_isolated_session
    def fetch_tileable_op_logs(
        self,
        tileable_op_key: str,
        offsets: Union[Dict[str, List[int]], str, int],
        sizes: Union[Dict[str, List[int]], str, int],
    ) -> Dict:
        pass  # pragma: no cover

    @implements(AbstractSyncSession.get_total_n_cpu)
    @_delegate_to_isolated_session
    def get_total_n_cpu(self):
        pass  # pragma: no cover

    @implements(AbstractSyncSession.get_web_endpoint)
    @_delegate_to_isolated_session
    def get_web_endpoint(self) -> Optional[str]:
        pass  # pragma: no cover

    @implements(AbstractSyncSession.get_cluster_versions)
    @_delegate_to_isolated_session
    def get_cluster_versions(self) -> List[str]:
        pass  # pragma: no cover

    @implements(AbstractSyncSession.create_mutable_tensor)
    def create_mutable_tensor(
        self,
        shape: tuple,
        dtype: Union[np.dtype, str],
        name: str = None,
        default_value: Union[int, float] = 0,
        chunk_size: Union[int, Tuple] = None,
    ):
        coro = self._isolated_session.create_mutable_tensor(
            shape, dtype, name, default_value, chunk_size
        )
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        tensor_info, mutable_api = fut.result()
        return MutableTensor.create(tensor_info, mutable_api, self._loop)

    @implements(AbstractSyncSession.get_mutable_tensor)
    def get_mutable_tensor(self, name: str):
        coro = self._isolated_session.get_mutable_tensor(name)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        tensor_info, mutable_api = fut.result()
        return MutableTensor.create(tensor_info, mutable_api, self._loop)

    def destroy(self):
        coro = self._isolated_session.destroy()
        asyncio.run_coroutine_threadsafe(coro, self._loop).result()
        self.reset_default()

    def stop_server(self, isolation=True):
        try:
            coro = self._isolated_session.stop_server()
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            future.result(timeout=5)
        finally:
            self.reset_default()
            if isolation:
                stop_isolation()

    def close(self):
        self.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


async def _execute_with_progress(
    execution_info: ExecutionInfo,
    progress_bar: ProgressBar,
    progress_update_interval: Union[int, float],
    cancelled: asyncio.Event,
):
    with progress_bar:
        while not cancelled.is_set():
            done, _pending = await asyncio.wait(
                [execution_info.get_future()], timeout=progress_update_interval
            )
            if not done:
                if not cancelled.is_set() and execution_info.progress() is not None:
                    progress_bar.update(execution_info.progress() * 100)
            else:
                # done
                if not cancelled.is_set():
                    progress_bar.update(100)
                break


async def _execute(
    *tileables: Tuple[TileableType],
    session: IsolatedAsyncSession = None,
    wait: bool = True,
    show_progress: Union[bool, str] = "auto",
    progress_update_interval: Union[int, float] = 1,
    cancelled: asyncio.Event = None,
    **kwargs,
):
    execution_info = await session.execute(*tileables, **kwargs)

    def _attach_session(future: asyncio.Future):
        if future.exception() is None:
            for t in execution_info.to_execute_tileables:
                t._attach_session(session)

    execution_info.add_done_callback(_attach_session)
    cancelled = cancelled or asyncio.Event()

    if wait:
        progress_bar = ProgressBar(show_progress)
        if progress_bar.show_progress:
            await _execute_with_progress(
                execution_info, progress_bar, progress_update_interval, cancelled
            )
        else:
            exec_task = asyncio.ensure_future(execution_info)
            cancel_task = asyncio.ensure_future(cancelled.wait())
            await asyncio.wait(
                [exec_task, cancel_task], return_when=asyncio.FIRST_COMPLETED
            )
        if cancelled.is_set():
            execution_info.remove_done_callback(_attach_session)
            execution_info.cancel()
        else:
            # set cancelled to avoid wait task leak
            cancelled.set()
        await execution_info
    else:
        return execution_info


def execute(
    tileable: TileableType,
    *tileables: Tuple[TileableType],
    session: SyncSession = None,
    wait: bool = True,
    new_session_kwargs: dict = None,
    show_progress: Union[bool, str] = None,
    progress_update_interval=1,
    **kwargs,
):
    if isinstance(tileable, (tuple, list)) and len(tileables) == 0:
        tileable, tileables = tileable[0], tileable[1:]
    if session is None:
        session = get_default_or_create(**(new_session_kwargs or dict()))
    session = _ensure_sync(session)
    return session.execute(
        tileable,
        *tileables,
        wait=wait,
        show_progress=show_progress,
        progress_update_interval=progress_update_interval,
        **kwargs,
    )


async def _fetch(
    tileable: TileableType,
    *tileables: Tuple[TileableType],
    session: IsolatedAsyncSession = None,
    **kwargs,
):
    if isinstance(tileable, tuple) and len(tileables) == 0:
        tileable, tileables = tileable[0], tileable[1:]
    session = _get_isolated_session(session)
    data = await session.fetch(tileable, *tileables, **kwargs)
    return data[0] if len(tileables) == 0 else data


async def _fetch_infos(
    tileable: TileableType,
    *tileables: Tuple[TileableType],
    session: IsolatedAsyncSession = None,
    fields: List[str] = None,
    **kwargs,
):
    if isinstance(tileable, tuple) and len(tileables) == 0:
        tileable, tileables = tileable[0], tileable[1:]
    session = _get_isolated_session(session)
    data = await session.fetch_infos(tileable, *tileables, fields=fields, **kwargs)
    return data[0] if len(tileables) == 0 else data


def fetch(
    tileable: TileableType,
    *tileables: Tuple[TileableType],
    session: SyncSession = None,
    **kwargs,
):
    if isinstance(tileable, (tuple, list)) and len(tileables) == 0:
        tileable, tileables = tileable[0], tileable[1:]
    if session is None:
        session = get_default_session()
        if session is None:  # pragma: no cover
            raise ValueError("No session found")

    session = _ensure_sync(session)
    return session.fetch(tileable, *tileables, **kwargs)


def fetch_infos(
    tileable: TileableType,
    *tileables: Tuple[TileableType],
    fields: List[str],
    session: SyncSession = None,
    **kwargs,
):
    if isinstance(tileable, tuple) and len(tileables) == 0:
        tileable, tileables = tileable[0], tileable[1:]
    if session is None:
        session = get_default_session()
        if session is None:  # pragma: no cover
            raise ValueError("No session found")
    session = _ensure_sync(session)
    return session.fetch_infos(tileable, *tileables, fields=fields, **kwargs)


def fetch_log(*tileables: TileableType, session: SyncSession = None, **kwargs):
    if len(tileables) == 1 and isinstance(tileables[0], (list, tuple)):
        tileables = tileables[0]
    if session is None:
        session = get_default_session()
        if session is None:  # pragma: no cover
            raise ValueError("No session found")
    session = _ensure_sync(session)
    return session.fetch_log(list(tileables), **kwargs)


def ensure_isolation_created(kwargs):
    loop = kwargs.pop("loop", None)
    use_uvloop = kwargs.pop("use_uvloop", "auto")

    try:
        return get_isolation()
    except KeyError:
        if loop is None:
            if not use_uvloop:
                loop = asyncio.new_event_loop()
            else:
                try:
                    import uvloop

                    loop = uvloop.new_event_loop()
                except ImportError:
                    if use_uvloop == "auto":
                        loop = asyncio.new_event_loop()
                    else:  # pragma: no cover
                        raise
        return new_isolation(loop=loop)


def _new_session_id():
    return "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(24)
    )


async def _new_session(
    address: str,
    session_id: str = None,
    backend: str = "mars",
    default: bool = False,
    **kwargs,
) -> AbstractSession:
    if session_id is None:
        session_id = _new_session_id()

    session = await AsyncSession.init(
        address, session_id=session_id, backend=backend, new=True, **kwargs
    )
    if default:
        session.as_default()
    return session


def new_session(
    address: str = None,
    session_id: str = None,
    backend: str = "mars",
    default: bool = True,
    new: bool = True,
    **kwargs,
) -> AbstractSession:
    # load third party extensions.
    init_extension_entrypoints()
    ensure_isolation_created(kwargs)

    if address is None:
        address = "127.0.0.1"
        if "init_local" not in kwargs:
            kwargs["init_local"] = True

    if session_id is None:
        session_id = _new_session_id()

    session = SyncSession.init(
        address, session_id=session_id, backend=backend, new=new, **kwargs
    )
    if default:
        session.as_default()
    return session


def get_default_session() -> Optional[SyncSession]:
    if AbstractSession.default is None:
        return
    return SyncSession.from_isolated_session(AbstractSession.default)


def clear_default_session():
    AbstractSession.reset_default()


def get_default_async_session() -> Optional[AsyncSession]:
    if AbstractSession.default is None:
        return
    return AsyncSession.from_isolated_session(AbstractSession.default)


def get_default_or_create(**kwargs):
    with AbstractSession._lock:
        session = AbstractSession.default
        if session is None:
            # no session attached, try to create one
            warnings.warn(warning_msg)
            session = new_session("127.0.0.1", init_local=True, **kwargs)
            session.as_default()
    if isinstance(session, IsolatedAsyncSession):
        session = SyncSession.from_isolated_session(session)
    return _ensure_sync(session)


def stop_server():
    if AbstractSession.default:
        SyncSession.from_isolated_session(AbstractSession.default).stop_server()


def _get_isolated_session(session: AbstractSession) -> IsolatedAsyncSession:
    if hasattr(session, "_isolated_session"):
        return session._isolated_session
    return session


def _ensure_sync(session: AbstractSession) -> SyncSession:
    if isinstance(session, SyncSession):
        return session
    isolated_session = _get_isolated_session(session)
    return SyncSession.from_isolated_session(isolated_session)
