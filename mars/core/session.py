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
import concurrent.futures
import functools
import os
import threading
import uuid
import warnings
from abc import ABC, ABCMeta, abstractmethod
from typing import Callable, Dict, List, Optional, Type, Tuple, Union

from ..config import options, option_context, get_global_option
from ..core import TileableGraph, enter_mode
from ..core.operand import Fetch
from ..lib.aio import create_lock
from ..typing import TileableType
from ..utils import classproperty, copy_tileables, build_fetch, implements


class ExecutionInfo(ABC):
    def __init__(self,
                 future: asyncio.Future):
        self.future = self.aio_future = future

    @abstractmethod
    def progress(self) -> float:
        """
        Get execution progress.

        Returns
        -------
        progress : float
        """

    def __getattr__(self, attr):
        return getattr(self.future, attr)

    def __await__(self):
        return self.future.__await__()


class AbstractSession(ABC):
    _default_session_local = threading.local()

    def __init__(self,
                 address: str,
                 session_id: str):
        self._address = address
        self._session_id = session_id

    @property
    def address(self):
        return self._address

    @property
    def session_id(self):
        return self._session_id

    def __eq__(self, other):
        return isinstance(other, AbstractSession) and \
               self._address == other.address and \
               self._session_id == other.session_id

    def __hash__(self):
        return hash((AbstractSession, self._address, self._session_id))

    @property
    @abstractmethod
    def is_sync(self) -> bool:
        """
        Is synchronous session or not

        Returns
        -------
        is_sync
        """

    def as_default(self):
        """
        Mark current session as default session.
        """
        AbstractSession._default_session_local.default_session = self
        return self

    @classmethod
    def reset_default(cls):
        AbstractSession._default_session_local.default_session = None

    @classproperty
    def default(self):
        return getattr(AbstractSession._default_session_local,
                       'default_session', None)

    @abstractmethod
    def to_async(self):
        """
        Get async session.

        Returns
        -------
        async_session
        """

    @abstractmethod
    def to_sync(self):
        """
        Get sync session.

        Returns
        -------
        sync_session
        """


class AbstractAsyncSession(AbstractSession, metaclass=ABCMeta):
    name = None

    @classmethod
    @abstractmethod
    async def init(cls,
                   address: str,
                   session_id: str,
                   new: bool = True,
                   **kwargs) -> "AbstractSession":
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

    @property
    @implements(AbstractSession.is_sync)
    def is_sync(self) -> bool:
        return False

    @abstractmethod
    async def destroy(self):
        """
        Destroy a session.
        """
        self.reset_default()

    @abstractmethod
    async def execute(self,
                      *tileables,
                      **kwargs) -> ExecutionInfo:
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
    async def decref(self, *tileable_keys):
        """
        Decref tileables.

        Parameters
        ----------
        tileable_keys
            Tileable keys.
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
    async def fetch_tileable_op_logs(self,
                                     tileable_op_key: str,
                                     offsets: Union[Dict[str, List[int]], str, int],
                                     sizes: Union[Dict[str, List[int]], str, int]) -> Dict:
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

    async def stop_server(self):
        """
        Stop server.
        """

    @implements(AbstractSession.to_async)
    def to_async(self):
        return self

    @implements(AbstractSession.to_sync)
    def to_sync(self):
        return SyncSession(self)


class AbstractSyncSession(AbstractSession, metaclass=ABCMeta):
    @property
    @implements(AbstractSession.is_sync)
    def is_sync(self) -> bool:
        return True

    @abstractmethod
    def execute(self,
                tileable,
                *tileables,
                **kwargs) -> Union[List[TileableType], TileableType, ExecutionInfo]:
        """
        Execute tileables.

        Parameters
        ----------
        tileable
            Tileable.
        tileables
            Tileables.
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
    def decref(self, *tileables_keys):
        """
        Decref tileables.

        Parameters
        ----------
        tileables_keys : list
            Tileables' keys
        """

    @abstractmethod
    def fetch_tileable_op_logs(self,
                               tileable_op_key: str,
                               offsets: Union[Dict[str, List[int]], str, int],
                               sizes: Union[Dict[str, List[int]], str, int]) -> Dict:
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

    def fetch_log(self,
                  tileables: List[TileableType],
                  offsets: List[int] = None,
                  sizes: List[int] = None):
        from .custom_log import fetch

        return fetch(tileables, self, offsets=offsets, sizes=sizes)

    @implements(AbstractSession.to_sync)
    def to_sync(self):
        return self


_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_pid: Optional[int] = None
_get_session_lock: Optional[asyncio.Lock] = None
_pool = concurrent.futures.ThreadPoolExecutor(1)
_gc_pool = concurrent.futures.ThreadPoolExecutor()


def _ensure_loop():
    global _loop_pid, _loop, _get_session_lock, _pool
    if _loop_pid != os.getpid():
        _loop_pid = os.getpid()
        _loop = asyncio.new_event_loop()
        _get_session_lock = create_lock(_loop)
    return _loop


def _wrap_in_thread(pool_or_func):
    """
    Function is those wrapping async function,
    when there is a running event loop,
    error will raise (see GH#2108),
    so we wrap them to run in a thread.
    """

    def _wrap(func: Callable,
              executor: concurrent.futures.ThreadPoolExecutor):
        def sync_default_session(sess: "AbstractSession"):
            if sess:
                sess.as_default()
            else:
                AbstractSession.reset_default()

        def inner(*args, **kwargs):
            default_session = get_default_session()
            config = get_global_option().to_dict()

            def run_in_thread():
                with option_context(config):
                    # set default session in this thread
                    sync_default_session(default_session)
                    return func(*args, **kwargs), get_default_session()

            _ensure_loop()
            fut = executor.submit(run_in_thread)
            result, default_session_in_thread = fut.result()
            sync_default_session(default_session_in_thread)

            return result
        return inner

    if callable(pool_or_func):
        return _wrap(pool_or_func, _pool)
    else:
        return functools.partial(_wrap, executor=pool_or_func)


@enter_mode(build=True, kernel=True)
def gen_submit_tileable_graph(
        session: "AbstractSession",
        result_tileables: List[TileableType]):
    tileable_to_copied = dict()
    result = [None] * len(result_tileables)
    graph = TileableGraph(result)

    q = list(result_tileables)
    while q:
        tileable = q.pop()
        if tileable in tileable_to_copied:
            if tileable in result_tileables:
                result[result_tileables.index(tileable)] = \
                    tileable_to_copied[tileable]
            continue
        outputs = tileable.op.outputs
        inputs = tileable.inputs \
            if session not in tileable._executed_sessions else []
        new_inputs = []
        all_inputs_processed = True
        for inp in inputs:
            if inp in tileable_to_copied:
                new_inputs.append(tileable_to_copied[inp])
            elif session in inp._executed_sessions:
                # executed, gen fetch
                fetch_input = build_fetch(inp).data
                tileable_to_copied[inp] = fetch_input
                graph.add_node(fetch_input)
                new_inputs.append(fetch_input)
            else:
                # some input not processed before
                all_inputs_processed = False
                # put back tileable
                q.append(tileable)
                q.append(inp)
                break
        if all_inputs_processed:
            if isinstance(tileable.op, Fetch):
                new_outputs = [tileable]
            elif session in tileable._executed_sessions:
                new_outputs = []
                for out in outputs:
                    fetch_out = tileable_to_copied.get(out, build_fetch(out).data)
                    new_outputs.append(fetch_out)
            else:
                new_outputs = [t.data for t
                               in copy_tileables(outputs, inputs=new_inputs)]
            for out, new_out in zip(outputs, new_outputs):
                tileable_to_copied[out] = new_out
                if out in result_tileables:
                    result[result_tileables.index(out)] = new_out
                graph.add_node(new_out)
                for new_inp in new_inputs:
                    graph.add_edge(new_inp, new_out)

    return graph


_type_name_to_session_cls: Dict[str, Type[AbstractSession]] = dict()


def register_session_cls(session_cls: Type[AbstractSession]):
    _type_name_to_session_cls[session_cls.name] = session_cls
    return session_cls


async def _new_session(address: str,
                       session_id: str = None,
                       backend: str = 'oscar',
                       default: bool = False,
                       **kwargs) -> AbstractSession:
    if session_id is None:
        session_id = str(uuid.uuid4())

    session_cls = _type_name_to_session_cls[backend]
    session = await session_cls.init(
        address, session_id=session_id,
        new=True, **kwargs)
    if default:
        session.as_default()
    return session


async def _get_session(address: str,
                       session_id: str,
                       backend: str = 'oscar',
                       default: bool = False):
    session_cls = _type_name_to_session_cls[backend]
    session = await session_cls.init(
        address, session_id=session_id, new=False)
    if default:
        session.as_default()
    return session


def get_default_session() -> AbstractSession:
    return AbstractSession.default


def stop_server():
    if AbstractSession.default:
        SyncSession(AbstractSession.default).stop_server()


warning_msg = """No session found, local session \
will be created in the background, \
it may take a while before execution. \
If you want to new a local session by yourself, \
run code below:

```
import mars

mars.new_session(default=True)
```
"""


async def _get_default_or_create(**kwargs):
    async with _get_session_lock:
        session = get_default_session()
        if session is None:
            # no session attached, try to create one
            warnings.warn(warning_msg)
            session = await _new_session(
                '127.0.0.1', default=True, init_local=True, **kwargs)
    return session


@_wrap_in_thread
def get_default_or_create(**kwargs):
    return _loop.run_until_complete(
        _get_default_or_create(**kwargs))


def _new_progress():  # pragma: no cover
    try:
        from tqdm.auto import tqdm
    except ImportError:
        raise ImportError('tqdm is required to show progress')

    with tqdm(total=100) as progress_bar:
        last_progress = progress = 0
        while True:
            last_progress = max(last_progress, progress)
            progress = yield
            progress_bar.update(progress - last_progress)


async def _execute(*tileables: Tuple[TileableType],
                   session: AbstractSession = None,
                   wait: bool = True,
                   show_progress: Union[bool, str] = 'auto',
                   progress_update_interval: Union[int, float] = 1,
                   **kwargs):

    def _attach_session(fut: asyncio.Future):
        fut.result()
        for t in tileables:
            t._attach_session(session)

    async_session = session.to_async()
    execution_info = await async_session.execute(*tileables, **kwargs)
    execution_info.add_done_callback(_attach_session)

    if wait:
        if show_progress:  # pragma: no cover
            try:
                progress = _new_progress()
                progress.send(None)
            except ImportError:
                if show_progress != 'auto':
                    raise
                else:
                    await execution_info
            else:
                while True:
                    try:
                        await asyncio.wait_for(asyncio.shield(execution_info),
                                               progress_update_interval)
                        # done
                        progress.send(100)
                        break
                    except asyncio.TimeoutError:
                        # timeout
                        progress.send(execution_info.progress() * 100)
        else:
            await execution_info
    else:
        return execution_info


def execute(tileable: TileableType,
            *tileables: Tuple[TileableType],
            session: AbstractSession = None,
            wait: bool = True,
            backend: str = 'oscar',
            new_session_kwargs: dict = None,
            show_progress: Union[bool, str] = None,
            progress_update_interval=1, **kwargs):
    if session is None:
        session = get_default_or_create(
            backend=backend, **(new_session_kwargs or dict()))
    if not session.is_sync:
        sync_session = SyncSession(session)
    else:
        sync_session = session
    return sync_session.execute(tileable, *tileables, wait=wait,
                                show_progress=show_progress,
                                progress_update_interval=progress_update_interval,
                                **kwargs)


async def _fetch(tileable: TileableType,
                 *tileables: Tuple[TileableType],
                 session: AbstractSession = None,
                 **kwargs):
    if isinstance(tileable, tuple) and len(tileables) == 0:
        tileable, tileables = tileable[0], tileable[1:]
    data = await session.fetch(tileable, *tileables, **kwargs)
    return data[0] if len(tileables) == 0 else data


def fetch(*tileables: Tuple[TileableType],
          session: AbstractSession = None,
          **kwargs):
    if session is None:
        session = get_default_session()
        if session is None:  # pragma: no cover
            raise ValueError('No session found')
    if not session.is_sync:
        sync_session = SyncSession(session)
    else:
        sync_session = session
    return sync_session.fetch(*tileables, **kwargs)


def fetch_log(*tileables: Tuple[TileableType],
              session: AbstractSession = None,
              **kwargs):
    if session is None:
        session = get_default_session()
        if session is None:  # pragma: no cover
            raise ValueError('No session found')
    if not session.is_sync:
        sync_session = SyncSession(session)
    else:
        sync_session = session
    return sync_session.fetch_log(tileables, **kwargs)


class SyncSession(AbstractSyncSession):
    def __init__(self,
                 session: AbstractAsyncSession):
        super().__init__(session.address, session.session_id)
        self._session = session

    @implements(AbstractSyncSession.execute)
    @_wrap_in_thread
    def execute(self,
                tileable: TileableType,
                *tileables: TileableType,
                show_progress: Union[bool, str] = None,
                **kwargs) -> Union[List[TileableType], TileableType, ExecutionInfo]:
        wait = kwargs.get('wait', True)
        if show_progress is None:
            show_progress = options.show_progress
        to_execute_tileables = []
        for t in (tileable,) + tileables:
            to_execute_tileables.extend(t.op.outputs)
        execution_info = _loop.run_until_complete(_execute(
            *set(to_execute_tileables), session=self,
            show_progress=show_progress, **kwargs))
        if wait:
            return tileable if len(tileables) == 0 else \
                [tileable] + list(tileables)
        else:
            fut = execution_info.aio_future

            def run():
                _loop.run_until_complete(fut)
                return tileable if len(tileables) == 0 else \
                    [tileable] + list(tileables)

            execution_info.future = _pool.submit(run)
            return execution_info

    @implements(AbstractSyncSession.fetch)
    @_wrap_in_thread
    def fetch(self, *tileables, **kwargs) -> list:
        return _loop.run_until_complete(
            _fetch(*tileables, session=self._session, **kwargs))

    @implements(AbstractSyncSession.decref)
    @_wrap_in_thread(_gc_pool)
    def decref(self, *tileables_keys):
        return _loop.run_until_complete(
            self._session.decref(*tileables_keys))

    @_wrap_in_thread
    def _get_ref_counts(self) -> Dict[str, int]:
        return _loop.run_until_complete(
            self._session._get_ref_counts())

    @implements(AbstractSyncSession.fetch_tileable_op_logs)
    @_wrap_in_thread
    def fetch_tileable_op_logs(self,
                               tileable_op_key: str,
                               offsets: Union[Dict[str, List[int]], str, int],
                               sizes: Union[Dict[str, List[int]], str, int]) -> Dict:
        return _loop.run_until_complete(
            self._session.fetch_tileable_op_logs(tileable_op_key,
                                                 offsets, sizes))

    @implements(AbstractSyncSession.get_total_n_cpu)
    @_wrap_in_thread
    def get_total_n_cpu(self):
        return _loop.run_until_complete(
            self._session.get_total_n_cpu())

    @_wrap_in_thread
    def destroy(self):
        return _loop.run_until_complete(
            self._session.destroy())

    @_wrap_in_thread
    def stop_server(self):
        return _loop.run_until_complete(
            self._session.stop_server())

    @implements(AbstractSession.to_async)
    def to_async(self):
        return self._session

    def close(self):
        self.destroy()
        if AbstractSession.default is self._session:
            AbstractSession.reset_default()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@_wrap_in_thread
def new_session(address: str = None,
                session_id: str = None,
                backend: str = 'oscar',
                default: bool = False,
                **kwargs):
    if address is None:
        address = '127.0.0.1'
        if 'init_local' not in kwargs:
            kwargs['init_local'] = True
    session = _loop.run_until_complete(
        _new_session(address, session_id=session_id,
                     backend=backend, default=default, **kwargs))
    return SyncSession(session)


@_wrap_in_thread
def get_session(address: str,
                session_id: str,
                backend: str = 'oscar',
                default: bool = False):
    session = _loop.run_until_complete(
        _get_session(address, session_id=session_id,
                     backend=backend, default=default))
    return SyncSession(session)
