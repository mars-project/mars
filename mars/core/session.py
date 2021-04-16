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
import threading
import uuid
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Type, Tuple, Union

from ..utils import classproperty
from .typing import TileableType


_loop = asyncio.new_event_loop()
_get_session_lock = asyncio.Lock(loop=_loop)


class ExecutionInfo(ABC):
    def __init__(self,
                 future: asyncio.Future):
        self._future = future

    @abstractmethod
    def progress(self) -> float:
        """
        Get execution progress.

        Returns
        -------
        progress : float
        """

    def result(self):
        return self._future.result()

    def exception(self):
        return self._future.exception()

    def done(self):
        return self._future.done()

    def cancel(self):
        return self._future.cancel()

    def add_done_callback(self, cb):
        return self._future.add_done_callback(cb)

    def __await__(self):
        return self._future.__await__()


class AbstractSession(ABC):
    name = None
    _default_session_local = threading.local()

    def __init__(self,
                 address: str,
                 session_id: str):
        self._address = address
        self._session_id = session_id

    @property
    def address(self):
        return self._address

    @staticmethod
    @abstractmethod
    async def init(cls,
                   address: str,
                   session_id: str,
                   **kwargs) -> "AbstractSession":
        """
        Init a new session.

        Parameters
        ----------
        address : str
            Address.
        session_id : str
            Session ID.
        kwargs

        Returns
        -------
        session
        """

    @abstractmethod
    async def destroy(self):
        """
        Destroy a session.
        """

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
    async def fetch(self, *tileables) -> list:
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

    def as_default(self):
        AbstractSession._default_session_local.default_session = self
        return self

    @classmethod
    def reset_default(cls):
        AbstractSession._default_session_local.default_session = None

    @classproperty
    def default(self):
        return getattr(AbstractSession._default_session_local,
                       'default_session', None)


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
        address, session_id=session_id, **kwargs)
    if default:
        session.as_default()
    return session


def get_default_session() -> AbstractSession:
    return AbstractSession.default


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
                   backend: str = 'oscar',
                   new_session_kwargs: dict = None,
                   show_progress: Union[bool, str] = 'auto',
                   progress_update_interval=1, **kwargs):
    if session is None:
        session = await _get_default_or_create(
            backend=backend, **(new_session_kwargs or dict()))

    def _attach_session(fut: asyncio.Future):
        fut.result()
        for t in tileables:
            t._attach_session(session)

    execution_info = await session.execute(*tileables, **kwargs)
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
            show_progress: Union[bool, str] = 'auto',
            progress_update_interval=1, **kwargs):
    _loop.run_until_complete(_execute(
        tileable, *tileables, session=session, wait=wait,
        backend=backend, new_session_kwargs=new_session_kwargs,
        show_progress=show_progress,
        progress_update_interval=progress_update_interval, **kwargs))
    if len(tileables) == 0:
        return tileable
    else:
        return [tileable] + list(tileables)


async def _fetch(tileable: TileableType,
                 *tileables: Tuple[TileableType],
                 session: AbstractSession = None):
    if session is None:
        session = get_default_session()
        if session is None:  # pragma: no cover
            raise ValueError('No session found')

    data = await session.fetch(tileable, *tileables)
    if len(tileables) == 0:
        return data[0]
    else:
        return data


def fetch(*tileables, session: AbstractSession = None):
    return _loop.run_until_complete(
        _fetch(*tileables, session=session))


class SyncSession:
    def __init__(self,
                 session: AbstractSession):
        self._session = session

    def execute(self,
                *tileables,
                **kwargs):
        return execute(*tileables, session=self._session, **kwargs)

    def fetch(self, *tileables):
        return fetch(*tileables, session=self._session)

    def destroy(self):
        return _loop.run_until_complete(
            self._session.destroy())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.destroy()
        if AbstractSession.default is self._session:
            AbstractSession.reset_default()


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
