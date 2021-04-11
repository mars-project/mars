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
from abc import ABC, abstractmethod
from typing import Dict, Type

from ..utils import classproperty


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

    @classproperty
    def default(self):
        return getattr(AbstractSession._default_session_local,
                       'default_session', None)


_type_name_to_session_cls: Dict[str, Type[AbstractSession]] = dict()


def register_session_cls(session_cls: Type[AbstractSession]):
    _type_name_to_session_cls[session_cls.name] = session_cls
    return session_cls


async def new_session(address: str,
                      session_id: str = None,
                      backend='oscar',
                      default=False,
                      **kwargs):
    if session_id is None:
        session_id = str(uuid.uuid4())

    session_cls = _type_name_to_session_cls[backend]
    session = await session_cls.init(
        address, session_id=session_id, **kwargs)
    if default:
        session.as_default()
    return session
