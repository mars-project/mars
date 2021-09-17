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

from typing import Union, TypeVar

from ....lib.aio import alru_cache
from .... import oscar as mo
from ...session.supervisor.core import SessionActor, SessionManagerActor
from .core import AbstractMutableAPI


APIType = TypeVar('APIType', bound='MutableAPI')


class MutableAPI(AbstractMutableAPI):
    def __init__(self,
                 address: str,
                 session_manager: Union[SessionManagerActor, mo.ActorRef]):
        self._address = address
        self._session_manager_ref = session_manager

    @classmethod
    @alru_cache(cache_exceptions=False)
    async def create(cls, address: str, **kwargs) -> "MutableAPI":
        if kwargs:  # pragma: no cover
            raise TypeError(f'SessionAPI.create '
                            f'got unknown arguments: {list(kwargs)}')
        session_manager = await mo.actor_ref(
            address, SessionManagerActor.default_uid())
        return MutableAPI(address, session_manager)

    @alru_cache(cache_exceptions=False)
    async def _get_session_ref(self, session_id: str) -> Union[SessionActor, mo.ActorRef]:
        return await self._session_manager_ref.get_session_ref(session_id)

    async def create_mutable_tensor(self, session_id: str, shape: tuple, dtype: str, chunk_size, name: str=None, default_value=0):
        session = await self._get_session_ref(session_id)
        return await session.create_mutable_tensor(shape, dtype, chunk_size, name, default_value)

    async def get_mutable_tensor(self, session_id: str, name: str):
        session = await self._get_session_ref(session_id)
        return await session.get_mutable_tensor(name)
