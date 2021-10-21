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

from typing import Tuple, Type, TypeVar, Union

import numpy as np

from .... import oscar as mo
from ....lib.aio import alru_cache
from ..core import MutableTensorInfo
from ..supervisor import MutableObjectManagerActor, MutableTensorActor
from .core import AbstractMutableAPI


APIType = TypeVar("APIType", bound="MutableAPI")


class MutableAPI(AbstractMutableAPI):
    def __init__(
        self,
        address: str,
        mutable_mananger: Union[MutableObjectManagerActor, mo.ActorRef],
    ):
        self._address = address
        self._mutable_manager_ref = mutable_mananger

    @classmethod
    @alru_cache(cache_exceptions=False)
    async def create(cls, session_id: str, address: str) -> "MutableAPI":
        mutable_manager = await mo.actor_ref(
            address, MutableObjectManagerActor.gen_uid(session_id)
        )
        return MutableAPI(address, mutable_manager)

    @alru_cache(cache_exceptions=False)
    async def _get_mutable_tensor_ref(
        self, name: str
    ) -> Union[MutableTensorActor, mo.ActorRef]:
        return await self._mutable_manager_ref.get_mutable_tensor(name)

    async def create_mutable_tensor(
        self,
        shape: tuple,
        dtype: Union[np.dtype, str],
        name: str = None,
        default_value: Union[int, float] = 0,
        chunk_size: Union[int, Tuple] = None,
    ) -> MutableTensorInfo:
        actor_ref = await self._mutable_manager_ref.create_mutable_tensor(
            name=name,
            shape=shape,
            dtype=dtype,
            chunk_size=chunk_size,
            default_value=default_value,
        )
        return await actor_ref.info()

    @alru_cache(cache_exceptions=False)
    async def get_mutable_tensor(self, name: str):
        actor_ref = await self._mutable_manager_ref.get_mutable_tensor(name)
        return await actor_ref.info()

    async def seal_mutable_tensor(self, name: str, timestamp=None):
        # invalidate the `get_mutable_tensor` cache first.
        self.get_mutable_tensor.invalidate()
        return await self._mutable_manager_ref.seal_mutable_tensor(
            name, timestamp=timestamp
        )

    async def read(self, name: str, index, timestamp=None):
        tensor_ref = await self._get_mutable_tensor_ref(name)
        return await tensor_ref.read(index, timestamp)

    async def write(self, name: str, index, value, timestamp=None):
        tensor_ref = await self._get_mutable_tensor_ref(name)
        return await tensor_ref.write(index, value, timestamp)


class MockMutableAPI(MutableAPI):
    @classmethod
    async def create(cls: Type[APIType], session_id: str, address: str) -> "MutableAPI":
        mutable_managger = await mo.create_actor(
            MutableObjectManagerActor,
            session_id,
            address=address,
            uid=MutableObjectManagerActor.gen_uid(session_id),
        )
        return MockMutableAPI(address, mutable_managger)

    @classmethod
    async def cleanup(cls: Type[APIType], session_id: str, address: str):
        await mo.destroy_actor(
            await mo.actor_ref(address, MutableObjectManagerActor.gen_uid(session_id))
        )
