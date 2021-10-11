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

from typing import List, TypeVar, Union
import uuid

import numpy as np

from .... import oscar as mo
from ....lib.aio import alru_cache
from ....utils import to_binary
from ...cluster.api.oscar import ClusterAPI
from ...core import NodeRole
from ..supervisor.core import MutableTensor
from ..supervisor.service import MutableTensorActor
from .core import AbstractMutableAPI


APIType = TypeVar('APIType', bound='MutableAPI')


class MutableAPI(AbstractMutableAPI):
    def __init__(self,
                 session_id: str,
                 address: str,
                 cluster_api: ClusterAPI):
        self._session_id = session_id
        self._address = address
        self._cluster_api = cluster_api
        self._mutable_objects = dict()

    @classmethod
    @alru_cache(cache_exceptions=False)
    async def create(cls,
                     session_id: str,
                     address: str) -> "MutableAPI":
        cluster_api = await ClusterAPI.create(address)
        return MutableAPI(session_id, address, cluster_api)

    async def create_mutable_tensor(self,
                                    shape: tuple,
                                    dtype: Union[np.dtype, int],
                                    chunk_size: Union[int, tuple],
                                    name: str = None,
                                    default_value: Union[int, float] = 0):
        workers: List[str] = list(await self._cluster_api.get_nodes_info(role=NodeRole.WORKER))
        if name is None:
            name = str(uuid.uuid1())
        if name in self._mutable_objects:
            raise ValueError("Mutable tensor %s already exists!" % name)
        ref = await mo.create_actor(
            MutableTensorActor, self._session_id, workers,
            shape, dtype, chunk_size, name, default_value,
            address=self._address, uid=to_binary(name))
        tensor = await MutableTensor.create(ref)
        self._mutable_objects[name] = tensor
        return tensor

    async def get_mutable_tensor(self, name: str):
        if name in self._mutable_objects:
            return self._mutable_objects[name]
        else:
            raise ValueError("Mutable tensor %s doesn't exist!" % name)

    async def seal_mutable_tensor(self, name: str, timestamp=None):
        if name in self._mutable_objects:
            await self._mutable_objects[name].seal(timestamp=timestamp)
            await mo.destroy_actor(self._mutable_objects[name])
            self._mutable_objects.pop(name)
        else:  # pragma: no cover
            raise ValueError("Mutable tensor %s doesn't exist!" % name)
