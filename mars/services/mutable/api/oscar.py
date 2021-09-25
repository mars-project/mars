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

import uuid
from typing import TypeVar
from collections import OrderedDict

from ....lib.aio import alru_cache
from .... import oscar as mo
from ....utils import to_binary
from ...core import NodeRole
from ...cluster.api.oscar import ClusterAPI
from ..supervisor.service import MutableTensorActor
from ..supervisor.core import MutableTensor
from .core import AbstractMutableAPI


APIType = TypeVar('APIType', bound='MutableAPI')


class MutableAPI(AbstractMutableAPI):
    def __init__(self,
                 address: str,
                 cluster_api):
        self._address = address
        self._cluster_api = cluster_api
        self._tensor_check = OrderedDict()

    @classmethod
    @alru_cache(cache_exceptions=False)
    async def create(cls, address: str) -> "MutableAPI":
        cluster_api = await ClusterAPI.create(address)
        return MutableAPI(address, cluster_api)

    async def create_mutable_tensor(self, shape: tuple, dtype: str, chunk_size, name: str=None, default_value=0):
        worker_pools: dict = await self._cluster_api.get_all_bands(role=NodeRole.WORKER)
        if name is None:
            name = str(uuid.uuid1())
        ref = await mo.create_actor(
            MutableTensorActor, shape, dtype, chunk_size, worker_pools, name, default_value, address=self._address, uid=to_binary(name))
        wrapper = await MutableTensor.create(ref)
        self._tensor_check[name] = wrapper
        return wrapper

    async def get_mutable_tensor(self, name: str) -> mo.ActorRef:
        if name in self._tensor_check.keys():
            return self._tensor_check[name]
        else:
            raise ValueError('invalid name!')
