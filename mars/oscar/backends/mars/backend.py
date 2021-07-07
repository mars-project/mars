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

from typing import Dict

from ...backend import BaseActorBackend, register_backend
from ..context import MarsActorContext
from .driver import MarsActorDriver
from .pool import MainActorPool


__all__ = ['MarsActorBackend']


def build_pool_kwargs(n_process: int, kwargs: Dict):
    n_io_process = kwargs.pop('n_io_process', 0)
    if n_io_process:
        n_process += n_io_process

        labels = kwargs['labels']
        envs = kwargs['envs']
        # sub-pools for IO(transfer and spill)
        for _ in range(n_io_process):
            if envs:  # pragma: no cover
                envs.append(dict())
            labels.append('io')
    return n_process, kwargs


@register_backend
class MarsActorBackend(BaseActorBackend):
    @staticmethod
    def name():
        # return None because Mars is default scheme
        return

    @staticmethod
    def get_context_cls():
        return MarsActorContext

    @staticmethod
    def get_driver_cls():
        return MarsActorDriver

    @staticmethod
    async def create_actor_pool(
            address: str,
            n_process: int = None,
            **kwargs):
        from ..pool import create_actor_pool

        n_process, kwargs = build_pool_kwargs(n_process, kwargs)
        return await create_actor_pool(
            address, pool_cls=MainActorPool, n_process=n_process, **kwargs)
