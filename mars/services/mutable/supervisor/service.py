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

from typing import List, Union, Dict, Tuple

import numpy as np

from .... import oscar as mo
from ....core import tile
from ....utils import build_fetch
from ...meta import MetaAPI
from ..worker.service import MutableTensorChunkActor


class MutableTensorActor(mo.Actor):
    def __init__(self,
                 session_id: str,
                 workers: List[str],
                 shape: Tuple,
                 dtype: Union[np.dtype, str],
                 chunk_size: Union[int, Tuple],
                 name: str = None,
                 default_value : Union[int, float] = 0):
        self._session_id = session_id
        self._workers = workers
        self._shape = shape
        self._dtype = dtype
        self._chunk_size = chunk_size
        self._name = name
        self._default_value = default_value

        self._sealed = False

        self._fetch = None
        self._chunk_actors = []
        # chunk to actor: {chunk index -> actor uid}
        self._chunk_to_actor = dict()

    async def __post_create__(self):
        self._meta_api = await MetaAPI.create(self._session_id, self.address)

        # tiling a random tensor to generate keys, but we doesn't actually execute
        # the random generator
        from ....tensor.random import rand
        self._fetch = build_fetch(tile(rand(*self._shape, dtype=self._dtype,
                                            chunk_size=self._chunk_size)))

        chunk_groups = np.array_split(self._fetch.chunks, len(self._workers))
        for idx, (worker, chunks) in enumerate(zip(self._workers, chunk_groups)):
            if len(chunks) == 0:
                break
            uid = MutableTensorChunkActor.gen_uid(self._name, idx)
            chunk_actor = await mo.create_actor(
                MutableTensorChunkActor, self._session_id, self.address,
                list(chunks), dtype=self._dtype, default_value=self._default_value,
                address=worker, uid=uid)
            self._chunk_actors.append(chunk_actor)
            for chunk in chunks:
                self._chunk_to_actor[chunk.index] = (worker, uid)

    async def dtype(self) -> Union[np.dtype, str]:
        return self._dtype

    async def default_value(self) -> str:
        return self._default_value

    async def fetch(self):
        return self._fetch

    async def chunk_to_actor(self) -> Dict[Tuple, str]:
        return self._chunk_to_actor

    @mo.extensible
    async def _seal_chunk(self, chunk_actor, timestamp):
        await chunk_actor.seal(timestamp)
        await mo.destroy_actor(chunk_actor)

    async def seal(self, timestamp):
        if self._sealed:
            return self._fetch
        self._sealed = True
        seal_tasks = []
        for chunk_actor in self._chunk_actors:
            seal_tasks.append(self._seal_chunk.delay(chunk_actor, timestamp))
        await self._seal_chunk.batch(*seal_tasks)
        return self._fetch
