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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .... import oscar as mo
from ....core import tile
from ....utils import build_fetch
from ...core import NodeRole
from ...cluster import ClusterAPI
from ...meta import MetaAPI
from ..core import MutableTensorInfo
from ..utils import (
    getitem_to_records,
    setitem_to_records,
    normalize_name,
    normalize_timestamp,
)
from ..worker import MutableTensorChunkActor


class MutableObjectManagerActor(mo.Actor):
    def __init__(self, session_id: str):
        self._session_id = session_id
        self._cluster_api: Optional[ClusterAPI] = None

        self._mutable_objects = dict()

    async def __post_create__(self):
        self._cluster_api = await ClusterAPI.create(self.address)

    async def __pre_destroy__(self):
        await asyncio.gather(
            *[mo.destroy_actor(ref) for ref in self._mutable_objects.values()]
        )

    @classmethod
    def gen_uid(cls, session_id: str):
        return f"mutable-object-manager-{session_id}"

    async def create_mutable_tensor(self, *args, name: Optional[str] = None, **kwargs):
        name = normalize_name(name)
        if name in self._mutable_objects:
            raise ValueError(f"Mutable tensor {name} already exists!")

        workers: List[str] = list(
            await self._cluster_api.get_nodes_info(role=NodeRole.WORKER)
        )

        tensor_ref = await mo.create_actor(
            MutableTensorActor,
            self._session_id,
            name,
            workers,
            *args,
            **kwargs,
            address=self.address,
            uid=MutableTensorActor.gen_uid(self._session_id, name),
        )
        self._mutable_objects[name] = tensor_ref
        return tensor_ref

    async def get_mutable_tensor(self, name: str):
        tensor_ref = self._mutable_objects.get(name, None)
        if tensor_ref is None:
            raise ValueError(f"Mutable tensor {name} doesn't exist!")
        return tensor_ref

    async def seal_mutable_tensor(self, name: str, timestamp=None):
        tensor_ref = self._mutable_objects.get(name, None)
        if tensor_ref is None:
            raise ValueError(f"Mutable tensor {name} doesn't exist!")
        tensor = await tensor_ref.seal(timestamp)
        await mo.destroy_actor(tensor_ref)
        self._mutable_objects.pop(name)
        return tensor


class MutableTensorActor(mo.Actor):
    def __init__(
        self,
        session_id: str,
        name: str,
        workers: List[str],
        shape: Tuple,
        dtype: Union[np.dtype, str],
        default_value: Union[int, float] = 0,
        chunk_size: Union[int, Tuple] = None,
    ):
        self._session_id = session_id
        self._name = name
        self._workers = workers
        self._shape = shape
        self._dtype = dtype
        self._default_value = default_value
        self._chunk_size = chunk_size

        self._sealed = False

        self._fetch = None
        self._chunk_actors = []
        # chunk to actor: {chunk index -> actor uid}
        self._chunk_to_actor: Dict[
            Tuple, Union[MutableTensorChunkActor, mo.ActorRef]
        ] = dict()

    async def __post_create__(self):
        self._meta_api = await MetaAPI.create(self._session_id, self.address)

        # tiling a random tensor to generate keys, but we doesn't actually execute
        # the random generator
        from ....tensor.random import rand

        self._fetch = build_fetch(
            tile(rand(*self._shape, dtype=self._dtype, chunk_size=self._chunk_size))
        )

        chunk_groups = np.array_split(self._fetch.chunks, len(self._workers))
        for idx, (worker, chunks) in enumerate(zip(self._workers, chunk_groups)):
            if len(chunks) == 0:
                break
            chunk_actor_ref = await mo.create_actor(
                MutableTensorChunkActor,
                self._session_id,
                self.address,
                list(chunks),
                dtype=self._dtype,
                default_value=self._default_value,
                address=worker,
                uid=MutableTensorChunkActor.gen_uid(self._session_id, self._name, idx),
            )
            self._chunk_actors.append(chunk_actor_ref)
            for chunk in chunks:
                self._chunk_to_actor[chunk.index] = chunk_actor_ref

    async def __pre_destroy__(self):
        await asyncio.gather(*[mo.destroy_actor(ref) for ref in self._chunk_actors])

    @classmethod
    def gen_uid(cls, session_id, name):
        return f"mutable-tensor-{session_id}-{name}"

    async def info(self) -> "MutableTensorInfo":
        return MutableTensorInfo(
            self._shape, self._dtype, self._name, self._default_value
        )

    @mo.extensible
    async def _read_chunk(
        self, chunk_actor_ref, chunk_index, records, chunk_value_shape, timestamp
    ):
        return await chunk_actor_ref.read(
            chunk_index, records, chunk_value_shape, timestamp
        )

    async def read(self, index, timestamp=None):
        """
        Read value from mutable tensor.

        Parameters
        ----------
        index:
            Index to read from the tensor.

        timestamp: optional
            Timestamp to read value that happened before then.
        """
        timestamp = normalize_timestamp(timestamp)
        records, output_shape = getitem_to_records(self._fetch, index)

        read_tasks, chunk_indices = [], []
        for chunk_index, (records, chunk_value_shape, indices) in records.items():
            chunk_actor_ref = self._chunk_to_actor[chunk_index]
            read_tasks.append(
                self._read_chunk.delay(
                    chunk_actor_ref, chunk_index, records, chunk_value_shape, timestamp
                )
            )
            chunk_indices.append(indices)
        chunks = await self._read_chunk.batch(*read_tasks)
        result = np.full(output_shape, fill_value=self._default_value)
        for chunk, indices in zip(chunks, chunk_indices):
            result[indices] = chunk
        return result

    @mo.extensible
    async def _write_chunk(self, chunk_actor_ref, chunk_index, records):
        await chunk_actor_ref.write(chunk_index, records)

    async def write(self, index, value, timestamp=None):
        """
        Write value to mutable tensor.

        Parameters
        ----------
        index:
            Index to write to the tensor.

        value:
            The value that will be filled into the mutable tensor according to `index`.

        timestamp: optional
            Timestamp to associated with the newly touched value.
        """
        timestamp = normalize_timestamp(timestamp)
        records = setitem_to_records(self._fetch, index, value, timestamp)

        write_tasks = []
        for chunk_index, records in records.items():
            chunk_actor_ref = self._chunk_to_actor[chunk_index]
            write_tasks.append(
                self._write_chunk.delay(chunk_actor_ref, chunk_index, records)
            )
        await self._write_chunk.batch(*write_tasks)

    @mo.extensible
    async def _seal_chunk(self, chunk_actor_ref, timestamp):
        await chunk_actor_ref.seal(timestamp)

    async def seal(self, timestamp=None):
        if self._sealed:
            return self._fetch

        timestamp = normalize_timestamp(timestamp)
        self._sealed = True
        seal_tasks = []
        for chunk_actor_ref in self._chunk_actors:
            seal_tasks.append(self._seal_chunk.delay(chunk_actor_ref, timestamp))
        await self._seal_chunk.batch(*seal_tasks)
        self._chunk_actors = []
        return self._fetch
