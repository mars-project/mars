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

from typing import Dict, List, Tuple, Union

import numpy as np

from .... import oscar as mo
from ....tensor.core import Tensor
from ....serialization.core import Serializer, buffered, serialize, deserialize
from ..utils import getitem_to_records, setitem_to_records, normalize_timestamp
from .service import MutableTensorActor


class MutableTensor:
    def __init__(self,
                 ref: MutableTensorActor,
                 fetch: Tensor,
                 chunk_to_actor_key: Dict[Tuple, Tuple[str, str]],
                 chunk_to_actor: Dict[Tuple, mo.ActorRef],
                 dtype: Union[np.dtype, str],
                 default_value: Union[int, float] = 0):
        self._ref = ref
        self._fetch = fetch
        self._chunk_to_actor_key = chunk_to_actor_key
        self._chunk_to_actor = chunk_to_actor
        self._dtype = dtype
        self._default_value = default_value

    @classmethod
    async def create(self, ref: mo.ActorRef) -> "MutableTensor":
        fetch = await ref.fetch()
        dtype = await ref.dtype()
        default_value = await ref.default_value()
        chunk_to_actor_key = await ref.chunk_to_actor()
        chunk_to_actor = dict()
        for chunk_index, (worker, uid) in chunk_to_actor_key.items():
            chunk_to_actor[chunk_index] = await mo.actor_ref(uid=uid, address=worker)
        return MutableTensor(ref, fetch, chunk_to_actor_key, chunk_to_actor,
                             dtype, default_value)

    async def ensure_chunk_actors(self):
        """
        Initialize the chunk actors in WebSession.
        """
        if self._chunk_to_actor is None:
            for chunk_index, (worker, uid) in self._chunk_to_actor_key.items():
                self._chunk_to_actor[chunk_index] = await mo.actor_ref(uid=uid, address=worker)

    async def __getitem__(self, index: Union[int, List[int]]):
        '''
        Read value from the mutable tensor.
        '''
        return await self.read(index)

    @mo.extensible
    async def _read_chunk(self, chunk_actor, chunk_index, records, chunk_value_shape, timestamp):
        return await chunk_actor.read(chunk_index, records, chunk_value_shape, timestamp)

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
            chunk_actor = self._chunk_to_actor[chunk_index]
            read_tasks.append(self._read_chunk.delay(chunk_actor, chunk_index, records,
                                                     chunk_value_shape, timestamp))
            chunk_indices.append(indices)
        chunks = await self._read_chunk.batch(*read_tasks)
        result = np.full(output_shape, fill_value=self._default_value)
        for chunk, indices in zip(chunks, chunk_indices):
            result[indices] = chunk
        return result

    @mo.extensible
    async def _write_chunk(self, chunk_actor, chunk_index, records):
        await chunk_actor.write(chunk_index, records)

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
            chunk_actor = self._chunk_to_actor[chunk_index]
            write_tasks.append(self._write_chunk.delay(chunk_actor, chunk_index, records))
        await self._write_chunk.batch(*write_tasks)

    async def seal(self, timestamp=None):
        timestamp = normalize_timestamp(timestamp)
        return await self._ref.seal(timestamp)


class MutableTensorSerializer(Serializer):
    serializer_name = 'mutable_tensor'

    @buffered
    def serialize(self, tensor: MutableTensor, context: Dict):
        values = {
            'fetch': tensor._fetch,
            'dtype': tensor._dtype,
            'default_value': tensor._default_value,
            'chunk_to_actor_key': tensor._chunk_to_actor_key,
        }
        return serialize(values, context=context)

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        values = deserialize(header, buffers, context)

        fetch = values['fetch']
        dtype = values['dtype']
        default_value = values['default_value']
        chunk_to_actor_key = values['chunk_to_actor_key']
        chunk_to_actor = None
        # FIXME: MutableTensor shouldn't relay on the `actor_ref` directly.
        #
        # that means seal doesn't work with web session, and it will be fixed
        # later, as `read/write` doesn't work with web session as well.
        return MutableTensor(None, fetch, chunk_to_actor_key, chunk_to_actor,
                             dtype, default_value)


MutableTensorSerializer.register(MutableTensor)
