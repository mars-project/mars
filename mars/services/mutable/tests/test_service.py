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

import pytest
import numpy as np

from ....deploy.oscar.local import new_cluster
from ..supervisor.core import MutableTensor


@pytest.mark.asyncio
async def test_mutable_tensor_actor():
    import platform
    if(platform.system() == 'Windows'):
        return
    client = await new_cluster(n_worker=2,
                               n_cpu=2)
    async with client:
        client.session.as_default()
        session = client.session
        tensor_useless: MutableTensor = await session.create_mutable_tensor(shape=(100, 100, 100), dtype=np.int64,
        chunk_size=(10, 10, 10), default_value=100)
        tensor: MutableTensor = await session.create_mutable_tensor(shape=(100, 100, 100), dtype=np.int64,
        chunk_size=(10, 10, 10), name="mytensor", default_value=100)
        tensor1: MutableTensor = await session.get_mutable_tensor("mytensor")
        try:
            tensor = await session.get_mutable_tensor("notensor")
        except Exception as e:
            assert str(e) == 'invalid name!'

        await tensor.write(((11, 2, 3), (14, 5, 6), (17, 8, 9)), 1)
        await tensor1.write(((12, 2, 3), (15, 5, 6), (16, 8, 9)), 10)
        await tensor_useless.write(((0,), (0,), (0,)), 1, 1)
        [t] = await tensor1[0, 0, 0]
        assert t == 100
        [t] = await tensor1[11, 14, 17]
        assert t == 1
        [t] = await tensor1[(3,), (6,), (9,)]
        assert t == 10
