from mars.services.mutable.supervisor.service import MutableTensor
import pytest
import numpy as np
from mars.deploy.oscar.local import new_cluster


@pytest.mark.asyncio
async def test_mutable_tensor_actor():
    client = await new_cluster(n_worker=2,
                               n_cpu=2)
    async with client:
        client.session.as_default()
        session = client.session
        tensor:MutableTensor = await session.create_mutable_tensor(shape=(100,100,100),dtype=np.int64,
        chunk_size=(10,10,10),name="mytensor",default_value=100)
        tensor1:MutableTensor = await session.get_mutable_tensor("mytensor")

        await tensor.write(((11,2,3),(14,5,6),(17,8,9)),1)
        await tensor1.write(((12,2,3),(15,5,6),(16,8,9)),10)
        [t] = await tensor1[0,0,0]
        assert t == 100
        [t] = await tensor1[11,14,17]
        assert t == 1
        [t] = await tensor1[3,6,9]
        assert t == 10
