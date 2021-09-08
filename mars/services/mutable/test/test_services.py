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
        chunk_size=(10,10,10),name="mytensor")
        await tensor.write(((11,2,3),(14,5,6),(17,8,9)),1)
        await tensor.write(((12,2,3),(15,5,6),(16,8,9)),10)
        t = await tensor[(11,12,2),(14,15,5),(17,16,8)]
        assert t == [10,1,10]
