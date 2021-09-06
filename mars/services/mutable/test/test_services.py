from _pytest.main import Session
from numpy import dtype
import pytest

import mars.oscar as mo
import numpy as np
from mars.deploy.oscar.local import new_cluster
from mars.utils import to_binary
from mars.services.mutable.supervisor import MutableTensorActor

@pytest.mark.asyncio
async def test_mutable_tensor_actor():
    client = await new_cluster(n_worker=2,
                               n_cpu=2)
    async with client:
        client.session.as_default()
        session = client.session
        tensor = await session.create_mutable_tensor(shape=(100,100,100),dtype=np.int64,
        chunksize=(10,10,10),name="mytensor")
        await tensor.write(((11,2,3),(14,5,6),(17,8,9)),1)
        await tensor.write(((12,2,3),(15,5,6),(16,8,9)),10)
        t = await tensor.read(((11,12,2),(14,15,5),(17,16,8)))
        assert t==[10,1,10]