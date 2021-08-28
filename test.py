import asyncio
import numpy as np
import os

import mars.dataframe as md
import mars.tensor as mt
import mars.remote as mr
from mars.services.session.supervisor.core import SessionActor
from mars.tensor.utils import *

from mars.deploy.oscar.local import new_cluster
from mars.lib.aio import new_isolation

async def work(session):
    tmp = await session.create_mutable_tensor(shape=(1000,1000),dtype=np.double,
    chunksize=(50,50),name="test2")
    await tmp.get_chunks()
    await tmp.assign_chunks()
    for i in range(2,1000,100):
        for j in range(2,1000,100):
            await tmp.write((i,j))
    # t = mt.ones((1024, 1024), chunk_size=256)
    # info = await session.execute(t)
    # await info
    # r = await session.fetch(t)
    # print(r)

async def main():
    client = await new_cluster(n_worker=2,
                               n_cpu=2)
    async with client:
        client.session.as_default()
        await work(client.session)

if __name__ == '__main__':
    # chunk_size = (3,3)
    # print((1)*3)
    # shape=[10,10]
    # shape = normalize_shape(shape)
    # chunk_size = normalize_chunk_sizes(shape,chunk_size)
    # print(shape,chunk_size)
    isolation = new_isolation()
    future = asyncio.run_coroutine_threadsafe(main(), isolation.loop)
    result = future.result()