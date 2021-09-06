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
    tensor = await session.create_mutable_tensor(shape=(100,100,100),dtype=np.double,
    chunksize=(10,10,10),name="mytensor")
    await tensor.write(((11,2,3),(14,5,6),(17,8,9)),1)
    await tensor.write(((12,2,3),(15,5,6),(16,8,9)),10)
    t = await tensor.read(((11,12,2),(14,15,5),(17,16,8)))
    print(t)

async def main():
    client = await new_cluster(n_worker=2,
                               n_cpu=2)
    async with client:
        client.session.as_default()
        await work(client.session)

if __name__ == '__main__':
    isolation = new_isolation()
    future = asyncio.run_coroutine_threadsafe(main(), isolation.loop)
    result = future.result()