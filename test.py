import asyncio
from mars.services.mutable.supervisor.service import MutableTensor
from mars.tensor.utils import *
from mars.deploy.oscar.local import new_cluster
from mars.lib.aio import new_isolation

async def work(session):
    tensor:MutableTensor = await session.create_mutable_tensor(shape=(100,100,100),dtype=np.double,
    chunk_size=(10,10,10),name="mytensor")
    await tensor.write(((11,),(10,),(9,)),2)
    # await tensor.write(((12,2,3),(15,5,6),(16,8,9)),10)
    t = await tensor[((11,),(10,),(9,))]
    print(t)

async def main():
    client = await new_cluster(n_worker=3,
                               n_cpu=4)
    async with client:
        client.session.as_default()
        await work(client.session)

if __name__ == '__main__':
    isolation = new_isolation()
    future = asyncio.run_coroutine_threadsafe(main(), isolation.loop)
    result = future.result()