# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
import sys
import multiprocessing
from typing import Union, List, Tuple, Type, Dict

import numpy as np
import pandas as pd
import pytest

from mars.lib.aio import AioEvent
from mars.oscar.backends.mars.communication import \
    SocketChannel, SocketServer, UnixSocketServer, \
    DummyChannel, DummyServer, get_client_type, \
    SocketClient, UnixSocketClient, DummyClient, Server
from mars.utils import get_next_port
from mars.tests.core import require_cudf, require_cupy


test_data = np.random.RandomState(0).rand(10, 10)
port = get_next_port()


# server_type, config, con
params: List[Tuple[Type[Server], Dict, str]] = [
    (SocketServer, dict(host='127.0.0.1', port=port), f'127.0.0.1:{port}'),
]
if sys.platform != 'win32':
    params.append((UnixSocketServer, dict(process_index='0'), f'unixsocket:///0'))
local_params = params.copy()
local_params.append((DummyServer, dict(), 'dummy://0'))


@pytest.mark.parametrize(
    'server_type, config, con',
    local_params
)
@pytest.mark.asyncio
async def test_comm(server_type, config, con):
    async def check_data(chan: Union[SocketChannel, DummyChannel]):
        np.testing.assert_array_equal(test_data, await chan.recv())
        await chan.send('success')

    config = config.copy()
    config['handle_channel'] = check_data

    # create server
    server = await server_type.create(config)
    await server.start()
    assert isinstance(server.info, dict)

    # create client
    client = await server_type.client_type.connect(con)
    assert isinstance(client.info, dict)
    assert isinstance(client.channel.info, dict)
    await client.send(test_data)

    assert 'success' == await client.recv()

    await client.close()
    assert client.closed

    # create client2
    async with await server_type.client_type.connect(con) as client2:
        assert not client2.closed
    assert client2.closed

    await server.join(.001)
    await server.stop()

    assert server.stopped

    async with await server_type.create(config) as server2:
        assert not server2.stopped
    assert server2.stopped


def _wrap_test(server_started_event, conf, tp):
    async def _test():
        async def check_data(chan: SocketChannel):
            np.testing.assert_array_equal(test_data, await chan.recv())
            await chan.send('success')

        nonlocal conf
        conf = conf.copy()
        conf['handle_channel'] = check_data

        # create server
        server = await tp.create(conf)
        await server.start()
        server_started_event.set()
        await server.join()

    asyncio.run(_test())


@pytest.mark.parametrize(
    'server_type, config, con',
    params
)
@pytest.mark.asyncio
async def test_multiprocess_comm(server_type, config, con):
    server_started = multiprocessing.Event()

    p = multiprocessing.Process(target=_wrap_test,
                                args=(server_started, config, server_type))
    p.daemon = True
    p.start()

    await AioEvent(server_started).wait()

    # create client
    client = await server_type.client_type.connect(con)
    await client.channel.send(test_data)

    assert 'success' == await client.recv()

    await client.close()
    assert client.closed


cupy_data = np.arange(100).reshape((10, 10))
cudf_data = pd.DataFrame({'col1': np.arange(10),
                          'col2': [f's{i}' for i in range(10)]})


def _wrap_cuda_test(server_started_event, conf, tp):
    async def _test():
        async def check_data(chan: SocketChannel):
            import cudf
            import cupy

            r = await chan.recv()

            if isinstance(r, cupy.ndarray):
                cupy.testing.assert_array_equal(
                    r, cupy.asarray(cupy_data))
            else:
                cudf.testing.assert_frame_equal(
                    r, cudf.DataFrame(cudf_data))
            await chan.send('success')

        conf['handle_channel'] = check_data

        # create server
        server = await tp.create(conf)
        await server.start()
        server_started_event.set()
        await server.join()

    asyncio.run(_test())


@require_cupy
@require_cudf
@pytest.mark.asyncio
async def test_multiprocess_cuda_comm():
    import cupy
    import cudf

    mp_ctx = multiprocessing.get_context('spawn')

    server_started = mp_ctx.Event()
    port = get_next_port()
    p = mp_ctx.Process(target=_wrap_cuda_test,
                       args=(server_started,
                             dict(host='127.0.0.1', port=port),
                             SocketServer))
    p.daemon = True
    p.start()

    await AioEvent(server_started).wait()

    # create client
    client = await SocketServer.client_type.connect(f'127.0.0.1:{port}')

    await client.channel.send(cupy.asarray(cupy_data))
    assert 'success' == await client.recv()

    client = await SocketServer.client_type.connect(f'127.0.0.1:{port}')

    await client.channel.send(cudf.DataFrame(cudf_data))
    assert 'success' == await client.recv()

    await client.close()


def test_get_client_type():
    assert issubclass(get_client_type('127.0.0.1'), SocketClient)
    assert issubclass(get_client_type('unixsocket:///1'), UnixSocketClient)
    assert issubclass(get_client_type('dummy://'), DummyClient)
