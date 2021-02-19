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
from typing import Union

import numpy as np
import pytest

from mars.lib.aio import AioEvent
from mars.oscar.backends.mars.communication.socket import \
    SocketChannel, SocketServer, UnixSocketServer
from mars.oscar.backends.mars.communication.dummy import \
    DummyChannel, DummyServer
from mars.utils import get_next_port


test_data = np.random.RandomState(0).rand(10, 10)
port = get_next_port()


# server_type, config, con
params = [
    (SocketServer, dict(host='127.0.0.1', port=port), f'127.0.0.1:{port}'),
]
if sys.platform != 'win32':
    params.append((UnixSocketServer, dict(process_index='0'), f'unixsocket:///0'))
local_params = params.copy()
local_params.append((DummyServer, dict(), 'dummy://'))


@pytest.mark.skipif(sys.version_info < (3, 7),
                    reason="requires Python3.7 or higher")
@pytest.mark.parametrize(
    'server_type, config, con',
    local_params
)
@pytest.mark.asyncio
async def test_comm(server_type, config, con):
    success = AioEvent()

    async def check_data(chan: Union[SocketChannel, DummyChannel]):
        np.testing.assert_array_equal(test_data, await chan.recv())
        success.set()

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
    await client.channel.send(test_data)

    assert success.wait(10)

    await client.close()
    await server.join(.001)
    await server.shutdown()

    assert client.closed
    assert server.is_shutdown


def _wrap_test(server_started_event, success_event, conf, tp):
    async def _test():
        async def check_data(chan: SocketChannel):
            np.testing.assert_array_equal(test_data, await chan.recv())
            success_event.set()

        nonlocal conf
        conf = conf.copy()
        conf['handle_channel'] = check_data

        # create server
        server = await tp.create(conf)
        await server.start()
        server_started_event.set()
        await server.join()

    asyncio.run(_test())


@pytest.mark.skipif(sys.version_info < (3, 7),
                    reason="requires Python3.7 or higher")
@pytest.mark.parametrize(
    'server_type, config, con',
    params
)
@pytest.mark.asyncio
async def test_multiprocess_comm(server_type, config, con):
    success = multiprocessing.Event()
    server_started = multiprocessing.Event()

    p = multiprocessing.Process(target=_wrap_test,
                                args=(server_started, success, config, server_type))
    p.daemon = True
    p.start()

    await AioEvent(server_started).wait()

    # create client
    client = await server_type.client_type.connect(con)
    await client.channel.send(test_data)

    assert success.wait(10)

    await client.close()
    assert client.closed
