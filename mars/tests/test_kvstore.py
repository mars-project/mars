#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import os
import sys
import unittest

from mars.tests.core import EtcdProcessHelper, aio_case
from mars.kvstore import get, PathResult


@aio_case
class Test(unittest.TestCase):
    async def testLocalPathStore(self):
        kvstore = get(':inproc:')
        await kvstore.write('/node/subnode/v1', 'value1')
        await kvstore.write('/node/v2', 'value2')

        res = await kvstore.read('/node', sort=True)
        expected = PathResult(key='/node', dir=True, children=[
            PathResult(key='/node/subnode', dir=True),
            PathResult(key='/node/v2', value='value2'),
        ])
        self.assertEqual(repr(res), repr(expected))

        res = await kvstore.read('/node', recursive=True, sort=True)
        expected = PathResult(key='/node', dir=True, children=[
            PathResult(key='/node/subnode/v1', value='value1'),
            PathResult(key='/node/v2', value='value2'),
        ])
        self.assertEqual(repr(res), repr(expected))

        await kvstore.write('/node/v3', 'value3')
        with self.assertRaises(KeyError):
            await kvstore.write('/node/v2/invalid_value', value='invalid')

        res = await kvstore.read('/', recursive=False, sort=True)
        expected = PathResult(key='/', dir=True, children=[
            PathResult(key='/node', dir=True),
        ])
        self.assertEqual(repr(res), repr(expected))

        res = await kvstore.read('/', recursive=True, sort=True)
        expected = PathResult(key='/', dir=True, children=[
            PathResult(key='/node/subnode/v1', value='value1'),
            PathResult(key='/node/v2', value='value2'),
            PathResult(key='/node/v3', value='value3'),
        ])
        self.assertEqual(repr(res), repr(expected))

        await kvstore.write('/node/subnode2/v4', 'value4')

        with self.assertRaises(KeyError):
            await kvstore.delete('/node/subnode', dir=True)

        await kvstore.delete('/node/subnode/v1')
        res = await kvstore.read('/', recursive=True, sort=True)
        expected = PathResult(key='/', dir=True, children=[
            PathResult(key='/node/subnode', dir=True),
            PathResult(key='/node/subnode2/v4', value='value4'),
            PathResult(key='/node/v2', value='value2'),
            PathResult(key='/node/v3', value='value3'),
        ])
        self.assertEqual(repr(res), repr(expected))

        await kvstore.delete('/node/subnode2', dir=True, recursive=True)
        res = await kvstore.read('/', recursive=True, sort=True)
        expected = PathResult(key='/', dir=True, children=[
            PathResult(key='/node/subnode', dir=True),
            PathResult(key='/node/v2', value='value2'),
            PathResult(key='/node/v3', value='value3')
        ])
        self.assertEqual(repr(res), repr(expected))

    @unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
    @unittest.skipIf('CI' not in os.environ and not EtcdProcessHelper().is_installed(),
                     'does not run without etcd')
    async def testEtcdPathStore(self):
        with EtcdProcessHelper(port_range_start=51342).run():
            kvstore = get('etcd://localhost:51342')
            await kvstore.write('/node/subnode/v1', 'value1')
            await kvstore.write('/node/v2', 'value2')

            res = await kvstore.read('/node', sort=True)
            expected = PathResult(key='/node', dir=True, children=[
                PathResult(key='/node/subnode', dir=True),
                PathResult(key='/node/v2', value='value2'),
            ])
            self.assertEqual(repr(res), repr(expected))

            res = await kvstore.read('/node', recursive=True, sort=True)
            expected = PathResult(key='/node', dir=True, children=[
                PathResult(key='/node/subnode/v1', value='value1'),
                PathResult(key='/node/v2', value='value2'),
            ])
            self.assertEqual(repr(res), repr(expected))

            await kvstore.write('/node/v3', 'value3')
            with self.assertRaises(KeyError):
                await kvstore.write('/node/v2/invalid_value', value='invalid')

            res = await kvstore.read('/', recursive=False, sort=True)
            expected = PathResult(key='/', dir=True, children=[
                PathResult(key='/node', dir=True),
            ])
            self.assertEqual(repr(res), repr(expected))

            res = await kvstore.read('/', recursive=True, sort=True)
            expected = PathResult(key='/', dir=True, children=[
                PathResult(key='/node/subnode/v1', value='value1'),
                PathResult(key='/node/v2', value='value2'),
                PathResult(key='/node/v3', value='value3'),
            ])
            self.assertEqual(repr(res), repr(expected))

            await kvstore.write('/node/subnode2/v4', 'value4')

            with self.assertRaises(KeyError):
                await kvstore.delete('/node/subnode', dir=True)

            await kvstore.delete('/node/subnode/v1')
            res = await kvstore.read('/', recursive=True, sort=True)
            expected = PathResult(key='/', dir=True, children=[
                PathResult(key='/node/subnode', dir=True),
                PathResult(key='/node/subnode2/v4', value='value4'),
                PathResult(key='/node/v2', value='value2'),
                PathResult(key='/node/v3', value='value3'),
            ])
            self.assertEqual(repr(res), repr(expected))

            await kvstore.delete(u'/node', recursive=True, dir=True)

    async def testLocalWatch(self):
        kvstore = get(':inproc:')
        await kvstore.write('/node/subnode/v1', 'value1')
        await kvstore.write('/node/v2', 'value2')

        async def watcher():
            return await kvstore.watch('/node/v2', timeout=10)

        async def writer():
            await asyncio.sleep(0.5)
            await kvstore.write('/node/v2', 'value2\'')

        asyncio.ensure_future(writer())
        self.assertEqual((await watcher()).value, 'value2\'')

        await kvstore.delete('/node/v2')

        async def watcher():
            return await kvstore.watch('/node/subnode', timeout=10, recursive=True)

        async def writer():
            await asyncio.sleep(0.5)
            await kvstore.write('/node/subnode/v1', 'value1\'')

        asyncio.ensure_future(writer())
        self.assertEqual((await watcher()).children[0].value, 'value1\'')

        await kvstore.write('/node/subnode/v3', '-1')

        async def watcher():
            results = []
            idx = 0
            async for result in kvstore.eternal_watch('/node/subnode/v3'):
                results.append(int(result.value))
                if idx == 4:
                    break
                idx += 1
            return results

        async def writer():
            await asyncio.sleep(0.1)
            for v in range(5):
                await kvstore.write('/node/subnode/v3', str(v))
                await asyncio.sleep(0.1)

        asyncio.ensure_future(writer())
        self.assertEqual(await watcher(), list(range(5)))

        await kvstore.delete('/node', dir=True, recursive=True)

    @unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
    @unittest.skipIf('CI' not in os.environ and not EtcdProcessHelper().is_installed(),
                     'does not run without etcd')
    async def testEtcdWatch(self):
        with EtcdProcessHelper(port_range_start=51342).run():
            kvstore = get('etcd://localhost:51342')
            await kvstore.write('/node/subnode/v1', 'value1')
            await kvstore.write('/node/v2', 'value2')

            async def watcher():
                return await kvstore.watch('/node/v2', timeout=10)

            async def writer():
                await asyncio.sleep(0.5)
                await kvstore.write('/node/v2', 'value2\'')

            asyncio.ensure_future(writer())
            self.assertEqual((await watcher()).value, 'value2\'')

            await kvstore.delete('/node/v2')

            async def watcher():
                return await kvstore.watch('/node/subnode', timeout=10, recursive=True)

            async def writer():
                await asyncio.sleep(0.5)
                await kvstore.write('/node/subnode/v1', 'value1\'')

            asyncio.ensure_future(writer())
            self.assertEqual((await watcher()).children[0].value, 'value1\'')

            await kvstore.write('/node/subnode/v3', '-1')

            async def watcher():
                results = []
                idx = 0
                async for result in kvstore.eternal_watch('/node/subnode/v3'):
                    results.append(int(result.value))
                    if idx == 4:
                        break
                    idx += 1
                return results

            async def writer():
                await asyncio.sleep(0.1)
                for v in range(5):
                    await kvstore.write('/node/subnode/v3', str(v))
                    await asyncio.sleep(0.1)

            asyncio.ensure_future(writer())
            self.assertEqual(await watcher(), list(range(5)))

            await kvstore.delete('/node', dir=True, recursive=True)
