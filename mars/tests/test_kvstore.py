#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import gevent

from mars.tests.core import TestBase
from mars.kvstore import LocalKVStore
from mars.kvstore import PathResult


class Test(TestBase):
    def testLocalPathStore(self):
        kvstore = LocalKVStore()
        kvstore.write('/node/subnode/v1', 'value1')
        kvstore.write('/node/v2', 'value2')

        res = kvstore.read('/node', sort=True)
        expected = PathResult(key='/node', dir=True, children=[
            PathResult(key='/node/subnode', dir=True),
            PathResult(key='/node/v2', value='value2'),
        ])
        self.assertEqual(repr(res), repr(expected))

        res = kvstore.read('/node', recursive=True, sort=True)
        expected = PathResult(key='/node', dir=True, children=[
            PathResult(key='/node/subnode/v1', value='value1'),
            PathResult(key='/node/v2', value='value2'),
        ])
        self.assertEqual(repr(res), repr(expected))

        kvstore.write('/node/v3', 'value3')
        with self.assertRaises(KeyError):
            kvstore.write('/node/v2/invalid_value', value='invalid')

        res = kvstore.read('/', recursive=False, sort=True)
        expected = PathResult(key='/', dir=True, children=[
            PathResult(key='/node', dir=True),
        ])
        self.assertEqual(repr(res), repr(expected))

        res = kvstore.read('/', recursive=True, sort=True)
        expected = PathResult(key='/', dir=True, children=[
            PathResult(key='/node/subnode/v1', value='value1'),
            PathResult(key='/node/v2', value='value2'),
            PathResult(key='/node/v3', value='value3'),
        ])
        self.assertEqual(repr(res), repr(expected))

        kvstore.write('/node/subnode2/v4', 'value4')

        with self.assertRaises(KeyError):
            kvstore.delete('/node/subnode', dir=True)

        kvstore.delete('/node/subnode/v1')
        res = kvstore.read('/', recursive=True, sort=True)
        expected = PathResult(key='/', dir=True, children=[
            PathResult(key='/node/subnode', dir=True),
            PathResult(key='/node/subnode2/v4', value='value4'),
            PathResult(key='/node/v2', value='value2'),
            PathResult(key='/node/v3', value='value3'),
        ])
        self.assertEqual(repr(res), repr(expected))

        kvstore.delete('/node/subnode2', dir=True, recursive=True)
        res = kvstore.read('/', recursive=True, sort=True)
        expected = PathResult(key='/', dir=True, children=[
            PathResult(key='/node/subnode', dir=True),
            PathResult(key='/node/v2', value='value2'),
            PathResult(key='/node/v3', value='value3')
        ])
        self.assertEqual(repr(res), repr(expected))

    def testLocalWatch(self):
        kvstore = LocalKVStore()
        kvstore.write('/node/subnode/v1', 'value1')
        kvstore.write('/node/v2', 'value2')

        def watcher():
            return kvstore.watch('/node/v2', timeout=10)

        def writer():
            gevent.sleep(1)
            kvstore.write('/node/v2', 'value2\'')

        g1 = gevent.spawn(writer)
        g2 = gevent.spawn(watcher)

        gevent.joinall([g1, g2])
        self.assertEqual(g2.value.value, 'value2\'')

        def watcher():
            return kvstore.watch('/node/subnode', timeout=10, recursive=True)

        def writer():
            gevent.sleep(1)
            kvstore.write('/node/subnode/v1', 'value1\'')

        g1 = gevent.spawn(writer)
        g2 = gevent.spawn(watcher)

        gevent.joinall([g1, g2])
        self.assertEqual(g2.value.children[0].value, 'value1\'')
