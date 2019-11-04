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

import os
import sys
import unittest

import gevent

from mars.tests.core import EtcdProcessHelper
from mars.kvstore import get, PathResult
from mars.utils import get_next_port


class Test(unittest.TestCase):
    def testLocalPathStore(self):
        kvstore = get(':inproc:')
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

    @unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
    @unittest.skipIf('CI' not in os.environ and not EtcdProcessHelper().is_installed(),
                     'does not run without etcd')
    def testEtcdPathStore(self):
        etcd_port = get_next_port()
        etcd_internal_port = get_next_port()
        with EtcdProcessHelper(port_range_start=etcd_port,
                               internal_port_range_start=etcd_internal_port).run():
            kvstore = get(u'etcd://localhost:%d' % etcd_port)
            kvstore.write(u'/node/subnode/v1', u'value1')
            kvstore.write(u'/node/v2', u'value2')

            res = kvstore.read(u'/node', sort=True)
            expected = PathResult(key=u'/node', dir=True, children=[
                PathResult(key=u'/node/subnode', dir=True),
                PathResult(key=u'/node/v2', value=u'value2'),
            ])
            self.assertEqual(repr(res), repr(expected))

            res = kvstore.read(u'/node', recursive=True, sort=True)
            expected = PathResult(key=u'/node', dir=True, children=[
                PathResult(key=u'/node/subnode/v1', value=u'value1'),
                PathResult(key=u'/node/v2', value=u'value2'),
            ])
            self.assertEqual(repr(res), repr(expected))

            kvstore.write(u'/node/v3', u'value3')
            with self.assertRaises(KeyError):
                kvstore.write(u'/node/v2/invalid_value', value=u'invalid')

            res = kvstore.read('/', recursive=False, sort=True)
            expected = PathResult(key='/', dir=True, children=[
                PathResult(key=u'/node', dir=True),
            ])
            self.assertEqual(repr(res), repr(expected))

            res = kvstore.read('/', recursive=True, sort=True)
            expected = PathResult(key='/', dir=True, children=[
                PathResult(key=u'/node/subnode/v1', value=u'value1'),
                PathResult(key=u'/node/v2', value=u'value2'),
                PathResult(key=u'/node/v3', value=u'value3'),
            ])
            self.assertEqual(repr(res), repr(expected))

            kvstore.write(u'/node/subnode2/v4', u'value4')

            with self.assertRaises(KeyError):
                kvstore.delete(u'/node/subnode', dir=True)

            kvstore.delete(u'/node/subnode/v1')
            res = kvstore.read('/', recursive=True, sort=True)
            expected = PathResult(key='/', dir=True, children=[
                PathResult(key=u'/node/subnode', dir=True),
                PathResult(key=u'/node/subnode2/v4', value=u'value4'),
                PathResult(key=u'/node/v2', value=u'value2'),
                PathResult(key=u'/node/v3', value=u'value3'),
            ])
            self.assertEqual(repr(res), repr(expected))

            kvstore.delete(u'/node', recursive=True, dir=True)

    def testLocalWatch(self):
        kvstore = get(':inproc:')
        kvstore.write('/node/subnode/v1', 'value1')
        kvstore.write('/node/v2', 'value2')

        def watcher():
            return kvstore.watch('/node/v2', timeout=10)

        def writer():
            gevent.sleep(0.5)
            kvstore.write('/node/v2', 'value2\'')

        g1 = gevent.spawn(writer)
        g2 = gevent.spawn(watcher)
        gevent.joinall([g1, g2])
        self.assertEqual(g2.value.value, 'value2\'')

        kvstore.delete('/node/v2')

        def watcher():
            return kvstore.watch('/node/subnode', timeout=10, recursive=True)

        def writer():
            gevent.sleep(0.5)
            kvstore.write('/node/subnode/v1', 'value1\'')

        g1 = gevent.spawn(writer)
        g2 = gevent.spawn(watcher)
        gevent.joinall([g1, g2])
        self.assertEqual(g2.value.children[0].value, 'value1\'')

        kvstore.write('/node/subnode/v3', '-1')

        def watcher():
            results = []
            for idx, result in enumerate(kvstore.eternal_watch('/node/subnode/v3')):
                results.append(int(result.value))
                if idx == 4:
                    break
            return results

        def writer():
            gevent.sleep(0.1)
            for v in range(5):
                kvstore.write('/node/subnode/v3', str(v))
                gevent.sleep(0.1)

        g1 = gevent.spawn(writer)
        g2 = gevent.spawn(watcher)
        gevent.joinall([g1, g2])
        self.assertEqual(g2.value, list(range(5)))

        kvstore.delete('/node', dir=True, recursive=True)

    @unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
    @unittest.skipIf('CI' not in os.environ and not EtcdProcessHelper().is_installed(),
                     'does not run without etcd')
    def testEtcdWatch(self):
        etcd_port = get_next_port()
        etcd_internal_port = get_next_port()
        with EtcdProcessHelper(port_range_start=etcd_port,
                               internal_port_range_start=etcd_internal_port).run():
            kvstore = get('etcd://localhost:%d' % etcd_port)
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

            kvstore.delete('/node/v2')

            def watcher():
                return kvstore.watch('/node/subnode', timeout=10, recursive=True)

            def writer():
                gevent.sleep(1)
                kvstore.write('/node/subnode/v1', 'value1\'')

            g1 = gevent.spawn(writer)
            g2 = gevent.spawn(watcher)
            gevent.joinall([g1, g2])
            self.assertEqual(g2.value.children[0].value, 'value1\'')

            kvstore.write('/node/subnode/v3', '-1')

            def watcher():
                results = []
                for idx, result in enumerate(kvstore.eternal_watch('/node/subnode/v3')):
                    results.append(int(result.value))
                    if idx == 4:
                        break
                return results

            def writer():
                gevent.sleep(0.1)
                for v in range(5):
                    kvstore.write('/node/subnode/v3', str(v))
                    gevent.sleep(0.1)

            g1 = gevent.spawn(writer)
            g2 = gevent.spawn(watcher)
            gevent.joinall([g1, g2])
            self.assertEqual(g2.value, list(range(5)))

            kvstore.delete('/node', dir=True, recursive=True)
