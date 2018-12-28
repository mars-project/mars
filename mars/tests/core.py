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
import logging
import shutil
import subprocess
import tempfile
import time
import unittest
from collections import Iterable
from weakref import ReferenceType

import numpy as np

from mars import compat
from mars.compat import zip_longest
from mars.compat import six
from mars.serialize import serializes, deserializes, \
    ProtobufSerializeProvider, JsonSerializeProvider


if compat.PY27:
    try:
        import mock
    except ImportError:
        mock = None
else:
    from unittest import mock
    _mock = mock


class TestCase(unittest.TestCase):
    pass


class MultiGetDict(dict):
    def __getitem__(self, item):
        if isinstance(item, tuple):
            return tuple(super(MultiGetDict, self).__getitem__(it)
                         for it in item)
        return super(MultiGetDict, self).__getitem__(item)


class TestBase(unittest.TestCase):
    def setUp(self):
        self.pb_serialize = lambda *args, **kw: \
            serializes(ProtobufSerializeProvider(), *args, **kw)
        self.pb_deserialize = lambda *args, **kw: \
            deserializes(ProtobufSerializeProvider(), *args, **kw)
        self.json_serialize = lambda *args, **kw: \
            serializes(JsonSerializeProvider(), *args, **kw)
        self.json_deserialize = lambda *args, **kw: \
            deserializes(JsonSerializeProvider(), *args, **kw)

    @classmethod
    def _serial(cls, obj):
        from mars.operands.core import Operand
        from mars.tensor.core import Tensor, TensorData, Chunk, ChunkData

        if isinstance(obj, (Tensor, Chunk)):
            obj = obj.data

        to_serials = set()

        def serial(ob):
            if ob in to_serials:
                return
            if isinstance(ob, TensorData):
                to_serials.add(ob)
                [serial(i) for i in (ob.chunks or [])]
                serial(ob.op)
            elif isinstance(ob, ChunkData):
                to_serials.add(ob)
                [serial(c) for c in (ob.composed or [])]
                serial(ob.op)
            else:
                try:
                    assert isinstance(ob, Operand)
                except AssertionError:
                    raise
                to_serials.add(ob)
                [serial(i) for i in (ob.inputs or [])]
                [serial(i) for i in (ob.outputs or []) if i is not None]

        serial(obj)
        return to_serials

    def _pb_serial(self, obj):
        to_serials = list(self._serial(obj))

        return MultiGetDict(zip(to_serials, self.pb_serialize(to_serials)))

    def _pb_deserial(self, serials_d):
        objs = list(serials_d)
        serials = list(serials_d[o] for o in objs)

        return MultiGetDict(zip(objs, self.pb_deserialize([type(o) for o in objs], serials)))

    def _json_serial(self, obj):
        to_serials = list(self._serial(obj))

        return MultiGetDict(zip(to_serials, self.json_serialize(to_serials)))

    def _json_deserial(self, serials_d):
        objs = list(serials_d)
        serials = list(serials_d[o] for o in objs)

        return MultiGetDict(zip(objs, self.json_deserialize([type(o) for o in objs], serials)))

    def base_equal(self, ob1, ob2):
        if type(ob1) != type(ob2):
            return False

        def cmp(obj1, obj2):
            if isinstance(obj1, np.ndarray):
                return np.array_equal(obj1, obj2)
            elif isinstance(obj1, Iterable) and \
                    not isinstance(obj1, six.string_types) and \
                    isinstance(obj2, Iterable) and \
                    not isinstance(obj2, six.string_types):
                return all(cmp(it1, it2) for it1, it2 in zip_longest(obj1, obj2))
            elif hasattr(obj1, 'key') and hasattr(obj2, 'key'):
                return obj1.key == obj2.key
            elif isinstance(obj1, ReferenceType) and isinstance(obj2, ReferenceType):
                return cmp(obj1(), obj2())
            else:
                return obj1 == obj2

        for slot in ob1.__slots__:
            if not cmp(getattr(ob1, slot, None), getattr(ob2, slot, None)):
                return False

        return True

    def assertBaseEqual(self, ob1, ob2):
        return self.assertTrue(self.base_equal(ob1, ob2))


class EtcdProcessHelper(object):

    def __init__(
            self,
            base_directory=None,
            proc_name='etcd',
            port_range_start=4001,
            internal_port_range_start=7001,
            cluster=False,
            tls=False
    ):
        if base_directory is None:
            import mars
            base_directory = os.path.join(os.path.dirname(os.path.dirname(mars.__file__)), 'bin')

        self.base_directory = base_directory
        self.proc_name = proc_name
        self.port_range_start = port_range_start
        self.internal_port_range_start = internal_port_range_start
        self.processes = {}
        self.cluster = cluster
        self.schema = 'http://'
        if tls:
            self.schema = 'https://'

    def run(self, number=1, proc_args=None):
        proc_args = proc_args or []
        if number > 1:
            initial_cluster = ",".join([
                "test-node-{}={}127.0.0.1:{}".format(slot, 'http://', self.internal_port_range_start + slot)
                for slot in range(0, number)
            ])
            proc_args.extend([
                '-initial-cluster', initial_cluster,
                '-initial-cluster-state', 'new'
            ])
        else:
            proc_args.extend([
                '-initial-cluster', 'test-node-0=http://127.0.0.1:{}'.format(self.internal_port_range_start),
                '-initial-cluster-state', 'new'
            ])

        for i in range(0, number):
            self.add_one(i, proc_args)

    def stop(self):
        for key in [k for k in self.processes.keys()]:
            self.kill_one(key)

    def add_one(self, slot, proc_args=None):
        log = logging.getLogger()
        directory = tempfile.mkdtemp(
            dir=self.base_directory,
            prefix='etcd-gevent.%d-' % slot)

        log.debug('Created directory %s' % directory)
        client = '%s127.0.0.1:%d' % (self.schema, self.port_range_start + slot)
        peer = '%s127.0.0.1:%d' % ('http://', self.internal_port_range_start
                                   + slot)
        daemon_args = [
            self.proc_name,
            '-data-dir', directory,
            '-name', 'test-node-%d' % slot,
            '-initial-advertise-peer-urls', peer,
            '-listen-peer-urls', peer,
            '-advertise-client-urls', client,
            '-listen-client-urls', client
        ]

        if proc_args:
            daemon_args.extend(proc_args)

        daemon = subprocess.Popen(daemon_args)
        log.debug('Started %d' % daemon.pid)
        log.debug('Params: %s' % daemon_args)
        time.sleep(2)
        self.processes[slot] = (directory, daemon)

    def kill_one(self, slot):
        log = logging.getLogger()
        data_dir, process = self.processes.pop(slot)
        process.kill()
        time.sleep(2)
        log.debug('Killed etcd pid:%d', process.pid)
        shutil.rmtree(data_dir)
        log.debug('Removed directory %s' % data_dir)
