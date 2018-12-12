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

import time
from functools import partial

import gevent

from mars.compat import unittest, mock
from mars.utils import get_next_port
from mars.actors import create_actor_pool
from mars.promise import PromiseActor
from mars.worker import *


class TaskActor(PromiseActor):
    def __init__(self, queue_name, call_records):
        super(TaskActor, self).__init__()
        self._queue_name = queue_name
        self._call_records = call_records
        self._dispatch_ref = None

    def post_create(self):
        self._dispatch_ref = self.promise_ref(DispatchActor.default_name())
        self._dispatch_ref.register_free_slot(self.uid, self._queue_name)

    def queued_call(self, key, delay):
        try:
            self._call_records[key] = time.time()
            gevent.sleep(delay)
        finally:
            self._dispatch_ref.register_free_slot(self.uid, self._queue_name)


class RunTaskTestActor(PromiseActor):
    def __init__(self, call_records):
        super(RunTaskTestActor, self).__init__()
        self._dispatch_ref = None
        self._finished = False
        self._call_records = call_records

    def post_create(self):
        self._dispatch_ref = self.promise_ref(DispatchActor.default_name())

    def get_finished(self):
        return self._finished

    def run_tasks(self, group_size):
        from mars.promise import Promise
        p = Promise()
        p.step_next()

        def _call_on_dispatched(uid, key=None):
            if uid is None:
                self._call_records[key] = 'NoneUID'
            else:
                self.promise_ref(uid).queued_call(key, 2, _tell=True)

        for idx in range(group_size + 1):
            p = p.then(lambda *_: self._dispatch_ref.get_free_slot('g1', _promise=True)) \
                .then(partial(_call_on_dispatched, key='%d_1' % idx)) \
                .then(lambda *_: self._dispatch_ref.get_free_slot('g2', _promise=True)) \
                .then(partial(_call_on_dispatched, key='%d_2' % idx))

        p.then(lambda *_: self._dispatch_ref.get_free_slot('g3', _promise=True)) \
            .then(partial(_call_on_dispatched, key='N_1')) \
            .then(lambda *_: setattr(self, '_finished', True))


class Test(unittest.TestCase):
    @mock.patch('mars.worker.utils.WorkerActor._init_chunk_store')
    def testDispatch(self, *_):
        call_records = dict()
        group_size = 4

        mock_scheduler_addr = '127.0.0.1:%d' % get_next_port()
        with create_actor_pool(n_process=1, backend='gevent',
                               address=mock_scheduler_addr) as pool:
            dispatch_ref = pool.create_actor(DispatchActor, uid=DispatchActor.default_name())
            # actors of g1
            [pool.create_actor(TaskActor, 'g1', call_records) for _ in range(group_size)]
            [pool.create_actor(TaskActor, 'g2', call_records) for _ in range(group_size)]

            self.assertEqual(len(dispatch_ref.get_slots('g1')), group_size)
            self.assertEqual(len(dispatch_ref.get_slots('g2')), group_size)
            self.assertEqual(len(dispatch_ref.get_slots('g3')), 0)

            self.assertEqual(dispatch_ref.get_hash_slot('g1', 'hash_str'),
                             dispatch_ref.get_hash_slot('g1', 'hash_str'))

            # tasks within [0, group_size - 1] will run almost simultaneously,
            # while the last one will be delayed due to lack of

            def run_tasks():
                test_ref = pool.create_actor(RunTaskTestActor, call_records)
                test_ref.run_tasks(group_size)
                while not test_ref.get_finished():
                    gevent.sleep(1)
                test_ref.destroy()

            gl = gevent.spawn(run_tasks)
            gl.join()

            self.assertEqual(call_records['N_1'], 'NoneUID')
            self.assertLess(sum(abs(call_records['%d_1' % idx] - call_records['0_1'])
                                for idx in range(group_size)), 1)
            self.assertGreater(call_records['%d_1' % group_size] - call_records['0_1'], 1)
            self.assertLess(call_records['%d_1' % group_size] - call_records['0_1'], 3)

            dispatch_ref.destroy()
