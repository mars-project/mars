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

import logging
import os
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.testing import assert_allclose

import mars.tensor as mt
from mars.session import new_session
from mars.scheduler.tests.integrated.base import SchedulerIntegratedTest

logger = logging.getLogger(__name__)


@unittest.skipIf(sys.platform == 'win32', "plasma don't support windows")
class Test(SchedulerIntegratedTest):
    def _submit_tileable(self, tileable):
        def submitter():
            sess = new_session(self.session_manager_ref.address)
            return tileable.execute(session=sess, timeout=self.timeout).fetch(session=sess)

        pool = ThreadPoolExecutor(1)
        return pool.submit(lambda: submitter())

    def testCommonOperandFailover(self):
        delay_file = self.add_state_file('OP_DELAY_STATE_FILE')
        open(delay_file, 'w').close()

        terminate_file = self.add_state_file('OP_TERMINATE_STATE_FILE')

        self.start_processes(modules=['mars.scheduler.tests.integrated.op_delayer'], log_worker=True)

        np_a = np.random.random((100, 100))
        np_b = np.random.random((100, 100))

        a = mt.array(np_a, chunk_size=30) * 2 + 1
        b = mt.array(np_b, chunk_size=30) * 2 + 1
        c = a.dot(b) * 2 + 1

        future = self._submit_tileable(c)

        while not os.path.exists(terminate_file):
            time.sleep(0.01)

        self.kill_process_tree(self.proc_workers[0])
        logger.warning('Worker %s KILLED!\n\n', self.proc_workers[0].pid)
        self.proc_workers = self.proc_workers[1:]
        os.unlink(delay_file)

        result = future.result(timeout=self.timeout)
        expected = (np_a * 2 + 1).dot(np_b * 2 + 1) * 2 + 1
        assert_allclose(result, expected)

    def testShuffleFailoverBeforeSuccStart(self):
        pred_finish_file = self.add_state_file('SHUFFLE_ALL_PRED_FINISHED_FILE')
        succ_start_file = self.add_state_file('SHUFFLE_START_SUCC_FILE')

        self.start_processes(modules=['mars.scheduler.tests.integrated.op_delayer'], log_worker=True)

        a = mt.ones((31, 27), chunk_size=10)
        b = a.reshape(27, 31)
        b.op.extra_params['_reshape_with_shuffle'] = True

        future = self._submit_tileable(b)
        time.sleep(1)
        while not os.path.exists(pred_finish_file):
            time.sleep(0.01)

        self.kill_process_tree(self.proc_workers[0])
        logger.warning('Worker %s KILLED!\n\n', self.proc_workers[0].pid)
        self.proc_workers = self.proc_workers[1:]
        open(succ_start_file, 'w').close()

        result = future.result(timeout=self.timeout)
        assert_allclose(result, np.ones((27, 31)))

    def testShuffleFailoverBeforeAllSuccFinish(self):
        pred_finish_file = self.add_state_file('SHUFFLE_ALL_PRED_FINISHED_FILE')
        succ_finish_file = self.add_state_file('SHUFFLE_HAS_SUCC_FINISH_FILE')

        self.start_processes(modules=['mars.scheduler.tests.integrated.op_delayer'], log_worker=True)

        a = mt.ones((31, 27), chunk_size=10)
        b = a.reshape(27, 31)
        b.op.extra_params['_reshape_with_shuffle'] = True
        r = mt.inner(b + 1, b + 1)

        future = self._submit_tileable(r)
        time.sleep(1)
        while not os.path.exists(succ_finish_file):
            time.sleep(0.01)

        self.kill_process_tree(self.proc_workers[0])
        logger.warning('Worker %s KILLED!\n\n', self.proc_workers[0].pid)
        self.proc_workers = self.proc_workers[1:]

        os.unlink(pred_finish_file)
        os.unlink(succ_finish_file)

        result = future.result(timeout=self.timeout)
        assert_allclose(result, np.inner(np.ones((27, 31)) + 1, np.ones((27, 31)) + 1))

    def testShuffleFailoverAfterAllSuccFinish(self):
        all_succ_finish_file = self.add_state_file('SHUFFLE_ALL_SUCC_FINISH_FILE')

        self.start_processes(modules=['mars.scheduler.tests.integrated.op_delayer'],
                             log_worker=True)

        a = mt.ones((31, 27), chunk_size=10)
        b = a.reshape(27, 31)
        b.op.extra_params['_reshape_with_shuffle'] = True
        r = mt.inner(b + 1, b + 1)

        future = self._submit_tileable(r)
        time.sleep(1)
        while not os.path.exists(all_succ_finish_file):
            time.sleep(0.01)

        self.kill_process_tree(self.proc_workers[0])
        logger.warning('Worker %s KILLED!\n\n', self.proc_workers[0].pid)
        self.proc_workers = self.proc_workers[1:]

        os.unlink(all_succ_finish_file)

        result = future.result()
        assert_allclose(result, np.inner(np.ones((27, 31)) + 1, np.ones((27, 31)) + 1))
