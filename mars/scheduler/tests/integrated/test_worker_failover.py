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

import json
import logging
import os
import sys
import unittest
import uuid

import numpy as np
from numpy.testing import assert_allclose

from mars import tensor as mt
from mars.serialize.dataserializer import loads
from mars.scheduler.tests.integrated.base import SchedulerIntegratedTest
from mars.actors.core import new_client
from mars.scheduler.graph import GraphState

logger = logging.getLogger(__name__)


@unittest.skipIf(sys.platform == 'win32', "plasma don't support windows")
class Test(SchedulerIntegratedTest):
    def testCommonOperandFailover(self):
        delay_file = self.add_state_file('OP_DELAY_STATE_FILE')
        open(delay_file, 'w').close()

        terminate_file = self.add_state_file('OP_TERMINATE_STATE_FILE')

        self.start_processes(modules=['mars.scheduler.tests.integrated.op_delayer'], log_worker=True)

        session_id = uuid.uuid1()
        actor_client = new_client()
        session_ref = actor_client.actor_ref(self.session_manager_ref.create_session(session_id))

        np_a = np.random.random((100, 100))
        np_b = np.random.random((100, 100))

        a = mt.array(np_a, chunk_size=30) * 2 + 1
        b = mt.array(np_b, chunk_size=30) * 2 + 1
        c = a.dot(b) * 2 + 1
        graph = c.build_graph()
        targets = [c.key]
        graph_key = uuid.uuid1()
        session_ref.submit_tileable_graph(
            json.dumps(graph.to_json()), graph_key, target_tileables=targets)

        while not os.path.exists(terminate_file):
            actor_client.sleep(0.01)

        self.kill_process_tree(self.proc_workers[0])
        logger.warning('Worker %s KILLED!\n\n', self.proc_workers[0].pid)
        self.proc_workers = self.proc_workers[1:]
        os.unlink(delay_file)

        state = self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        result = session_ref.fetch_result(graph_key, c.key)
        expected = (np_a * 2 + 1).dot(np_b * 2 + 1) * 2 + 1
        assert_allclose(loads(result), expected)

    def testShuffleFailoverBeforeSuccStart(self):
        pred_finish_file = self.add_state_file('SHUFFLE_ALL_PRED_FINISHED_FILE')
        succ_start_file = self.add_state_file('SHUFFLE_START_SUCC_FILE')

        self.start_processes(modules=['mars.scheduler.tests.integrated.op_delayer'], log_worker=True)

        session_id = uuid.uuid1()
        actor_client = new_client()
        session_ref = actor_client.actor_ref(self.session_manager_ref.create_session(session_id))

        a = mt.ones((31, 27), chunk_size=10)
        b = a.reshape(27, 31)
        b.op.extra_params['_reshape_with_shuffle'] = True
        graph = b.build_graph()
        targets = [b.key]
        graph_key = uuid.uuid1()
        session_ref.submit_tileable_graph(json.dumps(graph.to_json()),
                                          graph_key, target_tileables=targets)
        actor_client.sleep(1)

        while not os.path.exists(pred_finish_file):
            actor_client.sleep(0.01)

        self.kill_process_tree(self.proc_workers[0])
        logger.warning('Worker %s KILLED!\n\n', self.proc_workers[0].pid)
        self.proc_workers = self.proc_workers[1:]
        open(succ_start_file, 'w').close()

        state = self.wait_for_termination(actor_client, session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        result = session_ref.fetch_result(graph_key, b.key)
        assert_allclose(loads(result), np.ones((27, 31)))
