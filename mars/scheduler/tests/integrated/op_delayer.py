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

import os

from mars.scheduler.operands import OperandActor, ShuffleProxyActor
from mars.actors.core import register_actor_implementation


def _write_state_file(var_name):
    if var_name in os.environ:
        try:
            open(os.environ[var_name], 'w').close()
        except OSError:
            pass
        os.environ.pop(var_name)


class DelayedOperandActor(OperandActor):
    def _on_ready(self):
        for ctrl_file_var in ('OP_DELAY_STATE_FILE', 'SHUFFLE_ALL_PRED_FINISHED_FILE',
                              'SHUFFLE_ALL_SUCC_FINISH_FILE'):
            if ctrl_file_var in os.environ and os.path.exists(os.environ[ctrl_file_var]):
                self.ctx.sleep(1)

        while 'SHUFFLE_HAS_SUCC_FINISH_FILE' in os.environ and \
                os.path.exists(os.environ['SHUFFLE_HAS_SUCC_FINISH_FILE']):
            self.ctx.sleep(0.1)

        super()._on_ready()

    def _on_finished(self):
        super()._on_finished()
        if self._is_terminal:
            _write_state_file('OP_TERMINATE_STATE_FILE')


class DelayedShuffleProxyActor(ShuffleProxyActor):
    def _start_successors(self):
        _write_state_file('SHUFFLE_ALL_PRED_FINISHED_FILE')

        if 'SHUFFLE_START_SUCC_FILE' in os.environ:
            while not os.path.exists(os.environ['SHUFFLE_START_SUCC_FILE']):
                self.ctx.sleep(0.1)

        super()._start_successors()

    def add_finished_successor(self, op_key, worker):
        try:
            return super().add_finished_successor(op_key, worker)
        finally:
            _write_state_file('SHUFFLE_HAS_SUCC_FINISH_FILE')

    def free_predecessors(self):
        _write_state_file('SHUFFLE_ALL_SUCC_FINISH_FILE')
        super().free_predecessors()


register_actor_implementation(OperandActor, DelayedOperandActor)
register_actor_implementation(ShuffleProxyActor, DelayedShuffleProxyActor)
