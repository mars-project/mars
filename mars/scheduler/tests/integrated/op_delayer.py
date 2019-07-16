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

from mars.scheduler.operands import OperandActor, OperandPosition, ShuffleProxyActor
from mars.actors.core import register_actor_implementation


class DelayedOperandActor(OperandActor):
    def _on_ready(self):
        if 'OP_DELAY_STATE_FILE' in os.environ and os.path.exists(os.environ['OP_DELAY_STATE_FILE']):
            self.ctx.sleep(1)
        if 'SHUFFLE_ALL_PRED_FINISHED_FILE' in os.environ and \
                os.path.exists(os.environ['SHUFFLE_ALL_PRED_FINISHED_FILE']):
            self.ctx.sleep(1)

        if 'SHUFFLE_HAS_SUCC_FINISH_FILE' in os.environ:
            while os.path.exists(os.environ['SHUFFLE_HAS_SUCC_FINISH_FILE']):
                self.ctx.sleep(1)
        super(DelayedOperandActor, self)._on_ready()

    def _on_finished(self):
        super(DelayedOperandActor, self)._on_finished()
        if self._position == OperandPosition.TERMINAL and 'OP_TERMINATE_STATE_FILE' in os.environ:
            try:
                open(os.environ['OP_TERMINATE_STATE_FILE'], 'w').close()
            except OSError:
                pass


class DelayedShuffleProxyActor(ShuffleProxyActor):
    def _start_successors(self):
        if 'SHUFFLE_ALL_PRED_FINISHED_FILE' in os.environ:
            try:
                open(os.environ['SHUFFLE_ALL_PRED_FINISHED_FILE'], 'w').close()
            except OSError:
                pass
        if 'SHUFFLE_START_SUCC_FILE' in os.environ:
            while not os.path.exists(os.environ['SHUFFLE_START_SUCC_FILE']):
                self.ctx.sleep(1)
        super(DelayedShuffleProxyActor, self)._start_successors()

    def add_finished_successor(self, op_key, worker):
        r = super(DelayedShuffleProxyActor, self).add_finished_successor(op_key, worker)
        if 'SHUFFLE_HAS_SUCC_FINISH_FILE' in os.environ:
            open(os.environ['SHUFFLE_HAS_SUCC_FINISH_FILE'], 'w').close()
            os.environ.pop('SHUFFLE_HAS_SUCC_FINISH_FILE')
        return r


register_actor_implementation(OperandActor, DelayedOperandActor)
register_actor_implementation(ShuffleProxyActor, DelayedShuffleProxyActor)
