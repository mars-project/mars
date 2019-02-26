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

from mars.scheduler.operand import OperandActor, OperandPosition

_old_on_running = OperandActor._on_running
_old_on_finished = OperandActor._on_finished


def _on_running(self):
    _old_on_running(self)
    if 'DELAY_STATE_FILE' in os.environ and os.path.exists(os.environ['DELAY_STATE_FILE']):
        self.ctx.sleep(1)


def _on_finished(self):
    _old_on_finished(self)
    if self._position == OperandPosition.TERMINAL and 'TERMINATE_STATE_FILE' in os.environ:
        try:
            open(os.environ['TERMINATE_STATE_FILE'], 'w').close()
        except OSError:
            pass


OperandActor._on_running = _on_running
OperandActor._on_finished = _on_finished
