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

import contextlib
import logging
from enum import Enum

from ...actors import ActorNotExist
from ...errors import WorkerDead
from ...utils import classproperty, TypeDispatcher

logger = logging.getLogger(__name__)


class OperandState(Enum):
    __order__ = 'UNSCHEDULED READY RUNNING FINISHED CACHED FREED FATAL CANCELLING CANCELLED'

    UNSCHEDULED = 'unscheduled'
    READY = 'ready'
    RUNNING = 'running'
    FINISHED = 'finished'
    CACHED = 'cached'
    FREED = 'freed'
    FATAL = 'fatal'
    CANCELLING = 'cancelling'
    CANCELLED = 'cancelled'

    @classproperty
    def STORED_STATES(self):
        """
        States on which the data of the operand is stored
        """
        return self.FINISHED, self.CACHED

    @classproperty
    def SUCCESSFUL_STATES(self):
        """
        States on which the operand is executed successfully
        """
        return self.FINISHED, self.CACHED, self.FREED

    @classproperty
    def TERMINATED_STATES(self):
        """
        States on which the operand has already terminated
        """
        return self.FINISHED, self.CACHED, self.FREED, self.FATAL, self.CANCELLED


@contextlib.contextmanager
def rewrite_worker_errors(ignore_error=False):
    rewrite = False
    try:
        yield
    except (BrokenPipeError, ConnectionRefusedError, ActorNotExist):
        # we don't raise here, as we do not want
        # the actual stack be dumped
        logger.exception('Worker connection error detected, will raise WorkerDead')
        rewrite = not ignore_error
    if rewrite:
        raise WorkerDead


op_dispatcher = TypeDispatcher()
register_operand_class = op_dispatcher.register
get_operand_actor_class = op_dispatcher.get_handler
