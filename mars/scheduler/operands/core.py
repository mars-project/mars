# Copyright 1999-2019 Alibaba Group Holding Ltd.
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

from ...actors import ActorNotExist
from ...compat import Enum, BrokenPipeError, ConnectionRefusedError, TimeoutError  # pylint: disable=W0622
from ...errors import WorkerDead
from ...utils import classproperty


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


class OperandPosition(Enum):
    INITIAL = 'initial'
    TERMINAL = 'terminal'


@contextlib.contextmanager
def rewrite_worker_errors(ignore_error=False):
    rewrite = False
    try:
        yield
    except (BrokenPipeError, ConnectionRefusedError, ActorNotExist, TimeoutError):
        # we don't raise here, as we do not want
        # the actual stack be dumped
        rewrite = not ignore_error
    if rewrite:
        raise WorkerDead


_op_cls_to_actor = dict()


def get_operand_actor_class(op_cls):
    if op_cls in _op_cls_to_actor:
        return _op_cls_to_actor[op_cls]
    for cls in op_cls.__mro__:
        if cls in _op_cls_to_actor:
            actor_cls = _op_cls_to_actor[op_cls] = _op_cls_to_actor[cls]
            return actor_cls
    raise ValueError('Operand type %s not supported.' % op_cls.__name__)  # pragma: no cover


def register_operand_class(op_cls, actor_cls):
    _op_cls_to_actor[op_cls] = actor_cls
