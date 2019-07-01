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

import array

from .. import compat
from ..compat import six
from ..cluster_info import ClusterInfoActor, HasClusterInfoActor
from ..utils import classproperty
from ..promise import PromiseActor


class OperandState(compat.Enum):
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


class GraphState(compat.Enum):
    UNSCHEDULED = 'unscheduled'
    PREPARING = 'preparing'
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    CANCELLING = 'cancelling'
    CANCELLED = 'cancelled'

    @classproperty
    def TERMINATED_STATES(self):
        """
        States on which the graph has already terminated
        """
        return self.SUCCEEDED, self.FAILED


class SchedulerClusterInfoActor(ClusterInfoActor):
    @classmethod
    def default_uid(cls):
        return 's:h1:%s' % cls.__name__


class SchedulerHasClusterInfoActor(HasClusterInfoActor):
    cluster_info_uid = SchedulerClusterInfoActor.default_uid()


class SchedulerActor(SchedulerHasClusterInfoActor, PromiseActor):
    @classmethod
    def default_uid(cls):
        return 's:h1:{0}'.format(cls.__name__)

    @property
    def chunk_meta(self):
        try:
            return self._chunk_meta_client
        except AttributeError:
            from .chunkmeta import ChunkMetaClient
            self._chunk_meta_client = ChunkMetaClient(self.ctx, self._cluster_info_ref)
            return self._chunk_meta_client


def remove_shuffle_chunks(chunks):
    return list(n for n in chunks if not getattr(n.op, '_shuffle_source', False))


if six.PY3:
    def array_to_bytes(typecode, initializer):
        return array.array(typecode, initializer).tobytes()
else:
    def array_to_bytes(typecode, initializer):
        return array.array(typecode, initializer).tostring()
