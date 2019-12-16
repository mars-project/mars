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

import array
from enum import Enum

from ..cluster_info import ClusterInfoActor, HasClusterInfoActor
from ..utils import classproperty
from ..promise import PromiseActor


class GraphState(Enum):
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


class CombinedFutureWaiter(object):
    def __init__(self, futures):
        self._futures = futures

    def result(self):
        return [f.result() for f in self._futures]


def array_to_bytes(typecode, initializer):
    return array.array(typecode, initializer).tobytes()
