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

from ..compat import six, Enum
from ..cluster_info import HasClusterInfoActor
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


class SchedulerActor(HasClusterInfoActor, PromiseActor):
    def __init__(self, **kwargs):
        super(SchedulerActor, self).__init__()

        if 'balancer' in kwargs:
            self._balancer_ref = kwargs['balancer']

    @classmethod
    def default_name(cls):
        return 's:{0}'.format(cls.__name__)


if six.PY3:
    def array_to_bytes(typecode, initializer):
        return array.array(typecode, initializer).tobytes()
else:
    def array_to_bytes(typecode, initializer):
        return array.array(typecode, initializer).tostring()
