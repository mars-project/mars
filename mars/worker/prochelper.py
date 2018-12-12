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

import logging

from .utils import WorkerActor

logger = logging.getLogger(__name__)


class ProcessHelperActor(WorkerActor):
    """
    Actor handling utils on every process
    """
    def __init__(self):
        super(ProcessHelperActor, self).__init__()
        self._dispatch_ref = None

    def post_create(self):
        from .dispatcher import DispatchActor

        super(ProcessHelperActor, self).post_create()
        self._dispatch_ref = self.promise_ref(DispatchActor.default_name())
        self._dispatch_ref.register_free_slot(self.uid, 'process_helper')

    def free_mkl_buffers(self):
        """
        Free MKL buffer
        """
        from ..lib.mkl_interface import mkl_free_buffers
        if mkl_free_buffers is None:
            return
        mkl_free_buffers()
