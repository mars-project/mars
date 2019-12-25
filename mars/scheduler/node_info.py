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

import logging
import os

from ..actors import FunctionActor
from ..node_info import gather_node_info

logger = logging.getLogger(__name__)


class NodeInfoActor(FunctionActor):
    def __init__(self):
        super().__init__()
        self._node_info = None

    @classmethod
    def default_uid(cls):
        return 's:h1:' + cls.__name__

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        self.ref().gather_info()

    def gather_info(self):
        self._node_info = gather_node_info()
        self.ref().gather_info(_tell=True, _delay=1)

    def get_info(self):
        return self._node_info
