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

import time
from dataclasses import dataclass, field
from typing import Dict

from ..core import NodeRole


@dataclass
class NodeInfo:
    role: NodeRole
    update_time: float = field(default_factory=time.time)
    env: Dict = field(default_factory=dict)
    resource: Dict = field(default_factory=dict)
    state: Dict = field(default_factory=dict)
