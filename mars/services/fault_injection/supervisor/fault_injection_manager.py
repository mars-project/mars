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

from typing import Dict

from .... import oscar as mo

_option_keys = {'fault_count'}


class FaultInjectionManagerActor(mo.Actor):
    def __init__(self):
        self._options = {}
        self._fault_count = 0

    def set_options(self, options: Dict):
        invalid_keys = options.keys() - _option_keys
        if invalid_keys:
            raise ValueError(f"options has invalid keys: {invalid_keys}")
        self._options = options
        self._fault_count = options.get('fault_count', 0)

    def on_execute_operand(self):
        if self._fault_count > 0:
            self._fault_count -= 1
            return True
        return False
