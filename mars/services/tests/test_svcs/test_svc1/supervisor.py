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

import mars.oscar as mo


class SvcActor1(mo.Actor):
    def __init__(self, arg):
        super().__init__()
        self._arg = arg

    def get_arg(self):
        return self._arg


async def start(config: dict, address: None):
    svc_config = config['test_svc1']
    await mo.create_actor(
        SvcActor1, uid=svc_config['uid'], arg=svc_config['arg1'],
        address=address)
