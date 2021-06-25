# Copyright 1999-2021 Alibaba Group Holding Ltd.
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


class SvcActor2(mo.Actor):
    def __init__(self, arg, ref_uid):
        super().__init__()
        self._arg = arg
        self._ref_uid = ref_uid
        self._ref = None

    async def __post_create__(self):
        self._ref = await mo.actor_ref(self._ref_uid, address=self.address)

    async def get_arg(self):
        return await self._ref.get_arg() + ':' + self._arg


async def start(config: dict, address: str = None):
    svc_config = config['test_svc2']
    await mo.create_actor(
        SvcActor2, uid=svc_config['uid'], arg=svc_config['arg2'],
        ref_uid=svc_config['ref'], address=address)
