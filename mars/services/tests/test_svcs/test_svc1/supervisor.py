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

from ..... import oscar as mo
from ....core import AbstractService


class SvcActor1(mo.Actor):
    def __init__(self, arg):
        super().__init__()
        self._arg = arg

    def get_arg(self):
        return self._arg


class SvcSessionActor1(mo.Actor):
    @classmethod
    def gen_uid(cls, session_id: str):
        return f"{session_id}_svc_session_actor1"


class TestService1(AbstractService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def start(self):
        svc_config = self._config["test_svc1"]
        await mo.create_actor(
            SvcActor1,
            uid=svc_config["uid"],
            arg=svc_config["arg1"],
            address=self._address,
        )

    async def stop(self):
        svc_config = self._config["test_svc1"]
        await mo.destroy_actor(
            mo.create_actor_ref(uid=svc_config["uid"], address=self._address)
        )

    async def create_session(self, session_id: str):
        await mo.create_actor(
            SvcSessionActor1,
            uid=SvcSessionActor1.gen_uid(session_id),
            address=self._address,
        )

    async def destroy_session(self, session_id: str):
        await mo.destroy_actor(
            mo.create_actor_ref(
                uid=SvcSessionActor1.gen_uid(session_id), address=self._address
            )
        )
