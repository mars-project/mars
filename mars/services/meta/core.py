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

from ... import oscar as mo
from .store import get_meta_store


class MetaStoreActor(mo.Actor):
    def __init__(self,
                 meta_store_name: str,
                 session_id: str,
                 **meta_store_kwargs):
        meta_store_type = get_meta_store(meta_store_name)
        self._store = meta_store_type(session_id, **meta_store_kwargs)

    @staticmethod
    def gen_uid(session_id):
        return f'{session_id}_meta'

    def __getattr__(self, attr):
        return getattr(self._store, attr)
