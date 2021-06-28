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

import os.path
from collections import defaultdict
from typing import Dict, Tuple

from .... import oscar as mo


class CustomLogMetaActor(mo.Actor):
    # {tileable_op_key -> {chunk_op_key -> (worker_addr, path)}}
    _custom_log_path_store: Dict[str, Dict[str, Tuple[str, str]]]

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._custom_log_path_store = dict()

    @classmethod
    def gen_uid(cls, session_id: str):
        return f'custom_log_{session_id}'

    async def __post_create__(self):
        from ..worker.custom_log import CustomLogActor

        worker_address_to_paths = defaultdict(set)
        for address, path in self._custom_log_path_store.values():
            log_dir = os.path.dirname(path)
            worker_address_to_paths[address].add(log_dir)
        for address, paths in worker_address_to_paths.items():
            ref = await mo.actor_ref(address, CustomLogActor.default_uid())
            await ref.clear_custom_log_dirs(list(paths))

    def register_custom_log_path(self,
                                 tileable_op_key: str,
                                 chunk_op_key: str,
                                 worker_address: str,
                                 log_path: str):
        if tileable_op_key not in self._custom_log_path_store:
            self._custom_log_path_store[tileable_op_key] = dict()
        self._custom_log_path_store[tileable_op_key][chunk_op_key] = \
            (worker_address, log_path)

    def get_tileable_op_log_paths(self,
                                  tileable_op_key: str) -> Dict[str, Tuple[str, str]]:
        return self._custom_log_path_store.get(tileable_op_key)
